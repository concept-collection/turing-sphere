/*
 * The same run as `npm run bench`, on upstream SHTNS' own CUDA transforms.
 *
 *   ./shtbench_gpu --preset schnak-spots --lmax 63 --steps 2000
 *   ./shtbench_gpu --mode transform --lmax 63 --steps 2000
 *
 * This is the like-for-like comparison the WGSL transforms exist to be measured
 * against: single precision, everything resident on the GPU, nothing read back
 * inside the loop. The only difference between this and `npm run bench` is what
 * runs the transforms — SHTNS' hand-written CUDA kernels and cuFFT/VkFFT here,
 * generated WGSL and a WGSL FFT there — on the same device.
 *
 * It keeps the state on the GPU the same way the WebGPU side does: cu_* are the
 * on-device entry points, asynchronous on SHTNS' compute stream, and a batch of
 * steps is launched before anything is waited for. `--batch` matches
 * `npm run bench --batch`, so both sides can be made to synchronize equally
 * often.
 *
 * Two things are measured, selected by --mode:
 *
 *  - solver:    one IMEX Euler timestep of models/<key>.m — 2*nspecies
 *               transforms, one reaction kernel on the grid, one spectral
 *               update kernel.
 *  - transform: one spectral -> grid -> spectral round trip and nothing else.
 *
 * Two SHTNS details worth knowing when reading the numbers:
 *
 *  - in fp32 mode SHTNS runs the *Legendre recurrence* in fp64 when
 *    lmax <= 128 and the GPU has usable fp64 (SHT_L_RESCALE_FLY_FLOAT in
 *    sht_private.h), which WebGPU cannot do at all. Set
 *    SHTNS_GPU_REC_PREC=1 to force the recurrence into fp32 and get the closer
 *    comparison; the run prints which it got.
 *  - the spatial layout defaults to theta-contiguous, which is SHTNS' native
 *    and fastest. --layout phi matches what the WGSL side uses. The spectral
 *    layout and normalization are identical either way, so a state comparison
 *    is valid in both.
 */
#include "spec.h"

#include <cuda_runtime.h>
#include <shtns.h>
#include <shtns_cuda.h>

static const char *USAGE =
    "usage: ./shtbench_gpu [options]\n"
    "\n"
    "  --mode solver|transform  what to measure (default solver)\n"
    "  --preset <key>           schnak-spots | schnak-coarse | schnak-fine | brussel |\n"
    "                             allencahn  (default schnak-spots)\n"
    "  --lmax <n>               spherical harmonic truncation (default 63)\n"
    "  --steps <n>              timed steps, or round trips (default 2000)\n"
    "  --warmup <n>             untimed steps first (default 100)\n"
    "  --batch <n>              steps launched per synchronization (default 16), as in\n"
    "                             `npm run bench -- --batch`\n"
    "  --seed <n>               seed of the initial noise / spectrum (default 1)\n"
    "  --fp64                   use SHTNS' double-precision GPU transforms instead of\n"
    "                             single. Not comparable to WebGPU, which has no fp64;\n"
    "                             useful as an accuracy and cost reference\n"
    "  --layout theta|phi       spatial layout: theta-contiguous is SHTNS' native and\n"
    "                             fastest, phi-contiguous is what the WGSL side uses\n"
    "                             (default theta)\n"
    "  --polar-eps <x>          SHTNS polar-optimization threshold (default 0)\n"
    "  --device <n>             CUDA device index (default 0)\n"
    "  --digest                 after timing, re-run exactly --steps steps from the seed\n"
    "                             and print a digest of the final spectral state\n"
    "  --dump-state <f>         like --digest, and write the state to <f> as JSON, for\n"
    "                             scripts/compare-native.mjs to diff\n"
    "  --<param> <v>            any parameter of the preset's model, e.g. --dt 0.05\n"
    "  --json                   machine-readable output\n"
    "  --help\n"
    "\n"
    "Run the same spec through the WGSL transforms with:\n"
    "  npm run bench -- --preset <key> --lmax <n> --steps <n>       (solver)\n"
    "  npm run bench:sht -- --lmax <n> --steps <n>                  (transform)";

#define CU_CHECK(call)                                                                     \
  do {                                                                                     \
    cudaError_t err_ = (call);                                                              \
    if (err_ != cudaSuccess) {                                                              \
      fprintf(stderr, "shtbench_gpu: %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
              cudaGetErrorString(err_));                                                    \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

/* ------------------------------------------------------------------- kernels
 *
 * The counterparts of the generated WGSL kernels: one thread per output
 * element, reading the same inputs and doing the same arithmetic (see
 * shtb_react / shtb_imex in spec.h, transcribed from the .m).
 */

template <typename real>
__global__ void k_react(shtb_step_const<real> c, const real *u, const real *v, real *r1, real *r2,
                        long npts) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npts) return;
  real a = 0, b = 0;
  shtb_react<real>(c, u[i], v ? v[i] : (real)0, &a, &b);
  r1[i] = a;
  if (r2) r2[i] = b;
}

/* Latitude lines are `stride` apart in the theta-contiguous layout, so the flat
 * kernel above would step over padding. This one is used when there is any. */
template <typename real>
__global__ void k_react_strided(shtb_step_const<real> c, const real *u, const real *v, real *r1,
                                real *r2, long linelen, long stride) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= linelen) return;
  const long o = (long)blockIdx.y * stride + i;
  real a = 0, b = 0;
  shtb_react<real>(c, u[o], v ? v[o] : (real)0, &a, &b);
  r1[o] = a;
  if (r2) r2[o] = b;
}

/* One thread per real, over the 2 x nlm spectral layout — `lam` carries l(l+1)
 * duplicated across the real and imaginary halves, exactly as on the WGSL side. */
template <typename real>
__global__ void k_imex(shtb_step_const<real> c, int k, real *U, const real *R, const real *lam,
                       long n2) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  U[i] = shtb_imex<real>(c, k, U[i], R[i], lam[i]);
}

/* Scatter the seeded perturbation, which is generated in [ilat*nphi + iphi]
 * order, into whichever layout SHTNS is using. Also used to fill the uniform
 * background. */
template <typename real>
__global__ void k_fill(real *f, real value, const float *noise, long nlat, long nphi,
                       long stride_lat, long stride_phi) {
  const long ilat = (long)blockIdx.y;
  const long iphi = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (iphi >= nphi || ilat >= nlat) return;
  const real n = noise ? (real)noise[ilat * nphi + iphi] : (real)0;
  f[ilat * stride_lat + iphi * stride_phi] = value + n;
}

/* --------------------------------------------------------------------- setup */

/* SHTNS' own device-buffer sizes, from init_cuda_buffer_fft() in sht_gpu.cu.
 * Its kernels write a little past nlm ("one more data per m") and the Fourier
 * stage needs room for the R2C form, so allocating exactly nlm or nlat*nphi is
 * not enough. WARPSZE is 32 on every CUDA GPU SHTNS supports. */
static long spec_alloc_reals(long nlm, int mmax) {
  const long nlm2 = nlm + (mmax + 1);
  return ((2 * nlm2 + 31) / 32) * 32;
}
static long spat_alloc_reals(long nlat_padded, long nphi, int mmax) {
  const long extra = (nphi / 2 == mmax) ? 1 : 0;
  return ((nlat_padded * (nphi + extra) + 31) / 32) * 32;
}

struct Layout {
  long nlat, nphi, nlat_padded;
  long stride_lat, stride_phi;
  long linelen, nlines, stride; /* contiguous runs, for the reaction kernel */
  int padded;
};

static Layout layout_of(shtns_cfg sht, int which) {
  Layout l;
  l.nlat = sht->nlat;
  l.nphi = sht->nphi;
  l.nlat_padded = sht->nlat_padded;
  if (which == SHTB_LAYOUT_PHI) {
    l.stride_lat = l.nphi;
    l.stride_phi = 1;
    l.nlines = l.nlat;
    l.linelen = l.nphi;
    l.stride = l.nphi;
    l.padded = 0;
  } else {
    l.stride_lat = 1;
    l.stride_phi = l.nlat_padded;
    l.nlines = l.nphi;
    l.linelen = l.nlat;
    l.stride = l.nlat_padded;
    l.padded = l.nlat_padded != l.nlat;
  }
  return l;
}

/* --------------------------------------------------------------- the run body
 *
 * Templated on the transform precision so fp32 and fp64 are the same code. The
 * fp32 instantiation is the one that matters; fp64 is there as a reference.
 */
template <typename real>
struct Run {
  shtns_cfg sht;
  const shtb_spec *spec;
  /* The stream SHTNS was told to compute on, so our kernels are ordered against
   * its transforms and one synchronization waits for the whole step. */
  cudaStream_t stream;
  Layout lay;
  long nlm, n2, npts;
  int nsp;
  shtb_step_const<real> c;
  double base[2];

  real *dU[2], *dR[2], *dspat[2], *drspat[2], *dlam;
  real *dTq[2]; /* transform mode ping-pong */
  int tcur;
  float *dnoise;
  float *hnoise;
  float *hstate;

  void sync() { CU_CHECK(cudaStreamSynchronize(stream)); }

  void alloc() {
    const long spec_n = spec_alloc_reals(nlm, sht->mmax);
    long spat_n = spat_alloc_reals(lay.nlat_padded, lay.nphi, sht->mmax);
    if ((long)sht->nspat > spat_n) spat_n = (long)sht->nspat;
    for (int k = 0; k < nsp; k++) {
      CU_CHECK(cudaMalloc(&dU[k], sizeof(real) * (size_t)spec_n));
      CU_CHECK(cudaMalloc(&dR[k], sizeof(real) * (size_t)spec_n));
      CU_CHECK(cudaMalloc(&dspat[k], sizeof(real) * (size_t)spat_n));
      CU_CHECK(cudaMalloc(&drspat[k], sizeof(real) * (size_t)spat_n));
      CU_CHECK(cudaMemset(dU[k], 0, sizeof(real) * (size_t)spec_n));
      CU_CHECK(cudaMemset(dR[k], 0, sizeof(real) * (size_t)spec_n));
      CU_CHECK(cudaMemset(dspat[k], 0, sizeof(real) * (size_t)spat_n));
      CU_CHECK(cudaMemset(drspat[k], 0, sizeof(real) * (size_t)spat_n));
    }
    dTq[0] = dU[0];
    dTq[1] = dR[0];
    tcur = 0;
    CU_CHECK(cudaMalloc(&dlam, sizeof(real) * (size_t)n2));
    CU_CHECK(cudaMalloc(&dnoise, sizeof(float) * (size_t)npts));
    hnoise = (float *)malloc(sizeof(float) * (size_t)npts);
    hstate = (float *)malloc(sizeof(float) * (size_t)(n2));

    real *lam = (real *)malloc(sizeof(real) * (size_t)n2);
    for (long lm = 0; lm < nlm; lm++) {
      const int l = sht->li[lm];
      lam[2 * lm] = lam[2 * lm + 1] = (real)(l * (l + 1));
    }
    CU_CHECK(cudaMemcpy(dlam, lam, sizeof(real) * (size_t)n2, cudaMemcpyHostToDevice));
    free(lam);
  }

  void free_all() {
    for (int k = 0; k < nsp; k++) {
      cudaFree(dU[k]);
      cudaFree(dR[k]);
      cudaFree(dspat[k]);
      cudaFree(drspat[k]);
    }
    cudaFree(dlam);
    cudaFree(dnoise);
    free(hnoise);
    free(hstate);
  }

  void synth(real *qlm, real *spat);
  void analys(real *spat, real *qlm);

  void seed_state() {
    shtb_seeded_noise(npts, spec->model->seed_amp, (uint32_t)spec->seed, hnoise);
    CU_CHECK(cudaMemcpy(dnoise, hnoise, sizeof(float) * (size_t)npts, cudaMemcpyHostToDevice));
    const int tpb = 256;
    dim3 grid((unsigned)((lay.nphi + tpb - 1) / tpb), (unsigned)lay.nlat);
    k_fill<real><<<grid, tpb, 0, stream>>>(dspat[0], (real)base[0], dnoise, lay.nlat, lay.nphi,
                                           lay.stride_lat, lay.stride_phi);
    analys(dspat[0], dU[0]);
    if (nsp > 1) {
      k_fill<real><<<grid, tpb, 0, stream>>>(dspat[1], (real)base[1], NULL, lay.nlat, lay.nphi,
                                             lay.stride_lat, lay.stride_phi);
      analys(dspat[1], dU[1]);
    }
    sync();
  }

  /* One IMEX Euler timestep. Nothing is synchronized: the calls queue on SHTNS'
   * compute stream, which is what makes a batch of steps one submission's worth
   * of work, as on the WebGPU side. */
  void step() {
    const int tpb = 256;
    for (int k = 0; k < nsp; k++) synth(dU[k], dspat[k]);
    if (lay.padded) {
      dim3 grid((unsigned)((lay.linelen + tpb - 1) / tpb), (unsigned)lay.nlines);
      k_react_strided<real><<<grid, tpb, 0, stream>>>(c, dspat[0], nsp > 1 ? dspat[1] : NULL,
                                                      drspat[0], nsp > 1 ? drspat[1] : NULL,
                                                      lay.linelen, lay.stride);
    } else {
      k_react<real><<<(unsigned)((npts + tpb - 1) / tpb), tpb, 0, stream>>>(
          c, dspat[0], nsp > 1 ? dspat[1] : NULL, drspat[0], nsp > 1 ? drspat[1] : NULL, npts);
    }
    for (int k = 0; k < nsp; k++) analys(drspat[k], dR[k]);
    for (int k = 0; k < nsp; k++)
      k_imex<real><<<(unsigned)((n2 + tpb - 1) / tpb), tpb, 0, stream>>>(c, k, dU[k], dR[k], dlam,
                                                                         n2);
  }

  void seed_spectrum() {
    shtb_seeded_spectrum(spec->lmax, spec->lmax, (uint32_t)spec->seed, hstate);
    tcur = 0;
    if (sizeof(real) == sizeof(float)) {
      CU_CHECK(cudaMemcpy(dTq[0], hstate, sizeof(float) * (size_t)n2, cudaMemcpyHostToDevice));
    } else {
      double *tmp = (double *)malloc(sizeof(double) * (size_t)n2);
      for (long i = 0; i < n2; i++) tmp[i] = (double)hstate[i];
      CU_CHECK(cudaMemcpy(dTq[0], tmp, sizeof(double) * (size_t)n2, cudaMemcpyHostToDevice));
      free(tmp);
    }
    sync();
  }

  void round_trip() {
    synth(dTq[tcur], dspat[0]);
    analys(dspat[0], dTq[tcur ^ 1]);
    tcur ^= 1;
  }

  /* the final spectral state of species 0 (or of the round trip), as float */
  const float *read_state() {
    sync();
    const real *src = spec->mode == SHTB_MODE_TRANSFORM ? dTq[tcur] : dU[0];
    if (sizeof(real) == sizeof(float)) {
      CU_CHECK(cudaMemcpy(hstate, src, sizeof(float) * (size_t)n2, cudaMemcpyDeviceToHost));
    } else {
      double *tmp = (double *)malloc(sizeof(double) * (size_t)n2);
      CU_CHECK(cudaMemcpy(tmp, src, sizeof(double) * (size_t)n2, cudaMemcpyDeviceToHost));
      for (long i = 0; i < n2; i++) hstate[i] = (float)tmp[i];
      free(tmp);
    }
    return hstate;
  }

  /* species 0 on the grid, for the range check */
  void read_field(double *mn, double *mx, int *finite) {
    for (int k = 0; k < nsp; k++) synth(dU[k], dspat[k]);
    sync();
    const long n = lay.nlines * lay.stride;
    real *h = (real *)malloc(sizeof(real) * (size_t)n);
    CU_CHECK(cudaMemcpy(h, dspat[0], sizeof(real) * (size_t)n, cudaMemcpyDeviceToHost));
    *mn = INFINITY;
    *mx = -INFINITY;
    *finite = 1;
    for (long l = 0; l < lay.nlines; l++)
      for (long i = 0; i < lay.linelen; i++) {
        const double x = (double)h[l * lay.stride + i];
        if (x < *mn) *mn = x;
        if (x > *mx) *mx = x;
        if (!isfinite(x)) *finite = 0;
      }
    free(h);
  }
};

template <>
void Run<float>::synth(float *qlm, float *spat) {
  cu_SH_to_spat_float(sht, (cplx_f *)qlm, spat, sht->lmax);
}
template <>
void Run<float>::analys(float *spat, float *qlm) {
  cu_spat_to_SH_float(sht, spat, (cplx_f *)qlm, sht->lmax);
}
template <>
void Run<double>::synth(double *qlm, double *spat) {
  cu_SH_to_spat(sht, (cplx *)qlm, spat, sht->lmax);
}
template <>
void Run<double>::analys(double *spat, double *qlm) {
  cu_spat_to_SH(sht, spat, (cplx *)qlm, sht->lmax);
}

/* ----------------------------------------------------------------------- main */

template <typename real>
static int run(shtns_cfg sht, cudaStream_t stream, const shtb_spec &spec, const char *adapter,
               const char *runtime, const char *cfg_info) {
  Run<real> r;
  r.sht = sht;
  r.spec = &spec;
  r.stream = stream;
  r.lay = layout_of(sht, spec.layout);
  r.nlm = (long)sht->nlm;
  r.n2 = 2 * r.nlm;
  r.npts = r.lay.nlat * r.lay.nphi;
  r.nsp = spec.model->nspecies;
  r.c = shtb_make_step_const<real>(spec.model, &spec.params);
  shtb_background(spec.model, &spec.params, r.base);
  r.alloc();

  const int transform_mode = spec.mode == SHTB_MODE_TRANSFORM;
  const char *precision = sizeof(real) == 4 ? "fp32" : "fp64";

  if (!spec.json) {
    printf("shtbench_gpu — upstream SHTNS on CUDA, %s only\n\n",
           transform_mode ? "transforms" : "solver");
    printf("  mode      %s\n", transform_mode
                                   ? "transform (one synth + one analys per step)"
                                   : "solver (one IMEX Euler timestep per step)");
    if (!transform_mode) {
      printf("  preset    %s  (models/%s.m: %d species)\n", spec.preset->label, spec.model->key,
             r.nsp);
      printf("  params    ");
      for (int i = 0; i < spec.model->nparams; i++)
        printf("%s=%g  ", spec.model->params[i].key,
               *shtb_field_c(&spec.params, spec.model->params[i].off));
      printf("\n");
    }
    printf("  grid      lmax %d · %ldx%ld · nlm %ld\n", spec.lmax, r.lay.nlat, r.lay.nphi, r.nlm);
    printf("  layout    %s%s\n",
           spec.layout == SHTB_LAYOUT_PHI ? "phi-contiguous" : "theta-contiguous (native)",
           r.lay.padded ? ", padded" : "");
    printf("  backend   %s\n            %s, %s\n            %s\n", adapter, precision, runtime,
           cfg_info ? cfg_info : "(no GPU config info)");
    printf("  run       %d warmup + %d timed steps in batches of %d, seed %d\n\n", spec.warmup,
           spec.steps, spec.batch, spec.seed);
  }

  if (transform_mode)
    r.seed_spectrum();
  else
    r.seed_state();

  for (int i = 0; i < spec.warmup; i++) transform_mode ? r.round_trip() : r.step();
  r.sync();

  /* --- throughput: a batch launched together, waited for once ------------- */
  const int batches = (spec.steps + spec.batch - 1) / spec.batch;
  double launch_ms = 0;
  int done = 0;
  const double t0 = shtb_now_ms();
  for (int b = 0; b < batches; b++) {
    const int n = spec.steps - done < spec.batch ? spec.steps - done : spec.batch;
    const double e0 = shtb_now_ms();
    for (int i = 0; i < n; i++) transform_mode ? r.round_trip() : r.step();
    launch_ms += shtb_now_ms() - e0;
    r.sync();
    done += n;
  }
  const double throughput_ms = (shtb_now_ms() - t0) / done;

  /* --- latency: one step per synchronization, for the distribution -------- */
  const int lat_steps = spec.steps < 200 ? spec.steps : 200;
  double *samples = (double *)malloc(sizeof(double) * (size_t)lat_steps);
  for (int i = 0; i < lat_steps; i++) {
    const double a = shtb_now_ms();
    transform_mode ? r.round_trip() : r.step();
    r.sync();
    samples[i] = shtb_now_ms() - a;
  }

  shtb_report rep;
  memset(&rep, 0, sizeof(rep));
  char libbuf[192], adapterbuf[160], runtimebuf[160];
  rep.library = shtb_json_safe(libbuf, sizeof(libbuf), shtns_get_build_info());
  rep.runtime = shtb_json_safe(runtimebuf, sizeof(runtimebuf), runtime);
  rep.adapter = shtb_json_safe(adapterbuf, sizeof(adapterbuf), adapter);
  rep.precision = precision;
  rep.fourier = "cufft/vkfft";
  rep.nlm = r.nlm;
  rep.ops_per_step = transform_mode ? 2 : 2 * r.nsp + 1 + r.nsp;
  rep.ms_per_step = throughput_ms;
  rep.encode_ms_per_step = launch_ms / done;
  rep.latency = shtb_stats(samples, lat_steps);
  rep.have_latency = 1;
  rep.steps_run = spec.warmup + spec.steps + lat_steps;
  rep.model_t = transform_mode ? 0 : rep.steps_run * spec.params.dt;

  if (transform_mode) {
    const float *s = r.read_state();
    rep.field_min = INFINITY;
    rep.field_max = -INFINITY;
    rep.finite = shtb_all_finite(s, r.n2);
    for (long i = 0; i < r.n2; i++) {
      if (s[i] < rep.field_min) rep.field_min = s[i];
      if (s[i] > rep.field_max) rep.field_max = s[i];
    }
  } else {
    r.read_field(&rep.field_min, &rep.field_max, &rep.finite);
  }

  /* A reproducible state to compare against a WebGPU run: exactly --steps steps
   * from the seed, separate from the timed runs above. */
  if (spec.digest) {
    if (transform_mode) {
      r.seed_spectrum();
      rep.input_digest = shtb_digest_of(r.hstate, r.n2);
      rep.have_input_digest = 1;
      for (int i = 0; i < spec.steps; i++) r.round_trip();
    } else {
      r.seed_state();
      for (int i = 0; i < spec.steps; i++) r.step();
    }
    rep.digest = shtb_digest_of(r.read_state(), r.n2);
    rep.have_digest = 1;
  }

  if (spec.json) {
    shtb_print_json(&spec, &rep);
  } else {
    printf("  %.3f ms/step   %.1f steps/s", rep.ms_per_step, 1000.0 / rep.ms_per_step);
    if (!transform_mode) printf("   %.2f model time/s", spec.params.dt * 1000.0 / rep.ms_per_step);
    printf("   (batches of %d)\n", spec.batch);
    printf("  of which CPU kernel launching: %.3f ms/step (%.0f%% — the rest is the GPU)\n",
           rep.encode_ms_per_step, 100.0 * rep.encode_ms_per_step / rep.ms_per_step);
    printf("  one step per sync: %.3f ms mean · median %.3f · p05 %.3f · p95 %.3f · min %.3f\n",
           rep.latency.mean_ms, rep.latency.median_ms, rep.latency.p05_ms, rep.latency.p95_ms,
           rep.latency.min_ms);
    if (transform_mode)
      printf("  i.e. %.3f ms per single transform\n", rep.ms_per_step / 2);
    else
      printf("  after %d steps: t = %.2f, field ∈ [%.4f, %.4f] (contrast %.4f)%s\n",
             rep.steps_run, rep.model_t, rep.field_min, rep.field_max,
             rep.field_max - rep.field_min, rep.finite ? "" : "  — NOT FINITE");
    if (rep.have_digest) {
      printf("\n  state after %d steps from seed %d:\n", spec.steps, spec.seed);
      printf("    n=%ld min=%.9g max=%.9g mean=%.9g rms=%.9g\n", rep.digest.n, rep.digest.min,
             rep.digest.max, rep.digest.mean, rep.digest.rms);
    }
    printf("\n  Compare with `npm run bench --json` on this machine: same GPU, same\n"
           "  precision, same grid — the difference is the transform implementation.\n"
           "  scripts/compare-native.mjs runs both and lines the numbers up.\n");
  }

  if (spec.dump_state && rep.have_digest) {
    if (shtb_dump_state(spec.dump_state, &spec, &rep, r.hstate, r.n2) != 0) {
      fprintf(stderr, "shtbench_gpu: cannot write %s\n", spec.dump_state);
      return 1;
    }
    if (!spec.json) printf("\n  wrote %s\n", spec.dump_state);
  }

  free(samples);
  r.free_all();
  return rep.finite ? 0 : 1;
}

int main(int argc, char **argv) {
  shtb_spec spec;
  double polar_eps = 0.0;
  int device = 0;

  /* --polar-eps and --device are ours, not part of the shared spec. */
  int argc2 = 0;
  char **argv2 = (char **)malloc(sizeof(char *) * (size_t)argc);
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--polar-eps") == 0 && i + 1 < argc) {
      polar_eps = atof(argv[++i]);
      continue;
    }
    if (strncmp(argv[i], "--polar-eps=", 12) == 0) {
      polar_eps = atof(argv[i] + 12);
      continue;
    }
    if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = atoi(argv[++i]);
      continue;
    }
    if (strncmp(argv[i], "--device=", 9) == 0) {
      device = atoi(argv[i] + 9);
      continue;
    }
    argv2[argc2++] = argv[i];
  }
  int rc = shtb_parse_spec(argc2, argv2, &spec, USAGE);
  free(argv2);
  if (rc) return rc == 1 ? 0 : rc;

  CU_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CU_CHECK(cudaGetDeviceProperties(&prop, device));
  char adapter[128];
  snprintf(adapter, sizeof(adapter), "%s (sm_%d%d, %d SMs)", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
  int rtv = 0, drv = 0;
  cudaRuntimeGetVersion(&rtv);
  cudaDriverGetVersion(&drv);
  char runtime[128];
  snprintf(runtime, sizeof(runtime), "CUDA runtime %d.%d, driver %d.%d", rtv / 1000,
           (rtv % 1000) / 10, drv / 1000, (drv % 1000) / 10);

  shtns_verbose(0);
  shtns_cfg sht = shtns_create(spec.lmax, spec.lmax, 1, sht_orthonormal);
  if (!sht) {
    fprintf(stderr, "shtbench_gpu: shtns_create failed\n");
    return 1;
  }
  /* Our reaction and update kernels have to be ordered against SHTNS'
   * transforms, so both go on one stream we own. cushtns_set_streams must come
   * before shtns_set_grid, which is where the GPU (and its FFT plan) is set up. */
  cudaStream_t stream = 0;
  CU_CHECK(cudaStreamCreate(&stream));
  cushtns_set_streams(sht, stream, 0);

  int flags = sht_gauss | SHT_ALLOW_GPU | SHT_SCALAR_ONLY;
  flags |= spec.layout == SHTB_LAYOUT_PHI ? SHT_PHI_CONTIGUOUS : SHT_THETA_CONTIGUOUS;
  if (spec.fp32) flags |= SHT_FP32;
  if (shtns_set_grid(sht, (enum shtns_type)flags, polar_eps, spec.nlat, spec.nphi) <= 0) {
    fprintf(stderr, "shtbench_gpu: shtns_set_grid failed for lmax %d on a %dx%d grid\n", spec.lmax,
            spec.nlat, spec.nphi);
    return 1;
  }
  if ((int)sht->nlat != spec.nlat || (int)sht->nphi != spec.nphi) {
    fprintf(stderr, "shtbench_gpu: SHTNS chose a %ux%u grid, not the %dx%d asked for\n", sht->nlat,
            sht->nphi, spec.nlat, spec.nphi);
    return 1;
  }

  /* cushtns_get_cfg_info() returns NULL when the GPU was never initialized, which
   * is how SHT_ALLOW_GPU failing shows up — and with SHT_FP32 the CPU fallback
   * would read fp64 out of fp32 buffers, so stop rather than produce a number
   * for the wrong thing. */
  const char *cfg_info = cushtns_get_cfg_info(sht);
  if (!cfg_info) {
    fprintf(stderr,
            "shtbench_gpu: SHTNS did not initialize the GPU for this grid (lmax %d, %dx%d).\n"
            "  Was it built with --enable-cuda, and does nlat %% 4 == 0 hold for the\n"
            "  theta-contiguous layout? Try --layout phi.\n",
            spec.lmax, spec.nlat, spec.nphi);
    return 1;
  }

  const int status = spec.fp32 ? run<float>(sht, stream, spec, adapter, runtime, cfg_info)
                               : run<double>(sht, stream, spec, adapter, runtime, cfg_info);
  shtns_destroy(sht);
  cudaStreamDestroy(stream);
  return status;
}
