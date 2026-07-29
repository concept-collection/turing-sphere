/*
 * The same run as `npm run bench`, on upstream SHTNS on the CPU (fp64).
 *
 *   ./shtbench --preset schnak-spots --lmax 63 --steps 2000
 *   ./shtbench --mode transform --lmax 63 --steps 2000
 *
 * Two things are measured, selected by --mode:
 *
 *  - solver:    the IMEX Euler timestep of models/<key>.m — 4 transforms, a
 *               reaction on the grid, and the spectral update — which is what
 *               the app's `solver` number and `npm run bench` measure.
 *  - transform: one spectral -> grid -> spectral round trip and nothing else,
 *               which is the library-against-library number. The solver does
 *               one of these per species per step.
 *
 * This is fp64 throughout, because SHTNS' single precision exists only on the
 * GPU. So it is not a like-for-like comparison against the fp32 WGSL
 * transforms — it is the accuracy reference (how far has fp32 drifted?) and the
 * "what does a well-optimized CPU do" reference. shtbench_gpu is the
 * like-for-like one.
 *
 * The reaction is a transcription of the .m rather than the .m itself; see
 * spec.h. It is checked, not trusted: `compare-native.mjs --check` diffs the
 * final state against a WebGPU run of the actual .m.
 */
#include "spec.h"

#include <shtns.h>

static const char *USAGE =
    "usage: ./shtbench [options]\n"
    "\n"
    "  --mode solver|transform  what to measure (default solver)\n"
    "  --preset <key>           schnak-spots | schnak-coarse | schnak-fine | brussel |\n"
    "                             allencahn  (default schnak-spots)\n"
    "  --lmax <n>               spherical harmonic truncation (default 63)\n"
    "  --steps <n>              timed steps, or round trips (default 2000)\n"
    "  --warmup <n>             untimed steps first (default 100)\n"
    "  --seed <n>               seed of the initial noise / spectrum (default 1)\n"
    "  --threads <n>            OpenMP threads (default: the library's choice)\n"
    "  --batch <n>              accepted for symmetry with shtbench_gpu and ignored:\n"
    "                             every CPU transform call is synchronous, so there is\n"
    "                             nothing to batch\n"
    "  --layout theta|phi       spatial layout: theta-contiguous is SHTNS' native and\n"
    "                             fastest, phi-contiguous is what the WGSL side uses\n"
    "                             (default theta)\n"
    "  --polar-eps <x>          SHTNS polar-optimization threshold. 0 disables it, which\n"
    "                             is what the WGSL transforms do; SHTNS' own default is\n"
    "                             1e-10 (default 0)\n"
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

/*
 * The layout arithmetic. SHTNS stores a spatial field either theta-contiguous
 * (its native layout) or phi-contiguous (what the WGSL side uses), possibly
 * with padding between lines. The reaction is pointwise so it does not care,
 * but the seeded perturbation is indexed by (ilat, iphi) and does.
 *
 * Either way the field is `nlines` contiguous runs of `linelen` doubles,
 * `stride` apart.
 */
struct Grid {
  long nlat, nphi, nlat_padded;
  int layout;
  long nlines, linelen, stride;
};

static Grid grid_of(shtns_cfg sht, int layout) {
  Grid g;
  g.nlat = sht->nlat;
  g.nphi = sht->nphi;
  g.nlat_padded = sht->nlat_padded;
  g.layout = layout;
  if (layout == SHTB_LAYOUT_PHI) {
    g.nlines = g.nlat;
    g.linelen = g.nphi;
    g.stride = g.nphi;
  } else {
    g.nlines = g.nphi;
    g.linelen = g.nlat;
    g.stride = g.nlat_padded;
  }
  return g;
}

static inline long spat_index(const Grid &g, long ilat, long iphi) {
  return g.layout == SHTB_LAYOUT_PHI ? ilat * g.nphi + iphi : iphi * g.nlat_padded + ilat;
}

static void fill_uniform(const Grid &g, double *f, double value) {
  for (long l = 0; l < g.nlines; l++)
    for (long i = 0; i < g.linelen; i++) f[l * g.stride + i] = value;
}

/* value + the seeded perturbation, which arrives in [ilat*nphi + iphi] order */
static void fill_perturbed(const Grid &g, double *f, double value, const float *noise) {
  for (long ilat = 0; ilat < g.nlat; ilat++)
    for (long iphi = 0; iphi < g.nphi; iphi++)
      f[spat_index(g, ilat, iphi)] = value + (double)noise[ilat * g.nphi + iphi];
}

static void react_field(const Grid &g, const shtb_step_const<double> &c, const double *u,
                        const double *v, double *r1, double *r2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long l = 0; l < g.nlines; l++) {
    const long o = l * g.stride;
    for (long i = 0; i < g.linelen; i++) {
      double a = 0, b = 0;
      shtb_react<double>(c, u[o + i], v ? v[o + i] : 0.0, &a, &b);
      r1[o + i] = a;
      if (r2) r2[o + i] = b;
    }
  }
}

static void imex_update(const shtb_step_const<double> &c, int nspecies, long nlm, double **U,
                        const double *const *R, const float *lam) {
  const long n2 = 2 * nlm;
  for (int k = 0; k < nspecies; k++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < n2; i++)
      U[k][i] = shtb_imex<double>(c, k, U[k][i], R[k][i], (double)lam[i]);
  }
}

static void field_range(const double *f, long nlines, long linelen, long stride, double *mn,
                        double *mx, int *finite) {
  *mn = INFINITY;
  *mx = -INFINITY;
  *finite = 1;
  for (long l = 0; l < nlines; l++)
    for (long i = 0; i < linelen; i++) {
      double x = f[l * stride + i];
      if (x < *mn) *mn = x;
      if (x > *mx) *mx = x;
      if (!isfinite(x)) *finite = 0;
    }
}

int main(int argc, char **argv) {
  shtb_spec spec;
  double polar_eps = 0.0;

  /* --polar-eps is ours, not part of the shared spec; take it out first. */
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
    argv2[argc2++] = argv[i];
  }
  int rc = shtb_parse_spec(argc2, argv2, &spec, USAGE);
  free(argv2);
  if (rc) return rc == 1 ? 0 : rc;
  /* Every CPU transform call is synchronous, so there is nothing to batch. */
  spec.batch = 1;

  const int quiet = spec.json;
  const int transform_mode = spec.mode == SHTB_MODE_TRANSFORM;
  shtns_verbose(0);
  const int threads = shtns_use_threads(spec.threads);

  shtns_cfg sht = shtns_create(spec.lmax, spec.lmax, 1, sht_orthonormal);
  if (!sht) {
    fprintf(stderr, "shtbench: shtns_create failed\n");
    return 1;
  }
  const int layout_flag =
      spec.layout == SHTB_LAYOUT_PHI ? SHT_PHI_CONTIGUOUS : SHT_THETA_CONTIGUOUS;
  if (shtns_set_grid(sht, (enum shtns_type)(sht_gauss | layout_flag | SHT_SCALAR_ONLY), polar_eps,
                     spec.nlat, spec.nphi) <= 0) {
    fprintf(stderr, "shtbench: shtns_set_grid failed for lmax %d on a %dx%d grid\n", spec.lmax,
            spec.nlat, spec.nphi);
    return 1;
  }
  if ((int)sht->nlat != spec.nlat || (int)sht->nphi != spec.nphi) {
    fprintf(stderr, "shtbench: SHTNS chose a %ux%u grid, not the %dx%d asked for\n", sht->nlat,
            sht->nphi, spec.nlat, spec.nphi);
    return 1;
  }

  const Grid g = grid_of(sht, spec.layout);
  const long nlm = (long)sht->nlm;
  const long nspat = (long)sht->nspat;
  const int nsp = spec.model->nspecies;

  /* Laplace-Beltrami eigenvalues, 2 x nlm with the value duplicated across the
   * real and imaginary halves — the layout eigenvalues() builds in
   * src/mgpu/model.ts. Held in float so both sides divide by the same number. */
  float *lam = (float *)malloc(sizeof(float) * (size_t)(2 * nlm));
  for (long lm = 0; lm < nlm; lm++) {
    const int l = sht->li[lm];
    lam[2 * lm] = lam[2 * lm + 1] = (float)(l * (l + 1));
  }

  double *spat[2] = {NULL, NULL};
  double *rspat[2] = {NULL, NULL};
  cplx *Q[2] = {NULL, NULL};
  cplx *R[2] = {NULL, NULL};
  for (int k = 0; k < nsp; k++) {
    spat[k] = (double *)shtns_malloc(sizeof(double) * (size_t)nspat);
    rspat[k] = (double *)shtns_malloc(sizeof(double) * (size_t)nspat);
    Q[k] = (cplx *)shtns_malloc(sizeof(cplx) * (size_t)nlm);
    R[k] = (cplx *)shtns_malloc(sizeof(cplx) * (size_t)nlm);
    memset(spat[k], 0, sizeof(double) * (size_t)nspat);
    memset(rspat[k], 0, sizeof(double) * (size_t)nspat);
    /* through the double view: a cplx array is [re, im] pairs, and memset on
     * std::complex itself is a non-trivial-type warning */
    memset((double *)Q[k], 0, sizeof(double) * (size_t)(2 * nlm));
    memset((double *)R[k], 0, sizeof(double) * (size_t)(2 * nlm));
  }
  float *noise = (float *)malloc(sizeof(float) * (size_t)(g.nlat * g.nphi));
  float *state32 = (float *)malloc(sizeof(float) * (size_t)(2 * nlm));

  const shtb_step_const<double> c = shtb_make_step_const<double>(spec.model, &spec.params);
  double base[2];
  shtb_background(spec.model, &spec.params, base);

  /* --- solver: init and one timestep, from models/<key>.m ------------------ */
  auto seed_state = [&]() {
    shtb_seeded_noise(g.nlat * g.nphi, spec.model->seed_amp, (uint32_t)spec.seed, noise);
    fill_perturbed(g, spat[0], base[0], noise);
    spat_to_SH(sht, spat[0], Q[0]);
    if (nsp > 1) {
      fill_uniform(g, spat[1], base[1]);
      spat_to_SH(sht, spat[1], Q[1]);
    }
  };
  auto step = [&]() {
    for (int k = 0; k < nsp; k++) SH_to_spat(sht, Q[k], spat[k]);
    react_field(g, c, spat[0], nsp > 1 ? spat[1] : NULL, rspat[0], nsp > 1 ? rspat[1] : NULL);
    for (int k = 0; k < nsp; k++) spat_to_SH(sht, rspat[k], R[k]);
    double *Ud[2] = {(double *)Q[0], (double *)Q[1]};
    const double *Rd[2] = {(const double *)R[0], (const double *)R[1]};
    imex_update(c, nsp, nlm, Ud, Rd, lam);
  };

  /* --- transform: one synth + one analys, ping-ponging the two buffers ----- */
  cplx *tq[2] = {Q[0], R[0]};
  int tcur = 0;
  auto seed_spectrum = [&]() {
    shtb_seeded_spectrum(spec.lmax, spec.lmax, (uint32_t)spec.seed, state32);
    tcur = 0;
    double *q = (double *)tq[0];
    for (long i = 0; i < 2 * nlm; i++) q[i] = (double)state32[i];
  };
  auto round_trip = [&]() {
    SH_to_spat(sht, tq[tcur], spat[0]);
    spat_to_SH(sht, spat[0], tq[tcur ^ 1]);
    tcur ^= 1;
  };

  if (transform_mode)
    seed_spectrum();
  else
    seed_state();

  if (!quiet) {
    printf("shtbench — upstream SHTNS on the CPU, %s only\n\n",
           transform_mode ? "transforms" : "solver");
    printf("  mode      %s\n", transform_mode
                                   ? "transform (one synth + one analys per step)"
                                   : "solver (one IMEX Euler timestep per step)");
    if (!transform_mode) {
      printf("  preset    %s  (models/%s.m: %d species)\n", spec.preset->label, spec.model->key,
             nsp);
      printf("  params    ");
      for (int i = 0; i < spec.model->nparams; i++)
        printf("%s=%g  ", spec.model->params[i].key,
               *shtb_field_c(&spec.params, spec.model->params[i].off));
      printf("\n");
    }
    printf("  grid      lmax %d · %ldx%ld · nlm %ld\n", spec.lmax, g.nlat, g.nphi, nlm);
    printf("  layout    %s%s\n",
           spec.layout == SHTB_LAYOUT_PHI ? "phi-contiguous" : "theta-contiguous (native)",
           (long)sht->nlat_padded != g.nlat ? ", padded" : "");
    printf("  backend   %s\n            fp64, %d thread%s, polar opt %g\n",
           shtns_get_build_info(), threads, threads == 1 ? "" : "s", polar_eps);
    printf("  run       %d warmup + %d timed steps, seed %d\n\n", spec.warmup, spec.steps,
           spec.seed);
  }

  for (int i = 0; i < spec.warmup; i++) {
    if (transform_mode)
      round_trip();
    else
      step();
  }

  double *samples = (double *)malloc(sizeof(double) * (size_t)spec.steps);
  const double t0 = shtb_now_ms();
  for (int i = 0; i < spec.steps; i++) {
    const double a = shtb_now_ms();
    if (transform_mode)
      round_trip();
    else
      step();
    samples[i] = shtb_now_ms() - a;
  }
  const double total = shtb_now_ms() - t0;

  shtb_report rep;
  memset(&rep, 0, sizeof(rep));
  char libbuf[192], adapterbuf[64];
  rep.library = shtb_json_safe(libbuf, sizeof(libbuf), shtns_get_build_info());
  rep.runtime = "cpu";
  snprintf(adapterbuf, sizeof(adapterbuf), "CPU, %d thread%s", threads, threads == 1 ? "" : "s");
  rep.adapter = adapterbuf;
  rep.precision = "fp64";
  rep.fourier = "fftw";
  rep.nlm = nlm;
  rep.ops_per_step = transform_mode ? 2 : 2 * nsp + 2;
  rep.ms_per_step = total / spec.steps;
  rep.encode_ms_per_step = 0; /* nothing is deferred: every call is synchronous */
  rep.latency = shtb_stats(samples, spec.steps);
  rep.have_latency = 1;
  /* the range reported below is the state as it stands now: warmup included */
  rep.steps_run = spec.warmup + spec.steps;
  rep.model_t = transform_mode ? 0 : rep.steps_run * spec.params.dt;

  /* Did the run stay finite and develop contrast? The app shows the same range
   * for the first species under its stats line. */
  if (transform_mode) {
    field_range((const double *)tq[tcur], 1, 2 * nlm, 0, &rep.field_min, &rep.field_max,
                &rep.finite);
  } else {
    for (int k = 0; k < nsp; k++) SH_to_spat(sht, Q[k], spat[k]);
    field_range(spat[0], g.nlines, g.linelen, g.stride, &rep.field_min, &rep.field_max,
                &rep.finite);
  }

  /* A reproducible state to compare against a WebGPU run: exactly --steps steps
   * from the seed, separate from the timed run above (which has warmup in it). */
  if (spec.digest) {
    if (transform_mode) {
      seed_spectrum();
      rep.input_digest = shtb_digest_of(state32, 2 * nlm);
      rep.have_input_digest = 1;
      for (int i = 0; i < spec.steps; i++) round_trip();
    } else {
      seed_state();
      for (int i = 0; i < spec.steps; i++) step();
    }
    const double *q = (const double *)(transform_mode ? tq[tcur] : Q[0]);
    for (long i = 0; i < 2 * nlm; i++) state32[i] = (float)q[i];
    rep.digest = shtb_digest_of(state32, 2 * nlm);
    rep.have_digest = 1;
  }

  if (spec.json) {
    shtb_print_json(&spec, &rep);
  } else {
    printf("  %.3f ms/step   %.1f steps/s", rep.ms_per_step, 1000.0 / rep.ms_per_step);
    if (!transform_mode) printf("   %.2f model time/s", spec.params.dt * 1000.0 / rep.ms_per_step);
    printf("\n");
    printf("  per step: %.3f ms mean · median %.3f · p05 %.3f · p95 %.3f · min %.3f\n",
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
    printf("\n  This is fp64. The like-for-like fp32 comparison against the WGSL\n"
           "  transforms is ./shtbench_gpu; this run is the accuracy reference.\n");
  }

  if (spec.dump_state && rep.have_digest) {
    if (shtb_dump_state(spec.dump_state, &spec, &rep, state32, 2 * nlm) != 0) {
      fprintf(stderr, "shtbench: cannot write %s\n", spec.dump_state);
      return 1;
    }
    if (!spec.json) printf("\n  wrote %s\n", spec.dump_state);
  }

  free(samples);
  free(noise);
  free(state32);
  free(lam);
  for (int k = 0; k < nsp; k++) {
    shtns_free(spat[k]);
    shtns_free(rspat[k]);
    shtns_free(Q[k]);
    shtns_free(R[k]);
  }
  shtns_destroy(sht);
  return rep.finite ? 0 : 1;
}
