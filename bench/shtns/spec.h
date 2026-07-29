/*
 * One run, described the same way src/bench/runSpec.ts describes it.
 *
 * This is the native half of the comparison: the same grid rule, the same
 * presets and defaults, the same seeded perturbation, so that
 *
 *     npm run bench -- --preset schnak-spots --lmax 63 --steps 2000
 *     ./shtbench      --preset schnak-spots --lmax 63 --steps 2000
 *
 * describe the same computation, one through the WGSL transforms and one
 * through upstream SHTNS.
 *
 * The tables below are a hand copy of src/mgpu/registry.ts and
 * src/sht/layout.ts, which is unavoidable — C cannot import the TypeScript.
 * It is also the one place the two sides could silently drift apart, so both
 * emit their resolved spec in --json and scripts/compare-native.mjs refuses to
 * compare two runs whose specs do not match.
 *
 * Compiled as C++ (by g++ for the CPU benchmark, by nvcc for the CUDA one), so
 * the reaction can be written once as a template and used at float and double.
 */
#ifndef SHTB_SPEC_H
#define SHTB_SPEC_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__CUDACC__)
#define SHTB_HD __host__ __device__
#else
#define SHTB_HD
#endif

/* ------------------------------------------------------------------ models */

enum shtb_model_id { SHTB_SCHNAKENBERG = 0, SHTB_BRUSSELATOR, SHTB_ALLENCAHN };

/* Every tunable scalar of every model. A model uses the subset its .m names as
 * arguments; the tables below say which, in the .m's declared order. */
struct shtb_params {
  double a, b;   /* schnakenberg */
  double A, B;   /* brusselator  */
  double eps2;   /* allen-cahn   */
  double D1, D2; /* diffusivities (allen-cahn uses eps2 instead) */
  double dt;
};

struct shtb_param_def {
  const char *key;
  size_t off; /* offset into shtb_params */
  double dflt;
};

#define SHTB_P(field, dflt) \
  { #field, offsetof(struct shtb_params, field), dflt }

static const struct shtb_param_def SHTB_SCHNAK_PARAMS[] = {
    SHTB_P(a, 0.1), SHTB_P(b, 0.9), SHTB_P(D1, 4e-4), SHTB_P(D2, 8e-3), SHTB_P(dt, 0.05),
};
static const struct shtb_param_def SHTB_BRUSSEL_PARAMS[] = {
    SHTB_P(A, 3.0), SHTB_P(B, 9.0), SHTB_P(D1, 3.33e-3), SHTB_P(D2, 1.67e-2), SHTB_P(dt, 0.02),
};
static const struct shtb_param_def SHTB_ALLENCAHN_PARAMS[] = {
    SHTB_P(eps2, 1e-3), SHTB_P(dt, 0.02),
};

struct shtb_model_def {
  int id;
  const char *key;
  const char *label;
  int nspecies;
  int pdeg;         /* polynomial degree of the reaction, for dealiasing */
  double seed_amp;  /* amplitude of the seeded perturbation */
  const struct shtb_param_def *params;
  int nparams;
};

static const struct shtb_model_def SHTB_MODELS[] = {
    {SHTB_SCHNAKENBERG, "schnakenberg", "Schnakenberg", 2, 3, 1e-2, SHTB_SCHNAK_PARAMS, 5},
    {SHTB_BRUSSELATOR, "brusselator", "Brusselator", 2, 3, 1e-2, SHTB_BRUSSEL_PARAMS, 5},
    {SHTB_ALLENCAHN, "allencahn", "Allen-Cahn", 1, 3, 1e-2, SHTB_ALLENCAHN_PARAMS, 2},
};
#define SHTB_NMODELS 3

struct shtb_override {
  const char *key;
  double value;
};

struct shtb_preset {
  const char *key;
  const char *label;
  int model;
  struct shtb_override over[2]; /* .key == NULL terminates */
};

static const struct shtb_preset SHTB_PRESETS[] = {
    {"schnak-spots", "Schnakenberg - spots", SHTB_SCHNAKENBERG, {{NULL, 0}, {NULL, 0}}},
    {"schnak-coarse", "Schnakenberg - coarse spots", SHTB_SCHNAKENBERG,
     {{"D1", 1e-3}, {"D2", 2e-2}}},
    {"schnak-fine", "Schnakenberg - fine spots", SHTB_SCHNAKENBERG,
     {{"D1", 1.6e-4}, {"D2", 3.2e-3}}},
    {"brussel", "Brusselator - stripes & spots", SHTB_BRUSSELATOR, {{NULL, 0}, {NULL, 0}}},
    {"allencahn", "Allen-Cahn - coarsening", SHTB_ALLENCAHN, {{NULL, 0}, {NULL, 0}}},
};
#define SHTB_NPRESETS 5

static inline double *shtb_field(struct shtb_params *p, size_t off) {
  return (double *)((char *)p + off);
}
static inline const double *shtb_field_c(const struct shtb_params *p, size_t off) {
  return (const double *)((const char *)p + off);
}

/* --------------------------------------------------- the arithmetic per step */

/*
 * Everything a timestep needs, in the working precision. Products the .m forms
 * from parameters — `(dt * D1)`, `(B + 1)` — are formed here in that same
 * precision, so a float run multiplies floats exactly as the WGSL kernel does.
 */
template <typename real>
struct shtb_step_const {
  int model;
  real p0, p1; /* (a, b) | (A, B) | unused */
  real dt;
  real dtD[2]; /* dt * D_k, one per species */
};

/*
 * The reaction, transcribed from models/<key>.m. One line of MATLAB per line
 * here, and `u.^3` is written out as `u*u*u` because that is what the WGSL
 * backend emits for it (pow_i3; see emitPower in src/mgpu/wgsl.ts).
 *
 *   schnakenberg:  Un = (U + dt*analys(a - u + uuv)) ./ (1 + (dt*D1)*lam)
 *                  Vn = (V + dt*analys(b - uuv))     ./ (1 + (dt*D2)*lam)
 *   brusselator:   Un = (U + dt*analys(A - (B+1)*u + uuv)) ./ ...
 *                  Vn = (V + dt*analys(B*u - uuv))         ./ ...
 *   allencahn:     Un = (U + dt*analys(u - u.^3)) ./ (1 + (dt*eps2)*lam)
 */
template <typename real>
SHTB_HD inline void shtb_react(const shtb_step_const<real> &c, real u, real v, real *r1,
                               real *r2) {
  if (c.model == SHTB_ALLENCAHN) {
    *r1 = u - u * u * u;
    return;
  }
  const real uuv = u * u * v;
  if (c.model == SHTB_SCHNAKENBERG) {
    *r1 = c.p0 - u + uuv;
    *r2 = c.p1 - uuv;
  } else {
    *r1 = c.p0 - (c.p1 + (real)1) * u + uuv;
    *r2 = c.p1 * u - uuv;
  }
}

/* One element of the IMEX update, for species k. `lam` is l(l+1) of that
 * coefficient; the array is 2*nlm long with the value duplicated across the
 * real and imaginary halves, matching the 2 x nlm spectral layout the .m sees. */
template <typename real>
SHTB_HD inline real shtb_imex(const shtb_step_const<real> &c, int k, real U, real R, real lam) {
  return (U + c.dt * R) / ((real)1 + c.dtD[k] * lam);
}

/* ------------------------------------------------------------------ the spec */

enum shtb_mode { SHTB_MODE_SOLVER = 0, SHTB_MODE_TRANSFORM };
enum shtb_layout { SHTB_LAYOUT_THETA = 0, SHTB_LAYOUT_PHI };

/* Defaults, from src/bench/runSpec.ts. */
#define SHTB_DEFAULT_LMAX 63
#define SHTB_DEFAULT_SEED 1
#define SHTB_DEFAULT_STEPS 2000
#define SHTB_DEFAULT_WARMUP 100
#define SHTB_DEFAULT_BATCH 16

struct shtb_spec {
  const struct shtb_preset *preset;
  const struct shtb_model_def *model;
  struct shtb_params params;
  int lmax, nlat, nphi;
  int seed, steps, warmup, batch;
  int mode;
  int layout;
  int fp32;    /* GPU only: use SHTNS' single-precision transforms */
  int threads; /* CPU only: OpenMP threads, 0 = library default */
  int json, digest;
  const char *dump_state;
};

/* Grid sizes for a given lmax, dealiased for a reaction of polynomial degree
 * pdeg. Identical to gridForLmax() in src/sht/layout.ts, including rounding
 * nphi up to a power of two — which the WGSL side needs for its FFT path and
 * which SHTNS does not, but the grids have to match to compare anything. */
static inline void shtb_grid_for_lmax(int lmax, int pdeg, int *nlat, int *nphi) {
  double min_lat = ((double)(pdeg + 1) * lmax + 1) / 2.0;
  if (min_lat < lmax + 1) min_lat = lmax + 1;
  *nlat = 2 * (int)ceil(min_lat / 2.0);
  int n = 1;
  while (n < (pdeg + 1) * lmax + 1) n *= 2;
  *nphi = n;
}

static inline long shtb_nlm_calc(int lmax, int mmax) {
  return (long)(mmax + 1) * (lmax + 1) - (long)mmax * (mmax + 1) / 2;
}

/* -------------------------------------------------------------- the seeding */

/*
 * mulberry32 + Box-Muller, transcribed from src/mgpu/noise.ts so an integer
 * seed means the same perturbation on both sides. The integer state evolves
 * bit-identically; the Box-Muller step then goes through log/sqrt/sin/cos, so a
 * libm that rounds differently from V8's can differ in the last bit. Both sides
 * report the perturbation's RMS for that reason.
 */
struct shtb_rng {
  uint32_t s;
  int have_spare;
  double spare;
};

static inline void shtb_rng_init(struct shtb_rng *r, uint32_t seed) {
  r->s = seed;
  r->have_spare = 0;
  r->spare = 0;
}

/* uniform in [0, 1) */
static inline double shtb_rand(struct shtb_rng *r) {
  r->s = r->s + 0x6d2b79f5u;
  uint32_t t = r->s;
  t = (t ^ (t >> 15)) * (t | 1u);
  t ^= t + (t ^ (t >> 7)) * (t | 61u);
  return (double)(t ^ (t >> 14)) / 4294967296.0;
}

static inline double shtb_randn(struct shtb_rng *r) {
  if (r->have_spare) {
    r->have_spare = 0;
    return r->spare;
  }
  double u = 0;
  while (u == 0) u = shtb_rand(r);
  double rad = sqrt(-2 * log(u));
  double th = 2 * M_PI * shtb_rand(r);
  r->spare = rad * sin(th);
  r->have_spare = 1;
  return rad * cos(th);
}

/* amp-scaled normal deviates, one per grid point, in [ilat*nphi + iphi] order —
 * the order src/mgpu/noise.ts produces them in. Rounded to float, because the
 * browser stores them in a Float32Array; the fp64 run then differs from the
 * fp32 one only in the arithmetic, not in the initial condition. */
static inline void shtb_seeded_noise(long npts, double amp, uint32_t seed, float *out) {
  struct shtb_rng r;
  shtb_rng_init(&r, seed);
  for (long i = 0; i < npts; i++) out[i] = (float)(amp * shtb_randn(&r));
}

/*
 * A seeded spectrum for the transform-only benchmark: uniform in [-1, 1), and
 * bit-identical to the TypeScript side because it never leaves integer
 * arithmetic and exactly-representable doubles. The m = 0 imaginary parts are
 * zeroed, since a real field has none and the two libraries need not agree on
 * what to do with a coefficient that cannot occur.
 *
 * qlm is interleaved [re, im] per coefficient, SHTNS LM ordering.
 */
static inline void shtb_seeded_spectrum(int lmax, int mmax, uint32_t seed, float *qlm) {
  struct shtb_rng r;
  shtb_rng_init(&r, seed);
  long lm = 0;
  for (int m = 0; m <= mmax; m++) {
    for (int l = m; l <= lmax; l++, lm++) {
      qlm[2 * lm] = (float)(2 * shtb_rand(&r) - 1);
      float im = (float)(2 * shtb_rand(&r) - 1);
      qlm[2 * lm + 1] = (m == 0) ? 0.0f : im;
    }
  }
}

/* -------------------------------------------------------------- statistics */

static inline double shtb_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

struct shtb_timing {
  double mean_ms, median_ms, p05_ms, p95_ms, min_ms;
};

static int shtb_cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static inline struct shtb_timing shtb_stats(double *samples, int n) {
  qsort(samples, (size_t)n, sizeof(double), shtb_cmp_double);
  double total = 0;
  for (int i = 0; i < n; i++) total += samples[i];
  int i50 = (int)(0.50 * n), i05 = (int)(0.05 * n), i95 = (int)(0.95 * n);
  if (i50 >= n) i50 = n - 1;
  if (i05 >= n) i05 = n - 1;
  if (i95 >= n) i95 = n - 1;
  struct shtb_timing t;
  t.mean_ms = total / n;
  t.median_ms = samples[i50];
  t.p05_ms = samples[i05];
  t.p95_ms = samples[i95];
  t.min_ms = samples[0];
  return t;
}

/* The same five numbers digestOf() computes in src/mgpu/digest.ts. */
struct shtb_digest {
  long n;
  double min, max, mean, rms;
};

static inline struct shtb_digest shtb_digest_of(const float *v, long n) {
  struct shtb_digest d;
  d.n = n;
  d.min = INFINITY;
  d.max = -INFINITY;
  double sum = 0, sumsq = 0;
  for (long i = 0; i < n; i++) {
    double x = v[i];
    if (x < d.min) d.min = x;
    if (x > d.max) d.max = x;
    sum += x;
    sumsq += x * x;
  }
  d.mean = sum / (double)n;
  d.rms = sqrt(sumsq / (double)n);
  return d;
}

static inline int shtb_all_finite(const float *v, long n) {
  for (long i = 0; i < n; i++)
    if (!isfinite(v[i])) return 0;
  return 1;
}

/* ----------------------------------------------------------- spec resolution */

static inline const struct shtb_model_def *shtb_model_by_id(int id) {
  for (int i = 0; i < SHTB_NMODELS; i++)
    if (SHTB_MODELS[i].id == id) return &SHTB_MODELS[i];
  return NULL;
}

static inline void shtb_default_params(const struct shtb_model_def *m, struct shtb_params *p) {
  memset(p, 0, sizeof(*p));
  for (int i = 0; i < m->nparams; i++) *shtb_field(p, m->params[i].off) = m->params[i].dflt;
}

/* Homogeneous background each species starts from, and the diffusivity each is
 * advanced with. From the init/step functions of models/<key>.m; only species 0
 * gets the seeded perturbation. */
static inline void shtb_background(const struct shtb_model_def *m, const struct shtb_params *p,
                                   double base[2]) {
  base[0] = base[1] = 0;
  if (m->id == SHTB_SCHNAKENBERG) {
    double us = p->a + p->b;
    base[0] = us;
    base[1] = p->b / (us * us);
  } else if (m->id == SHTB_BRUSSELATOR) {
    base[0] = p->A;
    base[1] = p->B / p->A;
  }
}

static inline void shtb_diffusivity(const struct shtb_model_def *m, const struct shtb_params *p,
                                    double d[2]) {
  if (m->id == SHTB_ALLENCAHN) {
    d[0] = p->eps2;
    d[1] = 0;
  } else {
    d[0] = p->D1;
    d[1] = p->D2;
  }
}

template <typename real>
static inline shtb_step_const<real> shtb_make_step_const(const struct shtb_model_def *m,
                                                         const struct shtb_params *p) {
  double d[2];
  shtb_diffusivity(m, p, d);
  shtb_step_const<real> c;
  c.model = m->id;
  c.p0 = (real)(m->id == SHTB_BRUSSELATOR ? p->A : p->a);
  c.p1 = (real)(m->id == SHTB_BRUSSELATOR ? p->B : p->b);
  c.dt = (real)p->dt;
  /* (dt * D_k) as one product in the working precision, as the .m writes it */
  c.dtD[0] = (real)p->dt * (real)d[0];
  c.dtD[1] = (real)p->dt * (real)d[1];
  return c;
}

/* ------------------------------------------------------------ argument parsing
 *
 * `--key value` or `--key=value`, in any order — the same grammar parseArgs()
 * accepts in src/bench/runSpec.ts, plus the flags only a native run has.
 */

static inline int shtb_parse_spec(int argc, char **argv, struct shtb_spec *s, const char *usage) {
  const struct shtb_preset *preset = &SHTB_PRESETS[0];

  /* --preset first: it decides which parameter names are legal. */
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    const char *v = NULL;
    if (strcmp(a, "--preset") == 0 && i + 1 < argc)
      v = argv[i + 1];
    else if (strncmp(a, "--preset=", 9) == 0)
      v = a + 9;
    if (!v) continue;
    const struct shtb_preset *found = NULL;
    for (int k = 0; k < SHTB_NPRESETS; k++)
      if (strcmp(SHTB_PRESETS[k].key, v) == 0) found = &SHTB_PRESETS[k];
    if (!found) {
      fprintf(stderr, "shtbench: unknown preset '%s' (have:", v);
      for (int k = 0; k < SHTB_NPRESETS; k++) fprintf(stderr, " %s", SHTB_PRESETS[k].key);
      fprintf(stderr, ")\n");
      return 2;
    }
    preset = found;
  }

  memset(s, 0, sizeof(*s));
  s->preset = preset;
  s->model = shtb_model_by_id(preset->model);
  shtb_default_params(s->model, &s->params);
  for (int k = 0; k < 2; k++)
    if (preset->over[k].key) {
      for (int i = 0; i < s->model->nparams; i++)
        if (strcmp(s->model->params[i].key, preset->over[k].key) == 0)
          *shtb_field(&s->params, s->model->params[i].off) = preset->over[k].value;
    }
  s->lmax = SHTB_DEFAULT_LMAX;
  s->seed = SHTB_DEFAULT_SEED;
  s->steps = SHTB_DEFAULT_STEPS;
  s->warmup = SHTB_DEFAULT_WARMUP;
  s->batch = SHTB_DEFAULT_BATCH;
  s->mode = SHTB_MODE_SOLVER;
  s->layout = SHTB_LAYOUT_THETA;
  s->fp32 = 1;
  s->threads = 0;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (strncmp(a, "--", 2) != 0) {
      fprintf(stderr, "shtbench: unexpected argument '%s'\n\n%s\n", a, usage);
      return 2;
    }
    if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
      printf("%s\n", usage);
      return 1;
    }
    if (strcmp(a, "--json") == 0) {
      s->json = 1;
      continue;
    }
    if (strcmp(a, "--digest") == 0) {
      s->digest = 1;
      continue;
    }
    if (strcmp(a, "--fp64") == 0) {
      s->fp32 = 0;
      continue;
    }

    /* split key / value */
    char key[64];
    const char *val = NULL;
    const char *eq = strchr(a, '=');
    if (eq) {
      size_t n = (size_t)(eq - a - 2);
      if (n >= sizeof(key)) n = sizeof(key) - 1;
      memcpy(key, a + 2, n);
      key[n] = 0;
      val = eq + 1;
    } else {
      snprintf(key, sizeof(key), "%s", a + 2);
      if (i + 1 >= argc) {
        fprintf(stderr, "shtbench: --%s needs a value\n", key);
        return 2;
      }
      val = argv[++i];
    }

    if (strcmp(key, "preset") == 0) continue; /* handled above */
    if (strcmp(key, "lmax") == 0) {
      s->lmax = atoi(val);
      continue;
    }
    if (strcmp(key, "seed") == 0) {
      s->seed = atoi(val);
      continue;
    }
    if (strcmp(key, "steps") == 0) {
      s->steps = atoi(val);
      continue;
    }
    if (strcmp(key, "warmup") == 0) {
      s->warmup = atoi(val);
      continue;
    }
    if (strcmp(key, "batch") == 0) {
      s->batch = atoi(val);
      continue;
    }
    if (strcmp(key, "threads") == 0) {
      s->threads = atoi(val);
      continue;
    }
    if (strcmp(key, "dump-state") == 0) {
      s->dump_state = val;
      s->digest = 1;
      continue;
    }
    if (strcmp(key, "mode") == 0) {
      if (strcmp(val, "solver") == 0)
        s->mode = SHTB_MODE_SOLVER;
      else if (strcmp(val, "transform") == 0)
        s->mode = SHTB_MODE_TRANSFORM;
      else {
        fprintf(stderr, "shtbench: --mode must be 'solver' or 'transform' (got '%s')\n", val);
        return 2;
      }
      continue;
    }
    if (strcmp(key, "layout") == 0) {
      if (strcmp(val, "theta") == 0)
        s->layout = SHTB_LAYOUT_THETA;
      else if (strcmp(val, "phi") == 0)
        s->layout = SHTB_LAYOUT_PHI;
      else {
        fprintf(stderr, "shtbench: --layout must be 'theta' or 'phi' (got '%s')\n", val);
        return 2;
      }
      continue;
    }

    int matched = 0;
    for (int p = 0; p < s->model->nparams; p++)
      if (strcmp(s->model->params[p].key, key) == 0) {
        *shtb_field(&s->params, s->model->params[p].off) = atof(val);
        matched = 1;
      }
    if (!matched) {
      fprintf(stderr, "shtbench: unknown option --%s\nparameters of %s:", key, s->model->label);
      for (int p = 0; p < s->model->nparams; p++)
        fprintf(stderr, " --%s", s->model->params[p].key);
      fprintf(stderr, "\n");
      return 2;
    }
  }

  if (s->lmax < 1) {
    fprintf(stderr, "shtbench: --lmax must be >= 1\n");
    return 2;
  }
  if (s->steps < 1 || s->warmup < 0 || s->batch < 1) {
    fprintf(stderr, "shtbench: --steps and --batch must be >= 1, --warmup >= 0\n");
    return 2;
  }
  shtb_grid_for_lmax(s->lmax, s->model->pdeg, &s->nlat, &s->nphi);
  return 0;
}

/* ------------------------------------------------------------- JSON emission
 *
 * Deliberately shaped like the object scripts/bench.ts prints with --json, so
 * scripts/compare-native.mjs can read a WebGPU run and a native run the same
 * way. "step" means one solver timestep in solver mode and one
 * spectral->grid->spectral round trip in transform mode.
 */

/* Strings from the library go into JSON, so strip anything that would break it.
 * Returns `dst`, for use inline. */
static inline char *shtb_json_safe(char *dst, size_t cap, const char *src) {
  size_t j = 0;
  for (size_t i = 0; src && src[i] && j + 1 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    dst[j++] = (c < 0x20 || c == '"' || c == '\\' || c == 0x7f) ? ' ' : (char)c;
  }
  dst[j] = 0;
  return dst;
}

struct shtb_report {
  const char *library;  /* e.g. "SHTNS 3.7.5" */
  const char *runtime;  /* e.g. "cuda 12.4, vkfft" */
  const char *adapter;  /* GPU name, or the CPU's thread count */
  const char *precision;/* "fp32" | "fp64" */
  const char *fourier;  /* which FFT the library used */
  long nlm;
  int ops_per_step;
  double ms_per_step;
  double encode_ms_per_step;
  struct shtb_timing latency;
  int have_latency;
  struct shtb_digest digest;
  int have_digest;
  struct shtb_digest input_digest; /* transform mode: the seeded spectrum */
  int have_input_digest;
  double field_min, field_max;
  int finite;
  double model_t;
  int steps_run;
};

static inline void shtb_print_json(const struct shtb_spec *s, const struct shtb_report *r) {
  printf("{\n");
  printf("  \"mode\": \"%s\",\n", s->mode == SHTB_MODE_SOLVER ? "solver" : "transform");
  printf("  \"spec\": {\n");
  printf("    \"preset\": \"%s\",\n", s->preset->key);
  printf("    \"lmax\": %d,\n", s->lmax);
  printf("    \"seed\": %d,\n", s->seed);
  printf("    \"steps\": %d,\n", s->steps);
  printf("    \"warmup\": %d,\n", s->warmup);
  printf("    \"params\": {");
  for (int i = 0; i < s->model->nparams; i++)
    printf("%s\"%s\": %.17g", i ? ", " : "", s->model->params[i].key,
           *shtb_field_c(&s->params, s->model->params[i].off));
  printf("}\n  },\n");
  printf("  \"model\": \"%s\",\n", s->model->key);
  printf("  \"backend\": {\"library\": \"%s\", \"runtime\": \"%s\", \"adapter\": \"%s\", "
         "\"precision\": \"%s\", \"layout\": \"%s\"},\n",
         r->library, r->runtime, r->adapter, r->precision,
         s->layout == SHTB_LAYOUT_PHI ? "phi-contiguous" : "theta-contiguous");
  printf("  \"grid\": {\"lmax\": %d, \"nlat\": %d, \"nphi\": %d, \"nlm\": %ld},\n", s->lmax,
         s->nlat, s->nphi, r->nlm);
  printf("  \"compiled\": {\"opsPerStep\": %d},\n", r->ops_per_step);
  printf("  \"throughput\": {\"batch\": %d, \"msPerStep\": %.17g, \"stepsPerSec\": %.17g, "
         "\"encodeMsPerStep\": %.17g},\n",
         s->batch, r->ms_per_step, 1000.0 / r->ms_per_step, r->encode_ms_per_step);
  if (r->have_latency)
    printf("  \"latency\": {\"meanMs\": %.17g, \"medianMs\": %.17g, \"p05Ms\": %.17g, "
           "\"p95Ms\": %.17g, \"minMs\": %.17g},\n",
           r->latency.mean_ms, r->latency.median_ms, r->latency.p05_ms, r->latency.p95_ms,
           r->latency.min_ms);
  else
    printf("  \"latency\": null,\n");
  if (r->have_digest)
    printf("  \"digest\": {\"n\": %ld, \"min\": %.17g, \"max\": %.17g, \"mean\": %.17g, "
           "\"rms\": %.17g, \"fourier\": \"%s\", \"adapter\": \"%s\"},\n",
           r->digest.n, r->digest.min, r->digest.max, r->digest.mean, r->digest.rms, r->fourier,
           r->adapter);
  else
    printf("  \"digest\": null,\n");
  if (r->have_input_digest)
    printf("  \"input\": {\"n\": %ld, \"min\": %.17g, \"max\": %.17g, \"mean\": %.17g, "
           "\"rms\": %.17g},\n",
           r->input_digest.n, r->input_digest.min, r->input_digest.max, r->input_digest.mean,
           r->input_digest.rms);
  else
    printf("  \"input\": null,\n");
  printf("  \"state\": {\"t\": %.17g, \"steps\": %d, \"min\": %.17g, \"max\": %.17g, "
         "\"contrast\": %.17g, \"finite\": %s}\n",
         r->model_t, r->steps_run, r->field_min, r->field_max, r->field_max - r->field_min,
         r->finite ? "true" : "false");
  printf("}\n");
}

/* The state file scripts/compare-native.mjs diffs, in the same shape
 * `npm run bench -- --dump-state` writes. */
static inline int shtb_dump_state(const char *path, const struct shtb_spec *s,
                                  const struct shtb_report *r, const float *state, long n) {
  FILE *f = fopen(path, "w");
  if (!f) return -1;
  fprintf(f, "{\"spec\":{\"preset\":\"%s\",\"lmax\":%d,\"seed\":%d,\"steps\":%d,\"warmup\":%d,"
             "\"params\":{",
          s->preset->key, s->lmax, s->seed, s->steps, s->warmup);
  for (int i = 0; i < s->model->nparams; i++)
    fprintf(f, "%s\"%s\":%.17g", i ? "," : "", s->model->params[i].key,
            *shtb_field_c(&s->params, s->model->params[i].off));
  fprintf(f, "}},\"mode\":\"%s\",\"backend\":{\"library\":\"%s\",\"adapter\":\"%s\","
             "\"precision\":\"%s\"},",
          s->mode == SHTB_MODE_SOLVER ? "solver" : "transform", r->library, r->adapter,
          r->precision);
  fprintf(f, "\"digest\":{\"n\":%ld,\"min\":%.17g,\"max\":%.17g,\"mean\":%.17g,\"rms\":%.17g,"
             "\"fourier\":\"%s\",\"adapter\":\"%s\"},",
          r->digest.n, r->digest.min, r->digest.max, r->digest.mean, r->digest.rms, r->fourier,
          r->adapter);
  if (r->have_input_digest)
    fprintf(f, "\"input\":{\"n\":%ld,\"min\":%.17g,\"max\":%.17g,\"mean\":%.17g,\"rms\":%.17g},",
            r->input_digest.n, r->input_digest.min, r->input_digest.max, r->input_digest.mean,
            r->input_digest.rms);
  fprintf(f, "\"state\":[");
  for (long i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", (double)state[i]);
  fprintf(f, "]}\n");
  fclose(f);
  return 0;
}

#endif /* SHTB_SPEC_H */
