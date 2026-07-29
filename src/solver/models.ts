/**
 * Model metadata and the reference reaction terms.
 *
 * The parameter metadata here (names, labels, defaults, slider ranges) is what
 * the app uses; `src/mgpu/registry.ts` attaches each model's .m source to it, so
 * the .m running on the GPU and the reference solver used by the tests cannot be
 * configured differently. The `reaction` / `init` closures below are the
 * reference implementation only — the app runs the .m instead.
 *
 * Reaction-diffusion model presets, ported from websph's
 * SphericalReactionDiffusionDriver.m. Each species k solves
 *
 *    d(u_k)/dt = D_k * lap_s(u_k) + f_k(t, x, y, z, u_1, ..., u_N)
 *
 * on the unit sphere. Reactions are vectorized over the grid.
 */

export type Params = Record<string, number>;

export interface ParamSpec {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
}

export interface ModelSpec {
  key: string;
  label: string;
  blurb: string;
  species: string[];
  params: ParamSpec[];
  /** Polynomial degree of the reaction in the fields (for dealiasing). */
  pdeg: number;
  /** Amplitude of the random perturbation seeded into the initial state. */
  seedAmp: number;
  diffusivities(p: Params): number[];
  /** Fill out[k][i] with f_k evaluated at every grid point. */
  reaction(
    p: Params,
    t: number,
    x: Float64Array,
    y: Float64Array,
    z: Float64Array,
    V: ArrayLike<number>[],
    out: Float64Array[],
  ): void;
  /** Fill out[k][i] with the initial condition (noise added via randn). */
  init(
    p: Params,
    x: Float64Array,
    y: Float64Array,
    z: Float64Array,
    randn: () => number,
    out: Float64Array[],
  ): void;
}

const schnakenberg: ModelSpec = {
  key: 'schnakenberg',
  label: 'Schnakenberg',
  blurb:
    'Turing spots. The homogeneous state is stable to uniform perturbations ' +
    'but unstable to degrees 14 ≤ l ≤ 40, most strongly at l = 24.',
  species: ['u', 'v'],
  params: [
    { key: 'a', label: 'a', value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'b', label: 'b', value: 0.9, min: 0.1, max: 2, step: 0.05 },
    { key: 'D1', label: 'D₁', value: 4e-4, min: 1e-5, max: 5e-3, step: 1e-5 },
    { key: 'D2', label: 'D₂', value: 8e-3, min: 1e-4, max: 5e-2, step: 1e-4 },
    { key: 'dt', label: 'dt', value: 0.05, min: 0.005, max: 0.5, step: 0.005 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  diffusivities: (p) => [p.D1, p.D2],
  reaction(p, _t, _x, _y, _z, V, out) {
    const [u, v] = V;
    const [fu, fv] = out;
    const a = p.a, b = p.b;
    const n = fu.length;
    for (let i = 0; i < n; i++) {
      const ui = u[i], vi = v[i];
      const uuv = ui * ui * vi;
      fu[i] = a - ui + uuv;
      fv[i] = b - uuv;
    }
  },
  init(p, x, _y, _z, randn, out) {
    const [u, v] = out;
    const a = p.a, b = p.b;
    const us = a + b;
    const vs = b / (us * us);
    const n = x.length;
    for (let i = 0; i < n; i++) {
      u[i] = us + this.seedAmp * randn();
      v[i] = vs;
    }
  },
};

const brusselator: ModelSpec = {
  key: 'brusselator',
  label: 'Brusselator',
  blurb:
    'Turing stripes and spots, from a smaller diffusivity contrast than ' +
    'Schnakenberg but with a stiffer reaction.',
  species: ['u', 'v'],
  params: [
    { key: 'A', label: 'A', value: 3, min: 0.5, max: 6, step: 0.1 },
    { key: 'B', label: 'B', value: 9, min: 1, max: 15, step: 0.25 },
    { key: 'D1', label: 'D₁', value: 3.33e-3, min: 1e-4, max: 2e-2, step: 1e-4 },
    { key: 'D2', label: 'D₂', value: 1.67e-2, min: 1e-3, max: 1e-1, step: 1e-3 },
    { key: 'dt', label: 'dt', value: 0.02, min: 0.002, max: 0.1, step: 0.002 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  diffusivities: (p) => [p.D1, p.D2],
  reaction(p, _t, _x, _y, _z, V, out) {
    const [u, v] = V;
    const [fu, fv] = out;
    const A = p.A, B = p.B;
    const n = fu.length;
    for (let i = 0; i < n; i++) {
      const ui = u[i], vi = v[i];
      const uuv = ui * ui * vi;
      fu[i] = A - (B + 1) * ui + uuv;
      fv[i] = B * ui - uuv;
    }
  },
  init(p, x, _y, _z, randn, out) {
    const [u, v] = out;
    const n = x.length;
    for (let i = 0; i < n; i++) {
      u[i] = p.A + this.seedAmp * randn();
      v[i] = p.B / p.A;
    }
  },
};

const allenCahn: ModelSpec = {
  key: 'allencahn',
  label: 'Allen–Cahn',
  blurb:
    'A single species: interfaces form and then coarsen until one domain ' +
    'swallows the sphere.',
  species: ['u'],
  params: [
    { key: 'eps2', label: 'ε²', value: 1e-3, min: 1e-4, max: 1e-2, step: 1e-4 },
    { key: 'dt', label: 'dt', value: 0.02, min: 0.002, max: 0.2, step: 0.002 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  diffusivities: (p) => [p.eps2],
  reaction(_p, _t, _x, _y, _z, V, out) {
    const [u] = V;
    const [fu] = out;
    const n = fu.length;
    for (let i = 0; i < n; i++) {
      const ui = u[i];
      fu[i] = ui - ui * ui * ui;
    }
  },
  init(_p, x, _y, _z, randn, out) {
    const [u] = out;
    const n = x.length;
    for (let i = 0; i < n; i++) {
      u[i] = this.seedAmp * randn();
    }
  },
};

export const models: ModelSpec[] = [schnakenberg, brusselator, allenCahn];

export const defaultParams = (m: ModelSpec): Params =>
  Object.fromEntries(m.params.map((p) => [p.key, p.value]));

/** Named parameter presets shown in the UI dropdown. The pattern length
 *  scale goes as 1/sqrt(D), so scaling both diffusivities moves the spot
 *  size without changing the dynamics. */
export interface Preset {
  key: string;
  label: string;
  modelKey: string;
  /** Overrides applied on top of the model's default parameters. */
  params?: Params;
}

export const presets: Preset[] = [
  { key: 'schnak-spots', label: 'Schnakenberg — spots', modelKey: 'schnakenberg' },
  {
    key: 'schnak-coarse',
    label: 'Schnakenberg — coarse spots',
    modelKey: 'schnakenberg',
    params: { D1: 1e-3, D2: 2e-2 },
  },
  {
    key: 'schnak-fine',
    label: 'Schnakenberg — fine spots',
    modelKey: 'schnakenberg',
    params: { D1: 1.6e-4, D2: 3.2e-3 },
  },
  { key: 'brussel', label: 'Brusselator — stripes & spots', modelKey: 'brusselator' },
  { key: 'allencahn', label: 'Allen–Cahn — coarsening', modelKey: 'allencahn' },
];
