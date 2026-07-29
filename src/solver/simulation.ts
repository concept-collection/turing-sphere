/**
 * Reference solver — NOT what the app runs.
 *
 * The app executes the .m models under `models/` on the GPU (see `src/mgpu/`).
 * This TypeScript port remains as an independent implementation of the same
 * scheme, which is what makes it usable as the test oracle: `test/mgpuChecks.ts`
 * runs both from the same seed and compares. Keep the two in step.
 *
 * IMEX Euler reaction-diffusion timestepper on the sphere, ported from
 * websph's SphericalReactionDiffusion.m. Diffusion is implicit and diagonal
 * in spherical-harmonic space (Laplace-Beltrami eigenvalues -l(l+1));
 * reaction is explicit on the grid:
 *
 *    (I - dt*D_k*lap_s) u_k^{n+1} = u_k^n + dt*f_k(u^n)
 */
import type { ShtBackend } from './backend.ts';
import type { ModelSpec, Params } from './models.ts';
import { lmIndex } from '../sht/layout.ts';

/** Grid sizes for a given lmax, dealiased for a reaction of degree pdeg
 *  (see websph README): nlat >= ((pdeg+1)*lmax+1)/2, nlon >= (pdeg+1)*lmax+1.
 *  nphi is rounded up to a power of two to keep the GPU FFT path. */
export function gridForLmax(lmax: number, pdeg: number): { nlat: number; nphi: number } {
  const minLat = Math.max(lmax + 1, ((pdeg + 1) * lmax + 1) / 2);
  const nlat = 2 * Math.ceil(minLat / 2);
  let nphi = 1;
  while (nphi < (pdeg + 1) * lmax + 1) nphi *= 2;
  return { nlat, nphi };
}

/** Seeded normal deviates: mulberry32 + Box-Muller. */
export function makeRandn(seed: number): () => number {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    while (u === 0) u = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * rand();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

export class Simulation {
  readonly backend: ShtBackend;
  readonly model: ModelSpec;
  readonly params: Params;
  readonly nspecies: number;

  /** Spectral state, one interleaved-complex Float64Array (2*nlm) per species. */
  U: Float64Array[];
  /** Grid values per species as of the START of the last step (one step
   *  behind U; recomputed as the first stage of the next step). */
  V: (Float32Array | Float64Array)[];
  t = 0;
  stepCount = 0;

  /** Cartesian coordinates of the grid points (nlat*nphi each). */
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly z: Float64Array;
  /** Laplace-Beltrami eigenvalues l*(l+1) per spectral index (length nlm). */
  readonly lam: Float64Array;

  #R: Float64Array[]; // reaction scratch, one grid array per species
  #m0Imag: number[]; // interleaved-array positions of m=0 imaginary parts

  constructor(backend: ShtBackend, model: ModelSpec, params: Params) {
    this.backend = backend;
    this.model = model;
    this.params = params;
    this.nspecies = model.species.length;

    const { lmax, mmax, nlat, nphi } = backend.cfg;
    const npts = nlat * nphi;
    this.x = new Float64Array(npts);
    this.y = new Float64Array(npts);
    this.z = new Float64Array(npts);
    for (let i = 0; i < nlat; i++) {
      const ct = backend.cosTheta[i];
      const st = Math.sqrt(Math.max(0, 1 - ct * ct));
      for (let j = 0; j < nphi; j++) {
        const phi = (2 * Math.PI * j) / nphi;
        const idx = i * nphi + j;
        this.x[idx] = st * Math.cos(phi);
        this.y[idx] = st * Math.sin(phi);
        this.z[idx] = ct;
      }
    }

    this.lam = new Float64Array(backend.nlm);
    for (let m = 0; m <= mmax; m++) {
      for (let l = m; l <= lmax; l++) {
        this.lam[lmIndex(lmax, l, m)] = l * (l + 1);
      }
    }
    this.#m0Imag = [];
    for (let l = 0; l <= lmax; l++) {
      this.#m0Imag.push(2 * lmIndex(lmax, l, 0) + 1);
    }

    this.U = [];
    this.V = [];
    this.#R = [];
    for (let k = 0; k < this.nspecies; k++) {
      this.U.push(new Float64Array(2 * backend.nlm));
      this.V.push(new Float64Array(npts));
      this.#R.push(new Float64Array(npts));
    }
  }

  /** Project the initial conditions (band-limiting the seed noise). */
  async init(seed: number): Promise<void> {
    const grids = this.#R;
    this.model.init(this.params, this.x, this.y, this.z, makeRandn(seed), grids);
    for (let k = 0; k < this.nspecies; k++) {
      const q = await this.backend.analys(grids[k]);
      this.U[k].set(q);
      this.#cleanM0(this.U[k]);
      this.V[k] = await this.backend.synth(this.U[k]);
    }
    this.t = 0;
    this.stepCount = 0;
  }

  /** One IMEX Euler step. */
  async step(): Promise<void> {
    const dt = this.params.dt;
    const D = this.model.diffusivities(this.params);

    // Evaluate every species on the grid before reacting any of them
    for (let k = 0; k < this.nspecies; k++) {
      this.V[k] = await this.backend.synth(this.U[k]);
    }

    this.model.reaction(this.params, this.t, this.x, this.y, this.z, this.V, this.#R);

    for (let k = 0; k < this.nspecies; k++) {
      const Rlm = await this.backend.analys(this.#R[k]);
      const U = this.U[k];
      const dD = dt * D[k];
      for (let i = 0; i < this.backend.nlm; i++) {
        const fac = 1 / (1 + dD * this.lam[i]);
        U[2 * i] = (U[2 * i] + dt * Rlm[2 * i]) * fac;
        U[2 * i + 1] = (U[2 * i + 1] + dt * Rlm[2 * i + 1]) * fac;
      }
      this.#cleanM0(U);
    }

    this.t += dt;
    this.stepCount++;
  }

  /** m=0 coefficients of a real field are purely real; drop numerical junk. */
  #cleanM0(U: Float64Array): void {
    for (const p of this.#m0Imag) U[p] = 0;
  }
}
