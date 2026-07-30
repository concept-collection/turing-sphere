/**
 * One running model: grid, transforms, compiled .m, seeded state.
 *
 * Everything that is not rendering. The app, the desktop benchmark and the
 * tests all go through this, so there is one place that decides how a model is
 * turned into something running on the GPU — and nothing about it is
 * browser-specific beyond needing a GPUDevice.
 */
import { ShtPlan } from '../sht/sht.ts';
import { gridForLmax, type ShtConfig } from '../sht/layout.ts';
import { GpuModel, type ModelParams } from './model.ts';
import { seededNoise } from './noise.ts';
import type { MModel } from './registry.ts';

export interface ModelSessionOptions {
  device: GPUDevice;
  model: MModel;
  params: ModelParams;
  lmax: number;
  /** Override the model source — the editor's working copy. */
  source?: string;
  /** Linear render oversampling: read the species fields on a grid this many
   *  times finer than the solver's in each direction (default 1). The state is
   *  band-limited at lmax, so the finer evaluation is exact interpolation. */
  oversample?: number;
}

export class ModelSession {
  readonly device: GPUDevice;
  readonly model: MModel;
  readonly cfg: ShtConfig;
  readonly sht: ShtPlan;
  readonly gpu: GpuModel;
  readonly npts: number;

  /** Model time and step count since the last seeding. */
  t = 0;
  steps = 0;

  #params: ModelParams;
  /** Display-only transforms on the oversampled grid; null at 1x. */
  #displaySht: ShtPlan | null;
  #oversample: number;

  private constructor(init: {
    device: GPUDevice;
    model: MModel;
    cfg: ShtConfig;
    sht: ShtPlan;
    displaySht: ShtPlan | null;
    gpu: GpuModel;
    params: ModelParams;
    oversample: number;
  }) {
    this.device = init.device;
    this.model = init.model;
    this.cfg = init.cfg;
    this.sht = init.sht;
    this.gpu = init.gpu;
    this.npts = init.cfg.nlat * init.cfg.nphi;
    this.#oversample = init.oversample;
    this.#params = init.params;
    this.#displaySht = init.displaySht;
  }

  /** Linear render oversampling factor (1 = read on the solver grid). */
  get oversample(): number {
    return this.#oversample;
  }

  static async create(opts: ModelSessionOptions): Promise<ModelSession> {
    const { device, model, params, lmax } = opts;
    const oversample = Math.max(1, Math.round(opts.oversample ?? 1));
    const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const sht = await ShtPlan.create(device, cfg);
    let displaySht: ShtPlan | null = null;
    try {
      // The display plan shares nothing with the solver's beyond the
      // coefficients copied into it per readback; its grid is the solver's
      // scaled by the oversampling factor, so nphi stays a power of two (the
      // FFT path) for power-of-two factors.
      if (oversample > 1) {
        displaySht = await ShtPlan.create(device, {
          lmax,
          mmax: lmax,
          nlat: oversample * nlat,
          nphi: oversample * nphi,
        });
      }
      const gpu = await GpuModel.create({
        device,
        sht,
        cfg,
        source: opts.source ?? model.source,
        paramNames: model.params.map((p) => p.key),
        state: model.state,
        view: model.species,
      });
      gpu.setParams(params);
      return new ModelSession({
        device, model, cfg, sht, displaySht, gpu, params, oversample,
      });
    } catch (e) {
      // The transform plans own GPU buffers; do not leak them on a compile error.
      displaySht?.destroy();
      sht.destroy();
      throw e;
    }
  }

  /** The plan whose grid `readSpecies` samples on — the display plan when
   *  oversampling, otherwise the solver's. Its cosTheta/nphi define the mesh. */
  get viewSht(): ShtPlan {
    return this.#displaySht ?? this.sht;
  }

  /**
   * Change the display oversampling in place. Display-only: the simulation
   * state, time and parameters are untouched, so the run continues seamlessly
   * on the new render grid. The caller must not have a readSpecies in flight —
   * its readback maps a buffer of the plan being destroyed.
   */
  async setOversample(oversample: number): Promise<void> {
    const os = Math.max(1, Math.round(oversample));
    if (os === this.#oversample) return;
    const next =
      os > 1
        ? await ShtPlan.create(this.device, {
            lmax: this.cfg.lmax,
            mmax: this.cfg.mmax,
            nlat: os * this.cfg.nlat,
            nphi: os * this.cfg.nphi,
          })
        : null;
    const old = this.#displaySht;
    this.#displaySht = next;
    this.#oversample = os;
    old?.destroy();
  }

  /** Run `init` from a seeded perturbation, resetting model time. */
  seed(seed: number): void {
    this.gpu.init(seededNoise(this.npts, this.model.seedAmp, seed));
    this.t = 0;
    this.steps = 0;
  }

  setParams(params: ModelParams): void {
    this.#params = params;
    this.gpu.setParams(params);
  }

  /** Advance `n` steps. Synchronous: records and submits, nothing read back. */
  step(n = 1): void {
    this.gpu.step(n);
    this.t += n * (this.#params.dt ?? 0);
    this.steps += n;
  }

  /**
   * Wait for the submitted steps to finish, without reading anything back.
   * This is the honest way to time the solver: a readback would add a GPU->CPU
   * round trip, which in a browser also crosses a process boundary and can cost
   * more than the steps themselves.
   */
  sync(): Promise<undefined> {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Time a batch of `n` steps and return ms/step, leaving the simulation
   * exactly where it was: the spectral state is snapshotted before the batch
   * and restored after, and `t`/`steps` do not advance. One sync amortized
   * over the batch — the same measurement the desktop benchmark makes. The
   * grid view fields hold the batch's output until the next real step, so
   * step before reading them.
   */
  async measure(n: number): Promise<number> {
    this.gpu.snapshotState();
    const t0 = performance.now();
    this.gpu.step(n);
    await this.sync();
    const ms = (performance.now() - t0) / n;
    this.gpu.restoreState();
    return ms;
  }

  /** Read a named value (a grid field or the spectral state). */
  read(name: string): Promise<Float32Array> {
    return this.gpu.read(name);
  }

  /**
   * Read species `k` at render resolution (`viewSht`'s grid). Without
   * oversampling this is the grid field the .m returned. With oversampling the
   * spectral state is synthesized on the finer grid instead — the same field,
   * since the models define each species as synth of its state, evaluated
   * exactly on more points.
   */
  readSpecies(k: number): Promise<Float32Array> {
    if (!this.#displaySht) return this.read(this.model.species[k]);
    const state = this.model.state[k];
    const buf = this.gpu.valueBuffer(state);
    if (!buf) throw new Error(`readSpecies: no buffer for state '${state}'`);
    return this.#displaySht.synthFrom(buf);
  }

  describe(): { init: string[]; step: string[] } {
    return this.gpu.describe();
  }

  destroy(): void {
    this.gpu.destroy();
    this.#displaySht?.destroy();
    this.sht.destroy();
  }
}
