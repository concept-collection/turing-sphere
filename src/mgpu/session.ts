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

  private constructor(init: {
    device: GPUDevice;
    model: MModel;
    cfg: ShtConfig;
    sht: ShtPlan;
    gpu: GpuModel;
    params: ModelParams;
  }) {
    this.device = init.device;
    this.model = init.model;
    this.cfg = init.cfg;
    this.sht = init.sht;
    this.gpu = init.gpu;
    this.npts = init.cfg.nlat * init.cfg.nphi;
    this.#params = init.params;
  }

  static async create(opts: ModelSessionOptions): Promise<ModelSession> {
    const { device, model, params, lmax } = opts;
    const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const sht = await ShtPlan.create(device, cfg);
    try {
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
      return new ModelSession({ device, model, cfg, sht, gpu, params });
    } catch (e) {
      // The transform plan owns GPU buffers; do not leak them on a compile error.
      sht.destroy();
      throw e;
    }
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

  /** Read a named value (a grid field or the spectral state). */
  read(name: string): Promise<Float32Array> {
    return this.gpu.read(name);
  }

  describe(): { init: string[]; step: string[] } {
    return this.gpu.describe();
  }

  destroy(): void {
    this.gpu.destroy();
    this.sht.destroy();
  }
}
