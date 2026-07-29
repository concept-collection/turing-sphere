/**
 * Backend abstraction over the spherical harmonic transform, mirroring the
 * websph "porting boundary": the solver only ever needs coeffs->vals,
 * vals->coeffs, and the grid. Spectral layout is the SHTNS convention used
 * by shtns-webgpu (see src/sht/layout.ts): complex interleaved [re, im],
 * m >= 0 only, m-major ordering, orthonormal + Condon-Shortley.
 */
import { ShtPlan, requestShtDevice } from '../sht/sht.ts';
import { ShtReference } from '../sht/reference.ts';
import type { ShtConfig } from '../sht/layout.ts';

export interface ShtBackend {
  readonly cfg: ShtConfig;
  readonly nlm: number;
  /** cos(colatitude), length nlat, decreasing (north to south). */
  readonly cosTheta: Float64Array;
  readonly kind: 'webgpu' | 'cpu';
  synth(qlm: Float64Array): Promise<Float32Array | Float64Array>;
  analys(spat: Float64Array): Promise<Float32Array | Float64Array>;
  destroy(): void;
}

/** fp32 WebGPU backend (fast path). */
export class GpuBackend implements ShtBackend {
  readonly kind = 'webgpu';
  readonly cfg: ShtConfig;
  readonly nlm: number;
  readonly cosTheta: Float64Array;
  #plan: ShtPlan;
  #qlm32: Float32Array;
  #spat32: Float32Array;

  private constructor(plan: ShtPlan) {
    this.#plan = plan;
    this.cfg = plan.cfg;
    this.nlm = plan.nlm;
    this.cosTheta = plan.cosTheta;
    this.#qlm32 = new Float32Array(2 * plan.nlm);
    this.#spat32 = new Float32Array(plan.cfg.nlat * plan.cfg.nphi);
  }

  static async create(device: GPUDevice, cfg: ShtConfig): Promise<GpuBackend> {
    return new GpuBackend(await ShtPlan.create(device, cfg));
  }

  synth(qlm: Float64Array): Promise<Float32Array> {
    this.#qlm32.set(qlm);
    return this.#plan.synth(this.#qlm32);
  }

  analys(spat: Float64Array): Promise<Float32Array> {
    this.#spat32.set(spat);
    return this.#plan.analys(this.#spat32);
  }

  destroy(): void {
    this.#plan.destroy();
  }
}

/** f64 CPU backend by direct summation (slow; tests and no-WebGPU fallback). */
export class CpuBackend implements ShtBackend {
  readonly kind = 'cpu';
  readonly cfg: ShtConfig;
  readonly nlm: number;
  readonly cosTheta: Float64Array;
  #ref: ShtReference;

  constructor(cfg: ShtConfig) {
    this.#ref = new ShtReference(cfg);
    this.cfg = cfg;
    this.nlm = this.#ref.nlm;
    this.cosTheta = this.#ref.ct;
  }

  synth(qlm: Float64Array): Promise<Float64Array> {
    return Promise.resolve(this.#ref.synth(qlm));
  }

  analys(spat: Float64Array): Promise<Float64Array> {
    return Promise.resolve(this.#ref.analys(spat));
  }

  destroy(): void {}
}

export { requestShtDevice };
