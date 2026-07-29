/**
 * WebGPU spherical harmonic transform plan (scalar transforms, fp32).
 *
 * Mirrors the structure of the SHTNS CUDA backend (sht_gpu.cu):
 * host-side f64 precomputation of grid + recurrence coefficients, shader
 * source generated with sizes baked in (SHTNS uses NVRTC; WGSL is always
 * runtime-compiled), then per-transform: Legendre stage + Fourier stage.
 */
import { gaussNodesWeights } from './gauss.ts';
import { legendreCoeffs } from './coeffs.ts';
import { nlmCalc, validateConfig, isPowerOfTwo, type ShtConfig } from './layout.ts';
import { legSynthWGSL, legAnalysWGSL } from './wgsl/leg.ts';
import {
  fftSynthWGSL,
  fftAnalysWGSL,
  dftSynthWGSL,
  dftAnalysWGSL,
  fftThreads,
} from './wgsl/fourier.ts';

export type FourierMode = 'auto' | 'fft' | 'dft';

export interface ShtOptions {
  /** Fourier stage implementation.  'auto' picks fft when nphi is a power of two that fits in workgroup memory. */
  fourier?: FourierMode;
}

const WG_SYNTH = 64;
const WG_ANALYS = 256;

async function makePipeline(
  device: GPUDevice,
  code: string,
  entryPoint: string,
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label: entryPoint });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error(
      `WGSL compile error in ${entryPoint}:\n` +
        errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'),
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
    label: entryPoint,
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`pipeline ${entryPoint}: ${err.message}`);
  return pipeline;
}

export class ShtPlan {
  readonly cfg: ShtConfig;
  readonly nlm: number;
  readonly fourierMode: 'fft' | 'dft';
  /** Colatitudes theta_i (f64, increasing: north to south). */
  readonly theta: Float64Array;
  readonly cosTheta: Float64Array;
  readonly gaussWeights: Float64Array;

  private device: GPUDevice;
  private bufAb!: GPUBuffer;
  private bufAmm!: GPUBuffer;
  private bufCtstw!: GPUBuffer;
  private bufTrig!: GPUBuffer;
  /** Spectral input (synthesis) — write with queue.writeBuffer or use synth(). */
  readonly qlmIn!: GPUBuffer;
  /** Spectral output (analysis). */
  readonly qlmOut!: GPUBuffer;
  /** Fourier-space intermediate [(m)*nlat + ilat], complex f32. */
  readonly fmBuf!: GPUBuffer;
  /** Spatial field [ilat*nphi + iphi], f32. */
  readonly spatBuf!: GPUBuffer;
  private stageSpat!: GPUBuffer;
  private stageQ!: GPUBuffer;

  private pipeLegSynth!: GPUComputePipeline;
  private pipeLegAnalys!: GPUComputePipeline;
  private pipeFourSynth!: GPUComputePipeline;
  private pipeFourAnalys!: GPUComputePipeline;
  private bgLegSynth!: GPUBindGroup;
  private bgLegAnalys!: GPUBindGroup;
  private bgFourSynth!: GPUBindGroup;
  private bgFourAnalys!: GPUBindGroup;

  private constructor(device: GPUDevice, cfg: ShtConfig, fourierMode: 'fft' | 'dft') {
    this.device = device;
    this.cfg = cfg;
    this.nlm = nlmCalc(cfg.lmax, cfg.mmax);
    this.fourierMode = fourierMode;
    const { x, w } = gaussNodesWeights(cfg.nlat);
    this.cosTheta = x;
    this.gaussWeights = w;
    this.theta = new Float64Array(cfg.nlat);
    for (let i = 0; i < cfg.nlat; i++) this.theta[i] = Math.acos(x[i]);
  }

  static async create(device: GPUDevice, cfg: ShtConfig, opts: ShtOptions = {}): Promise<ShtPlan> {
    validateConfig(cfg);
    const want = opts.fourier ?? 'auto';
    const fftFits =
      isPowerOfTwo(cfg.nphi) &&
      16 * cfg.nphi <= device.limits.maxComputeWorkgroupStorageSize &&
      fftThreads(cfg.nphi) <= device.limits.maxComputeInvocationsPerWorkgroup;
    if (want === 'fft' && !fftFits) {
      throw new Error(
        `fourier:'fft' requires power-of-two nphi with 16*nphi <= maxComputeWorkgroupStorageSize ` +
          `(nphi=${cfg.nphi}, limit=${device.limits.maxComputeWorkgroupStorageSize})`,
      );
    }
    const mode: 'fft' | 'dft' = want === 'dft' ? 'dft' : fftFits ? 'fft' : 'dft';
    const plan = new ShtPlan(device, cfg, mode);
    await plan.init();
    return plan;
  }

  private async init(): Promise<void> {
    const { lmax, mmax, nlat, nphi } = this.cfg;
    const dev = this.device;
    const self = this as {
      -readonly [k in keyof ShtPlan]: ShtPlan[k];
    };

    // --- host precomputation (f64), then downcast to f32 for upload ---
    const { amm, ab } = legendreCoeffs(lmax, mmax);
    const ctstw = new Float32Array(3 * nlat);
    for (let i = 0; i < nlat; i++) {
      ctstw[i] = this.cosTheta[i];
      ctstw[nlat + i] = Math.sqrt(1 - this.cosTheta[i] * this.cosTheta[i]);
      ctstw[2 * nlat + i] = this.gaussWeights[i] * ((2 * Math.PI) / nphi);
    }
    // twiddle/phase table in f64 (device sin/cos is too inaccurate: ~2^-11 under Vulkan)
    const trig = new Float32Array(2 * nphi);
    for (let k = 0; k < nphi; k++) {
      trig[2 * k] = Math.cos((2 * Math.PI * k) / nphi);
      trig[2 * k + 1] = Math.sin((2 * Math.PI * k) / nphi);
    }

    const mkBuf = (label: string, size: number, usage: GPUBufferUsageFlags) =>
      dev.createBuffer({ label, size, usage });
    this.bufAb = mkBuf('sht-ab', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufAmm = mkBuf('sht-amm', 4 * (mmax + 1), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufCtstw = mkBuf('sht-ctstw', 4 * 3 * nlat, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufTrig = mkBuf('sht-trig', 8 * nphi, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    self.qlmIn = mkBuf('sht-qlm-in', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    self.qlmOut = mkBuf('sht-qlm-out', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    self.fmBuf = mkBuf('sht-fm', 8 * (mmax + 1) * nlat, GPUBufferUsage.STORAGE);
    self.spatBuf = mkBuf('sht-spat', 4 * nlat * nphi, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.stageSpat = mkBuf('sht-stage-spat', 4 * nlat * nphi, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.stageQ = mkBuf('sht-stage-q', 8 * this.nlm, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    dev.queue.writeBuffer(this.bufAb, 0, new Float32Array(ab));
    dev.queue.writeBuffer(this.bufAmm, 0, new Float32Array(amm));
    dev.queue.writeBuffer(this.bufCtstw, 0, ctstw);
    dev.queue.writeBuffer(this.bufTrig, 0, trig);

    // --- shaders / pipelines ---
    const legP = { lmax, mmax, nlat, wgSynth: WG_SYNTH, wgAnalys: WG_ANALYS };
    const fourP = { mmax, nlat, nphi };
    const [pLegS, pLegA, pFourS, pFourA] = await Promise.all([
      makePipeline(dev, legSynthWGSL(legP), 'leg_synth'),
      makePipeline(dev, legAnalysWGSL(legP), 'leg_analys'),
      makePipeline(
        dev,
        this.fourierMode === 'fft' ? fftSynthWGSL(fourP) : dftSynthWGSL(fourP),
        this.fourierMode === 'fft' ? 'fft_synth' : 'dft_synth',
      ),
      makePipeline(
        dev,
        this.fourierMode === 'fft' ? fftAnalysWGSL(fourP) : dftAnalysWGSL(fourP),
        this.fourierMode === 'fft' ? 'fft_analys' : 'dft_analys',
      ),
    ]);
    this.pipeLegSynth = pLegS;
    this.pipeLegAnalys = pLegA;
    this.pipeFourSynth = pFourS;
    this.pipeFourAnalys = pFourA;

    const entries = (bufs: GPUBuffer[]) =>
      bufs.map((buffer, binding) => ({ binding, resource: { buffer } }));
    this.bgLegSynth = dev.createBindGroup({
      layout: pLegS.getBindGroupLayout(0),
      entries: entries([this.bufAb, this.bufAmm, this.bufCtstw, this.qlmIn, this.fmBuf]),
    });
    this.bgLegAnalys = dev.createBindGroup({
      layout: pLegA.getBindGroupLayout(0),
      entries: entries([this.bufAb, this.bufAmm, this.bufCtstw, this.fmBuf, this.qlmOut]),
    });
    this.bgFourSynth = dev.createBindGroup({
      layout: pFourS.getBindGroupLayout(0),
      entries: entries([this.fmBuf, this.spatBuf, this.bufTrig]),
    });
    this.bgFourAnalys = dev.createBindGroup({
      layout: pFourA.getBindGroupLayout(0),
      entries: entries([this.spatBuf, this.fmBuf, this.bufTrig]),
    });
  }

  /** Record the synthesis (spectral qlmIn -> spatial spatBuf) into an encoder. */
  encodeSynth(encoder: GPUCommandEncoder): void {
    const { mmax, nlat, nphi } = this.cfg;
    const pass = encoder.beginComputePass({ label: 'sht-synth' });
    pass.setPipeline(this.pipeLegSynth);
    pass.setBindGroup(0, this.bgLegSynth);
    pass.dispatchWorkgroups(Math.ceil(nlat / WG_SYNTH), mmax + 1);
    pass.setPipeline(this.pipeFourSynth);
    pass.setBindGroup(0, this.bgFourSynth);
    if (this.fourierMode === 'fft') {
      pass.dispatchWorkgroups(nlat);
    } else {
      pass.dispatchWorkgroups(Math.ceil(nphi / 64), nlat);
    }
    pass.end();
  }

  /** Record the analysis (spatial spatBuf -> spectral qlmOut) into an encoder. */
  encodeAnalys(encoder: GPUCommandEncoder): void {
    const { mmax, nlat } = this.cfg;
    const pass = encoder.beginComputePass({ label: 'sht-analys' });
    pass.setPipeline(this.pipeFourAnalys);
    pass.setBindGroup(0, this.bgFourAnalys);
    if (this.fourierMode === 'fft') {
      pass.dispatchWorkgroups(nlat);
    } else {
      pass.dispatchWorkgroups(Math.ceil((mmax + 1) / 64), nlat);
    }
    pass.setPipeline(this.pipeLegAnalys);
    pass.setBindGroup(0, this.bgLegAnalys);
    pass.dispatchWorkgroups(mmax + 1);
    pass.end();
  }

  /**
   * Spectral -> spatial.  qlm: interleaved [re, im], SHTNS LM ordering,
   * length 2*nlm.  Returns the spatial field, length nlat*nphi.
   */
  async synth(qlm: Float32Array): Promise<Float32Array> {
    const { nlat, nphi } = this.cfg;
    if (qlm.length !== 2 * this.nlm) throw new Error(`qlm must have length ${2 * this.nlm}`);
    this.device.queue.writeBuffer(this.qlmIn, 0, qlm as Float32Array<ArrayBuffer>);
    const enc = this.device.createCommandEncoder();
    this.encodeSynth(enc);
    enc.copyBufferToBuffer(this.spatBuf, 0, this.stageSpat, 0, 4 * nlat * nphi);
    this.device.queue.submit([enc.finish()]);
    await this.stageSpat.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stageSpat.getMappedRange().slice(0));
    this.stageSpat.unmap();
    return out;
  }

  /** Spatial -> spectral.  spat: length nlat*nphi.  Returns interleaved qlm, length 2*nlm. */
  async analys(spat: Float32Array): Promise<Float32Array> {
    const { nlat, nphi } = this.cfg;
    if (spat.length !== nlat * nphi) throw new Error(`spat must have length ${nlat * nphi}`);
    this.device.queue.writeBuffer(this.spatBuf, 0, spat as Float32Array<ArrayBuffer>);
    const enc = this.device.createCommandEncoder();
    this.encodeAnalys(enc);
    enc.copyBufferToBuffer(this.qlmOut, 0, this.stageQ, 0, 8 * this.nlm);
    this.device.queue.submit([enc.finish()]);
    await this.stageQ.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stageQ.getMappedRange().slice(0));
    this.stageQ.unmap();
    return out;
  }

  destroy(): void {
    for (const b of [
      this.bufAb, this.bufAmm, this.bufCtstw, this.bufTrig, this.qlmIn, this.qlmOut,
      this.fmBuf, this.spatBuf, this.stageSpat, this.stageQ,
    ]) b?.destroy();
  }
}

/** Request an adapter/device suitable for the transforms. */
export async function requestShtDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  // ask for a larger workgroup storage if the adapter offers it (bigger FFTs)
  const wgStorage = Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768);
  return adapter.requestDevice({
    requiredLimits: { maxComputeWorkgroupStorageSize: wgStorage },
  });
}
