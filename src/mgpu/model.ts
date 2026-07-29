/**
 * A .m model, compiled and running on the GPU.
 *
 * A model file is ordinary MATLAB: it defines an `init` function that builds the
 * initial spectral state and a `step` function that advances it one timestep.
 * Each is specialized for the current grid and compiled into a ModelPlan, and
 * both operate on the same state buffers (see HostBuffers).
 *
 * Both functions return the new state followed by the grid fields the app
 * renders, so their signatures say exactly what they produce:
 *
 *   function [U, V, u, v] = init(noise, a, b)
 *   function [U, V, u, v] = step(U, V, lam, a, b, D1, D2, dt)
 *
 * The host supplies the things that are precomputation rather than algorithm:
 * the grid, the Laplace-Beltrami eigenvalues, the seeded initial noise, and the
 * parameter values. Each argument is matched to the .m's declared parameter
 * name, so the file documents its own interface.
 */
import { ShtPlan } from '../sht/sht.ts';
import { lmIndex, type ShtConfig } from '../sht/layout.ts';
import { HostBuffers, ModelPlan } from './plan.ts';
import { inFunction, inFunctionAsync, inModel } from './errors.ts';
import { CompiledModel, type Binding } from './compile.ts';

export interface ModelParams {
  [key: string]: number;
}

export interface GpuModelOptions {
  device: GPUDevice;
  sht: ShtPlan;
  cfg: ShtConfig;
  /** Model source (.m text). */
  source: string;
  /** Parameter names the .m may take as arguments. */
  paramNames: string[];
  /** Spectral state names, in order (e.g. ['U', 'V']). */
  state: string[];
  /** Grid fields to render, in order (e.g. ['u', 'v']). */
  view: string[];
}

/** Laplace-Beltrami eigenvalues l(l+1), duplicated across re/im so the array
 *  matches the 2 x nlm spectral layout element for element. */
export function eigenvalues(cfg: ShtConfig, nlm: number): Float32Array {
  const lam = new Float32Array(2 * nlm);
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = lmIndex(cfg.lmax, l, m);
      lam[2 * i] = l * (l + 1);
      lam[2 * i + 1] = l * (l + 1);
    }
  }
  return lam;
}

export class GpuModel {
  readonly paramNames: string[];
  readonly state: string[];
  readonly view: string[];
  readonly npts: number;
  readonly nlm: number;

  #device: GPUDevice;
  #host: HostBuffers;
  #initPlan: ModelPlan;
  #stepPlan: ModelPlan;
  #readback: GPUBuffer;
  /** Which function wrote the state most recently; see `read`. */
  #lastRan: 'init' | 'step' = 'init';

  private constructor(init: {
    device: GPUDevice;
    host: HostBuffers;
    initPlan: ModelPlan;
    stepPlan: ModelPlan;
    readback: GPUBuffer;
    paramNames: string[];
    state: string[];
    view: string[];
    npts: number;
    nlm: number;
  }) {
    this.#device = init.device;
    this.#host = init.host;
    this.#initPlan = init.initPlan;
    this.#stepPlan = init.stepPlan;
    this.#readback = init.readback;
    this.paramNames = init.paramNames;
    this.state = init.state;
    this.view = init.view;
    this.npts = init.npts;
    this.nlm = init.nlm;
  }

  static async create(opts: GpuModelOptions): Promise<GpuModel> {
    const { device, sht, cfg, source, paramNames, state, view } = opts;
    const npts = cfg.nlat * cfg.nphi;
    const nlm = sht.nlm;

    // What the .m may ask for by parameter name. Spectral state and the
    // eigenvalues are 2 x nlm; the seeded perturbation is a grid field.
    const bindings: Record<string, Binding> = {
      lam: { kind: 'tensor', shape: [2, nlm] },
      noise: { kind: 'tensor', shape: [npts, 1] },
      npts: { kind: 'const', value: npts },
      nlm: { kind: 'const', value: nlm },
    };
    for (const s of state) bindings[s] = { kind: 'tensor', shape: [2, nlm] };
    for (const p of paramNames) bindings[p] = { kind: 'param' };

    // Parsing belongs to the file, not to either function.
    const compiled = inModel(() => new CompiledModel(source, bindings, { npts, nlm }));
    // Both functions return the new state first, then the rendered grid fields.
    const nargout = state.length + view.length;
    const initFn = inFunction('init', () => compiled.specialize('init', nargout));
    const stepFn = inFunction('step', () => compiled.specialize('step', nargout));
    compiled.finish();

    // Only the state outputs feed back into the argument buffers; the grid
    // fields are read for display and then overwritten next call.
    const feedback = [...state, ...view.map(() => null)];

    const host = new HostBuffers(device);
    // The host owns the state and the inputs it uploads, whether or not a given
    // function happens to take them as arguments — `init` does not read `U`, but
    // it writes it, and `step` reads it back.
    for (const s of state) host.ensure(s, 2 * nlm);
    host.ensure('lam', 2 * nlm);
    host.ensure('noise', npts);

    const initPlan = await inFunctionAsync('init', () =>
      ModelPlan.create(device, sht, { fn: initFn, feedback }, host),
    );
    const stepPlan = await inFunctionAsync('step', () =>
      ModelPlan.create(device, sht, { fn: stepFn, feedback }, host),
    );

    host.upload('lam', eigenvalues(cfg, nlm));

    const readback = device.createBuffer({
      label: 'mgpu-readback',
      size: 4 * Math.max(npts, 2 * nlm),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    return new GpuModel({
      device, host, initPlan, stepPlan, readback,
      paramNames, state, view, npts, nlm,
    });
  }

  setParams(params: ModelParams): void {
    this.#initPlan.setParams(params);
    this.#stepPlan.setParams(params);
  }

  /**
   * Write a host-owned value directly — the spectral state, or one of the input
   * fields. Lets a test set up an exact initial condition (a single spherical-
   * harmonic mode, say) instead of going through `init`.
   */
  upload(name: string, data: Float32Array): void {
    this.#host.upload(name, data);
  }

  /** Upload the seeded perturbation and run `init`. */
  init(noise: Float32Array): void {
    this.#host.upload('noise', noise);
    const enc = this.#device.createCommandEncoder({ label: 'mgpu-init' });
    this.#initPlan.encodeSteps(enc, 1);
    this.#device.queue.submit([enc.finish()]);
    this.#lastRan = 'init';
  }

  /**
   * Advance `steps` timesteps. Synchronous — this only records commands and
   * submits them; nothing is read back and nothing is awaited.
   */
  step(steps = 1): void {
    const enc = this.#device.createCommandEncoder({ label: 'mgpu-step' });
    this.#stepPlan.encodeSteps(enc, steps);
    this.#device.queue.submit([enc.finish()]);
    this.#lastRan = 'step';
  }

  /**
   * Read a named value back to the CPU. The only await in the whole loop.
   *
   * Grid fields like `u` are produced by both functions, into separate buffers
   * (only the spectral state is shared), so this reads from whichever ran most
   * recently — which is what makes the first frame show the initial state
   * rather than an unwritten buffer.
   */
  async read(name: string): Promise<Float32Array> {
    const [first, second] =
      this.#lastRan === 'init'
        ? [this.#initPlan, this.#stepPlan]
        : [this.#stepPlan, this.#initPlan];
    const buffer = first.buffer(name) ?? second.buffer(name);
    const count = first.elementCount(name) ?? second.elementCount(name);
    if (!buffer || count === undefined) {
      throw new Error(`read: the model has no value named '${name}'`);
    }
    const enc = this.#device.createCommandEncoder({ label: `mgpu-read-${name}` });
    enc.copyBufferToBuffer(buffer, 0, this.#readback, 0, 4 * count);
    this.#device.queue.submit([enc.finish()]);
    await this.#readback.mapAsync(GPUMapMode.READ, 0, 4 * count);
    const out = new Float32Array(this.#readback.getMappedRange(0, 4 * count).slice(0));
    this.#readback.unmap();
    return out;
  }

  /** What the .m compiled to, for display. */
  describe(): { init: string[]; step: string[] } {
    return { init: this.#initPlan.describe(), step: this.#stepPlan.describe() };
  }

  destroy(): void {
    this.#initPlan.destroy();
    this.#stepPlan.destroy();
    this.#host.destroy();
    this.#readback.destroy();
  }
}
