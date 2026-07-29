import { requestShtDevice, ShtPlan } from './sht/sht.ts';
import { describeAdapter } from './solver/backend.ts';
import { gridForLmax, makeRandn } from './solver/simulation.ts';
import { presets, type Params } from './solver/models.ts';
import { GpuModel } from './mgpu/model.ts';
import { mModelByKey, type MModel } from './mgpu/registry.ts';
import { ModelCompileError, formatFailure } from './mgpu/errors.ts';
import { EXTERNAL_OPS } from './mgpu/externals.ts';
import { CodeEditor } from './editor/codeEditor.ts';
import {
  formatCommand,
  resolvePreset,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  type RunSpec,
} from './bench/runSpec.ts';
import {
  buildTopology,
  fillFieldValues,
  fillColors,
  type SphereMeshTopology,
} from './render/sphereMesh.ts';
import { SphereScene } from './render/SphereScene.ts';
import { Colorbar } from './render/colorbar.ts';
import { colormaps, colormapNames } from './render/colormaps.ts';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const elModel = $<HTMLSelectElement>('model');
const elLmax = $<HTMLSelectElement>('lmax');
const elColormap = $<HTMLSelectElement>('colormap');
const elRunPause = $<HTMLButtonElement>('runpause');
const elReseed = $<HTMLButtonElement>('reseed');
const elResetView = $<HTMLButtonElement>('resetview');
const elParams = $('params');
const elPanels = $('panels');
const elStats = $('stats');
const elCmd = $('cmd');
const elCopyCmd = $<HTMLButtonElement>('copycmd');
const elBlurb = $('blurb');
const elErr = $('err');
const elSource = $<HTMLTextAreaElement>('source');
const elHighlight = $('highlight');
const elCompiled = $('compiled');
const elEditorTitle = $('editor-title');
const elRecompile = $<HTMLButtonElement>('recompile');
const elRevert = $<HTMLButtonElement>('revert');

for (const p of presets) {
  const o = document.createElement('option');
  o.value = p.key;
  o.textContent = p.label;
  elModel.append(o);
}
for (const name of colormapNames) {
  const o = document.createElement('option');
  o.value = name;
  o.textContent = name;
  elColormap.append(o);
}
elColormap.value = 'jet';

/** The model source, with MATLAB highlighting. The host-provided operations are
 *  marked so the boundary between the model and what it is given is visible. */
const editor = new CodeEditor({
  textarea: elSource,
  overlay: elHighlight,
  external: EXTERNAL_OPS,
  onInput: (value) => {
    editedSource = value;
    elRecompile.textContent = 'Recompile *';
  },
});

/** Timesteps submitted per rendered frame. Nothing is read back between them,
 *  so the batch costs one submit and one readback regardless of size. */
const STEPS_PER_FRAME = 4;

// ---------------------------------------------------------------- state
let device: GPUDevice | null = null;
let sht: ShtPlan | null = null;
let gpu: GpuModel | null = null;
let topo: SphereMeshTopology | null = null;
let scenes: SphereScene[] = [];
let colorbars: Colorbar[] = [];
let valueBufs: Float32Array[] = [];
let colorBufs: Float32Array[] = [];
let ranges: { lo: number; hi: number }[] = [];
let resizeObs: ResizeObserver | null = null;

const initial = resolvePreset(presets[0].key);
let model: MModel = mModelByKey(initial.model.key)!;
let params: Params = initial.params;
/** The .m as edited in the page; `null` while it matches the file. */
let editedSource: string | null = null;
let seed = 1;
let running = false;
let adapterName = '';
let pumping = false;
let stepMs = 0;
let simTime = 0;
let stepCount = 0;
let generation = 0; // bumped on every rebuild to cancel stale pumps

const source = (): string => editedSource ?? model.source;

// ---------------------------------------------------------------- UI wiring
function buildParamInputs(): void {
  elParams.replaceChildren();
  for (const spec of model.params) {
    const label = document.createElement('label');
    label.textContent = `${spec.label} `;
    const input = document.createElement('input');
    input.type = 'number';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(params[spec.key]);
    input.addEventListener('change', () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) params[spec.key] = v;
      // Parameters are uniforms, not constants baked into the kernels, so a
      // change costs an upload rather than a recompile.
      gpu?.setParams(params);
      updateCommand();
    });
    label.append(input);
    elParams.append(label);
  }
}

function applyPreset(presetKey: string): void {
  const resolved = resolvePreset(presetKey);
  const next = mModelByKey(resolved.model.key);
  if (!next) {
    elErr.textContent = `No .m model for '${resolved.model.key}'`;
    return;
  }
  model = next;
  params = resolved.params;
  editedSource = null;
  editor.value = model.source;
  elEditorTitle.textContent =
    `models/${model.key}.m — init() and step(), compiled to WebGPU`;
  buildParamInputs();
  elBlurb.textContent = model.blurb;
  updateCommand();
}

/** The run currently on screen, as the benchmark's RunSpec. */
function currentSpec(): RunSpec {
  return {
    preset: elModel.value,
    lmax: Number(elLmax.value),
    backend: 'webgpu',
    seed,
    steps: DEFAULT_STEPS,
    warmup: DEFAULT_WARMUP,
    params,
  };
}

function updateCommand(): void {
  elCmd.textContent = formatCommand(currentSpec());
}

elModel.addEventListener('change', () => {
  applyPreset(elModel.value);
  void rebuild();
});
elLmax.addEventListener('change', () => void rebuild());
elColormap.addEventListener('change', () => void draw());

function setRunning(next: boolean): void {
  running = next;
  elRunPause.textContent = running ? 'Pause' : 'Run';
  if (running) void pump();
}

elRunPause.addEventListener('click', () => setRunning(!running));
elReseed.addEventListener('click', () => {
  seed = (Math.random() * 2 ** 31) >>> 0;
  setRunning(false);
  updateCommand();
  void reseed();
});
elResetView.addEventListener('click', () => {
  for (const s of scenes) s.resetCamera();
});

elRecompile.addEventListener('click', () => {
  editedSource = editor.value;
  void rebuild();
});
elRevert.addEventListener('click', () => {
  editedSource = null;
  editor.value = model.source;
  void rebuild();
});

// The command reproduces this run's parameters on the desktop; keep it
// selectable even where the clipboard API is unavailable.
elCopyCmd.addEventListener('click', () => {
  const text = elCmd.textContent ?? '';
  const flash = (msg: string): void => {
    elCopyCmd.textContent = msg;
    setTimeout(() => (elCopyCmd.textContent = 'Copy'), 1200);
  };
  const selectCommand = (): void => {
    const range = document.createRange();
    range.selectNodeContents(elCmd);
    const sel = getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
    flash('Selected');
  };
  if (!navigator.clipboard) return selectCommand();
  navigator.clipboard.writeText(text).then(() => flash('Copied'), selectCommand);
});

// ---------------------------------------------------------------- setup
function disposeView(): void {
  for (const s of scenes) s.dispose();
  scenes = [];
  colorbars = [];
  resizeObs?.disconnect();
  resizeObs = null;
  elPanels.replaceChildren();
}

/** Seeded perturbation, one normal deviate per grid point. */
function makeNoise(npts: number): Float32Array {
  const randn = makeRandn(seed);
  const noise = new Float32Array(npts);
  for (let i = 0; i < npts; i++) noise[i] = model.seedAmp * randn();
  return noise;
}

/** Report a compile failure, and select the offending text in the editor. */
function reportCompileError(e: unknown): void {
  elErr.textContent = formatFailure(e, source());
  elCompiled.textContent = '';
  if (e instanceof ModelCompileError && e.start !== undefined) {
    editor.select(e.start, e.end ?? e.start);
  }
}

async function rebuild(): Promise<void> {
  generation++;
  const gen = generation;
  setRunning(false);
  disposeView();
  gpu?.destroy();
  gpu = null;
  sht?.destroy();
  sht = null;
  stepMs = 0;
  simTime = 0;
  stepCount = 0;
  elErr.textContent = '';
  updateCommand();
  if (!device) return;

  const lmax = Number(elLmax.value);
  const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
  const cfg = { lmax, mmax: lmax, nlat, nphi };

  try {
    sht = await ShtPlan.create(device, cfg);
    gpu = await GpuModel.create({
      device,
      sht,
      cfg,
      source: source(),
      paramNames: model.params.map((p) => p.key),
      state: model.state,
      view: model.species,
    });
  } catch (e) {
    reportCompileError(e);
    gpu?.destroy();
    gpu = null;
    sht?.destroy();
    sht = null;
    return;
  }
  if (gen !== generation) return;

  gpu.setParams(params);
  gpu.init(makeNoise(nlat * nphi));

  const plan = gpu.describe();
  elCompiled.textContent =
    `one step compiled to ${plan.step.length} GPU operations:\n` +
    plan.step.map((l) => `  ${l}`).join('\n');
  elRecompile.textContent = 'Recompile';

  // mesh + scenes
  const phi = new Float64Array(nphi);
  for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
  topo = buildTopology(sht.cosTheta, phi);

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();
  for (let k = 0; k < model.species.length; k++) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    const box = document.createElement('div');
    box.className = 'sphere-box';
    const tag = document.createElement('div');
    tag.className = 'species-tag';
    tag.textContent = model.species[k];
    box.append(tag);
    const side = document.createElement('div');
    panel.append(box, side);
    elPanels.append(panel);

    const scene = new SphereScene(
      box,
      topo.numVertices,
      topo.indices,
      topo.sphereRef,
      sphereBg || undefined,
    );
    scene.fitCamera();
    scenes.push(scene);
    colorbars.push(new Colorbar(side));
    valueBufs[k] = new Float32Array(topo.numVertices);
    colorBufs[k] = new Float32Array(topo.numVertices * 3);
    ranges[k] = { lo: NaN, hi: NaN };
  }
  for (let k = 1; k < scenes.length; k++) scenes[0].syncCamerasWith(scenes[k]);

  resizeObs = new ResizeObserver(() => {
    const boxes = elPanels.querySelectorAll<HTMLElement>('.sphere-box');
    boxes.forEach((box, i) => {
      scenes[i]?.resize(box.clientWidth, box.clientHeight);
    });
  });
  elPanels
    .querySelectorAll<HTMLElement>('.sphere-box')
    .forEach((box) => resizeObs!.observe(box));

  await draw();
  updateStats();
  void pump();
}

async function reseed(): Promise<void> {
  if (!gpu || !sht) return;
  const gen = generation;
  gpu.init(makeNoise(sht.cfg.nlat * sht.cfg.nphi));
  simTime = 0;
  stepCount = 0;
  if (gen !== generation) return;
  for (const r of ranges) {
    r.lo = NaN;
    r.hi = NaN;
  }
  await draw();
  updateStats();
}

// ---------------------------------------------------------------- drawing
async function draw(): Promise<void> {
  if (!gpu || !topo) return;
  const gen = generation;
  const cmap = colormaps[elColormap.value] ?? colormaps.viridis;
  for (let k = 0; k < model.species.length; k++) {
    // The one readback per frame — the loop is otherwise entirely on the GPU.
    // A rebuild can land while this is in flight and destroy the buffer being
    // mapped, which rejects the map; that result is stale anyway, so drop it.
    let field: Float32Array;
    try {
      field = await gpu.read(model.species[k]);
    } catch (e) {
      if (gen !== generation) return;
      throw e;
    }
    if (gen !== generation || !topo) return;
    fillFieldValues(valueBufs[k], field, topo);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of valueBufs[k]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    // smooth the color range in both directions so the shading evolves
    // gently as the pattern grows (out-of-range values clamp meanwhile)
    const r = ranges[k];
    if (!Number.isFinite(r.lo)) {
      r.lo = lo;
      r.hi = hi;
    } else {
      const a = 0.15;
      r.lo += a * (lo - r.lo);
      r.hi += a * (hi - r.hi);
    }
    if (r.hi - r.lo < 1e-9) {
      const mid = (r.hi + r.lo) / 2;
      r.lo = mid - 5e-10;
      r.hi = mid + 5e-10;
    }
    fillColors(colorBufs[k], valueBufs[k], r.lo, r.hi, cmap);
    scenes[k]?.updateColors(colorBufs[k]);
    colorbars[k]?.update(cmap, r.lo, r.hi);
  }
}

function updateStats(): void {
  if (!gpu || !sht) return;
  const { nlat, nphi } = sht.cfg;
  const kind = `WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}`;
  const rate = stepMs > 0 ? `${(1000 / stepMs).toFixed(1)} steps/s` : '—';
  elStats.innerHTML =
    `<b>${kind}</b> · grid ${nlat}×${nphi} · nlm ${sht.nlm.toLocaleString()} · ` +
    `${stepMs > 0 ? stepMs.toFixed(1) : '—'} ms/step · ${rate} · ` +
    `t = <b>${simTime.toFixed(2)}</b> (${stepCount} steps)`;
}

// ---------------------------------------------------------------- sim loop
const nextFrame = () => new Promise<number>(requestAnimationFrame);

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  const gen = generation;
  try {
    while (running && gpu && gen === generation) {
      const t0 = performance.now();
      gpu.step(STEPS_PER_FRAME);
      // draw() awaits the readback, which also waits for the batch to finish,
      // so this measures the real end-to-end cost per step.
      await draw();
      if (gen !== generation) break;
      const dtMs = (performance.now() - t0) / STEPS_PER_FRAME;
      stepMs = stepMs === 0 ? dtMs : stepMs + 0.05 * (dtMs - stepMs);
      simTime += STEPS_PER_FRAME * (params.dt ?? 0);
      stepCount += STEPS_PER_FRAME;
      updateStats();
      await nextFrame();
    }
    if (gen === generation) {
      await draw();
      updateStats();
    }
  } finally {
    pumping = false;
  }
}

// ---------------------------------------------------------------- boot
async function boot(): Promise<void> {
  elModel.value = presets[0].key;
  applyPreset(presets[0].key);
  try {
    device = await requestShtDevice();
    adapterName = await describeAdapter(device);
  } catch (e) {
    device = null;
    elErr.textContent =
      `WebGPU is not available (${e instanceof Error ? e.message : e}). ` +
      `This demo compiles the MATLAB solver to WebGPU compute shaders, so it ` +
      `needs a WebGPU-capable browser (Chrome/Edge 113+).`;
    return;
  }
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      elErr.textContent = `WebGPU device lost: ${info.message}`;
    }
  });
  await rebuild();
}

void boot();
