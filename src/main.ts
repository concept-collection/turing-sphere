import { requestShtDevice, describeAdapter } from './sht/sht.ts';
import { ModelSession } from './mgpu/session.ts';
import { mModelByKey, presets, type MModel, type Params } from './mgpu/registry.ts';
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

/**
 * Steps in a solver-timing burst, and how often to run one.
 *
 * Timing the solver needs a `queue.onSubmittedWorkDone()` to know the work
 * finished, and in a browser that is an IPC round trip into the GPU process — a
 * fixed cost of a few milliseconds. Spread over one frame's four steps it would
 * swamp them on a fast GPU and make the solver look far slower than it is. So the
 * rate is measured in an occasional larger batch, where the single sync is
 * amortized the way the desktop benchmark amortizes its own. These are ordinary
 * steps: the simulation advances by them like any others.
 */
const MEASURE_BURST = 32;
const MEASURE_EVERY_MS = 2000;

// ---------------------------------------------------------------- state
let device: GPUDevice | null = null;
let session: ModelSession | null = null;
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
let solverMs = 0;
let frameMs = 0;
let lastMeasure = 0;
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
      session?.setParams(params);
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
  session?.destroy();
  session = null;
  solverMs = 0;
  frameMs = 0;
  lastMeasure = 0;
  elErr.textContent = '';
  updateCommand();
  if (!device) return;

  try {
    session = await ModelSession.create({
      device,
      model,
      params,
      lmax: Number(elLmax.value),
      source: source(),
    });
  } catch (e) {
    reportCompileError(e);
    return;
  }
  if (gen !== generation) return;

  session.seed(seed);

  const plan = session.describe();
  elCompiled.textContent =
    `one step compiled to ${plan.step.length} GPU operations:\n` +
    plan.step.map((l) => `  ${l}`).join('\n');
  elRecompile.textContent = 'Recompile';

  // mesh + scenes
  const { nphi } = session.cfg;
  const phi = new Float64Array(nphi);
  for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
  topo = buildTopology(session.sht.cosTheta, phi);

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
  if (!session) return;
  const gen = generation;
  session.seed(seed);
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
  if (!session || !topo) return;
  const gen = generation;
  const cmap = colormaps[elColormap.value] ?? colormaps.viridis;
  for (let k = 0; k < model.species.length; k++) {
    // The one readback per frame — the loop is otherwise entirely on the GPU.
    // A rebuild can land while this is in flight and destroy the buffer being
    // mapped, which rejects the map; that result is stale anyway, so drop it.
    let field: Float32Array;
    try {
      field = await session.read(model.species[k]);
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
  if (!session) return;
  const { nlat, nphi } = session.cfg;
  const kind = `WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}`;
  const solver =
    solverMs > 0
      ? `<b>${solverMs.toFixed(2)} ms/step</b> (${(1000 / solverMs).toFixed(0)} steps/s, ` +
        `batch of ${MEASURE_BURST}, no readback)`
      : '—';
  const frame =
    frameMs > 0
      ? `${frameMs.toFixed(1)} ms/frame (${STEPS_PER_FRAME} steps + readback + render)`
      : '—';
  elStats.innerHTML =
    `<b>${kind}</b> · grid ${nlat}×${nphi} · nlm ${session.sht.nlm.toLocaleString()} · ` +
    `${session.sht.fourierMode.toUpperCase()} · solver ${solver} · ${frame} · ` +
    `t = <b>${session.t.toFixed(2)}</b> (${session.steps} steps)`;
}

// ---------------------------------------------------------------- sim loop
const nextFrame = () => new Promise<number>(requestAnimationFrame);

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  const gen = generation;
  try {
    while (running && session && gen === generation) {
      // Occasionally, a burst purely to measure the solver rate: many steps,
      // one sync, nothing read back — directly comparable to the desktop
      // benchmark's throughput number.
      if (performance.now() - lastMeasure > MEASURE_EVERY_MS) {
        const m0 = performance.now();
        session.step(MEASURE_BURST);
        await session.sync();
        if (gen !== generation) break;
        solverMs = (performance.now() - m0) / MEASURE_BURST;
        lastMeasure = performance.now();
      }

      // The frame itself. No explicit sync here — draw()'s readback already
      // waits for the steps, so asking twice would only add a round trip.
      const t0 = performance.now();
      session.step(STEPS_PER_FRAME);
      await draw();
      if (gen !== generation) break;
      frameMs = frameMs === 0
        ? performance.now() - t0
        : frameMs + 0.05 * (performance.now() - t0 - frameMs);
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
