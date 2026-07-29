import {
  GpuBackend,
  CpuBackend,
  requestShtDevice,
  type ShtBackend,
} from './solver/backend.ts';
import { Simulation, gridForLmax } from './solver/simulation.ts';
import {
  models,
  presets,
  defaultParams,
  type ModelSpec,
  type Params,
} from './solver/models.ts';
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
const elBackend = $<HTMLSelectElement>('backend');
const elRunPause = $<HTMLButtonElement>('runpause');
const elReseed = $<HTMLButtonElement>('reseed');
const elResetView = $<HTMLButtonElement>('resetview');
const elParams = $('params');
const elPanels = $('panels');
const elStats = $('stats');
const elBlurb = $('blurb');
const elErr = $('err');

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

// ---------------------------------------------------------------- state
let device: GPUDevice | null = null;
let backend: ShtBackend | null = null;
let sim: Simulation | null = null;
let topo: SphereMeshTopology | null = null;
let scenes: SphereScene[] = [];
let colorbars: Colorbar[] = [];
let valueBufs: Float32Array[] = [];
let colorBufs: Float32Array[] = [];
let ranges: { lo: number; hi: number }[] = [];
let resizeObs: ResizeObserver | null = null;

let model: ModelSpec = models[0];
let params: Params = defaultParams(model);
let seed = 1;
let running = false;
let adapterName = '';
let pumping = false;
let stepMs = 0;
let generation = 0; // bumped on every rebuild to cancel stale pumps

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
    });
    label.append(input);
    elParams.append(label);
  }
}

function applyPreset(presetKey: string): void {
  const preset = presets.find((p) => p.key === presetKey) ?? presets[0];
  model = models.find((m) => m.key === preset.modelKey) ?? models[0];
  params = { ...defaultParams(model), ...preset.params };
  buildParamInputs();
  elBlurb.textContent = model.blurb;
}

elModel.addEventListener('change', () => {
  applyPreset(elModel.value);
  void rebuild();
});
elLmax.addEventListener('change', () => void rebuild());
elBackend.addEventListener('change', () => void rebuild());
elColormap.addEventListener('change', () => draw());
function setRunning(next: boolean): void {
  running = next;
  elRunPause.textContent = running ? 'Pause' : 'Run';
  if (running) void pump();
}

elRunPause.addEventListener('click', () => setRunning(!running));
elReseed.addEventListener('click', () => {
  seed = (Math.random() * 2 ** 31) >>> 0;
  setRunning(false);
  void reseed();
});
elResetView.addEventListener('click', () => {
  for (const s of scenes) s.resetCamera();
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

async function rebuild(): Promise<void> {
  generation++;
  const gen = generation;
  // a rebuild restarts from a fresh initial state, so pause like Re-seed does
  setRunning(false);
  disposeView();
  backend?.destroy();
  backend = null;
  sim = null;
  stepMs = 0;

  const lmax = Number(elLmax.value);
  const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
  const cfg = { lmax, mmax: lmax, nlat, nphi };
  const wantGpu = elBackend.value === 'webgpu' && device !== null;
  try {
    backend = wantGpu
      ? await GpuBackend.create(device!, cfg)
      : new CpuBackend(cfg);
  } catch (e) {
    elErr.textContent = `Failed to create transform plan: ${e}`;
    return;
  }
  if (gen !== generation) return;
  elErr.textContent =
    !wantGpu && lmax > 31
      ? 'Heads up: the CPU backend is a direct-summation f64 reference — expect well under 10 steps/s at this lmax.'
      : '';

  sim = new Simulation(backend, model, params);
  await sim.init(seed);
  if (gen !== generation) return;

  // mesh + scenes
  const phi = new Float64Array(nphi);
  for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
  topo = buildTopology(backend.cosTheta, phi);

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();
  for (let k = 0; k < sim.nspecies; k++) {
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

  draw();
  updateStats();
  void pump();
}

async function reseed(): Promise<void> {
  if (!sim) return;
  const gen = generation;
  await sim.init(seed);
  if (gen !== generation) return;
  for (const r of ranges) {
    r.lo = NaN;
    r.hi = NaN;
  }
  draw();
}

// ---------------------------------------------------------------- drawing
function draw(): void {
  if (!sim || !topo) return;
  const cmap = colormaps[elColormap.value] ?? colormaps.viridis;
  for (let k = 0; k < sim.nspecies; k++) {
    fillFieldValues(valueBufs[k], sim.V[k], topo);
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
  if (!sim || !backend) return;
  const { nlat, nphi } = backend.cfg;
  const kind =
    backend.kind === 'webgpu'
      ? `WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}`
      : 'CPU f64 (direct summation)';
  const rate = stepMs > 0 ? `${(1000 / stepMs).toFixed(1)} steps/s` : '—';
  elStats.innerHTML =
    `<b>${kind}</b> · grid ${nlat}×${nphi} · nlm ${backend.nlm.toLocaleString()} · ` +
    `${stepMs > 0 ? stepMs.toFixed(1) : '—'} ms/step · ${rate} · ` +
    `t = <b>${sim.t.toFixed(2)}</b> (${sim.stepCount} steps)`;
}

// ---------------------------------------------------------------- sim loop
const nextFrame = () => new Promise<number>(requestAnimationFrame);

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  const gen = generation;
  let lastYield = performance.now();
  try {
    while (running && sim && gen === generation) {
      const t0 = performance.now();
      await sim.step();
      const dtMs = performance.now() - t0;
      stepMs = stepMs === 0 ? dtMs : stepMs + 0.05 * (dtMs - stepMs);
      const now = performance.now();
      if (now - lastYield > 25 || backend?.kind === 'cpu') {
        draw();
        updateStats();
        await nextFrame();
        lastYield = performance.now();
      }
    }
    // final frame after pausing
    if (gen === generation) {
      draw();
      updateStats();
    }
  } finally {
    pumping = false;
  }
}

// ---------------------------------------------------------------- boot
/** Best-effort human-readable adapter name, so it is clear which GPU (or
 *  software rasterizer) is actually running the transforms. */
async function describeAdapter(dev: GPUDevice): Promise<string> {
  const fmt = (info: GPUAdapterInfo | undefined): string => {
    if (!info) return '';
    const parts = [info.description, info.device, info.vendor].filter(
      (s): s is string => !!s && s.length > 0,
    );
    const name = parts[0] ?? '';
    return info.architecture && !name.includes(info.architecture)
      ? `${name} (${info.architecture})`.trim()
      : name;
  };
  const own = fmt((dev as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo);
  if (own) return own;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return fmt(adapter?.info);
  } catch {
    return '';
  }
}

async function boot(): Promise<void> {
  elModel.value = presets[0].key;
  applyPreset(presets[0].key);
  try {
    device = await requestShtDevice();
    adapterName = await describeAdapter(device);
  } catch (e) {
    device = null;
    elLmax.value = '31';
    elBackend.value = 'cpu';
    elBackend.options[0].disabled = true;
    elErr.textContent =
      `WebGPU is not available (${e instanceof Error ? e.message : e}); ` +
      `falling back to the slow CPU transform at low resolution. ` +
      `Use a WebGPU-capable browser (Chrome/Edge 113+) for the full experience.`;
  }
  device?.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      elErr.textContent = `WebGPU device lost: ${info.message}`;
    }
  });
  await rebuild();
}

void boot();
