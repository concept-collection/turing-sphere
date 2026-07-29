/**
 * Browser validation: the fp32 WebGPU solver against the f64 CPU solver.
 * Runs identical seeded simulations on both backends and compares fields.
 * Results are posted to window.__RESULTS__ for the headless runner.
 */
import { GpuBackend, CpuBackend, requestShtDevice } from '../src/solver/backend.ts';
import { Simulation, gridForLmax } from '../src/solver/simulation.ts';
import { models, defaultParams } from '../src/solver/models.ts';
import { randomSpectrum } from '../src/sht/reference.ts';

declare global {
  interface Window {
    __RESULTS__?: { ok: boolean; fatal?: string; lines: string[] };
  }
}

const logEl = document.getElementById('log')!;
const lines: string[] = [];
let failures = 0;

function log(s: string): void {
  lines.push(s);
  logEl.textContent = lines.join('\n');
  console.log(s);
}

function check(name: string, ok: boolean, detail: string): void {
  log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
}

function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

/**
 * Solver-only soak (no rendering), selected with ?soak=<steps>&lmax=<n>.
 * Isolates the GPU transform loop from the three.js renderer.
 */
async function soak(steps: number, lmax: number): Promise<void> {
  const device = await requestShtDevice();
  const schnak = models[0];
  const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
  const gpu = await GpuBackend.create(device, { lmax, mmax: lmax, nlat, nphi });
  const sim = new Simulation(gpu, schnak, defaultParams(schnak));
  await sim.init(5);
  log(`soak: ${steps} steps at lmax ${lmax} (grid ${nlat}x${nphi}), solver only`);

  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    await sim.step();
    if ((s + 1) % 100 === 0) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of sim.V[0]) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const mem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      log(
        `  step ${s + 1}  u in [${lo.toFixed(4)}, ${hi.toFixed(4)}]` +
          (mem ? `  heap ${(mem.usedJSHeapSize / 1048576).toFixed(1)} MB` : ''),
      );
      // yield so the page stays responsive and the runner can poll
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const ms = (performance.now() - t0) / steps;
  let finite = true;
  for (const v of sim.V[0]) if (!Number.isFinite(v)) finite = false;
  check(`soak: ${steps} steps survived`, finite, `${ms.toFixed(1)} ms/step`);
  gpu.destroy();
  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

async function main(): Promise<void> {
  const q = new URLSearchParams(location.search);
  if (q.has('soak')) {
    return soak(Number(q.get('soak')) || 500, Number(q.get('lmax')) || 63);
  }
  const device = await requestShtDevice();

  // --- transform cross-check: GPU vs CPU on a random spectrum ---
  {
    const lmax = 31;
    const { nlat, nphi } = gridForLmax(lmax, 1);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const gpu = await GpuBackend.create(device, cfg);
    const cpu = new CpuBackend(cfg);
    const q = randomSpectrum(cfg, 42);
    const q64 = new Float64Array(q);
    const sGpu = await gpu.synth(q64);
    const sCpu = await cpu.synth(q64);
    const errSynth = relL2(sGpu, sCpu);
    const aGpu = await gpu.analys(new Float64Array(sCpu));
    const aCpu = await cpu.analys(new Float64Array(sCpu));
    const errAnalys = relL2(aGpu, aCpu);
    check('transforms: GPU vs CPU', errSynth < 1e-4 && errAnalys < 1e-4,
      `synth ${errSynth.toExponential(2)}, analys ${errAnalys.toExponential(2)}`);
    gpu.destroy();
  }

  // --- solver cross-check: identical seeded runs on both backends ---
  {
    const schnak = models[0];
    const params = defaultParams(schnak);
    const lmax = 31;
    const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const gpu = await GpuBackend.create(device, cfg);
    const cpu = new CpuBackend(cfg);
    const simGpu = new Simulation(gpu, schnak, { ...params });
    const simCpu = new Simulation(cpu, schnak, { ...params });
    await simGpu.init(7);
    await simCpu.init(7);
    const nsteps = 10;
    const t0 = performance.now();
    for (let s = 0; s < nsteps; s++) await simGpu.step();
    const gpuMs = (performance.now() - t0) / nsteps;
    for (let s = 0; s < nsteps; s++) await simCpu.step();
    let worst = 0;
    for (let k = 0; k < simGpu.nspecies; k++) {
      worst = Math.max(worst, relL2(simGpu.V[k], simCpu.V[k]));
    }
    let nan = false;
    for (let k = 0; k < simGpu.nspecies; k++) {
      for (const v of simGpu.V[k]) if (!Number.isFinite(v)) nan = true;
    }
    check('solver: GPU vs CPU after 10 steps', worst < 2e-3 && !nan,
      `worst rel L2 ${worst.toExponential(2)}${nan ? ', NaN!' : ''} (${gpuMs.toFixed(1)} ms/step GPU)`);
    gpu.destroy();
  }

  // --- longer GPU-only run stays finite and patterned ---
  {
    const schnak = models[0];
    const params = defaultParams(schnak);
    const lmax = 63;
    const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
    const gpu = await GpuBackend.create(device, { lmax, mmax: lmax, nlat, nphi });
    const sim = new Simulation(gpu, schnak, params);
    await sim.init(3);
    const nsteps = 100;
    const t0 = performance.now();
    for (let s = 0; s < nsteps; s++) await sim.step();
    const ms = (performance.now() - t0) / nsteps;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of sim.V[0]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const finite = Number.isFinite(lo) && Number.isFinite(hi);
    check('solver: 100 steps at lmax 63 stay finite', finite && lo > -10 && hi < 10,
      `u range [${lo.toFixed(4)}, ${hi.toFixed(4)}], ${ms.toFixed(1)} ms/step`);
    gpu.destroy();
  }

  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

main().catch((e) => {
  const msg = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
  log(`fatal: ${msg}`);
  window.__RESULTS__ = { ok: false, fatal: msg, lines };
});
