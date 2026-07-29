/**
 * Correctness of the .m -> WGSL path, against the TypeScript solver.
 *
 * src/solver/ is no longer what the app runs, but it is an independent
 * implementation of the same IMEX scheme, which makes it the oracle here: run
 * both from the same seeded perturbation, through the same fp32 transforms, and
 * compare the spectral state.
 *
 * The only difference between the two is where the reaction and the IMEX update
 * happen — f64 on the CPU for the reference, fp32 in generated WGSL for the .m.
 * The pattern-forming regime amplifies small differences, so this checks a
 * short run.
 */
import { GpuBackend } from '../src/solver/backend.ts';
import { Simulation, gridForLmax, makeRandn } from '../src/solver/simulation.ts';
import { models, defaultParams } from '../src/solver/models.ts';
import { ShtPlan } from '../src/sht/sht.ts';
import { GpuModel } from '../src/mgpu/model.ts';
import { mModels, type MModel } from '../src/mgpu/registry.ts';

type Check = (name: string, ok: boolean, detail: string) => void;
type Log = (s: string) => void;

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

const LMAX = 31;
const STEPS = 10;

/**
 * Kernels the step of each model should compile to — one per element-wise line
 * of MATLAB. This is a fusion guard: numbl's lowering emits one statement per
 * *operator*, and the inline pass folds those back into per-line expression
 * trees. If that stops happening the results stay correct but every operator
 * becomes its own dispatch, which is exactly the silent regression to catch.
 */
const EXPECTED_KERNELS: Record<string, number> = {
  schnakenberg: 5,
  brusselator: 5,
  allencahn: 2,
};

/** One model: compile it, run it, and compare against the reference solver. */
async function checkModel(
  device: GPUDevice,
  m: MModel,
  check: Check,
  log: Log,
): Promise<{ mgpuMs: number; refMs: number } | null> {
  const spec = models.find((x) => x.key === m.key);
  if (!spec) {
    check(`${m.key}: reference model exists`, false, 'no matching ModelSpec');
    return null;
  }
  const params = defaultParams(spec);
  const { nlat, nphi } = gridForLmax(LMAX, m.pdeg);
  const cfg = { lmax: LMAX, mmax: LMAX, nlat, nphi };
  const npts = nlat * nphi;

  const sht = await ShtPlan.create(device, cfg);
  const gpu = await GpuModel.create({
    device,
    sht,
    cfg,
    source: m.source,
    paramNames: m.params.map((p) => p.key),
    state: m.state,
    view: m.species,
  });
  gpu.setParams(params);

  const plan = gpu.describe();
  const kernels = plan.step.filter((l) => l.startsWith('kernel')).length;
  const xforms = plan.step.filter(
    (l) => l.startsWith('synth') || l.startsWith('analys'),
  ).length;
  log(
    `  ${m.key}.m -> ${plan.step.length} ops/step ` +
      `(${kernels} generated kernels, ${xforms} transforms)`,
  );
  const expected = EXPECTED_KERNELS[m.key];
  check(
    `${m.key}: element-wise lines fused into one kernel each`,
    kernels === expected,
    `${kernels} kernels (expected ${expected})`,
  );

  // One randn per grid point, in index order. Rounded to f32 once and fed to
  // BOTH sides, so the comparison is about the compute path, not the seed.
  const randn = makeRandn(1);
  const noise = new Float32Array(npts);
  for (let i = 0; i < npts; i++) noise[i] = m.seedAmp * randn();

  gpu.init(noise);

  // Reference, seeded from the same perturbation by handing the model's own
  // init the identical sequence.
  const backend = await GpuBackend.create(device, cfg);
  const ref = new Simulation(backend, spec, params);
  {
    let i = 0;
    const feed = (): number => noise[i++] / m.seedAmp;
    const grids = m.state.map(() => new Float64Array(npts));
    spec.init(params, ref.x, ref.y, ref.z, feed, grids);
    for (let k = 0; k < m.state.length; k++) {
      ref.U[k].set(await backend.analys(grids[k]));
    }
  }

  let worstInit = 0;
  for (let k = 0; k < m.state.length; k++) {
    worstInit = Math.max(worstInit, relL2(await gpu.read(m.state[k]), ref.U[k]));
  }
  check(
    `${m.key}: init matches reference`,
    worstInit < 1e-5,
    `rel L2 ${worstInit.toExponential(2)}`,
  );

  for (let s = 0; s < STEPS; s++) await ref.step();
  gpu.step(STEPS);

  let worst = 0;
  let nan = false;
  for (let k = 0; k < m.state.length; k++) {
    const got = await gpu.read(m.state[k]);
    worst = Math.max(worst, relL2(got, ref.U[k]));
    for (const v of got) if (!Number.isFinite(v)) nan = true;
  }
  check(
    `${m.key}: .m vs reference after ${STEPS} steps`,
    worst < 2e-3 && !nan,
    `rel L2 ${worst.toExponential(2)}${nan ? ', NaN!' : ''}`,
  );

  // Guard against "both sides computed nothing".
  const field = await gpu.read(m.species[0]);
  let peak = 0;
  for (const v of field) peak = Math.max(peak, Math.abs(v));
  check(
    `${m.key}: rendered field is non-trivial`,
    peak > 1e-4,
    `max |${m.species[0]}| ${peak.toExponential(2)}`,
  );

  // Step rate. The reference maps a staging buffer on every transform, so it
  // pays four driver round-trips per step; the .m path keeps everything in GPU
  // buffers and submits once.
  const TIMED = 50;
  const t0 = performance.now();
  gpu.step(TIMED);
  await device.queue.onSubmittedWorkDone();
  const mgpuMs = (performance.now() - t0) / TIMED;

  const t1 = performance.now();
  for (let s = 0; s < TIMED; s++) await ref.step();
  const refMs = (performance.now() - t1) / TIMED;

  gpu.destroy();
  backend.destroy();
  sht.destroy();
  return { mgpuMs, refMs };
}

export async function mgpuChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  check(
    'models: registry populated',
    mModels.length === models.length,
    `${mModels.length} .m models`,
  );

  for (const m of mModels) {
    const timing = await checkModel(device, m, check, log);
    if (!timing) continue;
    const { mgpuMs, refMs } = timing;
    log(
      `    step rate: .m ${mgpuMs.toFixed(2)} ms vs reference ` +
        `${refMs.toFixed(2)} ms (${(refMs / mgpuMs).toFixed(1)}x)`,
    );
    // A soft bound, not a performance target: on a software rasterizer the
    // transforms dominate and the round-trips this path avoids are a small
    // share of the total, so the ratio understates what it is worth on real
    // hardware. The check is only that executing the .m did not make it worse.
    check(
      `${m.key}: step rate no worse than the readback path`,
      mgpuMs < refMs * 1.15,
      `${mgpuMs.toFixed(2)} vs ${refMs.toFixed(2)} ms/step`,
    );
  }
}
