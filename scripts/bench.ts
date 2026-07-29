/**
 * Command-line benchmark: run exactly what the browser runs — the same .m
 * models, lowered by numbl and compiled to the same WGSL kernels, over the same
 * transforms — on desktop WebGPU (Google Dawn, via the optional `webgpu`
 * package), and report ms/step. The app prints the matching command under its
 * stats line; copy it and run it here for an apples-to-apples comparison.
 *
 *   npm run bench -- --preset schnak-spots --lmax 63 --steps 2000 --seed 1 \
 *     --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05
 *
 * The only thing missing here is the rendering: this is the solver alone.
 *
 * Two numbers are reported, because they answer different questions:
 *  - throughput: a batch of steps submitted together, awaited once. This is how
 *    the app runs, and what keeping the state in GPU buffers is for.
 *  - latency: one step per submit, each awaited. Comparable to a design that
 *    reads back every step, and the only way to get a per-step distribution.
 */
import { requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { presets } from '../src/mgpu/registry.ts';
import {
  parseArgs,
  modelForSpec,
  resolvePreset,
  formatCommand,
  BENCH_COMMAND,
  DEFAULT_LMAX,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  type RunSpec,
} from '../src/bench/runSpec.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';

const USAGE = `usage: ${BENCH_COMMAND} [options]

  --preset <key>    ${presets.map((p) => p.key).join(' | ')}
                    (default ${presets[0].key})
  --lmax <n>        spherical harmonic truncation (default ${DEFAULT_LMAX})
  --steps <n>       timed steps (default ${DEFAULT_STEPS})
  --warmup <n>      untimed steps first (default ${DEFAULT_WARMUP})
  --seed <n>        initial-noise seed (default ${DEFAULT_SEED})
  --batch <n>       steps per submit for the throughput number (default 16)
  --<param> <v>     any parameter of the preset's model, e.g. --dt 0.05
  --json            machine-readable output
  --help

The browser app shows the command for whatever it is currently simulating;
copy it from under the stats line to compare the same run here.`;

function fail(msg: string, code = 1): never {
  console.error(`bench: ${msg}`);
  process.exit(code);
}

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(USAGE);
  process.exit(0);
}
const wantJson = argv.includes('--json');
let batch = 16;
const rest: string[] = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--json') continue;
  if (argv[i] === '--batch') {
    batch = Number(argv[++i]);
    continue;
  }
  if (argv[i].startsWith('--batch=')) {
    batch = Number(argv[i].slice('--batch='.length));
    continue;
  }
  rest.push(argv[i]);
}
if (!Number.isInteger(batch) || batch < 1) fail(`--batch must be an integer >= 1`, 2);

let spec: RunSpec;
try {
  spec = parseArgs(rest);
} catch (e) {
  fail(`${errMsg(e)}\n\n${USAGE}`, 2);
}

// ---------------------------------------------------------------- statistics
interface Timing {
  meanMs: number;
  medianMs: number;
  p05Ms: number;
  p95Ms: number;
  minMs: number;
  totalMs: number;
  stepsPerSec: number;
}

function timing(samples: Float64Array): Timing {
  const sorted = Float64Array.from(samples).sort();
  const q = (p: number): number =>
    sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  let total = 0;
  for (const v of samples) total += v;
  const mean = total / samples.length;
  return {
    meanMs: mean,
    medianMs: q(0.5),
    p05Ms: q(0.05),
    p95Ms: q(0.95),
    minMs: sorted[0],
    totalMs: total,
    stepsPerSec: 1000 / mean,
  };
}

function fieldRange(v: ArrayLike<number>): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if (v[i] < min) min = v[i];
    if (v[i] > max) max = v[i];
  }
  return { min, max };
}

// ---------------------------------------------------------------- run
const model = modelForSpec(spec);
const { preset } = resolvePreset(spec.preset);

let device: GPUDevice | null = null;
let session: ModelSession | null = null;

try {
  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);

  session = await ModelSession.create({
    device,
    model,
    params: spec.params,
    lmax: spec.lmax,
  });
  session.seed(spec.seed);

  const plan = session.describe();
  const kernels = plan.step.filter((l) => l.startsWith('kernel')).length;
  const cfg = session.cfg;

  if (!wantJson) {
    console.log(`turing-sphere bench — solver only, no rendering\n`);
    console.log(`  preset    ${preset.label}  (models/${model.key}.m: ${model.species.join(', ')})`);
    console.log(
      `  params    ${model.params.map((p) => `${p.key}=${spec.params[p.key]}`).join('  ')}`,
    );
    console.log(
      `  grid      lmax ${cfg.lmax} · ${cfg.nlat}×${cfg.nphi} · nlm ${session.sht.nlm.toLocaleString()}`,
    );
    console.log(`  compiled  ${plan.step.length} GPU ops/step (${kernels} generated kernels)`);
    console.log(`  backend   WebGPU fp32${adapter ? ` — ${adapter}` : ''}\n            ${runtime}`);
    console.log(`  run       ${spec.warmup} warmup + ${spec.steps} timed steps, seed ${spec.seed}\n`);
  }

  const done = (): Promise<undefined> => device!.queue.onSubmittedWorkDone();

  session.step(spec.warmup);
  await done();

  // --- throughput: batches submitted together, awaited once each ---
  const batches = Math.max(1, Math.ceil(spec.steps / batch));
  const progress = !wantJson && process.stderr.isTTY;
  let lastReport = performance.now();
  const tp0 = performance.now();
  let stepsRun = 0;
  for (let b = 0; b < batches; b++) {
    const n = Math.min(batch, spec.steps - stepsRun);
    session.step(n);
    await done();
    stepsRun += n;
    if (progress && performance.now() - lastReport > 1000) {
      const so_far = (performance.now() - tp0) / stepsRun;
      process.stderr.write(
        `\r\x1b[K  ${stepsRun}/${spec.steps} steps · ${so_far.toFixed(2)} ms/step`,
      );
      lastReport = performance.now();
    }
  }
  const throughputMs = (performance.now() - tp0) / stepsRun;
  if (progress) process.stderr.write('\r\x1b[K');

  // --- latency: one step per submit, for the distribution ---
  const latencySteps = Math.min(spec.steps, 200);
  const samples = new Float64Array(latencySteps);
  for (let s = 0; s < latencySteps; s++) {
    const t0 = performance.now();
    session.step(1);
    await done();
    samples[s] = performance.now() - t0;
  }
  const t = timing(samples);

  const field = await session.read(model.species[0]);
  const range = fieldRange(field);
  let finite = true;
  for (const v of field) if (!Number.isFinite(v)) finite = false;

  if (wantJson) {
    console.log(
      JSON.stringify(
        {
          command: formatCommand(spec),
          spec,
          model: model.key,
          backend: { adapter, runtime },
          grid: { lmax: cfg.lmax, nlat: cfg.nlat, nphi: cfg.nphi, nlm: session.sht.nlm },
          compiled: { opsPerStep: plan.step.length, kernels },
          throughput: { batch, msPerStep: throughputMs, stepsPerSec: 1000 / throughputMs },
          latency: t,
          state: {
            t: session.t,
            steps: session.steps,
            species: model.species[0],
            min: range.min,
            max: range.max,
            contrast: range.max - range.min,
            finite,
          },
        },
        null,
        2,
      ),
    );
  } else {
    console.log(
      `  ${throughputMs.toFixed(2)} ms/step   ${(1000 / throughputMs).toFixed(1)} steps/s   ` +
        `${(spec.params.dt * (1000 / throughputMs)).toFixed(2)} model time/s` +
        `   (batches of ${batch})`,
    );
    console.log(
      `  one step per submit: ${t.meanMs.toFixed(2)} ms mean · median ${t.medianMs.toFixed(2)} · ` +
        `p05 ${t.p05Ms.toFixed(2)} · p95 ${t.p95Ms.toFixed(2)} · min ${t.minMs.toFixed(2)}`,
    );
    console.log(
      `  after ${session.steps} steps: t = ${session.t.toFixed(2)}, ` +
        `${model.species[0]} ∈ [${range.min.toFixed(4)}, ${range.max.toFixed(4)}] ` +
        `(contrast ${(range.max - range.min).toFixed(4)})${finite ? '' : '  — NOT FINITE'}`,
    );
    console.log(
      `\n  Compare with the ms/step in the app's stats line: same .m, same kernels,\n` +
        `  but measured while the page renders the spheres.`,
    );
  }

  session.destroy();
  device.destroy();
  process.exit(finite ? 0 : 1);
} catch (e) {
  session?.destroy();
  device?.destroy();
  fail(errMsg(e));
}
