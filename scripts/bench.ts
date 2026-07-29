/**
 * Command-line benchmark: run exactly the simulation the browser app is
 * running — same solver, same transforms, same parameters — on desktop WebGPU
 * (Google Dawn, via the optional `webgpu` package) or on the f64 CPU
 * reference, and report ms/step.  The app prints the matching command under
 * its stats line; copy it and run it here for an apples-to-apples comparison.
 *
 *   node scripts/bench.ts --preset schnak-spots --lmax 63 --backend webgpu \
 *     --steps 500 --seed 1 --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05
 *
 * The only thing missing here is the rendering: this is the solver alone.
 */
import {
  GpuBackend,
  CpuBackend,
  requestShtDevice,
  describeAdapter,
  type ShtBackend,
} from '../src/solver/backend.ts';
import { Simulation } from '../src/solver/simulation.ts';
import { presets } from '../src/solver/models.ts';
import {
  parseArgs,
  modelForSpec,
  configForSpec,
  resolvePreset,
  formatCommand,
  BENCH_COMMAND,
  DEFAULT_LMAX,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  DEFAULT_BACKEND,
  type RunSpec,
} from '../src/bench/runSpec.ts';

const USAGE = `usage: ${BENCH_COMMAND} [options]

  --preset <key>    ${presets.map((p) => p.key).join(' | ')}
                    (default ${presets[0].key})
  --lmax <n>        spherical harmonic truncation (default ${DEFAULT_LMAX})
  --backend <kind>  webgpu | cpu (default ${DEFAULT_BACKEND})
  --steps <n>       timed steps (default ${DEFAULT_STEPS})
  --warmup <n>      untimed steps first (default ${DEFAULT_WARMUP})
  --seed <n>        initial-noise seed (default ${DEFAULT_SEED})
  --<param> <v>     any parameter of the preset's model, e.g. --dt 0.05
  --json            machine-readable output
  --help

The browser app shows the command for whatever it is currently simulating;
copy it from under the stats line to compare the same run here.`;

function fail(msg: string, code = 1): never {
  console.error(`bench: ${msg}`);
  process.exit(code);
}
const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(USAGE);
  process.exit(0);
}
const wantJson = argv.includes('--json');
let spec: RunSpec;
try {
  spec = parseArgs(argv.filter((a) => a !== '--json'));
} catch (e) {
  fail(`${errMsg(e)}\n\n${USAGE}`, 2);
}

// ---------------------------------------------------------------- WebGPU
/**
 * Install Dawn under the globals the transform code expects (navigator.gpu,
 * GPUBufferUsage, ...), so src/ runs here unchanged — including
 * requestShtDevice(), which is the same device request the browser makes.
 * The specifier is indirect so that typechecking does not require the
 * optional package to be installed.
 */
async function installWebGpu(): Promise<string> {
  const specifier = 'webgpu';
  let mod: {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  try {
    mod = await import(specifier);
  } catch {
    throw new Error(
      'desktop WebGPU needs the optional `webgpu` package (prebuilt Google Dawn):\n' +
        '  npm install webgpu\n' +
        'or run with --backend cpu.',
    );
  }
  Object.assign(globalThis, mod.globals);
  // DAWN_FLAGS is ';'-separated because individual Dawn options take
  // comma-separated lists, e.g. 'enable-dawn-features=allow_unsafe_apis,timestamp_quantization'
  const dawnFlags = process.env.DAWN_FLAGS?.split(';').filter(Boolean) ?? [];
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: mod.create(dawnFlags) },
    configurable: true,
    writable: true,
  });
  const { version } = await import(`${specifier}/package.json`, {
    with: { type: 'json' },
  }).then(
    (m) => m.default as { version: string },
    () => ({ version: '?' }),
  );
  return `node-webgpu ${version} (Google Dawn)`;
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
const cfg = configForSpec(spec);

let device: GPUDevice | null = null;
let backend: ShtBackend | null = null;
let runtime = 'CPU (direct summation, f64)';
let adapter = '';

try {
  if (spec.backend === 'webgpu') {
    runtime = await installWebGpu();
    device = await requestShtDevice();
    adapter = await describeAdapter(device);
    backend = await GpuBackend.create(device, cfg);
  } else {
    backend = new CpuBackend(cfg);
  }

  const sim = new Simulation(backend, model, spec.params);
  await sim.init(spec.seed);

  if (!wantJson) {
    const kind =
      spec.backend === 'webgpu'
        ? `WebGPU fp32${adapter ? ` — ${adapter}` : ''}`
        : 'CPU f64';
    console.log(`turing-sphere bench — solver only, no rendering\n`);
    console.log(`  preset    ${preset.label}  (model ${model.key}: ${model.species.join(', ')})`);
    console.log(
      `  params    ${model.params.map((p) => `${p.key}=${spec.params[p.key]}`).join('  ')}`,
    );
    console.log(
      `  grid      lmax ${cfg.lmax} · ${cfg.nlat}×${cfg.nphi} · nlm ${backend.nlm.toLocaleString()}`,
    );
    console.log(`  backend   ${kind}\n            ${runtime}`);
    console.log(`  run       ${spec.warmup} warmup + ${spec.steps} timed steps, seed ${spec.seed}\n`);
  }

  for (let s = 0; s < spec.warmup; s++) await sim.step();

  const samples = new Float64Array(spec.steps);
  const progress = !wantJson && process.stderr.isTTY;
  let lastReport = performance.now();
  let running = 0;
  for (let s = 0; s < spec.steps; s++) {
    const t0 = performance.now();
    await sim.step();
    samples[s] = performance.now() - t0;
    running += samples[s];
    if (progress && performance.now() - lastReport > 1000) {
      process.stderr.write(
        `\r\x1b[K  ${s + 1}/${spec.steps} steps · ${(running / (s + 1)).toFixed(2)} ms/step`,
      );
      lastReport = performance.now();
    }
  }
  if (progress) process.stderr.write('\r\x1b[K');

  const t = timing(samples);
  const range = fieldRange(sim.V[0]);
  let finite = true;
  for (const v of sim.V[0]) if (!Number.isFinite(v)) finite = false;

  if (wantJson) {
    console.log(
      JSON.stringify(
        {
          command: formatCommand(spec),
          spec,
          model: model.key,
          backend: { kind: spec.backend, adapter, runtime },
          grid: { lmax: cfg.lmax, nlat: cfg.nlat, nphi: cfg.nphi, nlm: backend.nlm },
          timing: t,
          state: {
            t: sim.t,
            steps: sim.stepCount,
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
      `  ${t.meanMs.toFixed(2)} ms/step   ${t.stepsPerSec.toFixed(1)} steps/s   ` +
        `${(spec.params.dt * t.stepsPerSec).toFixed(2)} model time/s`,
    );
    console.log(
      `  median ${t.medianMs.toFixed(2)} · p05 ${t.p05Ms.toFixed(2)} · ` +
        `p95 ${t.p95Ms.toFixed(2)} · min ${t.minMs.toFixed(2)} ms  ` +
        `(${(t.totalMs / 1000).toFixed(1)} s total)`,
    );
    console.log(
      `  after ${sim.stepCount} steps: t = ${sim.t.toFixed(2)}, ` +
        `${model.species[0]} ∈ [${range.min.toFixed(4)}, ${range.max.toFixed(4)}] ` +
        `(contrast ${(range.max - range.min).toFixed(4)})${finite ? '' : '  — NOT FINITE'}`,
    );
    console.log(
      `\n  Compare with the ms/step in the app's stats line. That one is also the\n` +
        `  solver alone, but measured while the page renders the spheres.`,
    );
  }

  backend.destroy();
  device?.destroy();
  process.exit(finite ? 0 : 1);
} catch (e) {
  backend?.destroy();
  device?.destroy();
  fail(errMsg(e));
}
