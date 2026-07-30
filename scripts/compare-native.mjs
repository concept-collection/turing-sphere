/**
 * How do the WGSL transforms compare with upstream SHTNS?
 *
 *   node scripts/compare-native.mjs                    # transforms, lmax 63
 *   node scripts/compare-native.mjs --mode solver      # a whole IMEX timestep
 *   node scripts/compare-native.mjs --check            # and diff the final state
 *
 * Runs the same spec through every implementation available on this machine and
 * puts the numbers in one table:
 *
 *   webgpu       src/sht, fp32, through Dawn — what the app runs
 *   shtns cuda   SHTNS' own CUDA kernels, fp32 — the like-for-like comparison
 *   shtns cpu    SHTNS on the CPU, fp64 — the accuracy and "what a CPU does"
 *                reference (SHTNS has no CPU single precision)
 *
 * Everything runs in this one invocation, back to back, so a second process
 * competing for the GPU affects both sides rather than one. Missing
 * implementations are reported and skipped, not fatal: on a machine without
 * CUDA you still get webgpu against the CPU.
 *
 * With --check it also re-runs each side with --dump-state and diffs the final
 * spectral state, which is what makes the timing comparison mean anything: the
 * two are only comparable if they compute the same thing. The native solver is
 * a transcription of models/<key>.m rather than the .m itself (C cannot run
 * numbl), so this is the check on that transcription.
 *
 * Needs `bench/shtns/bootstrap.sh && make` in bench/shtns first, and desktop
 * WebGPU (the optional `webgpu` package) for the WGSL side.
 *
 * This compares implementations, all in the terminal. For the browser against
 * the terminal, see scripts/compare-perf.mjs.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const ROOT = new URL('..', import.meta.url).pathname;
const NATIVE = join(ROOT, 'bench', 'shtns');

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
const has = (name) => argv.includes(`--${name}`);
const flag = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1] !== undefined) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : dflt;
};
if (has('help') || argv.includes('-h')) {
  console.log(`usage: node scripts/compare-native.mjs [options]

  --mode transform|solver  what to compare (default transform)
  --lmax <n>               spherical harmonic truncation (default 63)
  --steps <n>              timed steps / round trips (default 1000)
  --warmup <n>             untimed steps first (default 50)
  --preset <key>           model preset; in transform mode only its dealiasing
                             degree is used (default schnak-spots)
  --batch <n>              steps per synchronization on both sides (default 16)
  --layout theta|phi       spatial layout for the native runs. theta is SHTNS'
                             native and fastest; phi is what the WGSL side uses
                             (default theta)
  --threads <n>            OpenMP threads for the SHTNS CPU run. 1 by default,
                             which is the reproducible per-core reference; 0 lets
                             the library use every core, which at small lmax is
                             often slower than one thread
  --check                  also diff the final state between implementations
  --check-steps <n>        steps for that state, kept short on purpose: fp32
                             round-off accumulates, and in solver mode the
                             unstable modes amplify it (default 20)
  --tolerance <x>          --check threshold on the relative L2 (default 2e-3)
  --no-cpu                 skip the SHTNS CPU run
  --json                   machine-readable output`);
  process.exit(0);
}
const mode = flag('mode', 'transform');
if (mode !== 'transform' && mode !== 'solver') {
  console.error(`compare-native: --mode must be 'transform' or 'solver'`);
  process.exit(2);
}
const lmax = flag('lmax', '63');
const steps = flag('steps', '1000');
const warmup = flag('warmup', '50');
const preset = flag('preset', 'schnak-spots');
const batch = flag('batch', '16');
const layout = flag('layout', 'theta');
const threads = flag('threads', '1');
const wantCheck = has('check');
const checkSteps = flag('check-steps', '20');
const wantJson = has('json');
const tolerance = Number(flag('tolerance', '2e-3'));
const wantCpu = !has('no-cpu');
const progress = !wantJson && process.stderr.isTTY;

const tmp = (tag) => join(tmpdir(), `turing-sphere-native-${tag}-${process.pid}.json`);
const cleanup = [];

// ------------------------------------------------------------------- runners
/**
 * Pull our JSON object out of stdout.
 *
 * A linked library shares the process's stdout, and SHTns writes to it
 * unconditionally — it announces the GPU it found, which FFT layout it chose and
 * which VkFFT it linked, none of it gated by shtns_verbose(). So the object may
 * not start at byte 0. Every producer here prints it with `{` alone on a line and
 * `}` alone on the last, which is enough to find it.
 */
function extractJson(out) {
  try {
    return JSON.parse(out);
  } catch {
    /* there is something else on stdout; find where the object starts */
  }
  const start = out.search(/^\{$/m);
  const ends = [...out.matchAll(/^\}$/gm)];
  if (start < 0 || !ends.length) return null;
  const last = ends[ends.length - 1];
  try {
    return JSON.parse(out.slice(start, last.index + 1));
  } catch {
    return null;
  }
}

/** Dawn reports two of these on every start-up; they are not the failure. */
const NOISE = /^Warning: max(Dynamic|Compute|Storage)/;

/** The tail of a failed run's output, which is where the actual error is. */
function failureDetail(r, cmd, args) {
  const lines = `${r.stderr ?? ''}\n${r.stdout ?? ''}`
    .split('\n')
    .map((s) => s.trimEnd())
    .filter((s) => s && !NOISE.test(s));
  const tail = lines.slice(-14).map((l) => `      ${l}`);
  return [
    `exit ${r.status}`,
    ...tail,
    `      re-run it alone to see everything:`,
    `        ${cmd} ${args.join(' ')}`,
  ].join('\n');
}

/**
 * Why a run that reported itself is still not usable. Every producer records the
 * device, the Fourier stage and whether its result stayed finite, so a wrong
 * answer can be described rather than dumped.
 */
function badResult(json) {
  const b = json.backend ?? {};
  const lines = [];
  const first = json.firstRoundTrip;
  if (first && first.finite === false) {
    lines.push('a single spectral -> grid -> spectral round trip returns no finite values,');
    lines.push('so the transforms are wrong on this device — nothing here is worth timing.');
  } else if (first && !(first.relL2 < 1e-3)) {
    lines.push(
      `a single round trip comes back ${Number(first.relL2).toExponential(2)} away from its`,
    );
    lines.push('input, far outside fp32 round-off (~1e-7). The transforms are wrong here.');
  } else if (json.state && json.state.finite === false) {
    lines.push('it ran, but the final state has no finite values — one round trip is fine,');
    lines.push('so something diverges over the length of the run.');
  } else {
    lines.push('it reported a failure without saying why; run it alone.');
  }
  lines.push(`device: ${b.adapter || '(unknown)'}`);
  if (json.fourier) lines.push(`Fourier stage: ${String(json.fourier).toUpperCase()}`);
  return lines.map((l, i) => (i === 0 ? l : `      ${l}`)).join('\n');
}

/** Run one side and parse its --json output. `ok: false` with a reason if it is
 *  not available here — a missing binary, no adapter, no CUDA. */
function run(label, cmd, args, statePath) {
  const full = statePath
    ? [...args, '--steps', checkSteps, '--warmup', '0', '--dump-state', statePath]
    : args;
  const r = spawnSync(cmd, full, {
    encoding: 'utf8',
    cwd: ROOT,
    maxBuffer: 256 * 1024 * 1024,
  });
  if (r.error) return { label, ok: false, why: r.error.message };
  const json = extractJson(r.stdout);
  if (r.status !== 0) {
    // A run that failed but still reported itself is the interesting case: it
    // computed something wrong rather than failing to start, and it already said
    // what and on which device. Use that instead of dumping its output.
    return { label, ok: false, why: json ? badResult(json) : failureDetail(r, cmd, full) };
  }
  if (!json) {
    return {
      label,
      ok: false,
      why: `printed no JSON object:\n${failureDetail(r, cmd, full)}`,
    };
  }
  let state = null;
  if (statePath && existsSync(statePath)) {
    cleanup.push(statePath);
    state = JSON.parse(readFileSync(statePath, 'utf8'));
  }
  return { label, ok: true, json, state };
}

const common = ['--lmax', lmax, '--steps', steps, '--warmup', warmup, '--preset', preset];
const nativeCommon = [...common, '--batch', batch, '--layout', layout, '--json'];

const jobs = [];
if (mode === 'transform') {
  jobs.push({
    label: 'webgpu',
    cmd: 'npx',
    args: ['vite-node', 'scripts/bench-sht.ts', '--json', ...common, '--batch', batch],
  });
} else {
  jobs.push({
    label: 'webgpu',
    cmd: 'npx',
    args: ['vite-node', 'scripts/bench.ts', '--json', ...common, '--batch', batch],
  });
}
const gpuBin = join(NATIVE, 'shtbench_gpu');
const cpuBin = join(NATIVE, 'shtbench');
const nativeMode = ['--mode', mode];
if (existsSync(gpuBin)) {
  jobs.push({ label: 'shtns cuda', cmd: gpuBin, args: [...nativeCommon, ...nativeMode] });
} else {
  jobs.push({
    label: 'shtns cuda',
    missing:
      `bench/shtns/shtbench_gpu is not built. On a machine with nvcc:\n` +
      `        cd bench/shtns && ./bootstrap.sh && make`,
  });
}
if (wantCpu) {
  if (existsSync(cpuBin)) {
    jobs.push({
      label: 'shtns cpu',
      cmd: cpuBin,
      args: [...nativeCommon, ...nativeMode, '--threads', threads],
    });
  } else {
    jobs.push({
      label: 'shtns cpu',
      missing: `bench/shtns/shtbench is not built:\n        cd bench/shtns && ./bootstrap.sh && make`,
    });
  }
}

if (!wantJson) {
  console.log(
    `comparing ${mode === 'transform' ? 'transforms' : 'solver timesteps'} — ` +
      `lmax ${lmax}, ${steps} steps, preset ${preset}` +
      (mode === 'transform' ? ' (for its grid rule)' : '') +
      `\n`,
  );
}

const results = [];
for (const job of jobs) {
  if (job.missing) {
    results.push({ label: job.label, ok: false, why: job.missing });
    continue;
  }
  if (progress) process.stderr.write(`\r\x1b[K  running ${job.label}...`);
  results.push(run(job.label, job.cmd, job.args, null));
  if (progress) process.stderr.write('\r\x1b[K');
}

/* The state comparison is a second, short run: the timing wants thousands of
 * steps and the state comparison wants as few as possible, since fp32 round-off
 * accumulates and a solver run amplifies it. */
const states = new Map();
if (wantCheck) {
  for (const job of jobs) {
    if (job.missing || !results.find((r) => r.label === job.label)?.ok) continue;
    if (progress) process.stderr.write(`\r\x1b[K  checking ${job.label}...`);
    const r = run(job.label, job.cmd, job.args, tmp(job.label.replace(/ /g, '-')));
    if (r.ok && r.state) states.set(job.label, r.state);
    if (progress) process.stderr.write('\r\x1b[K');
  }
}

const good = results.filter((r) => r.ok);
if (!good.length) {
  console.error('compare-native: nothing ran.\n');
  for (const r of results) console.error(`  ${r.label}: ${r.why}`);
  process.exit(1);
}

// ------------------------------------------------- are these the same problem?
// The native side keeps its own copy of the presets and the grid rule (C cannot
// import the TypeScript), so this is the one thing that can silently drift.
const gridOf = (r) => r.json.grid;
const ref = good[0];
const gridProblems = [];
for (const r of good.slice(1)) {
  const a = gridOf(ref);
  const b = gridOf(r);
  for (const k of ['lmax', 'nlat', 'nphi', 'nlm']) {
    if (a[k] !== b[k]) gridProblems.push(`${r.label}: ${k} ${b[k]} vs ${ref.label}'s ${a[k]}`);
  }
  const pa = ref.json.spec?.params;
  const pb = r.json.spec?.params;
  if (pa && pb) {
    for (const k of Object.keys(pa)) {
      if (Math.abs(Number(pa[k]) - Number(pb[k])) > 1e-12 * Math.max(1, Math.abs(Number(pa[k])))) {
        gridProblems.push(`${r.label}: ${k} = ${pb[k]} vs ${ref.label}'s ${pa[k]}`);
      }
    }
  }
}
if (gridProblems.length) {
  console.error(
    `compare-native: the two sides are not running the same problem, so there is\n` +
      `nothing to compare. bench/shtns/spec.h has drifted from src/mgpu/registry.ts\n` +
      `or src/sht/layout.ts:\n`,
  );
  for (const p of gridProblems) console.error(`  ${p}`);
  process.exit(1);
}

// -------------------------------------------------------------------- report
// Ratios are against the WGSL run, which is the point of the comparison. If that
// is the side that failed, fall back to whatever did run and say so, rather than
// printing "1.00x webgpu" for a run webgpu had no part in.
const rate = (r) => r.json.throughput.msPerStep;
const baseRun = good.find((r) => r.label === 'webgpu') ?? good[0];
const base = rate(baseRun);

if (wantJson) {
  console.log(
    JSON.stringify(
      {
        mode,
        spec: {
          lmax: Number(lmax),
          steps: Number(steps),
          warmup: Number(warmup),
          preset,
          batch: Number(batch),
          layout,
          threads: Number(threads),
        },
        grid: gridOf(ref),
        baseline: baseRun.label,
        runs: results.map((r) =>
          r.ok
            ? {
                label: r.label,
                msPerStep: rate(r),
                stepsPerSec: r.json.throughput.stepsPerSec,
                encodeMsPerStep: r.json.throughput.encodeMsPerStep,
                ratioToBaseline: rate(r) / base,
                precision: r.json.backend.precision,
                adapter: r.json.backend.adapter,
                library: r.json.backend.library,
                digest: r.json.digest ?? null,
              }
            : { label: r.label, ok: false, why: r.why },
        ),
      },
      null,
      2,
    ),
  );
} else {
  const unit = mode === 'transform' ? 'ms/round trip' : 'ms/step';
  console.log(
    `  grid lmax ${gridOf(ref).lmax} · ${gridOf(ref).nlat}×${gridOf(ref).nphi} · ` +
      `nlm ${gridOf(ref).nlm.toLocaleString()}` +
      (mode === 'transform' ? '   (one synthesis + one analysis per round trip)' : ''),
  );
  console.log();
  for (const r of results) {
    if (!r.ok) {
      console.log(`  ${r.label.padEnd(11)} not available — ${r.why}`);
      continue;
    }
    const ms = rate(r);
    const ratio = ms / base;
    console.log(
      `  ${r.label.padEnd(11)} ${ms.toFixed(3)} ${unit}   ` +
        `${r.json.throughput.stepsPerSec.toFixed(0)}/s   ` +
        `${r === baseRun ? '(baseline)' : `${ratio.toFixed(2)}x ${baseRun.label}`}   ` +
        `${r.json.backend.precision}`,
    );
    console.log(
      `  ${''.padEnd(11)} ${r.json.backend.adapter}` +
        (r.json.throughput.encodeMsPerStep
          ? `   ·   CPU-side launching ${r.json.throughput.encodeMsPerStep.toFixed(3)} ms/step`
          : ''),
    );
    if (r === baseRun && baseRun.label !== 'webgpu') {
      console.log(
        `  ${''.padEnd(11)} webgpu did not run, so this is the baseline instead — which is` +
          `\n  ${''.padEnd(11)} not the comparison you wanted. Fix that side first.`,
      );
    }
  }

  // Same caution compare-perf.mjs takes: a ratio between two different devices
  // is not a comparison of implementations.
  const wg = good.find((r) => r.label === 'webgpu');
  const cuda = good.find((r) => r.label === 'shtns cuda');
  const software = (a) => /swiftshader|llvmpipe|software|basic render/i.test(a ?? '');
  if (wg && cuda) {
    const a = wg.json.backend.adapter ?? '';
    const b = cuda.json.backend.adapter ?? '';
    if (software(a)) {
      console.log(
        `\n  STOP  the WGSL side is on a software renderer (${a}), so the ratio above\n` +
          `  compares a CPU emulation against a real GPU and means nothing. Dawn reaches\n` +
          `  the GPU through Vulkan; DAWN_FLAGS='backend=vulkan' makes it explain itself.`,
      );
    } else if (!sameDevice(a, b)) {
      console.log(
        `\n  NOTE  the two name different devices. If this machine has more than one GPU,\n` +
          `  they are not comparable — point Dawn and --device at the same one:\n` +
          `    webgpu:     ${a}\n    shtns cuda: ${b}`,
      );
    }
  }
  // A run whose CPU-side share is most of its wall time is not measuring the GPU
  // at all — it is measuring how long the host takes to queue the work. That is a
  // real cost, but it puts a floor under the number that has nothing to do with
  // the transform, and it means a ratio against it understates the gap in GPU
  // work. Say so rather than letting the headline ratio be read as compute.
  const LAUNCH_BOUND = 0.5;
  for (const r of good) {
    const share = (r.json.throughput.encodeMsPerStep ?? 0) / rate(r);
    if (share > LAUNCH_BOUND) {
      console.log(
        `\n  NOTE  ${r.label} spends ${(100 * share).toFixed(0)}% of its time on the CPU queueing\n` +
          `  work, so ${rate(r).toFixed(3)} ms is roughly what it costs to *submit* a round trip\n` +
          `  here, not what the GPU spends on one — the real GPU time is below that and\n` +
          `  this measurement cannot see it. Raise --lmax until the GPU dominates, or read\n` +
          `  any ratio involving this row as a lower bound on the difference in GPU work.`,
      );
    }
  }
  if (cuda && wg) {
    const ratio = rate(cuda) / rate(wg);
    console.log(
      `\n  ${
        ratio < 1
          ? `SHTNS' CUDA transforms are ${(1 / ratio).toFixed(2)}x faster than the WGSL ones`
          : `the WGSL transforms are ${ratio.toFixed(2)}x faster than SHTNS' CUDA ones`
      } on the same device, at the same precision and grid.`,
    );
    console.log(
      `  Things that are genuinely different, and worth checking before reading much\n` +
        `  into the number: SHTNS runs its Legendre recurrence in fp64 for lmax <= 128\n` +
        `  even in fp32 mode (SHTNS_GPU_REC_PREC=1 forces fp32, which is what WebGPU is\n` +
        `  restricted to); it uses cuFFT or VkFFT for the Fourier stage against a WGSL\n` +
        `  FFT; and --layout theta is its native layout, phi is the WGSL one.`,
    );
  }
}

// --------------------------------------------------------------------- check
let checkFailed = false;
if (wantCheck) {
  const labels = [...states.keys()].filter((k) => states.get(k).state?.length);
  if (labels.length < 2) {
    if (!wantJson) console.log(`\n  --check: fewer than two implementations produced a state.`);
  } else {
    if (!wantJson)
      console.log(
        `\n  --check: the spectral state after exactly ${checkSteps} ` +
          `${mode === 'transform' ? 'round trips' : 'steps'} from seed 1\n`,
      );
    const bl = labels[0];
    const b = states.get(bl);
    for (const label of labels) {
      const s = states.get(label);
      let note = '(reference)';
      if (label !== bl) {
        const rel = relL2(s.state, b.state);
        checkFailed = checkFailed || !(rel < tolerance);
        note = `relative L2 vs ${bl}: ${rel.toExponential(3)}`;
      }
      if (!wantJson) {
        console.log(`  ${label.padEnd(11)} ${digestLine(s.digest)}`);
        console.log(`  ${''.padEnd(11)} ${note}`);
      }
      // The seeded input has to match, or the two states are answers to
      // different questions and the L2 above says nothing about the transforms.
      if (label !== bl && s.input && b.input && Math.abs(s.input.rms - b.input.rms) > 1e-9) {
        console.log(
          `  ${''.padEnd(11)} MISMATCHED INPUT: seeded spectrum rms ${s.input.rms} vs ` +
            `${b.input.rms}.\n` +
            `  ${''.padEnd(11)} The two seeded generators disagree (shtb_seeded_spectrum in\n` +
            `  ${''.padEnd(11)} bench/shtns/spec.h against seededSpectrum in bench-sht.ts), so the\n` +
            `  ${''.padEnd(11)} difference above is not about the transforms.`,
        );
        checkFailed = true;
      }
    }
    if (!wantJson) {
      console.log(
        `\n  ${checkFailed ? 'FAIL' : 'PASS'}  every implementation agrees to better than ` +
          `${tolerance.toExponential(1)} relative L2`,
      );
      console.log(
        `        fp32 against fp64 lands near 1e-6 for a single transform and drifts\n` +
          `        upward with the step count; two fp32 implementations differ in\n` +
          `        fused-multiply-add and the other latitude fp32 allows. Raise\n` +
          `        --check-steps to watch the drift accumulate.`,
      );
    }
  }
}

for (const p of cleanup) {
  try {
    unlinkSync(p);
  } catch {
    /* best effort */
  }
}
process.exit(checkFailed ? 1 : 0);

// ------------------------------------------------------------------- helpers
function relL2(a, b) {
  let num = 0;
  let den = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

function digestLine(d) {
  if (!d) return '(no digest)';
  const g = (v) => Number(v).toPrecision(9);
  return `min=${g(d.min)} max=${g(d.max)} mean=${g(d.mean)} rms=${g(d.rms)}`;
}

/** Two adapter strings for the same GPU rarely match textually — Dawn says
 *  "NVIDIA GeForce RTX 4090" where CUDA says "NVIDIA GeForce RTX 4090 (sm_89,
 *  128 SMs)". Compare on the words they have in common instead. */
function sameDevice(a, b) {
  const words = (s) =>
    new Set(
      (s ?? '')
        .toLowerCase()
        .replace(/[^a-z0-9 ]+/g, ' ')
        .split(/\s+/)
        .filter((w) => w.length > 2),
    );
  const wa = words(a);
  const wb = words(b);
  let shared = 0;
  for (const w of wa) if (wb.has(w)) shared++;
  return shared >= 2;
}
