/**
 * One solver run, described in a single object shared by the browser app and
 * the command-line benchmark.  The app formats the run it is currently showing
 * into a `node scripts/bench.mjs ...` command; the benchmark parses that command
 * back into the same object and drives the same Simulation with it.  Neither
 * side keeps its own copy of the defaults, so the two runs cannot drift apart.
 */
import {
  models,
  presets,
  defaultParams,
  type ModelSpec,
  type Params,
  type Preset,
} from '../solver/models.ts';
import { gridForLmax } from '../solver/simulation.ts';
import type { ShtConfig } from '../sht/layout.ts';

export type BackendKind = 'webgpu' | 'cpu';

export interface RunSpec {
  /** Preset key from models.ts; fixes the model, params may still be edited. */
  preset: string;
  lmax: number;
  backend: BackendKind;
  /** Seed of the initial noise. */
  seed: number;
  /** Timed steps (the app runs forever; the benchmark stops here). */
  steps: number;
  /** Untimed steps run first, so shader/pipeline warm-up is not measured. */
  warmup: number;
  /** Full parameter set of the preset's model, as edited. */
  params: Params;
}

/** The command the app displays and the benchmark answers to.  It names the
 *  .mjs wrapper rather than bench.ts, so that it also runs on a Node that
 *  needs to be told to strip types (see scripts/bench.mjs). */
export const BENCH_COMMAND = 'node scripts/bench.mjs';
export const DEFAULT_LMAX = 63;
export const DEFAULT_SEED = 1;
/** Long enough that clock ramp-up and the occasional scheduling hiccup wash
 *  out: ~10 s of GPU stepping at lmax 63. */
export const DEFAULT_STEPS = 2000;
export const DEFAULT_WARMUP = 100;
export const DEFAULT_BACKEND: BackendKind = 'webgpu';

/** Model + starting parameters of a preset, for the app's dropdown and the
 *  benchmark's --preset flag. */
export function resolvePreset(key: string): {
  preset: Preset;
  model: ModelSpec;
  params: Params;
} {
  const preset = presets.find((p) => p.key === key);
  if (!preset) {
    throw new Error(
      `unknown preset '${key}' (have: ${presets.map((p) => p.key).join(', ')})`,
    );
  }
  const model = models.find((m) => m.key === preset.modelKey);
  if (!model) throw new Error(`preset '${key}' names unknown model '${preset.modelKey}'`);
  return { preset, model, params: { ...defaultParams(model), ...preset.params } };
}

export function modelForSpec(spec: RunSpec): ModelSpec {
  return resolvePreset(spec.preset).model;
}

/** Transform configuration implied by the spec (same rule as the app). */
export function configForSpec(spec: RunSpec): ShtConfig {
  const { nlat, nphi } = gridForLmax(spec.lmax, modelForSpec(spec).pdeg);
  return { lmax: spec.lmax, mmax: spec.lmax, nlat, nphi };
}

/** The command line that reproduces this run. Every knob the app exposes is
 *  written out explicitly, so the command stays valid if a preset changes. */
export function formatCommand(spec: RunSpec): string {
  const model = modelForSpec(spec);
  const parts = [
    BENCH_COMMAND,
    `--preset ${spec.preset}`,
    `--lmax ${spec.lmax}`,
    `--backend ${spec.backend}`,
    `--steps ${spec.steps}`,
    `--seed ${spec.seed}`,
    ...model.params.map((p) => `--${p.key} ${String(spec.params[p.key])}`),
  ];
  if (spec.warmup !== DEFAULT_WARMUP) parts.push(`--warmup ${spec.warmup}`);
  return parts.join(' ');
}

/** Inverse of formatCommand: `--key value` or `--key=value`, in any order.
 *  Throws with a usable message on anything it does not recognize. */
export function parseArgs(argv: string[]): RunSpec {
  const flags = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) throw new Error(`unexpected argument '${arg}'`);
    const eq = arg.indexOf('=');
    const key = eq >= 0 ? arg.slice(2, eq) : arg.slice(2);
    const value = eq >= 0 ? arg.slice(eq + 1) : argv[++i];
    if (value === undefined) throw new Error(`--${key} needs a value`);
    if (!key) throw new Error(`bad option '${arg}'`);
    flags.set(key, value);
  }
  const take = (key: string): string | undefined => {
    const v = flags.get(key);
    flags.delete(key);
    return v;
  };
  const number = (key: string, dflt: number): number => {
    const raw = take(key);
    if (raw === undefined) return dflt;
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`--${key} must be a number (got '${raw}')`);
    return v;
  };
  const count = (key: string, dflt: number, min: number): number => {
    const v = number(key, dflt);
    if (!Number.isInteger(v) || v < min) {
      throw new Error(`--${key} must be an integer >= ${min} (got '${v}')`);
    }
    return v;
  };

  const presetKey = take('preset') ?? presets[0].key;
  const { model, params } = resolvePreset(presetKey);
  const backend = take('backend') ?? DEFAULT_BACKEND;
  if (backend !== 'webgpu' && backend !== 'cpu') {
    throw new Error(`--backend must be 'webgpu' or 'cpu' (got '${backend}')`);
  }
  const spec: RunSpec = {
    preset: presetKey,
    lmax: count('lmax', DEFAULT_LMAX, 1),
    backend,
    seed: number('seed', DEFAULT_SEED),
    steps: count('steps', DEFAULT_STEPS, 1),
    warmup: count('warmup', DEFAULT_WARMUP, 0),
    params,
  };
  for (const p of model.params) {
    const raw = take(p.key);
    if (raw === undefined) continue;
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`--${p.key} must be a number (got '${raw}')`);
    params[p.key] = v;
  }
  if (flags.size) {
    throw new Error(
      `unknown option(s): ${[...flags.keys()].map((k) => `--${k}`).join(', ')}\n` +
        `parameters of ${model.label}: ${model.params.map((p) => `--${p.key}`).join(' ')}`,
    );
  }
  return spec;
}
