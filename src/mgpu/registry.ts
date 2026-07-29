/**
 * The available .m models.
 *
 * Parameter metadata (names, defaults, slider ranges), species names and the
 * dealiasing degree stay in src/solver/models.ts: the host owns those, and
 * sharing them means the reference solver used by the tests and the .m running
 * on the GPU are always configured identically. This module only attaches each
 * model's MATLAB source.
 *
 * Naming convention, relied on by the app and documented in each .m:
 *   `u`, `v`, ...  grid fields the model computes and the app renders
 *   `U`, `V`, ...  the corresponding spectral state (uppercase)
 */
import { models, type ModelSpec, type ParamSpec } from '../solver/models.ts';
import schnakenbergSource from '../../models/schnakenberg.m?raw';
import brusselatorSource from '../../models/brusselator.m?raw';
import allencahnSource from '../../models/allencahn.m?raw';

const sources: Record<string, string> = {
  schnakenberg: schnakenbergSource,
  brusselator: brusselatorSource,
  allencahn: allencahnSource,
};

export interface MModel {
  key: string;
  label: string;
  blurb: string;
  /** Grid fields to render, one panel each. */
  species: string[];
  /** Spectral state names the .m advances. */
  state: string[];
  params: ParamSpec[];
  /** Polynomial degree of the reaction, for grid dealiasing. */
  pdeg: number;
  /** Amplitude of the seeded perturbation. */
  seedAmp: number;
  /** MATLAB source — the algorithm itself. */
  source: string;
}

const fromSpec = (m: ModelSpec): MModel => ({
  key: m.key,
  label: m.label,
  blurb: m.blurb,
  species: m.species,
  state: m.species.map((s) => s.toUpperCase()),
  params: m.params,
  pdeg: m.pdeg,
  seedAmp: m.seedAmp,
  source: sources[m.key],
});

export const mModels: MModel[] = models
  .filter((m) => sources[m.key] !== undefined)
  .map(fromSpec);

export const mModelByKey = (key: string): MModel | undefined =>
  mModels.find((m) => m.key === key);
