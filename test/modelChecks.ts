/**
 * Every model the app offers: that it compiles, what it compiles to, and that it
 * runs stably and produces a pattern.
 *
 * Numerical correctness of the pipeline is analyticChecks.ts's job. This file is
 * about the models themselves and about the compilation staying as intended — in
 * particular the kernel count, which is a fusion guard: numbl's lowering emits
 * one statement per *operator*, and its inline pass folds those back into
 * per-line expression trees. If that stops happening the results stay correct
 * but every operator becomes its own dispatch, which is invisible except here.
 */
import { ModelSession } from '../src/mgpu/session.ts';
import { mModels, defaultParams } from '../src/mgpu/registry.ts';
import type { Check, Log } from './analyticChecks.ts';

/** Kernels the step of each model should compile to — one per element-wise line. */
const EXPECTED_KERNELS: Record<string, number> = {
  schnakenberg: 5,
  brusselator: 5,
  allencahn: 2,
};

const LMAX = 31;
const STEPS = 40;

export async function modelChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  check('models: registry populated', mModels.length === 3, `${mModels.length} models`);

  for (const model of mModels) {
    const session = await ModelSession.create({
      device,
      model,
      params: defaultParams(model),
      lmax: LMAX,
    });

    const plan = session.describe();
    const kernels = plan.step.filter((l) => l.startsWith('kernel')).length;
    const xforms = plan.step.filter(
      (l) => l.startsWith('synth') || l.startsWith('analys'),
    ).length;
    log(
      `  ${model.key}.m -> ${plan.step.length} ops/step ` +
        `(${kernels} generated kernels, ${xforms} transforms)`,
    );
    check(
      `${model.key}: element-wise lines fused into one kernel each`,
      kernels === EXPECTED_KERNELS[model.key],
      `${kernels} kernels (expected ${EXPECTED_KERNELS[model.key]})`,
    );

    session.seed(1);
    session.step(STEPS);

    // Every rendered field must be finite and have developed some contrast.
    for (const field of model.species) {
      const values = await session.read(field);
      let lo = Infinity;
      let hi = -Infinity;
      let finite = true;
      for (const v of values) {
        if (!Number.isFinite(v)) finite = false;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      check(
        `${model.key}: '${field}' is finite and patterned after ${STEPS} steps`,
        finite && hi - lo > 1e-6,
        finite
          ? `range [${lo.toFixed(5)}, ${hi.toFixed(5)}]`
          : 'contains NaN or Infinity',
      );
    }

    session.destroy();
  }
}
