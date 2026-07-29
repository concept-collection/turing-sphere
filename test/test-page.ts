/**
 * Browser validation, in the environment the demo actually ships to.
 *
 * Runs the same three check modules as `npm run test:node` — so both GPU stacks
 * (Dawn on the desktop, the browser's own here) get the same guarantees — plus a
 * long soak that only makes sense in a page.
 *
 * Results are posted to window.__RESULTS__ for the headless runner.
 */
import { requestShtDevice } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModels, defaultParams } from '../src/mgpu/registry.ts';
import { transformChecks } from './transformChecks.ts';
import { analyticChecks } from './analyticChecks.ts';
import { modelChecks } from './modelChecks.ts';

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

/**
 * Solver-only soak, selected with ?soak=<steps>&lmax=<n>. No rendering, so it
 * isolates the compiled .m and the transforms from three.js.
 */
async function soak(steps: number, lmax: number): Promise<void> {
  const device = await requestShtDevice();
  const model = mModels[0];
  const session = await ModelSession.create({
    device,
    model,
    params: defaultParams(model),
    lmax,
  });
  session.seed(5);
  log(
    `soak: ${steps} steps at lmax ${lmax} ` +
      `(grid ${session.cfg.nlat}x${session.cfg.nphi}), solver only`,
  );

  const BATCH = 25;
  const t0 = performance.now();
  for (let s = 0; s < steps; s += BATCH) {
    session.step(Math.min(BATCH, steps - s));
    const u = await session.read(model.species[0]);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of u) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if ((s + BATCH) % 100 === 0) {
      const mem = (performance as Performance & { memory?: { usedJSHeapSize: number } })
        .memory;
      log(
        `  step ${session.steps}  u in [${lo.toFixed(4)}, ${hi.toFixed(4)}]` +
          (mem ? `  heap ${(mem.usedJSHeapSize / 1048576).toFixed(1)} MB` : ''),
      );
      // yield so the page stays responsive and the runner can poll
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const ms = (performance.now() - t0) / steps;

  const final = await session.read(model.species[0]);
  let finite = true;
  for (const v of final) if (!Number.isFinite(v)) finite = false;
  check(`soak: ${steps} steps survived`, finite, `${ms.toFixed(1)} ms/step`);

  session.destroy();
  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

async function main(): Promise<void> {
  const q = new URLSearchParams(location.search);
  if (q.has('soak')) {
    return soak(Number(q.get('soak')) || 500, Number(q.get('lmax')) || 63);
  }
  const device = await requestShtDevice();

  await transformChecks(device, check, log);
  await analyticChecks(device, check, log);
  await modelChecks(device, check, log);

  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

main().catch((e) => {
  const msg = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
  log(`fatal: ${msg}`);
  window.__RESULTS__ = { ok: false, fatal: msg, lines };
});
