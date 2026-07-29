/**
 * Long-run sanity check (CPU f64, lmax 31): run Schnakenberg to t = 100 and
 * confirm the pattern saturates into O(1)-contrast spots rather than decaying
 * or blowing up.  Run: node scripts/longrun-node.ts
 */
import { CpuBackend } from '../src/solver/backend.ts';
import { Simulation, gridForLmax } from '../src/solver/simulation.ts';
import { models, defaultParams } from '../src/solver/models.ts';

const schnak = models[0];
const params = defaultParams(schnak);
const lmax = 31;
const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
const backend = new CpuBackend({ lmax, mmax: lmax, nlat, nphi });
const sim = new Simulation(backend, schnak, params);
await sim.init(1);

const nsteps = Math.round(100 / params.dt);
const t0 = performance.now();
for (let s = 0; s < nsteps; s++) {
  await sim.step();
  if ((s + 1) % 400 === 0) {
    let lo = Infinity, hi = -Infinity;
    for (const v of sim.V[0]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    console.log(
      `t=${sim.t.toFixed(1).padStart(5)}  u in [${lo.toFixed(4)}, ${hi.toFixed(4)}]  ` +
      `contrast ${(hi - lo).toFixed(4)}`,
    );
  }
}
console.log(`${((performance.now() - t0) / nsteps).toFixed(1)} ms/step CPU`);

let lo = Infinity, hi = -Infinity;
for (const v of sim.V[0]) {
  if (v < lo) lo = v;
  if (v > hi) hi = v;
}
const ok = Number.isFinite(lo) && hi - lo > 0.3 && hi - lo < 5;
console.log(ok ? 'PASS: saturated O(1) pattern' : 'FAIL: no saturated pattern');
process.exit(ok ? 0 : 1);
