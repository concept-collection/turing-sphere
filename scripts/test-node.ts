/**
 * Solver correctness tests against the f64 CPU transform backend.
 *
 *  A. Linear reaction + diffusion, single mode: every (l,m) mode of
 *     f = c*u with implicit diffusion follows the exact scalar recurrence
 *     g = (1 + dt*c) / (1 + dt*D*l(l+1)).
 *  B. Uniform state, nonlinear reaction: the l=0 mode follows the explicit
 *     Euler map of the reaction ODE exactly.
 *  C. Turing linear stability: a small single-mode perturbation of the
 *     Schnakenberg fixed point follows the 2x2 linearized IMEX recurrence,
 *     and the (24, 7) mode lies in the unstable band.
 *
 * Run: node scripts/test-node.ts
 */
import { CpuBackend } from '../src/solver/backend.ts';
import { Simulation, gridForLmax } from '../src/solver/simulation.ts';
import { models, defaultParams } from '../src/solver/models.ts';
import type { ModelSpec } from '../src/solver/models.ts';
import { lmIndex } from '../src/sht/layout.ts';

let failures = 0;
function check(name: string, ok: boolean, detail: string): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
}

// ---------------------------------------------------------------- test A
{
  const lmax = 15;
  const { nlat, nphi } = gridForLmax(lmax, 1);
  const backend = new CpuBackend({ lmax, mmax: lmax, nlat, nphi });
  const c = -0.3;
  const D = 0.01;
  const model: ModelSpec = {
    key: 'linear', label: 'linear', blurb: '', species: ['u'],
    params: [], pdeg: 1, seedAmp: 0,
    diffusivities: () => [D],
    reaction(_p, _t, _x, _y, _z, V, out) {
      for (let i = 0; i < out[0].length; i++) out[0][i] = c * V[0][i];
    },
    init() {},
  };
  const sim = new Simulation(backend, model, { dt: 0.1 });
  const l = 5, m = 2;
  const idx = lmIndex(lmax, l, m);
  sim.U[0][2 * idx] = 0.8;
  sim.U[0][2 * idx + 1] = -0.35;

  const nsteps = 20;
  for (let s = 0; s < nsteps; s++) await sim.step();

  const g = (1 + 0.1 * c) / (1 + 0.1 * D * l * (l + 1));
  const gn = Math.pow(g, nsteps);
  const errRe = Math.abs(sim.U[0][2 * idx] - 0.8 * gn);
  const errIm = Math.abs(sim.U[0][2 * idx + 1] - -0.35 * gn);
  let leak = 0;
  for (let i = 0; i < backend.nlm; i++) {
    if (i === idx) continue;
    leak = Math.max(leak, Math.abs(sim.U[0][2 * i]), Math.abs(sim.U[0][2 * i + 1]));
  }
  check('A: single-mode linear recurrence', errRe < 1e-12 && errIm < 1e-12,
    `err=(${errRe.toExponential(2)}, ${errIm.toExponential(2)})`);
  check('A: no leakage into other modes', leak < 1e-12, `leak=${leak.toExponential(2)}`);
}

// ---------------------------------------------------------------- test B
{
  const schnak = models[0];
  const p = defaultParams(schnak);
  const lmax = 15;
  const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
  const backend = new CpuBackend({ lmax, mmax: lmax, nlat, nphi });
  const uniform: ModelSpec = {
    ...schnak,
    seedAmp: 0,
    init(pp, x, _y, _z, _randn, out) {
      out[0].fill(1.2);
      out[1].fill(0.8);
      void pp; void x;
    },
  };
  const sim = new Simulation(backend, uniform, p);
  await sim.init(1);
  const nsteps = 50;
  for (let s = 0; s < nsteps; s++) await sim.step();

  // reference: explicit Euler on the 2-species ODE (l=0 is untouched by diffusion)
  let u = 1.2, v = 0.8;
  for (let s = 0; s < nsteps; s++) {
    const fu = p.a - u + u * u * v;
    const fv = p.b - u * u * v;
    u += p.dt * fu;
    v += p.dt * fv;
  }
  // the area mean is the l=0 coefficient of U (V lags U by one step)
  const sqrt4pi = Math.sqrt(4 * Math.PI);
  const i00 = 2 * lmIndex(lmax, 0, 0);
  const errU = Math.abs(sim.U[0][i00] / sqrt4pi - u);
  const errV = Math.abs(sim.U[1][i00] / sqrt4pi - v);
  check('B: uniform nonlinear reaction ODE', errU < 1e-10 && errV < 1e-10,
    `err=(${errU.toExponential(2)}, ${errV.toExponential(2)}) u=${u.toFixed(6)} v=${v.toFixed(6)}`);
}

// ---------------------------------------------------------------- test C
{
  const schnak = models[0];
  const p = defaultParams(schnak);
  const lmax = 31;
  const { nlat, nphi } = gridForLmax(lmax, schnak.pdeg);
  const backend = new CpuBackend({ lmax, mmax: lmax, nlat, nphi });
  const sim = new Simulation(backend, schnak, p);

  const us = p.a + p.b; // 1.0
  const vs = p.b / (us * us); // 0.9
  const sqrt4pi = Math.sqrt(4 * Math.PI);
  const l = 24, m = 7;
  const idx = lmIndex(lmax, l, m);
  const eps = 1e-6;
  const c0 = [eps, 0.5 * eps];
  // fixed point + single-mode perturbation, set directly in spectral space
  sim.U[0][2 * lmIndex(lmax, 0, 0)] = us * sqrt4pi;
  sim.U[1][2 * lmIndex(lmax, 0, 0)] = vs * sqrt4pi;
  sim.U[0][2 * idx] = c0[0];
  sim.U[1][2 * idx] = c0[1];

  const nsteps = 20;
  for (let s = 0; s < nsteps; s++) await sim.step();

  // linearized IMEX recurrence: c' = diag(1/(1+dt*Dk*lam)) * (I + dt*J) * c
  const lam = l * (l + 1);
  const J = [
    [-1 + 2 * us * vs, us * us],
    [-2 * us * vs, -us * us],
  ];
  let c = [...c0];
  for (let s = 0; s < nsteps; s++) {
    const r0 = c[0] + p.dt * (J[0][0] * c[0] + J[0][1] * c[1]);
    const r1 = c[1] + p.dt * (J[1][0] * c[0] + J[1][1] * c[1]);
    c = [r0 / (1 + p.dt * p.D1 * lam), r1 / (1 + p.dt * p.D2 * lam)];
  }
  const got = [sim.U[0][2 * idx], sim.U[1][2 * idx]];
  const errU = Math.abs(got[0] - c[0]) / Math.abs(c[0]);
  const errV = Math.abs(got[1] - c[1]) / Math.abs(c[1]);
  check('C: linearized Turing-mode recurrence', errU < 1e-4 && errV < 1e-4,
    `rel err=(${errU.toExponential(2)}, ${errV.toExponential(2)})`);
  check('C: (l=24, m=7) is growing', Math.abs(got[0]) > Math.abs(c0[0]),
    `|c|: ${Math.abs(c0[0]).toExponential(2)} -> ${Math.abs(got[0]).toExponential(2)}`);
}

console.log(failures === 0 ? '\nAll tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
