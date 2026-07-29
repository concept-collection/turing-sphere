# turing-sphere

Reaction–diffusion systems (Turing patterns) solved **live in the browser on the
surface of a sphere**, using a spectral spherical-harmonic method with the
transforms running on the GPU via WebGPU.

**Live demo:** <https://concept-collection.github.io/turing-sphere/>

## What it does

It solves the N-species system

```
d(u_k)/dt = D_k*lap_s(u_k) + f_k(t, x, y, z, u_1, ..., u_N),    k = 1, ..., N
```

on the unit sphere, where `lap_s` is the Laplace–Beltrami operator. Diffusion is
treated implicitly in spherical-harmonic coefficient space, where `lap_s` is
diagonal with eigenvalues `-l(l+1)`; reaction is treated explicitly on the grid.
The two are combined with a first-order IMEX Euler step — the entire time loop is

```
V_k  = synth(U_k)                          # spectral -> grid
R_k  = analys(f_k(t, x, y, z, V_1..V_N))   # reaction on grid -> spectral
U_k  = (U_k + dt*R_k) / (1 + dt*D_k*l(l+1))
```

You watch the patterns emerge in real time on orbitable 3D spheres (one per
species, cameras synced), with pause/resume, re-seeding, live parameter editing,
and colormap selection.

Three presets are included:

- **Schnakenberg** — Turing spots (unstable band 14 ≤ l ≤ 40, peak l = 24)
- **Brusselator** — stripes and spots from a stiffer reaction
- **Allen–Cahn** — a single species whose interfaces form and coarsen

## Provenance

This is the browser port of a MATLAB reference implementation
(`SphericalReactionDiffusion.m`, "websph"), which defines the solver through a
four-member porting boundary: `coeffs2vals`, `vals2coeffs`, `grid.lat`,
`grid.lon`. Profiling of the MATLAB version shows the transforms are ~96% of
compute, so this port swaps in:

- **Transforms:** [shtns-webgpu](https://github.com/concept-collection/shtns-webgpu) —
  fp32 spherical harmonic transforms in WGSL compute shaders, modeled on
  [SHTNS](https://nschaeff.bitbucket.io/shtns/). Its source is vendored under
  [`src/sht/`](src/sht/) (CECILL-2.1), including the f64 CPU reference
  transform used for testing and as a no-WebGPU fallback.
- **Rendering:** three.js spheres with per-vertex colormaps, adapted from the
  `SphereEmbedding` view in
  [figpack](https://github.com/flatironinstitute/figpack)'s experimental
  extension package ([`src/render/`](src/render/)).
- **Solver:** [`src/solver/simulation.ts`](src/solver/simulation.ts), a direct
  TypeScript port of the MATLAB IMEX loop, in f64 on the coefficients with the
  transforms in fp32 on the GPU.

## Numerics

- Grid: Gauss–Legendre × equispaced-phi, dealiased for the cubic reactions with
  the `(pdeg+1)` rule from the reference implementation:
  `nlat ≥ ((pdeg+1)·lmax+1)/2`, `nphi ≥ (pdeg+1)·lmax+1` (rounded up to a power
  of two for the GPU FFT path). At the default lmax 63 that is a 128×256 grid.
- Spectral layout: SHTNS conventions — orthonormal + Condon–Shortley, complex
  coefficients for m ≥ 0, m-major ordering.
- fp32 transforms introduce ~1e-6 relative error per step (verified against the
  f64 CPU path); for pattern formation from 1e-2 seeded noise this is
  inconsequential.

## Tests

- `npm run test:node` — f64 solver correctness in Node: exact single-mode
  linear recurrence, exact uniform-state reaction ODE, and the linearized
  Turing-mode 2×2 IMEX recurrence (all at ~1e-12).
- `npm run test:gpu` — builds and drives headless Chrome: GPU-vs-CPU transform
  and solver cross-checks, plus a 100-step stability run.
- `node scripts/longrun-node.ts` — CPU run to t = 100 confirming pattern
  saturation.
- `node scripts/screenshot.mjs out.png [light|dark] [minSteps]` — screenshot
  the demo after a number of steps.

## Development

```
npm install
npm run dev       # local dev server
npm run build     # type-check + production build to dist/
```

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are vendored).
