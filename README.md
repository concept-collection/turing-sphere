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

## Desktop vs browser

How much does running this in a browser cost? [`scripts/bench.ts`](scripts/bench.ts)
answers that by running the *same* code — same `Simulation`, same WGSL
transforms, same parameters — from Node on desktop WebGPU (Google Dawn), and
the app prints the command line that reproduces whatever it is currently
simulating:

```
node scripts/bench.mjs --preset schnak-spots --lmax 63 --backend webgpu --steps 2000 \
  --seed 1 --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05
```

Copy it from under the stats line, run it, and compare the `ms/step` it reports
with the app's. Both sides go through the one shared
[`src/bench/runSpec.ts`](src/bench/runSpec.ts) — the app formats a run into that
command, the benchmark parses it back — so there is no second copy of the
defaults for the two runs to drift apart on. Node runs the TypeScript sources
directly, so `src/` is literally the same code in both places, down to the
device request in `requestShtDevice()` (Dawn is installed under `navigator.gpu`
and the WebGPU globals, and the rest runs unchanged).

Desktop WebGPU comes from the `webgpu` package (prebuilt Dawn, ~70 MB), listed
as an optional dependency so that a platform it has no binaries for fails the
install of that package alone rather than the whole tree. `npm install` picks it
up; without it, only `--backend cpu` runs and the benchmark says so. Other
flags: `--steps`, `--warmup`, `--json`, `--help`; `DAWN_FLAGS='backend=vulkan'`
(`;`-separated) passes Dawn options through, e.g. to pick a backend or to
compare against Dawn's own software adapter.

What the comparison does and does not control for:

- the benchmark is **solver only**; the app's `ms/step` excludes `draw()` but is
  still measured on a page that renders two spheres between steps. For a browser
  number with no rendering at all, open `test.html?soak=2000&lmax=63`.
- each step is four transforms, each ending in a buffer readback, so both sides
  are dominated by submit-and-map latency rather than arithmetic — this measures
  a driver round-trip more than it measures a GPU.
- the browser adds its own GPU-process boundary and, for a page that is not
  cross-origin isolated, coarser timers.

## Tests

- `npm run bench -- --help` — the desktop benchmark above (see
  [Desktop vs browser](#desktop-vs-browser)).
- `npm run test:node` — f64 solver correctness in Node: exact single-mode
  linear recurrence, exact uniform-state reaction ODE, and the linearized
  Turing-mode 2×2 IMEX recurrence (all at ~1e-12).
- `npm run test:gpu` — builds and drives headless Chrome: GPU-vs-CPU transform
  and solver cross-checks, plus a 100-step stability run.
- `node scripts/longrun-node.ts` — CPU run to t = 100 confirming pattern
  saturation.
- `node scripts/soak.mjs [steps] [lmax] [backend]` — drive the demo for many
  steps, sampling JS heap and catching crashes. A 900-step run at lmax 63 on
  software WebGPU (SwiftShader) completes with a flat ~4 MB heap.
- `node scripts/screenshot.mjs out.png [light|dark] [minSteps]` — screenshot
  the demo after a number of steps.
- `node scripts/check-live.mjs [url]` — smoke-check a deployed URL in a real
  browser: load, press Run, confirm the solver advances.
- `test.html?soak=<steps>&lmax=<n>` — solver-only soak with no rendering.

### A note on canvas resizing

Early long runs killed the browser after ~700–800 steps. The cause was the
colorbar's min/max labels changing width as their digit count changed, which
reflowed the panel, fired the `ResizeObserver`, and called
`renderer.setSize()` — reallocating the WebGL drawing buffer. Assigning
`canvas.width` also blanks the canvas even when the value is unchanged, so the
same bug caused visible flicker. Fixed by giving the colorbar column a fixed
width and making `SphereScene.resize()` return early on no-op resizes.

## Development

```
npm install
npm run dev       # local dev server
npm run build     # type-check + production build to dist/
```

The `.ts` entry points under `scripts/` are run by Node directly, which strips
types without being asked only from Node 22.18 / 23.6 / 24 on. Everything here
works back to 22.6, where stripping exists but is flagged: the npm scripts pass
`--experimental-strip-types` themselves, and the benchmark — the one command
that gets copied to other machines — goes through
[`scripts/bench.mjs`](scripts/bench.mjs), which re-runs itself with the flag
when it has to. Invoking a `scripts/*.ts` file by hand on 22.6–22.17 needs the
flag spelled out.

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are vendored).
