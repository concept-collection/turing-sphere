# turing-sphere

Reaction–diffusion systems (Turing patterns) solved **live in the browser on the
surface of a sphere**, using a spectral spherical-harmonic method with the
transforms running on the GPU via WebGPU.

The solver itself is **MATLAB**. The `.m` files under [`models/`](models/) are the
algorithm — [numbl](https://numbl.org) parses and lowers them in the browser, and
each element-wise line becomes a WebGPU compute kernel. You can edit the MATLAB
on the page and watch the pattern change.

**Live demo:** <https://concept-collection.github.io/turing-sphere-2/>

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

Three models are included, one `.m` file each:

- **[Schnakenberg](models/schnakenberg.m)** — Turing spots (unstable band
  14 ≤ l ≤ 40, peak l = 24)
- **[Brusselator](models/brusselator.m)** — stripes and spots from a stiffer reaction
- **[Allen–Cahn](models/allencahn.m)** — a single species whose interfaces form
  and coarsen

## MATLAB, compiled to WebGPU

A model file is ordinary MATLAB defining two functions — `init` builds the initial
spectral state, `step` advances it one timestep:

```matlab
function [Un, Vn, u, v] = step(U, V, lam, a, b, D1, D2, dt)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;
  Un = (U + dt * analys(a - u + uuv)) ./ (1 + (dt * D1) * lam);
  Vn = (V + dt * analys(b - uuv))     ./ (1 + (dt * D2) * lam);
end
```

Getting from there to the GPU uses numbl for everything up to the IR, and this
repo only for the backend:

1. **numbl parses and lowers.** Each function is specialized for the concrete
   argument types of the current grid, via the same `specializeUserFunction`
   entry point numbl's own JIT uses. Types and array shapes are fixed at this
   point, so the backend never has to re-decide what an operation means.
2. **numbl's inline pass fuses.** Lowering emits one statement per *operator*
   (ANF); `inlinePass` folds single-use temps back into their consumer, so one
   line of MATLAB becomes one expression tree. `uuv = u .* u .* v` arrives as a
   single statement, not three.
3. **This repo emits WGSL** ([`src/mgpu/wgsl.ts`](src/mgpu/wgsl.ts)). Each
   element-wise statement becomes one compute kernel that computes one output
   element per invocation — the WebGPU counterpart of numbl's own C-side fused
   emitter. Anything it cannot express is refused at compile time with a source
   position, never silently mis-compiled.
4. **`synth` / `analys` are external operations.** numbl learns their type rules
   from a `.mtoc2.js` workspace file — its sanctioned extension point for a
   JS-defined builtin — and the backend maps each call onto the existing
   spherical-harmonic compute pipelines.

The Schnakenberg step above compiles to 11 GPU operations: 4 transforms, 5
generated kernels, and 2 buffer copies feeding the new state back.

Two consequences worth noting:

- **The step is synchronous.** WebGPU's encode path (`writeBuffer`, dispatch,
  `submit`) is all synchronous; only readback and pipeline creation are async, and
  every pipeline is built once at compile time. So a timestep is pure command
  recording — the whole batch goes out in one submit, and the only `await` in the
  loop is the single readback per rendered frame. numbl's own execution being
  synchronous is therefore not an obstacle: nothing about the algorithm needs to
  block.
- **Parameters are uniforms, not constants.** Tunable scalars are deliberately
  lowered without exact values, so moving a slider rewrites a small buffer
  instead of triggering a recompile. Editing the MATLAB recompiles; changing `dt`
  does not.

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
  transform used for testing.
- **Rendering:** three.js spheres with per-vertex colormaps, adapted from the
  `SphereEmbedding` view in
  [figpack](https://github.com/flatironinstitute/figpack)'s experimental
  extension package ([`src/render/`](src/render/)).
- **Solver:** the MATLAB stayed MATLAB. [`models/`](models/) holds the IMEX loop
  as `.m` files, executed on the GPU by [`src/mgpu/`](src/mgpu/).

[`src/solver/`](src/solver/) still holds the TypeScript port of the same loop. The
app no longer runs it, but it is an *independent* implementation of the scheme,
which makes it the test oracle: `npm run test:gpu` runs both from the same seeded
perturbation through the same transforms and compares. It is also where parameter
metadata (names, defaults, slider ranges) lives, so the two paths cannot be
configured differently.

Because the algorithm is now compiled to compute shaders, **WebGPU is required** —
there is no CPU fallback in the app (the f64 CPU transform remains, for tests).

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
runs the reference solver — same WGSL transforms, same parameters — from Node on
desktop WebGPU (Google Dawn), and the app prints the command line that
reproduces whatever it is currently simulating:

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
up; without it, only `--backend cpu` runs and the benchmark says so. Those
binaries need glibc 2.29+, which rules out older cluster images (RHEL/Rocky 8 is
2.28) unless you run inside a container with a newer base. Other
flags: `--steps`, `--warmup`, `--json`, `--help`; `DAWN_FLAGS='backend=vulkan'`
(`;`-separated) passes Dawn options through, e.g. to pick a backend or to
compare against Dawn's own software adapter.

What the comparison does and does not control for:

- **it is not the same solver.** The benchmark runs the TypeScript reference; the
  app runs the `.m` compiled to WGSL. Node cannot load numbl's TypeScript sources
  (its internal imports are extensionless-`.js`, which needs a bundler's
  resolution), so the `.m` path is browser-only for now. Same scheme, same
  transforms, same parameters — but the reaction and the IMEX update happen in f64
  on the CPU there and in fp32 on the GPU here.
- the benchmark is **solver only**; the app's `ms/step` includes the per-frame
  readback amortized over the step batch. For a browser number with no rendering,
  open `test.html?soak=2000&lmax=63` (that soak also runs the reference solver).
- the reference pays a buffer readback on *every* transform — four driver
  round-trips per step — so it measures submit-and-map latency more than
  arithmetic. The `.m` path keeps everything in GPU buffers and submits once per
  batch, which is where its advantage should come from. On the software rasterizer
  in CI the transforms dominate and the two come out within ~10% of each other;
  the gap on real hardware is untested.
- the browser adds its own GPU-process boundary and, for a page that is not
  cross-origin isolated, coarser timers.

## Tests

- `npm run bench -- --help` — the desktop benchmark above (see
  [Desktop vs browser](#desktop-vs-browser)).
- `npm run test:node` — f64 solver correctness in Node: exact single-mode
  linear recurrence, exact uniform-state reaction ODE, and the linearized
  Turing-mode 2×2 IMEX recurrence (all at ~1e-12).
- `npm run test:gpu` — builds and drives headless Chrome: GPU-vs-CPU transform
  and solver cross-checks, a 100-step stability run, and then for every `.m`
  model: that it compiles, that its element-wise lines each fuse into exactly one
  kernel, and that 10 steps agree with the reference solver from the same seed
  (they agree to ~1e-7 relative L2 — fp32 round-off).
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

### The numbl dependency

numbl is a local `file:../../numbl` dependency, so a sibling checkout of
[numbl](https://github.com/flatironinstitute/numbl) is required. We use its
compiler internals — parser, lowerer, IR, inline pass — which its package
`exports` map does not publish, so they are reached through the `numbl-src` path
alias in [`vite.config.ts`](vite.config.ts).

The exact surface we depend on is written down in
[`src/mgpu/numbl.d.ts`](src/mgpu/numbl.d.ts) and TypeScript checks against
*that*, not against numbl's sources. This keeps this project's compiler settings
independent of numbl's (its sources do not type-check under the stricter options
used here), and means a change to one of those shapes upstream breaks the build
here with a clear diff rather than deep inside numbl's tree.

The compiler is ~395 kB gzipped and lands in its own chunk. That is the cost of
compiling MATLAB in the page; a build-time lowering step could remove it at the
price of no longer being editable live.

CI clones numbl to the sibling path that `file:` dependency expects, pinned to a
commit. Two details make that work, both verified by building against a checkout
that had none of numbl's own dependencies installed:

- **numbl's `node_modules` are not needed.** The slice we import — parser,
  lowering, IR, inline pass — is self-contained TypeScript. (Other parts of numbl
  do import `three`, `react` and `fflate`; we never reach them.)
- **the install must pass `--ignore-scripts`.** npm runs a linked package's
  `prepare` script, and numbl's is `husky`, which is not installed in CI.

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
