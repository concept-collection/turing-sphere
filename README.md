# turing-sphere

Reaction–diffusion systems (Turing patterns) solved **live in the browser on the
surface of a sphere**, using a spectral spherical-harmonic method with the
transforms running on the GPU via WebGPU.

The solver itself is **MATLAB**. The `.m` files under [`models/`](models/) are the
algorithm — [numbl](https://numbl.org) parses and lowers them in the browser, and
each element-wise line becomes a WebGPU compute kernel. You can edit the MATLAB
on the page and watch the pattern change.

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
colormap selection, and movie download — the run is recomputed from t = 0 and
encoded to a captioned MP4 in the browser (WebCodecs). The display can
oversample the solver — the state is spectral, so evaluating it on a finer grid
for rendering is exact interpolation, not smoothing. This never touches the
solver or its grid; by default it turns on only when the solver grid is coarse.

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
  transform used for testing. [`bench/shtns/`](bench/shtns/) builds the real
  SHTNS and measures ours against it — see
  [Against upstream SHTNS](#against-upstream-shtns).
- **Rendering:** three.js spheres with per-vertex colormaps, adapted from the
  `SphereEmbedding` view in
  [figpack](https://github.com/flatironinstitute/figpack)'s experimental
  extension package ([`src/render/`](src/render/)).
- **Solver:** the MATLAB stayed MATLAB. [`models/`](models/) holds the IMEX loop
  as `.m` files, executed on the GPU by [`src/mgpu/`](src/mgpu/). There is no
  second implementation: the app, the desktop benchmark and the tests all compile
  and run the same `.m`.

An earlier version of this repo carried a TypeScript port of the loop alongside
the `.m`, and used it as the test oracle. That is gone. Two implementations
agreeing only shows they share assumptions, so the `.m` path is now checked
against closed-form answers instead — see [Tests](#tests). The one place a second
implementation is still the right oracle is the transforms themselves, where
[`src/sht/reference.ts`](src/sht/reference.ts) is shtns-webgpu's own f64
direct-summation twin.

Because the algorithm is compiled to compute shaders, **WebGPU is required** —
there is no CPU fallback (the f64 CPU transform remains, for tests).

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
runs the *same* thing — same `.m`, lowered by numbl into the same WGSL kernels,
over the same transforms — from Node on desktop WebGPU (Google Dawn), and the app
prints the command line that reproduces whatever it is currently simulating:

```
npm run bench -- --preset schnak-spots --lmax 63 --steps 2000 \
  --seed 1 --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05
```

Copy it from under the stats line, run it, and compare the `ms/step` it reports
with the app's. Both sides go through the one shared
[`src/bench/runSpec.ts`](src/bench/runSpec.ts) — the app formats a run into that
command, the benchmark parses it back — so there is no second copy of the
defaults for the two runs to drift apart on. Both then go through the same
[`ModelSession`](src/mgpu/session.ts), down to the device request in
`requestShtDevice()` (Dawn is installed under `navigator.gpu` and the WebGPU
globals, and the rest runs unchanged).

The benchmark runs under `vite-node`, which is what resolves numbl's compiler
sources and the `?raw` model imports — plain Node cannot (see
[The numbl dependency](#the-numbl-dependency)).

It reports two numbers, because they answer different questions:

```
  0.54 ms/step   1857.5 steps/s   92.87 model time/s   (batches of 16)
  one step per submit: 0.74 ms mean · median 0.60 · p05 0.51 · p95 1.29 · min 0.50
```

The first is throughput: a batch of steps submitted together and awaited once,
which is how the app runs and what keeping the state in GPU buffers is for. The
second is per-step latency, one submit each — comparable to a design that
synchronises every step, and the only way to get a distribution.

**What the GPU-resident design is worth.** At lmax 31 on an Intel Xe (Mesa, via
Dawn) this path runs at **0.25 ms/step**, against **3.01 ms/step** for the
TypeScript solver this repo used to carry — same machine, same transforms, same
parameters. A **~12x** difference, and almost all of it is the four per-step
buffer readbacks that version paid and this one does not. Note that CI, which
only has a software rasterizer, shows no such gap: there the transforms dominate
and both designs land within ~10% of each other. The saving is real but it is a
saving on driver round-trips, so it only appears once the GPU is fast.

Desktop WebGPU comes from the `webgpu` package (prebuilt Dawn, ~70 MB), listed
as an optional dependency so that a platform it has no binaries for fails the
install of that package alone rather than the whole tree. `npm install` picks it
up; without it there is no desktop GPU to run on and the benchmark says so.
Those binaries need glibc 2.29+, which rules out older cluster images
(RHEL/Rocky 8 is 2.28) unless you run inside a container with a newer base. Other
flags: `--steps`, `--warmup`, `--batch`, `--json`, `--help`;
`DAWN_FLAGS='backend=vulkan'` (`;`-separated) passes Dawn options through, e.g. to
pick a backend or to compare against Dawn's own software adapter.

### Comparing the two honestly

The app reports **two** numbers, and only the first is comparable to the
benchmark:

```
solver 0.58 ms/step (1724 steps/s) · 12.4 ms/frame incl. readback + render
```

`solver` is the batch of steps alone, waited for but not read back — the same
thing the benchmark's throughput number measures. `ms/frame` additionally carries
a GPU→CPU readback **per species**, the colormapping, and the vertex upload.

Those per-frame costs are fixed: they do not shrink when the GPU gets faster. So
the faster your GPU, the larger the ratio between them — on a quick discrete GPU
it is easy for a frame to cost ten times the four steps inside it, purely because
a `mapAsync` round trip in a browser has to drain the queue and cross into the GPU
process. **That is expected, and it is not the solver being slower in the
browser.** Compare `solver` with the benchmark's throughput line; comparing
`ms/frame` against it measures the readback, not the computation.

Other things the comparison does not control for:

- the browser's renderer→GPU-process boundary on every submit, where Dawn in Node
  is in-process; and, for a page that is not cross-origin isolated, coarser
  `performance.now()`.
- both sides are fp32 throughout, on the same generated kernels, so nothing here
  is a numerics comparison — only a cost one.

### Why the browser is slower, and how to find out by how much

Some gap is real and some is measurement. Four numbers, in increasing order of
what they include — walk down them and the gap attributes itself:

```
node scripts/compare-perf.mjs [--lmax 63] [--steps 300]
```

measures the same solver work in both — batched, nothing read back, no rendering
on either side — and reports each with its CPU-encoding share, the Fourier stage,
and the adapter. It stops you first if the two are not even the same device: a
browser quietly falling back to a software adapter is a common cause of "the
browser is much slower", and then the ratio compares different hardware and means
nothing.

Or press **Benchmark** in the app: it pauses rendering and runs batches
continuously for two seconds, reporting the same measurement the terminal makes,
plus the **ramp** — the first third of the run against the last. GPUs downclock
when idle and an animation-paced loop leaves them idle most of every frame, so a
large ramp means the steady-state number is limited by clocks rather than by the
work.

By hand, five numbers, in increasing order of what they include:

| number | includes |
|---|---|
| `npm run bench -- --lmax 63` | desktop solver: batched steps, one sync per batch, in-process Dawn |
| the app's **Benchmark** button | browser solver, sustained, no rendering, no pacing |
| `test.html?soak=2000&lmax=63` → `solver` | the same, without the page around it |
| the app's `solver` | browser solver, one batch of 32 every two seconds |
| the app's `ms/frame` | four steps **plus** a readback per species, colormapping and the vertex upload |

If the soak matches the benchmark, the solver is fine in the browser and
everything above it is readback and rendering. If the soak is itself slower, the
remaining suspects are:

- **the GPU-process boundary.** Every submit and every sync is IPC out of the
  renderer; Dawn in Node is in-process. This is a fixed per-batch cost, so it hurts
  most when the GPU is fast. `npm run bench -- --batch 4` makes the desktop pay a
  sync as often as the app's frame loop does, which shows how much of the gap is
  just amortization.
- **not CPU command encoding**, which is worth ruling out explicitly because it is
  the obvious suspect: a step is ~47 WebGPU calls, and 32 of them per burst is a
  lot of JS→GPU traffic. Measured, it goes the other way — 0.009 ms/step in Chrome
  against 0.062 ms/step under node-webgpu, because Chrome defers commands to the
  GPU process while node-webgpu validates them inline. Encoding is *cheaper* in
  the browser. Both `compare-perf.mjs` and the benchmark print it.
- **competing with the renderer.** The page draws two spheres through WebGL on the
  same GPU, in its own animation loop. The soak has no renderer, so comparing the
  soak against the app's `solver` separates contention from everything else.
- **clocks.** An animation-paced loop leaves the GPU idle for most of each 16 ms
  frame, so it may never leave its low-power state, while the benchmark hammers it
  continuously and boosts. On a thermally managed laptop this alone can be worth a
  factor of two, and it is not something the code can fix. The **Benchmark**
  button's ramp figure measures it directly.
- **anything else using the GPU.** Another process competing for it changes
  whichever run overlaps it, which makes a comparison across two separate
  invocations meaningless. `compare-perf.mjs` runs both sides back to back in one
  invocation partly for this reason.
- **not buffer robustness**, another plausible suspect: WebGPU clamps every array
  access for safety, which could cost real time in the transform kernels' inner
  loops. Measured with Dawn's `disable_robustness` toggle
  (`DAWN_FLAGS='enable-dawn-features=disable_robustness' npm run bench`), it makes
  no difference here at all — 0.59 ms/step either way.
- **which browser.** WebGPU implementations differ substantially in maturity;
  Chrome and Safari are not interchangeable for this.

None of these change *what* is computed — see below for how to confirm that
independently.

### Is it really the same computation?

```
node scripts/compare-env.mjs [--lmax 31] [--steps 200] [--preset schnak-spots]
```

runs one identical spec on the desktop and in a real browser and compares the
final spectral state. The pipeline is deterministic given (model source,
parameters, lmax, seed, steps) — a seeded PRNG, then fixed arithmetic — so the two
should agree to fp32 round-off. Both sides build their spec through the same
`parseArgs`, so neither can quietly use a different default.

They will *not* agree bit for bit; GPUs differ in fused-multiply-add and other
latitude fp32 allows. Between Intel Xe (via Dawn) and SwiftShader — about as
different as two implementations get — 200 steps at lmax 31 agree to a relative
L2 of **2e-6**.

It also reports which **Fourier stage** each side chose. `ShtPlan` picks FFT or
DFT from the device's workgroup-storage and invocation limits, and those are
genuinely different algorithms that round differently, so a mismatch there
explains a difference in the values rather than being a symptom of one. The app's
stats line and the benchmark both print the chosen stage for the same reason.

## Against upstream SHTNS

The transforms are a WGSL translation of
[SHTNS](https://nschaeff.bitbucket.io/shtns/), and the tests check them against
their own f64 CPU twin — which shows they are self-consistent, not how they
compare with the library they are modeled on. SHTNS itself runs on the CPU with
hand-tuned SIMD codelets, and on Nvidia GPUs with its own CUDA kernels, including
a single-precision mode. That is a direct comparison, and
[`bench/shtns/`](bench/shtns/) makes it:

```
cd bench/shtns && ./bootstrap.sh && make    # clone SHTns at a pinned commit, build
node scripts/compare-native.mjs --check     # then, from the repo root
```

`bootstrap.sh` adds CUDA support when `nvcc` is on `PATH`, so the same tree gives
you the CPU comparison anywhere and the GPU one on a machine with an Nvidia card.
`compare-native.mjs` runs every implementation present, back to back in one
invocation so a second process competing for the GPU affects both sides rather
than one, and prints them in one table. On an RTX PRO 6000 Blackwell, at the
app's default lmax:

```
  grid lmax 63 · 128×256 · nlm 2,080   (one synthesis + one analysis per round trip)

  webgpu      0.084 ms/round trip  11848/s   (baseline)     fp32
              NVIDIA (blackwell), via Dawn   ·   CPU-side launching 0.012 ms/step
  shtns cuda  0.021 ms/round trip  48009/s   0.25x webgpu   fp32
              NVIDIA RTX PRO 6000 (sm_120, 188 SMs)   ·   CPU-side launching 0.018 ms/step
  shtns cpu   0.072 ms/round trip  13982/s   0.85x webgpu   fp64
              CPU, 1 thread
```

Read that carefully rather than as "4x". The two GPU rows are limited by different
things: the WGSL row spends 14% of its time on the CPU and is genuinely GPU-bound,
while SHTNS spends **86%** — 0.018 ms of 0.021 — queueing its six-or-so kernels, so
its number is close to what it costs to *submit* a round trip on that host and its
actual GPU time is below that and unresolved. The 4x is a lower bound on the gap in
GPU work, not a measurement of it. `compare-native.mjs` flags any row above 50%
for this reason.

The other number worth noticing is the third row: one CPU core in fp64 is about
level with the WGSL transforms on a 188-SM datacentre GPU. At lmax 63 there are
2,080 coefficients on a 128×256 grid — far too little work to occupy that card, so
this says more about occupancy than about the shaders. Sweep lmax before drawing
conclusions, and stop at 511: above that `16*nphi` exceeds the workgroup-storage
limit, the FFT stage falls back to the O(nphi·mmax) DFT, and the comparison stops
being about the FFT.

Two things are measured, because they answer different questions:

- **transforms** (`npm run bench:sht` here, `--mode transform` there) — one
  spectral → grid → spectral round trip and nothing else. This is the
  library-against-library number, and since the transforms are ~96% of the
  solver's compute it is what decides how fast the solver can be.
- **solver** (`npm run bench` here, `--mode solver` there) — a whole IMEX Euler
  timestep, which is what the app's `solver` line reports.

`--check` diffs the final spectral state across implementations, which is what
makes the timing mean anything: two numbers are only comparable if they are the
cost of the same computation. That check is possible at all because the spectral
layout and normalization are SHTNS's own — orthonormal with Condon–Shortley,
coefficients grouped by `m`, `LM(l,m)` agreeing index for index — so a state can
be diffed element by element with no reindexing. Over 20 steps, fp32 WGSL against
fp64 SHTNS agrees to **~1e-6** relative L2, for every model.

It is also the check on the one second implementation this repo has. The native
solver cannot run `models/<key>.m` — C has no numbl — so `bench/shtns/spec.h`
restates the same arithmetic, one line per line of MATLAB. `--check` is what
keeps that transcription honest, and `compare-native.mjs` refuses to compare two
runs whose resolved grid or parameters disagree, which is the other way the two
sides could drift.

[`bench/shtns/README.md`](bench/shtns/README.md) lists what is *not* identical and
should be kept in mind when reading the ratio — SHTNS runs its Legendre
recurrence in fp64 even in fp32 mode for `lmax <= 128` (WebGPU has no fp64 at
all), the Fourier stages are cuFFT/VkFFT/FFTW against a WGSL FFT, and SHTNS'
polar optimization is off by default here because we have none.

How much the grid size matters is easiest to see on a weak GPU, where there is no
launch-overhead floor to hide behind. On an Intel Xe iGPU against one core of the
same laptop, one round trip costs:

| lmax | grid | WGSL (fp32) | SHTNS, 1 CPU core (fp64) |
|---|---|---|---|
| 31 | 64×128 | 0.183 ms | 0.017 ms |
| 63 | 128×256 | 0.250 ms | 0.110 ms |
| 127 | 256×512 | 0.733 ms | 0.602 ms |

10x behind at lmax 31, 1.2x at lmax 127 — the same comparison, on the same two
chips. Whatever a single number says, it is saying it about one grid size.

## Tests

There is no second implementation of the solver to diff against, so the `.m` path
is checked against **closed-form answers**. Each case is one whose evolution is
known exactly, run through the whole real pipeline — MATLAB source, numbl
lowering, generated WGSL, GPU transforms — and compared with arithmetic
([`test/analyticChecks.ts`](test/analyticChecks.ts)):

- **A** — a linear reaction `f(u) = c*u` leaves every spherical-harmonic mode
  independent, growing by exactly `(1 + dt*c) / (1 + dt*D*l(l+1))` per step. This
  pins the transform round-trip, the eigenvalue mapping, the IMEX update and the
  state feedback at once, and checks that nothing leaks between modes. Agrees to
  ~2e-7 over 20 steps.
- **B** — a nonlinear reaction on a *uniform* field stays uniform and diffusion
  cannot touch it, so each step is exactly the scalar ODE map. Agrees to 1.5e-8
  over 25 steps. Checks that a generated kernel evaluates a nonlinear reaction.
- **C** — a 1e-6 perturbation of the Schnakenberg fixed point follows the
  linearized 2x2 IMEX recurrence, and the `(l=24, m=7)` mode is confirmed
  unstable. Looser (~2e-3) because fp32 keeps only about four digits of a
  perturbation that small.

Two test models exist only for this: [`test/models/linear.m`](test/models/linear.m)
and [`test/models/logistic.m`](test/models/logistic.m).

Alongside those, [`test/modelChecks.ts`](test/modelChecks.ts) compiles every model
the app offers and asserts **how many kernels it compiles to**. That is a fusion
guard: numbl's lowering emits one statement per *operator* and its inline pass
folds them back into per-line expression trees, and if that stops happening the
results stay correct while every operator becomes its own dispatch. It is
invisible in the numbers, so it is asserted directly. (It has already caught one
regression.)

[`test/transformChecks.ts`](test/transformChecks.ts) is the one remaining
implementation-vs-implementation check inside the suite, comparing the WGSL
transforms against shtns-webgpu's f64 CPU twin. Comparing them against *upstream*
SHTNS is a separate, opt-in step, because it needs a native toolchain — see
[Against upstream SHTNS](#against-upstream-shtns).

All three modules run in **both** environments, so the two GPU stacks get the same
guarantees:

- `npm run test:node` — under Dawn on the desktop, via `vite-node`. Needs a GPU;
  pass `--skip-without-gpu` to let a machine without one say so and move on
  (which is what CI does, since the browser suite covers the same modules).
- `npm run test:gpu` — builds and drives headless Chrome, on SwiftShader in CI.
  Also runs the soak.

Other commands:

- `npm run bench -- --help` — the desktop benchmark (see
  [Desktop vs browser](#desktop-vs-browser)).
- `npm run bench:sht -- --help` — the transforms alone, with no solver around
  them, for comparing against upstream SHTNS.
- `npx vite-node scripts/diagnose-sht.ts` — when the transform tests fail on a GPU,
  say *which* stage is wrong. It reads the intermediate `fm` back out and scores
  the Legendre and Fourier stages of each direction separately against the f64
  reference, then breaks the error down by order `m` and by latitude.
- `npx vite-node scripts/diagnose-leg.ts [--m 0]` — the follow-up to that: read the
  Legendre recurrence out of the production shader term by term, by synthesizing a
  spectrum that is 1 at a single coefficient, and compare each `ỹ_l^m` with the f64
  reference. The first term that disagrees names the culprit.
- `npx vite-node scripts/longrun-node.ts [lmax]` — run to t = 100 and confirm the
  pattern saturates into O(1)-contrast spots rather than decaying or diverging.
- `node scripts/soak.mjs [steps] [lmax]` — drive the demo for many steps,
  sampling JS heap and catching crashes.
- `node scripts/screenshot.mjs out.png [light|dark] [minSteps]` — screenshot the
  demo after a number of steps.
- `node scripts/check-live.mjs [url]` — smoke-check a deployed URL in a real
  browser: load, press Run, confirm the solver advances.
- `node scripts/compare-env.mjs` — run one identical spec on the desktop and in a
  browser and compare the final state (see
  [Is it really the same computation?](#is-it-really-the-same-computation)).
- `node scripts/compare-perf.mjs` — measure the same solver work in both and split
  the difference (see
  [Why the browser is slower](#why-the-browser-is-slower-and-how-to-find-out-by-how-much)).
- `node scripts/compare-native.mjs` — run one spec through the WGSL transforms and
  through upstream SHTNS, and line the numbers up (see
  [Against upstream SHTNS](#against-upstream-shtns)). Needs
  [`bench/shtns/`](bench/shtns/) built first.
- `test.html?soak=<steps>&lmax=<n>` — solver-only soak with no rendering.

### A note on canvas resizing

Early long runs killed the browser after ~700–800 steps. The cause was the
colorbar's min/max labels changing width as their digit count changed, which
reflowed the panel, fired the `ResizeObserver`, and called
`renderer.setSize()` — reallocating the WebGL drawing buffer. Assigning
`canvas.width` also blanks the canvas even when the value is unchanged, so the
same bug caused visible flicker. Fixed by giving the colorbar column a fixed
width and making `SphereScene.resize()` return early on no-op resizes.

### A note on the Legendre recurrence on Blackwell

The first run on an Nvidia GPU — an RTX PRO 6000, driver 590.48, reached through
Dawn's Vulkan backend — failed 11 of the tests. `synth` was off by 5.5e+3 while
`analys` was accurate to 7.3e-7, and the solver produced NaN within 40 steps.

The two diagnostic scripts above were written for it and localized it in two
steps: `leg_synth` was the only wrong shader, and within it the recurrence was
right at `l = m` and `l = m+1` and then returned *exactly zero* at `l = m+2`, at
every latitude. That is not a precision failure. It is

```wgsl
let c0 = ab[base + (l + 2u - m)];
y0 = c0.x * ct * y1 + c0.y * y0;      // c0 reads as (0, 0) on the first iteration
```

with the `ab` read two lines later working fine. The buffer was not at fault:
`leg_analys` reads the same array correctly on the same device, and `m = 62, 63`
— the only orders whose loop breaks before that line — were the only correct
ones. Nothing about that WGSL is invalid, so it was a miscompiled load.

Fixed by giving the advance the shape `leg_analys` already used, which that
driver compiles correctly: both coefficients fetched unconditionally, and the new
`y0` carried in a temporary rather than assigned and then read back by the `y1`
update. Two shaders doing the same recurrence should have agreed on form anyway.

Worth knowing for what it says about the transforms in general: nothing had
exercised them on Nvidia hardware before, and the existing test caught it
immediately — it just could not say where. That is what the diagnostics are for.

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

The `scripts/*.ts` entry points that touch the compiler (the benchmark, the node
tests, the long run) go through `vite-node`, so they resolve imports exactly as the
browser build does — the `numbl-src` alias and the `?raw` model imports included.
Plain `node` cannot: numbl's sources import each other as `./foo.js` while the
files are `.ts`, which needs a bundler's resolution. Scripts that do not touch the
compiler (`soak.mjs`, `screenshot.mjs`, `check-live.mjs`, `test-gpu.mjs`) are plain
`.mjs` and run under `node` directly.

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are vendored).
