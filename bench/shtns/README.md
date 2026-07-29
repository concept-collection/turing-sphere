# The WGSL transforms against upstream SHTNS

[`src/sht/`](../../src/sht/) is a WGSL translation of
[SHTNS](https://nschaeff.bitbucket.io/shtns/), vendored from
[shtns-webgpu](https://github.com/concept-collection/shtns-webgpu). Its tests
check it against its own f64 CPU twin, which says it is self-consistent — not
how it compares with the library it is modeled on.

This directory answers that. It builds upstream SHTNS and runs the same spec
through it, so the numbers line up against `npm run bench`:

| | transforms | precision | where |
|---|---|---|---|
| `npm run bench:sht` | WGSL, via Dawn | fp32 | this repo |
| `./shtbench_gpu` | SHTNS' own CUDA kernels | fp32 | upstream |
| `./shtbench` | SHTNS on the CPU | fp64 | upstream |

`shtbench_gpu` is the like-for-like comparison: same GPU, same precision, same
grid, same spectral conventions — the only difference is who computes the
transform. `shtbench` is fp64 because SHTNS' single precision exists only on the
GPU, so it serves as the accuracy reference (how far has fp32 drifted?) and as
the "what does a well-optimized CPU do" reference.

## Build

```
./bootstrap.sh          # clone SHTNS at a pinned commit, configure, build
make                    # shtbench, plus shtbench_gpu if SHTNS got CUDA support
```

`bootstrap.sh` adds `--enable-cuda` when `nvcc` is on `PATH` and passes
`--enable-openmp` always; `--no-cuda` and `--cuda=ampere` override it. It writes
`shtns.mk` with the library name and the link flags `configure` decided on, which
the `Makefile` includes — so a host that needed MKL or a different FFTW still
links without editing anything here. Both the checkout and `shtns.mk` are
gitignored; `make distclean` throws them away.

Needs: a C++ compiler, FFTW3 headers (`libfftw3-dev`), and for the GPU half a
CUDA toolkit. SHTNS' `configure` wants `CUDA_PATH` set; `bootstrap.sh` derives it
from `nvcc`'s location if it is not.

## Run

```
./shtbench_gpu --mode transform --lmax 63 --steps 2000
./shtbench_gpu --mode solver    --lmax 63 --steps 2000 --preset schnak-spots
./shtbench --help
```

Both binaries take the same options as `npm run bench` — `--preset`, `--lmax`,
`--steps`, `--warmup`, `--seed`, `--batch`, any model parameter by name — plus
`--mode`, `--layout`, `--polar-eps`, `--json`, `--digest`, `--dump-state`.

Two things are measured, and they answer different questions:

- **`--mode transform`** is one spectral → grid → spectral round trip and nothing
  else. This is the library-against-library number. Profiling of the reference
  implementation puts the transforms at ~96% of the solver's compute, so this is
  what decides how fast the solver can be.
- **`--mode solver`** is one IMEX Euler timestep of `models/<key>.m`:
  2·species transforms, the reaction on the grid, the spectral update. This is
  the number `npm run bench` and the app's `solver` line report.

Both report throughput (a batch launched together, waited for once — what
`--batch` controls, matching `npm run bench -- --batch`) and a per-step
distribution from one synchronization per step.

## Compare

```
node scripts/compare-native.mjs                  # transforms, lmax 63
node scripts/compare-native.mjs --mode solver
node scripts/compare-native.mjs --check          # and diff the final state
```

from the repo root (or `npm run bench:native --`). It runs every implementation
present on the machine, back to back in one invocation so a second process
competing for the GPU affects both sides rather than one, and prints them in one
table. Missing implementations are reported and skipped, so this is still useful
on a machine with no CUDA.

`--check` adds a short second pass that diffs the final spectral state across
implementations. That is what makes the timing mean anything: two numbers are
only comparable if they are the cost of the same computation.

## What is and is not the same on the two sides

The comparison is exact where it can be:

- **Spectral layout and normalization are identical.** SHTNS' default
  `sht_orthonormal` with the Condon–Shortley phase, `mres = 1`, coefficients
  grouped by `m` — which is what [`src/sht/layout.ts`](../../src/sht/layout.ts)
  implements, down to `LM(l,m)` agreeing index for index. So a spectral state can
  be diffed element by element with no reindexing.
- **The grid is identical.** `shtb_grid_for_lmax` in [`spec.h`](spec.h) is
  `gridForLmax` from `src/sht/layout.ts`, including rounding `nphi` up to a power
  of two — which SHTNS does not need, but the grids have to match. Both sides
  assert they got the grid they asked for.
- **The seed is identical.** `shtb_seeded_noise` and `shtb_seeded_spectrum` are
  transcriptions of `src/mgpu/noise.ts`. The transform check deliberately uses a
  spectrum drawn with integer arithmetic only, so it is bit-identical on both
  sides and a difference in the result is a difference in the transforms.

And explicit where it cannot be:

- **The solver step is transcribed, not shared.** The app compiles
  `models/<key>.m` through numbl; C cannot, so `shtb_react` and `shtb_imex` in
  [`spec.h`](spec.h) restate the same arithmetic, one line per line of MATLAB
  (including writing `u.^3` as `u*u*u`, which is what the WGSL backend emits for
  it). This is the repo's one second implementation of the loop, and it exists
  only to be compared against — `compare-native.mjs --check` is what keeps it
  honest. It agrees with the real `.m` path to ~1e-6 over 20 steps, for every
  model.
- **The presets are duplicated.** `spec.h` copies the tables from
  `src/mgpu/registry.ts`. This is the one thing that could silently drift, so
  `compare-native.mjs` compares both sides' resolved grid and parameters and
  refuses to compare two runs that disagree.
- **SHTNS runs the Legendre recurrence in fp64 even in fp32 mode**, for
  `lmax <= 128` on a GPU with usable fp64 (`SHT_L_RESCALE_FLY_FLOAT` in its
  `sht_private.h`). WebGPU has no fp64 at all, so ours cannot. Set
  `SHTNS_GPU_REC_PREC=1` to force SHTNS' recurrence into fp32 for the closer
  comparison; the run prints which it used.
- **Different FFTs.** SHTNS uses cuFFT or VkFFT on the GPU and FFTW on the CPU;
  ours is a WGSL FFT (or a DFT when the device's workgroup limits do not fit one
  — the run says which). These are different algorithms with different cost and
  different rounding.
- **Polar optimization is off by default here** (`--polar-eps 0`), because the
  WGSL transforms do not have it. SHTNS' own default is `1e-10` and is worth a
  few percent; `--polar-eps 1e-10` turns it on.
- **The spatial layout defaults to theta-contiguous**, SHTNS' native and fastest.
  `--layout phi` is what the WGSL side uses. The reaction is pointwise and the
  spectral layout is unaffected, so a state comparison is valid either way — this
  only moves the cost. On the GPU, `--layout phi` needs SHTNS' VkFFT backend,
  which it uses whenever `vkfft/vkFFT.h` is in its tree (it normally is;
  `bootstrap.sh` says which Fourier stage you got). Its cuFFT fallback handles
  only theta-contiguous. SHTNS' own accuracy check runs at `shtns_set_grid` time
  and aborts on a mismatch, so this fails loudly rather than quietly — but run
  `compare-native.mjs --check` after changing the layout anyway.

## Reading the result

Small grids flatter the CPU: at `lmax 31` there are 528 coefficients, and a GPU
spends most of a transform on launch latency rather than arithmetic. The gap
closes with `lmax`, so run a sweep before concluding anything:

```
for l in 31 63 127 255; do node scripts/compare-native.mjs --lmax $l --steps 500; done
```

If the WGSL side lands on a software adapter, `compare-native.mjs` says so and
stops you — the ratio then compares a CPU emulation against a real GPU and means
nothing.

## Editing `shtbench_gpu.cu` without a GPU

`nvcc` is needed to build it, but not to typecheck it. Rewriting the launch
syntax as calls, against a handful of stub declarations, gets g++ to check
everything else:

```
sed -e 's/<<</\/*/g; s/>>>/*\//g' shtbench_gpu.cu > /tmp/check.cpp
g++ -fsyntax-only -std=c++14 -I. -Ishtns -I/path/to/cuda-stubs /tmp/check.cpp
```

where the stub directory holds a `cuda_runtime.h` defining `__global__`,
`dim3`, `blockIdx`/`threadIdx`/`blockDim`, `cudaStream_t`, and the dozen
`cuda*` functions used here.

## License

CECILL-2.1, as the rest of this repo — the same license SHTNS itself is under.
