/**
 * WGSL Fourier-stage kernels (the role cuFFT/VkFFT plays in SHTNS).
 *
 * Real fields, band-limited to |m| <= mmax < nphi/2:
 *  - synthesis: assemble a Hermitian spectrum from F_m (m >= 0) and do an
 *    inverse complex FFT along phi; take the real part.
 *  - analysis: forward complex FFT of the (real) row; keep m = 0..mmax.
 *
 * Two implementations, selected at plan creation:
 *  - 'fft': radix-2 Stockham in workgroup memory, one workgroup per
 *    latitude row.  Requires nphi a power of two and
 *    2 * 8 * nphi bytes <= maxComputeWorkgroupStorageSize.
 *  - 'dft': direct band-limited trigonometric summation, O(nphi * mmax)
 *    per row.  Works for any nphi; also useful as a cross-check.
 *
 * All trigonometric factors come from a host-precomputed (f64 -> f32)
 * table trig[k] = (cos, sin)(2*pi*k/nphi): device sin/cos is only
 * guaranteed to ~2^-11 absolute error under Vulkan, which would dominate
 * the fp32 transform error.
 */

export interface FourierParams {
  mmax: number;
  nlat: number;
  nphi: number;
}

const TRIG_BINDING = /* wgsl */ `
@group(0) @binding(2) var<storage, read> trig: array<vec2f>;  // (cos,sin)(2*pi*k/NPHI), k < NPHI
`;

function stockham(nphi: number, threads: number, sign: number): string {
  const log2n = Math.log2(nphi);
  if (!Number.isInteger(log2n)) throw new Error('fft requires power-of-two nphi');
  // twiddle for pass with half-block ns: w = e^{sign*i*pi*j/ns} = T[j * (N/(2*ns))]^sign
  return /* wgsl */ `
var<workgroup> bufA: array<vec2f, ${nphi}>;
var<workgroup> bufB: array<vec2f, ${nphi}>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn ld(sel: u32, i: u32) -> vec2f {
  if (sel == 0u) { return bufA[i]; }
  return bufB[i];
}
fn st_(sel: u32, i: u32, v: vec2f) {
  if (sel == 0u) { bufA[i] = v; } else { bufB[i] = v; }
}

// radix-2 Stockham, natural order in and out; data starts in bufA (sel 0)
// and ends in sel = LOG2N % 2.  Unnormalized: X_k = sum_j x_j e^{s*2*pi*i*jk/N}.
fn fft_inplace(lid: u32) {
  for (var p = 0u; p < ${log2n}u; p++) {
    workgroupBarrier();
    let ns = 1u << p;
    let sel = p & 1u;
    let stride = ${nphi / 2}u >> p;   // N/(2*ns)
    for (var t = lid; t < ${nphi / 2}u; t += ${threads}u) {
      let j = t & (ns - 1u);
      let tw = trig[j * stride];
      let w = vec2f(tw.x, ${sign > 0 ? '' : '-'}tw.y);
      let u = ld(sel, t);
      let v = cmul(ld(sel, t + ${nphi / 2}u), w);
      let idst = 2u * (t - j) + j;
      st_(1u - sel, idst, u + v);
      st_(1u - sel, idst + ns, u - v);
    }
  }
  workgroupBarrier();
}
const FFT_OUT_SEL: u32 = ${log2n % 2}u;
`;
}

/** Choose FFT workgroup size: enough threads for the butterflies, capped at 256. */
export function fftThreads(nphi: number): number {
  return Math.max(32, Math.min(256, nphi / 2));
}

export function fftSynthWGSL(p: FourierParams): string {
  const T = fftThreads(p.nphi);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> fm: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;
${TRIG_BINDING}
${stockham(p.nphi, T, +1)}

@compute @workgroup_size(${T})
fn fft_synth(@builtin(local_invocation_id) lid3: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  // assemble Hermitian spectrum: X[0] = Re F_0, X[m] = F_m, X[N-m] = conj(F_m)
  for (var k = lid; k < NPHI; k += ${T}u) {
    var v = vec2f(0.0);
    if (k == 0u) {
      v = vec2f(fm[ilat].x, 0.0);
    } else if (k <= MMAX) {
      v = fm[k * NLAT + ilat];
    } else if (k >= NPHI - MMAX) {
      let c = fm[(NPHI - k) * NLAT + ilat];
      v = vec2f(c.x, -c.y);
    }
    bufA[k] = v;
  }
  fft_inplace(lid);
  for (var k = lid; k < NPHI; k += ${T}u) {
    spat[ilat * NPHI + k] = ld(FFT_OUT_SEL, k).x;
  }
}
`;
}

export function fftAnalysWGSL(p: FourierParams): string {
  const T = fftThreads(p.nphi);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> spat: array<f32>;
@group(0) @binding(1) var<storage, read_write> fm: array<vec2f>;
${TRIG_BINDING}
${stockham(p.nphi, T, -1)}

@compute @workgroup_size(${T})
fn fft_analys(@builtin(local_invocation_id) lid3: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  for (var k = lid; k < NPHI; k += ${T}u) {
    bufA[k] = vec2f(spat[ilat * NPHI + k], 0.0);
  }
  fft_inplace(lid);
  for (var m = lid; m <= MMAX; m += ${T}u) {
    fm[m * NLAT + ilat] = ld(FFT_OUT_SEL, m);
  }
}
`;
}

export function dftSynthWGSL(p: FourierParams): string {
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> fm: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;
${TRIG_BINDING}

@compute @workgroup_size(64)
fn dft_synth(@builtin(global_invocation_id) gid: vec3u) {
  let iphi = gid.x;
  let ilat = gid.y;
  if (iphi >= NPHI) { return; }
  var v: f32 = fm[ilat].x;   // m = 0: real part
  for (var m = 1u; m <= MMAX; m++) {
    let w = trig[(m * iphi) % NPHI];     // e^{+i m phi}
    let c = fm[m * NLAT + ilat];
    v += 2.0 * (c.x * w.x - c.y * w.y);
  }
  spat[ilat * NPHI + iphi] = v;
}
`;
}

export function dftAnalysWGSL(p: FourierParams): string {
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> spat: array<f32>;
@group(0) @binding(1) var<storage, read_write> fm: array<vec2f>;
${TRIG_BINDING}

@compute @workgroup_size(64)
fn dft_analys(@builtin(global_invocation_id) gid: vec3u) {
  let m = gid.x;
  let ilat = gid.y;
  if (m > MMAX) { return; }
  var acc = vec2f(0.0);
  for (var j = 0u; j < NPHI; j++) {
    let w = trig[(m * j) % NPHI];        // conj => e^{-i m phi}
    let f = spat[ilat * NPHI + j];
    acc += f * vec2f(w.x, -w.y);
  }
  fm[m * NLAT + ilat] = acc;
}
`;
}
