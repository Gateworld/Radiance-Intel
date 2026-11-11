/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * Ported from libr4d/r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip at 9eaf10d: first pass by libxe/tools/hipport.py, then edited by hand.
 * Compiled as SYCL behind libxe/xe_lanes.h (sub-group 32); the original header follows. */
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip -- the gated-delta-net recurrent state
// update, run by decode (prefill runs the chunked scan). One kernel for the whole step:
//
//     g = -exp(A_log) . softplus(a + dt_bias)      beta = sigmoid(b)      q, k l2-normalised
//     S  *= e^g
//     u   = beta . (v - S k)                       S += u k^T              o = S q
//
// per token, against a paged state cache: the initial state is read from the slot the last
// accepted token wrote, and every candidate token writes its own slot, since which survives
// verification is unknown until after this layer. Replaces FLA's
// fused_sigmoid_gating_delta_rule_update; the non-speculative case is width 1.
//
// State traffic is the whole cost: one fp32 [V,K] state is 64 KB, written per candidate token per
// head (113 MB per layer per rank at width 9, at the memory bandwidth limit). So the design keeps
// the state in registers: one thread owns one v row and half its k range, so
//
//   * S k and S q are in-lane dot products closed by a single lane<->lane^1 exchange, not a
//     tree reduction;
//   * the outer-product update is 64 independent FMAs with no cross-lane traffic at all;
//   * a thread's 64 state elements are contiguous in k, so the load and every store is four
//     16-byte accesses over one 256-byte run.
//
// The state cache is fp32 by default (the model's HF config sets mamba_ssm_dtype=float32, which
// vLLM honours); the conv state is bf16. An fp16 state cache is also supported (SDT).
//
// Slot and head strides are arguments, not H*V*K: vLLM's hybrid cache pads a mamba page to the
// attention page size (e.g. 421888 elements per slot vs 393216 from the shape), so a shape-derived
// stride would read every slot but the first at the wrong offset there.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"
#include "r4d_gdn_conv_tap.h"
#include "r4d_q8.h"         // the 8-bit codes the folded quantiser writes: E4M3 or int8
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

#define RU_K   128              // head_k
#define RU_V   128              // head_v
#define RU_HK   64              // k elements per thread (RU_K / 2)
#define RU_THR (RU_V * 2)       // two threads per v row

// The state written per step (~12 MB a layer, ~660 MB a step) is read again only once, a step
// later, and cannot stay in the 64 MB last-level cache; its write-back slows the next GEMM (a
// (5120, 3072) GEMM goes from 14.9 us to 20.4 us behind a 12 MB write flood, 18.4 us when the
// flood is non-temporal). A non-temporal hint on the state load and store changes no result, and
// whether it pays depends on the flood size, so it is chosen per launch (RU_NT_MB below).
typedef float ru_f4 __attribute__((ext_vector_type(4)));
typedef _Float16 ru_h8 __attribute__((ext_vector_type(8)));

// SDT is the state cache's element type (0 = fp32, 1 = fp16); all arithmetic is fp32. At eight
// sequences x eight candidates a call is ~68 us, of which ~45 us is 24 MB of state traffic at 88%
// of achievable bandwidth; halving the state width saves about a third.
//
// Cost model (eight sequences, 16k context, anchor state): ~3.2 us per token walked plus ~57 us
// fixed per layer. State traffic largely overlaps the token loop (fp16 state saves only about a
// fifth of what its byte count predicts), and the loop is on the critical path.
//
// Shipped instantiation: 207 VGPRs, 7 waves/SIMD, no spill, 384 B LDS. At seven waves the serial
// `dot +=` and `on +=` chains are hidden; splitting them would only change the summation order.
//
// fp16 rather than bf16: state elements are O(0.1) (sums of beta (v - Sk) k^T with l2-normalised
// k), so range is no issue and fp16's mantissa gives ~5e-4 relative round-trip error. vLLM's
// --mamba-ssm-cache-dtype already accepts float16.
template <int SDT> struct ru_state { using T = float; };
template <> struct ru_state<1>     { using T = _Float16; };

// Eight elements per call: two 16-byte accesses in fp32, one in fp16.
template <int NT, int SDT>
__device__ __forceinline__ void ru_load8(float* d, const typename ru_state<SDT>::T* p) {
  if constexpr (SDT) {
    const ru_h8 t = NT ? __builtin_nontemporal_load((const ru_h8*)p) : *(const ru_h8*)p;
#pragma unroll
    for (int e = 0; e < 8; ++e) d[e] = (float)t[e];
  } else {
    const ru_f4* q = (const ru_f4*)p;
    *(ru_f4*)d       = NT ? __builtin_nontemporal_load(q)     : *q;
    *(ru_f4*)(d + 4) = NT ? __builtin_nontemporal_load(q + 1) : *(q + 1);
  }
}

template <int NT, int SDT>
__device__ __forceinline__ void ru_store8(typename ru_state<SDT>::T* p, const float* d) {
  if constexpr (SDT) {
    ru_h8 t;
#pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = (_Float16)d[e];
    if (NT) __builtin_nontemporal_store(t, (ru_h8*)p);
    else    *(ru_h8*)p = t;
  } else {
    ru_f4* q = (ru_f4*)p;
    if (NT) {
      __builtin_nontemporal_store(*(const ru_f4*)d,       q);
      __builtin_nontemporal_store(*(const ru_f4*)(d + 4), q + 1);
    } else {
      *q       = *(const ru_f4*)d;
      *(q + 1) = *(const ru_f4*)(d + 4);
    }
  }
}
// VS: workgroups sharing a head's 128 v rows. At one sequence the (sequences, heads) grid is 24
// workgroups on 64 CUs, and the working set fits the last-level cache. v rows are independent
// (dot products contract over k; the update is per (v, k)); only the fused RMS norm spans the
// row, so VS > 1 is compiled without it.
#define WGTARGET 64             // workgroups the v-row split aims for

__device__ __forceinline__ float ru_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// a and b are a chunk() of the ba projection (see r4d_gdn_conv_w4_h128_bf16.hip).
template <int ABF16>
__device__ __forceinline__ float ru_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// The layer's gated RMS norm, folded into the epilogue: with NORM a workgroup owns one (sequence,
// head) and so the whole 128-channel output row, and the norm is one workgroup reduction per
// token. Prefill owns half a row and uses r4d_gdn_gated_rmsnorm_h128_bf16 instead.
__device__ __forceinline__ float ru_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;
}

// One token's replay factors (anchor state), inside the head's block of a factor slot:
//
//     [0    .. 4*K) normalised k, FOUR copies -- one per fixed 32-row group
//     [4*K  .. 4*K+V) the v row, one float per v row
//     [4*K+V .. +8)   decay and beta, one pair per row group
//
// Four k copies avoid a cross-workgroup barrier: a head's v rows may be split over up to four
// workgroups, and fixed 32-row groups make every byte belong to exactly one workgroup at any split.
// Read-before-write within a workgroup is ordered by a barrier.
#define RU_FSTR (4 * RU_K + RU_V + 8)

// RU_TMAX bounds the candidate block STG can stage (LDS for four partials and two scales per
// token). The launcher checks the caller's width against it.
#define RU_TMAX 16

// RU_ANC_DUP=0 drops the two neighbouring-page copies of the replay factors (and, without the
// marker, of the anchor). Incorrect without RU_ANC_MARK (a moved page index reads an unwritten
// slot). With the marker it writes the factor block once; `-DRU_ANC_MARK=1 -DRU_ANC_DUP=0` is the
// leanest build, since the factor block (RU_TMAX * RU_FSTR) outweighs the state (RU_V * RU_K).
#ifndef RU_ANC_DUP
#define RU_ANC_DUP 1
#endif

// Shift marker: instead of writing the anchor three times (one per possible page shift), each of
// the three factor-block copies records the column it was written to. The reader finds `m` at its
// fixed factor column, so the page index moved by `m - 1`, and reads the anchor at the matching
// column; the anchor is written once (6.3 MB a layer instead of 18.9 at eight sequences, roughly a
// fifth of the layer's time).
//
// The marker lies past all factor data (RU_TMAX tokens of RU_FSTR) within one head's slice, so no
// workgroup reads a marker another is writing. Stored values are 1, 2, 3 (one-based; see the read),
// exact in fp16 and fp32.
//
// Invariant: the anchor is written at column 2 and read at 3 - m, never column 0. vLLM's
// precopy_mamba_align_fused_kernel copies state[bt[src_col + num_accepted - 1]] into this window's
// column 0 before the forward on a page crossing, so a read of column 0 (anchor at 1, read at
// 2 - m, shift +1) would see migrated data and degenerate intermittently.
#ifndef RU_ANC_MARK
#define RU_ANC_MARK 1
#endif
#define RU_MARK_OFF (RU_TMAX * RU_FSTR)

// RU_ANC_FRD=1 (diagnostic): use the displaced factor read (column 5 - m) while still writing all
// three factor copies, isolating the marker's read-column logic from the write side.
#ifndef RU_ANC_FRD
#define RU_ANC_FRD 0
#endif

// Decode convolution folded in (CV). This kernel is gdn_conv_update's only decode reader, so the
// fold saves a launch and a dependent-dispatch gap. Each workgroup computes the convolution its
// token loop reads (its key head's 128 q and 128 k channels and its value head's 128 v channels)
// from the projection rows into LDS; q, k and v never reach memory. The arithmetic is
// gdn_conv_update's (same taps, order, cp_silu and bf16 rounding), so results are byte-identical.
//
// The conv state is rewritten in place, and a key head's channels are read by all H / Hg value-head
// workgroups of its group, so none may rewrite them until all have read the history. Each
// workgroup arrives on a counter after reading; the last arriver rewrites the key head's q and k
// channels and resets the counter (no waiting). A value head's v channels have one reader and are
// rewritten by it. Two launches sharing a counter block must not overlap; blocks are keyed on
// (stream, output), as split_ctr in r4d_gemm_fp8a8.hip.
struct RuConv {
  const unsigned short* x; long xpitch;   // the projection's [q | k | v] columns, row pitch
  const unsigned short* w;                // [conv_dim, R4D_GDN_CONV_W]
  const unsigned short* b;                // [conv_dim] or null
  unsigned short* cs;                     // the conv state cache
  long cs_seq, cs_dim, cs_tok;
  const int* cidx; long ci_stride;        // conv state slot per sequence; negative is padding
  unsigned int* ctr;                      // one arrival counter per (sequence, key head)
  /* Intel port, run 112 (owner note 138, RADIANCE_XE_GDN_SPLIT): the split fold (VS > 1, no fused norm) writes
   * each row's f32 output here, [token][H][RU_V], for ru_split_norm_kernel; null otherwise */
  float* onf;
  /* Run 284 (briefs/2026-11-11-gdn-split-latency.md N1, RADIANCE_XE_GDN_LAT=7): one arrival counter per (sequence,
   * value head) for the split fold's own norm by the head's last arriving slice; null otherwise */
  unsigned int* nctr = nullptr;
};
// Max tokens per sequence for the folded convolution (preloaded into registers, static indices).
#define RU_CVT 8

// A channel's conv state rewritten as gdn_conv_update does: shifted by one, the old history's
// tail then this step's tokens. The launcher requires the window split to be R4D_GDN_CONV_W - 2,
// so these are the same stores.
__device__ __forceinline__ void ru_conv_rewrite(unsigned short* sp, const unsigned short* hist,
                                                const unsigned short* xs, int T, long cs_tok) {
  constexpr int NH = R4D_GDN_CONV_W - 2;
#pragma unroll
  for (int i = 0; i < NH; ++i) sp[(size_t)i * cs_tok] = hist[i + 1];
#pragma unroll
  for (int t = 0; t < RU_CVT; ++t)
    if (t < T) sp[(size_t)(t + NH) * cs_tok] = xs[t];
}

/* The sum over a v row's KT lanes (consecutive lanes of one sub-group). */
template <int KT>
__device__ __forceinline__ float ru_row_sum(float x) {
  if constexpr (KT >= 2) x += r4d_xor_lane<1>(x);
  if constexpr (KT >= 4) x += r4d_xor_lane<2>(x);
  if constexpr (KT >= 8) x += r4d_xor_lane<4>(x);
  return x;
}

/* Intel port, run 112 (owner note 138): the split fold's norm -- one (token, value head), a thread a row (128 rows), as
 * the fused path's arithmetic in the fused path's order at KT 8: on^2 a row; a 32-lane wave there holds four rows
 * (lanes 0 / 8 / 16 / 24) and its xor tree gives (r0^2 + r2^2) + (r1^2 + r3^2) (the other lanes add zeros); the 32
 * wave sums are added in order from 0; then the gate, the bf16 store and the quantiser as there. The same bits.
 * Run 284 (briefs/2026-11-11-gdn-split-latency.md N1): the body is a device function so the split kernel's last
 * arriving slice can run it itself (the same code, the same bits); s2 / ws / red are this (token, head)'s own LDS,
 * `active` false for a lane whose token is past T (it keeps the barriers, stores nothing). The barriers are the
 * workgroup's: every lane of the workgroup calls it together. */
constexpr int kSplitNormRowsAWave = 4;   /* RU_V rows over 1,024 threads at KT 8, 32 lanes a wave */
template <int NQ>
__device__ __forceinline__ void ru_split_norm_body(bool active, int tok, int hv, int row,
                                                   const float* __restrict__ onf, unsigned short* __restrict__ o,
                                                   unsigned char* __restrict__ oq, float* __restrict__ os,
                                                   long oq_pitch, long os_ld, const unsigned short* __restrict__ zg,
                                                   long zg_pitch, const float* __restrict__ nw, float neps, int nact,
                                                   int H, float* s2, float* ws, float* red) {
  float on = active ? onf[((size_t)tok * H + hv) * RU_V + row] : 0.0f;
  s2[row] = on * on;
  __syncthreads();
  if (row < RU_V / kSplitNormRowsAWave) {
    const int r0 = row * kSplitNormRowsAWave;
    ws[row] = (s2[r0] + s2[r0 + 2]) + (s2[r0 + 1] + s2[r0 + 3]);
  }
  __syncthreads();
  if (row == 0) {
    float tot = 0.0f;
    for (int i = 0; i < RU_V / kSplitNormRowsAWave; ++i) tot += ws[i];
    *red = tot;
  }
  __syncthreads();
  const float sc = __frsqrt_rn(*red * (1.0f / RU_V) + neps);
  const unsigned short zr = active ? zg[(size_t)tok * zg_pitch + (size_t)hv * RU_V + row] : (unsigned short)0;
  on *= sc * nw[row] * ru_act(bf2f(zr), nact);
  const unsigned short ob = f2bf(on);
  if (active) o[((size_t)tok * H + hv) * RU_V + row] = ob;
  if constexpr (NQ != 0) {
    const float ov = bf2f(ob);
    __syncthreads();                         /* red is read above by every thread */
    s2[row] = fabsf(ov);
    __syncthreads();
    if (row == 0) {
      float amax = 0.0f;
      for (int i = 0; i < RU_V; ++i) amax = fmaxf(amax, s2[i]);
      *red = amax;
    }
    __syncthreads();
    const float amax = *red;
    const float qs = amax > 0.0f ? amax * (1.0f / q8_top<NQ == 2>()) : 1.0f;
    if (active && row == 0) os[(size_t)tok * os_ld + hv] = qs;
    unsigned char code;
    if constexpr (NQ == 2) {
      code = (unsigned char)(int)rintf(fminf(fmaxf(ov * (1.0f / qs), -127.0f), 127.0f));
    } else {
      const float cvq = fminf(fmaxf(ov * (1.0f / qs), -448.0f), 448.0f);
      code = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(cvq, cvq, 0, false) & 0xff);
    }
    if (active) oq[(size_t)tok * oq_pitch + (size_t)hv * RU_V + row] = code;
  }
}

/* Run 283 (briefs/2026-11-11-gdn-split-latency.md; KG17): of the brief's three EXACT load-chain changes, L1 (the anchor
 * slot selected from the sidx registers instead of a dependent reload, -1.3 us a launch at T 4) is FOLDED into every
 * form above; L2 (the replay rows preloaded) was slower and is dropped; N1 -- the split fold's norm, gate, store and
 * quantiser (ru_split_norm_body<2>, the int8 codes) run by the head's last arriving slice over the head's T rows, the
 * norm launch skipped -- was exact over 3,000 stressed calls but slower (+3.9 us at T 4) and stays an opt-in RECORD:
 * LAT = 4 (RADIANCE_XE_GDN_LAT=7 on the VS 2 split at KT 8, the coordinator's switch value). LAT 0 is the kernel. */
template <int ABF16, int NORM, int VS, int NT, int ANC, int STG = 0, int SDT = 0,
          int CV = 0, int KT = 2, int LAT = 0>
__global__ __launch_bounds__(RU_V / VS * KT)
void r4d_gdn_recurrent_update_kernel(const unsigned short* __restrict__ q,
                                     const unsigned short* __restrict__ k,
                                     const unsigned short* __restrict__ v,
                                     const void* __restrict__ av,
                                     const void* __restrict__ bv, long ab_stride,
                                     const float* __restrict__ A_log,
                                     const float* __restrict__ dt_bias,
                                     typename ru_state<SDT>::T* __restrict__ state,
                                     long st_slot, long st_head,
                                     unsigned short* __restrict__ o,
                                     unsigned char* __restrict__ oq,
                                     float* __restrict__ os, long oq_pitch,
                                     long os_ld,
                                     const int* __restrict__ cu,
                                     const int* __restrict__ sidx, long sidx_stride,
                                     const int* __restrict__ naccept,
                                     const unsigned short* __restrict__ zg,
                                     long zg_pitch,
                                     const float* __restrict__ nw, float neps, int nact,
                                     int H, int Hg, float scale, float sp_thr,
                                     const RuConv cv)
{
  using ST = typename ru_state<SDT>::T;
  constexpr int NR = RU_V / VS;         // v rows this workgroup owns
  constexpr int THR = NR * KT;          // KT threads a v row (milestone 3: 4 for the decode shape)
  constexpr int HK = RU_K / KT;         // k elements a thread
  static_assert(KT == 2 || KT == 4 || KT == 8, "a v row's lanes must be a power of two in one sub-group");
  static_assert(!STG || KT == 2, "the staged path's thread layout is two halves");
  static_assert(!(NORM && VS > 1), "the fused row norm needs the whole 128-wide row");
  XE_SHARED(float, wsum, THR / 32);
  XE_SHARED(float, spart, STG ? RU_TMAX * 4 : 1);   // per (token, half): sum q^2, sum k^2
  XE_SHARED(float, sscl, STG ? RU_TMAX * 2 : 1);   // per token: the two closed scales
  const int n = blockIdx.x, hv = blockIdx.y;
  const int bos = cu[n], T = cu[n + 1] - bos;
  // With ANC, T == 0 is valid: it folds the previous step's accepted prefix into the anchor before
  // something else reads the state cache. Without ANC there is nothing to do.
  const bool empty = (T <= 0);
  if (empty && !ANC) return;

  const int tid = threadIdx.x;
  const int row = (tid / KT) + (VS > 1 ? blockIdx.z * NR : 0), half = tid & (KT - 1), k0 = half * HK;
  const int hk = hv / (H / Hg);
  /* Read early so their latency overlaps the prologue's loads. */
  const float a_log_h = A_log[hv], dtb = dt_bias[hv];

  // ---- CV: issue the convolution's inputs first so they are in flight during the anchor replay.
  // Thread tid owns one q|k channel of the key head (q below RU_K, k above) and, below RU_V, one v
  // channel of the value head.
  /* The decode shape: the fused norm at one workgroup a head, or (owner note 138) the split fold without it, VS 2
   * or 4 so a workgroup still has a thread for each of the key head's 256 q | k channels */
  /* Run 113 (owner note 138's rest): the anchor form splits too. A slice owns whole 32-row groups at VS <= 4, so
   * its replay reads and this step's factor writes (a group's k copy, decay / beta, its rows' v) touch only its own
   * part of the factor block; the shift marker, which every slice of a head reads, is stamped by the last arriver
   * on the key head's counter instead (see `stamp_group`). */
  static_assert(!CV || (STG == 0 && (NORM ? VS == 1 : (VS == 2 || VS == 4))),
                "the folded convolution is the decode shape");
  static_assert(!CV || RU_V / VS * KT >= 2 * RU_K, "a q | k channel a thread");
  constexpr int NVC = VS > 1 ? NR : RU_V;               // this workgroup's v channels (its rows)
  const int vrow0 = VS > 1 ? (int)blockIdx.z * NR : 0;
  static_assert(RU_K == RU_V, "one LDS row pitch serves the q, k and v taps");
  constexpr int CW = R4D_GDN_CONV_W, CS = R4D_GDN_CONV_W - 1;
  XE_SHARED(unsigned short, cvq, CV ? RU_CVT * RU_K : 1); XE_SHARED(unsigned short, cvk, CV ? RU_CVT * RU_K : 1);
  XE_SHARED(unsigned short, cvv, CV ? RU_CVT * RU_V : 1);
  XE_SHARED_VAR(int, cvlast);
  /* N1 (LAT & 4): the folded norm's LDS, one 128-row group a token (THR / RU_V groups a pass), and the arrival flag */
  static_assert(!(LAT & 4) || (CV && !NORM && VS == 2 && KT == 8 && THR % RU_V == 0), "N1 is the VS 2 split at KT 8");
  XE_SHARED(float, ns2, (LAT & 4) ? THR : 1);
  XE_SHARED(float, nws, (LAT & 4) ? THR / kSplitNormRowsAWave : 1);
  XE_SHARED(float, nred, (LAT & 4) ? THR / RU_V : 1);
  XE_SHARED_VAR(int, nlast);
  /* The key head's history for the final rewrite (two newest history entries and this step's
   * inputs, one channel per thread), parked in LDS: in registers across the token loop they spill. */
  XE_SHARED(unsigned short, cvh, CV ? (CS - 1 + RU_CVT) * THR : 1);
  /* The token loop's per-token global reads (a, b and the output gate row) are also staged: at
   * decode there is too little occupancy to hide a global load per token. They are issued here,
   * parked in LDS at the convolution's barrier and read from there: `ca_b` is one token's a
   * (tid < T) or b (T <= tid < 2T), `czg` this thread's strided share of the gate rows. */
  constexpr int ZPT = (RU_CVT * RU_V + THR - 1) / THR;  // gate values a thread stages
  XE_SHARED(float, sab, CV ? 2 * RU_CVT : 1);
  XE_SHARED(unsigned short, szg, CV ? RU_CVT * RU_V : 1);
  float ca_b = 0.0f;
  unsigned short czg[ZPT];
  const int G = H / Hg;                                 // workgroups that read one key head
  float cwa[CW], cwb[CW], cba = 0.0f, cbb = 0.0f;
  unsigned short cha[CS], chb[CS], cxa[RU_CVT], cxb[RU_CVT];
  unsigned short* spa = nullptr;
  unsigned short* spb = nullptr;
  unsigned int arrived = 0;
  const bool cvt = tid < 2 * RU_K;                      // this thread stages one q|k channel
  if constexpr (CV) {
    if (!empty) {
      const int cslot = cv.cidx[(size_t)n * cv.ci_stride];
      // A negative slot is padding: every workgroup of the sequence leaves here, so none arrives
      // on the counter.
      if (cslot < 0) return;
      const int off = naccept ? naccept[n] - 1 : 0;
      const int ca = tid < RU_K ? hk * RU_K + tid : (Hg + hk) * RU_K + (tid - RU_K);
      const int cb = (2 * Hg + hv) * RU_V + vrow0 + tid;   // split: this workgroup's own v rows
      const unsigned short* xr = cv.x + (size_t)bos * cv.xpitch;
      if (cvt) {
#pragma unroll
        for (int t = 0; t < RU_CVT; ++t)
          if (t < T) cxa[t] = xr[(size_t)t * cv.xpitch + ca];
        {
          const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)ca * CW);
          cwa[0] = bf2f(w4.x); cwa[1] = bf2f(w4.y); cwa[2] = bf2f(w4.z); cwa[3] = bf2f(w4.w);
          if (cv.b) cba = bf2f(cv.b[ca]);
        }
        spa = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)ca * cv.cs_dim;
#pragma unroll
        for (int i = 0; i < CS; ++i) cha[i] = spa[(size_t)(off + i) * cv.cs_tok];
      }
      if (tid < NVC) {
#pragma unroll
        for (int t = 0; t < RU_CVT; ++t)
          if (t < T) cxb[t] = xr[(size_t)t * cv.xpitch + cb];
        const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)cb * CW);
        cwb[0] = bf2f(w4.x); cwb[1] = bf2f(w4.y); cwb[2] = bf2f(w4.z); cwb[3] = bf2f(w4.w);
        if (cv.b) cbb = bf2f(cv.b[cb]);
        spb = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)cb * cv.cs_dim;
#pragma unroll
        for (int i = 0; i < CS; ++i) chb[i] = spb[(size_t)(off + i) * cv.cs_tok];
      }
      if (tid < 2 * T)
        ca_b = tid < T ? ru_ab<ABF16>(av, (size_t)(bos + tid) * ab_stride + hv)
                       : ru_ab<ABF16>(bv, (size_t)(bos + tid - T) * ab_stride + hv);
      if constexpr (NORM) {   /* the gate rows are the fused norm's; the split fold's norm reads them itself */
#pragma unroll
      for (int i = 0; i < ZPT; ++i) {
        const int zi = tid + i * THR;
        if (zi < T * RU_V)
          czg[i] = zg[(size_t)(bos + zi / RU_V) * zg_pitch + (size_t)hv * RU_V + (zi % RU_V)];
      }
      }
    }
  }

  // Without ANC the initial state is the one the last accepted token wrote; with ANC there is one
  // state per sequence and the accepted prefix is replayed onto it below.
  const int nacc = naccept ? naccept[n] : 1;
  // Paged anchor. vLLM addresses mamba state by page, re-derived each step as
  // (seq_lens - 1) / block_size, so the column holding a sequence's state can move. seq_lens
  // includes this step's query (the draft depth, which varies), so
  //     start' - start = ((ctx + acc + q') - 1)/bs - ((ctx + q) - 1)/bs
  // is +1 when the committed context crosses a page, usually 0, and -1 when a step drafts fewer
  // tokens than the previous one.
  //
  // With the shift in {-1, 0, +1}, a fixed read column works only if all three pages it can land
  // on were written. So the anchor is written to columns 0, 1, 2 and read from column 1, and the
  // replay factors are written to columns 3, 4, 5 and read from column 4:
  //     shift -1 -> next column 1 is this column 0     (and next column 4 is this column 3)
  //     shift  0 -> next column 1 is this column 1     (and next column 4 is this column 4)
  //     shift +1 -> next column 1 is this column 2     (and next column 4 is this column 5)
  // The six columns are distinct writable slots: vLLM's per-candidate path writes to the same
  // indices, and indices_stride is num_spec + 1 = 8 whatever the draft depth.
  //
  // vLLM also migrates state itself: precopy_mamba_align_fused_kernel copies
  // state[bt[src_col + num_accepted - 1]] into state[bt[dst_col]] on a page crossing, assuming the
  // per-candidate layout. The anchor and factor read columns resolve to pages it never writes.
  // Any change to these columns must be re-derived against that kernel too.
  //
  // Nothing else may write this cache: advancing one copy alone leaves the others stale. Every step
  // shape, including non-speculative decode (`kind == "decode1"` in radiance_gdn.py), goes through
  // the fused path, and any other step first folds the accepted prefix into a single state.
  const int sa0 = ANC ? sidx[(size_t)n * sidx_stride + 0] : 0;   // anchor, page before
  const int sa1 = ANC ? sidx[(size_t)n * sidx_stride + 1] : 0;   // anchor, written here
  const int sa2 = ANC ? sidx[(size_t)n * sidx_stride + 2] : 0;   // anchor, page after
  const int sf0 = ANC ? sidx[(size_t)n * sidx_stride + 3] : 0;   // factors, marker 0
  const int sf  = ANC ? sidx[(size_t)n * sidx_stride + 4] : 0;   // factors, marker 1, read here
  const int sf2 = ANC ? sidx[(size_t)n * sidx_stride + 5] : 0;   // factors, marker 2
  // A negative slot is padding; 0 is a real slot (as for the per-candidate read below).
  if (ANC && (sa0 < 0 || sa1 < 0 || sa2 < 0 || sf0 < 0 || sf < 0 || sf2 < 0)) return;
  ST* const fb  = ANC ? (state + (size_t)sf  * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb0 = ANC ? (state + (size_t)sf0 * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb2 = ANC ? (state + (size_t)sf2 * st_slot + (size_t)hv * st_head) : nullptr;
  // Anchor column: without markers, column 1 (all three copies written); with them, 3 - m, never 0
  // (column 0 is the migration's target, derived from the committed context; see RU_ANC_MARK).
  // A caller whose state does not page (this engine's linear pool: one slot per sequence for its
  // life) names the same slot in every column; then duplicates are skipped, saving two thirds of
  // the factor traffic.
  const bool dup_a = ANC && RU_ANC_DUP && (sa0 != sa2 || sa1 != sa2);
  const bool dup_f = ANC && RU_ANC_DUP && (sf0 != sf || sf2 != sf);
  int acol = RU_ANC_MARK ? 2 : 1;
  int mrk = 1;
  if (ANC && RU_ANC_MARK) {
    // The marker is one-based so a zeroed (freshly allocated) factor block reads as absent and
    // falls back to the default anchor column. Zero-based, it would steer the first read after a
    // prefill to column 3, a factor column of zeros, discarding the prefill's state.
    const int m = (int)((float)fb[RU_MARK_OFF] + 0.5f) - 1;
    mrk = (m >= 0 && m <= 2) ? m : 1;
    acol = 3 - mrk;
  }
  // With RU_ANC_DUP=0 (or RU_ANC_FRD) the factors are read at 5 - m (3, 4 or 5) and written only
  // at column 4. The marker itself stays at fixed column 4 and is stamped into all three copies.
  const ST* const fr = (ANC && RU_ANC_MARK && (RU_ANC_FRD || !RU_ANC_DUP))
      ? (const ST*)(state + (size_t)(mrk == 0 ? sf2 : (mrk == 1 ? sf : sf0)) * st_slot
                    + (size_t)hv * st_head)
      : (const ST*)fb;
  /* Run 283 (briefs/2026-11-11-gdn-split-latency.md L1, folded after KG17: exact in every form, -1.3 us a launch at
   * T 4): in the anchor form acol is 1, 2 or 3 and those columns' slots are in registers from the row's loads above
   * (sa1, sa2, sf0) -- select instead of a dependent reload: the same integer, one round trip fewer */
  const int si = ANC ? (acol == 1 ? sa1 : (acol == 2 ? sa2 : sf0)) : sidx[(size_t)n * sidx_stride + (nacc - 1)];
  // A negative slot is padding; 0 is a real slot.
  if (si < 0) return;
  /* Fixed 32-row group (see RU_FSTR), uniform per wave for any VS (a wave is 16 consecutive rows
   * from a multiple of 16). readfirstlane makes the factor addresses scalar instead of 64-bit
   * VGPR offsets live across the token loop. */
  const int grp = __builtin_amdgcn_readfirstlane(row >> 5);

  float h[HK];
  {
    const ST* p = state + (size_t)si * st_slot + (size_t)hv * st_head
                + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < HK; j += 8) ru_load8<NT, SDT>(&h[j], p + j);
  }

  if (ANC) {
    /* (the brief's L2 -- the replay rows preloaded four at a time -- was exact but slower in KG17, +2.2 us at T 4:
     * dropped; today's loop stays) */
    for (int t = 0; t < nacc; ++t) {
      const ST* f = fr + (size_t)t * RU_FSTR;
      float kk[HK];
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_load8<0, SDT>(&kk[j], f + grp * RU_K + k0 + j);
      const float vv = (float)f[4 * RU_K + row];
      const float eg = __expf((float)f[4 * RU_K + RU_V + 2 * grp]);
      const float bt = (float)f[4 * RU_K + RU_V + 2 * grp + 1];
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
      dot = ru_row_sum<KT>(dot);            // the row's lanes, no LDS round trip
      const float u = (vv - dot) * bt;
#pragma unroll
      for (int j = 0; j < HK; ++j) h[j] += u * kk[j];
    }
    // Publish the advanced anchor, then barrier: every replay read above must complete before the
    // token loop overwrites the factor block with this step's factors.
    const size_t rowoff = (size_t)hv * st_head + (size_t)row * RU_K + k0;
    ST* p  = state + (size_t)(RU_ANC_MARK ? sa2 : sa1) * st_slot + rowoff;
    ST* p0 = state + (size_t)sa0 * st_slot + rowoff;
    ST* p2 = state + (size_t)sa2 * st_slot + rowoff;
#pragma unroll
    for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
    if (dup_a && !RU_ANC_MARK) {
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p0 + j, &h[j]);
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p2 + j, &h[j]);
    }
    __syncthreads();
    // Stamp after the barrier: every thread read fb's marker at the top, and stamping earlier
    // could steer a late wave to a different column from the rest of the workgroup. Split (VS > 1):
    // the other slices of this head may not have read it yet, so the last arriver stamps instead.
    if (RU_ANC_MARK && VS == 1 && tid == 0) {           // the three copies say where they are
      fb0[RU_MARK_OFF] = (ST)1.0f;
      fb[RU_MARK_OFF]  = (ST)2.0f;
      fb2[RU_MARK_OFF] = (ST)3.0f;
    }
  }
  /* Split anchor: every value head of the key head, in the same order as the single-workgroup stamp above (so a
   * caller naming one slot in every column ends with the same value). Called by the last of the G x VS workgroups
   * to arrive, after all of them read their markers. */
  auto stamp_group = [&]() {
    for (int g2 = 0; g2 < G; ++g2) {
      const size_t ho = (size_t)(hk * G + g2) * st_head + RU_MARK_OFF;
      state[(size_t)sf0 * st_slot + ho] = (ST)1.0f;
      state[(size_t)sf  * st_slot + ho] = (ST)2.0f;
      state[(size_t)sf2 * st_slot + ho] = (ST)3.0f;
    }
  };
  if (empty) {
    /* A fold-only step (T == 0) never reaches the convolution's arrival, so a split one arrives here. */
    if constexpr (CV && ANC && RU_ANC_MARK && VS > 1) {
      if (tid == 0) {
        const unsigned int was = __hip_atomic_fetch_add(cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT);
        if (was == (unsigned)(G * VS - 1)) {
          __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          stamp_group();
        }
      }
    }
    return;
  }

  if constexpr (CV) {
    // gdn_conv_update's token body, one channel per thread, into LDS instead of q, k and v.
    auto conv = [&](const float* wt, float bs, const unsigned short* hist,
                    const unsigned short* xs, unsigned short* dst) {
      float win[CW];
#pragma unroll
      for (int i = 0; i < CS; ++i) win[i] = bf2f(hist[i]);
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t) {
        if (t < T) {
          win[CW - 1] = bf2f(xs[t]);
          float acc = bs;
#pragma unroll
          for (int i = 0; i < CW; ++i) acc += wt[i] * win[i];
          dst[t * RU_K] = f2bf(cp_silu(acc));
#pragma unroll
          for (int i = 0; i < CW - 1; ++i) win[i] = win[i + 1];
        }
      }
    };
    if (cvt) {
      conv(cwa, cba, cha, cxa, (tid < RU_K ? cvq : cvk) + (tid & (RU_K - 1)));
#pragma unroll
      for (int i = 1; i < CS; ++i) cvh[(i - 1) * THR + tid] = cha[i];
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t)
        if (t < T) cvh[(CS - 1 + t) * THR + tid] = cxa[t];
    }
    if (tid < NVC) {   /* a v channel has one reader, this workgroup (split: its own rows only) */
      conv(cwb, cbb, chb, cxb, cvv + vrow0 + tid);
      ru_conv_rewrite(spb, chb, cxb, T, cv.cs_tok);
    }
    if (tid < 2 * T) sab[tid] = ca_b;
    if constexpr (NORM) {
#pragma unroll
    for (int i = 0; i < ZPT; ++i) {
      const int zi = tid + i * THR;
      if (zi < T * RU_V) szg[zi] = czg[i];
    }
    }
    // All history is consumed (taps are in LDS), so after the barrier this workgroup no longer
    // reads the key head's conv state and may arrive on the counter. Split: G x VS readers.
    __syncthreads();
    if (G * VS > 1 && tid == 0)
      arrived = __hip_atomic_fetch_add(cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  }

  const float alog = __expf(a_log_h);

  if (STG) {
    // One lane per (token, half) computes the same sums each v row computes on the STG=0 path.
    if (tid < 2 * T) {
      const int ut = tid >> 1, uk0 = (tid & 1) * HK;
      const unsigned short* qp = q + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      const unsigned short* kp = k + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      float ssq = 0.0f, ssk = 0.0f;
#pragma unroll
      for (int j = 0; j < HK; j += 8) {
        const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
        const unsigned short* aq = (const unsigned short*)&wq;
        const unsigned short* ak = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float fq = bf2f(aq[e]), fk = bf2f(ak[e]);
          ssq += fq * fq; ssk += fk * fk;
        }
      }
      spart[tid * 2] = ssq;
      spart[tid * 2 + 1] = ssk;
    }
    __syncthreads();
    if (tid < T) {
      // Same order as the STG=0 exchange: own half plus the other half.
      const float ssq = spart[tid * 4]     + spart[tid * 4 + 2];
      const float ssk = spart[tid * 4 + 1] + spart[tid * 4 + 3];
      sscl[tid * 2]     = __frsqrt_rn(ssq + 1e-6f) * scale;
      sscl[tid * 2 + 1] = __frsqrt_rn(ssk + 1e-6f);
    }
    __syncthreads();
  }

  for (int t = 0; t < T; ++t) {
    const int tok = bos + t;
    const unsigned short* qp = CV ? cvq + t * RU_K + k0 : q + ((size_t)tok * Hg + hk) * RU_K + k0;
    const unsigned short* kp = CV ? cvk + t * RU_K + k0 : k + ((size_t)tok * Hg + hk) * RU_K + k0;

    float qq[HK], kk[HK];
#pragma unroll
    for (int j = 0; j < HK; j += 8) {
      const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
      for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq[e]); kk[j + e] = bf2f(sk[e]); }
    }

    // l2 norms: 64 in-lane terms, then one exchange with the lane holding the other half
    float qs, ks;
    if (STG) {
      qs = sscl[t * 2]; ks = sscl[t * 2 + 1];
    } else {
      float sq = 0.0f, sk = 0.0f;
#pragma unroll
      for (int j = 0; j < HK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
      sq = ru_row_sum<KT>(sq);
      sk = ru_row_sum<KT>(sk);
      qs = __frsqrt_rn(sq + 1e-6f) * scale; ks = __frsqrt_rn(sk + 1e-6f);
    }
#pragma unroll
    for (int j = 0; j < HK; ++j) { qq[j] *= qs; kk[j] *= ks; }

    const float ra  = CV ? sab[t]     : ru_ab<ABF16>(av, (size_t)tok * ab_stride + hv);
    const float rb  = CV ? sab[T + t] : ru_ab<ABF16>(bv, (size_t)tok * ab_stride + hv);
    const float g   = -alog * ru_softplus(ra + dtb, sp_thr);
    const float bt  = 1.0f / (1.0f + __expf(-rb));
    const float eg  = __expf(g);

    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    dot = ru_row_sum<KT>(dot);

    const float vraw = bf2f(CV ? cvv[t * RU_V + row] : v[((size_t)tok * H + hv) * RU_V + row]);
    const float u = (vraw - dot) * bt;

    if (ANC) {
      // This step's factors for the next step's replay: one thread pair per row group writes the
      // group's k copy and decay/beta; every row writes its own v.
      ST* f  = fb  + (size_t)t * RU_FSTR;
      ST* f0 = fb0 + (size_t)t * RU_FSTR;
      ST* f2 = fb2 + (size_t)t * RU_FSTR;
      if ((row & 31) == 0) {
#pragma unroll
        for (int j = 0; j < HK; j += 8) {
          float kn[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) kn[e] = kk[j + e];
          ru_store8<0, SDT>(f + grp * RU_K + k0 + j, kn);
          if (dup_f) {
            ru_store8<0, SDT>(f0 + grp * RU_K + k0 + j, kn);
            ru_store8<0, SDT>(f2 + grp * RU_K + k0 + j, kn);
          }
        }
        if (half == 0) {
          f[4 * RU_K + RU_V + 2 * grp]      = (ST)g;
          f[4 * RU_K + RU_V + 2 * grp + 1]  = (ST)bt;
          if (dup_f) {
            f0[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f0[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
            f2[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f2[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
          }
        }
      }
      if (half == 0) {
        f[4 * RU_K + row] = (ST)vraw;
        if (dup_f) { f0[4 * RU_K + row] = (ST)vraw; f2[4 * RU_K + row] = (ST)vraw; }
      }
    }

    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
    on = ru_row_sum<KT>(on);

    if (NORM) {
      // sum of squares over the row: one channel per thread pair, 16 per wave
      float sq = (half == 0) ? on * on : 0.0f;
      // compile-time masks: 16 uses v_permlanex16, the rest DPP row_xmask
      sq += r4d_xor_lane<16>(sq); sq += r4d_xor_lane<8>(sq); sq += r4d_xor_lane<4>(sq);
      sq += r4d_xor_lane<2>(sq);  sq += r4d_xor_lane<1>(sq);
      if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sq;
      __syncthreads();
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) tot += wsum[i];
      const float sc = __frsqrt_rn(tot * (1.0f / RU_V) + neps);
      const unsigned short zr = CV ? szg[t * RU_V + row]
                                   : zg[(size_t)tok * zg_pitch + (size_t)hv * RU_V + row];
      on *= sc * nw[row] * ru_act(bf2f(zr), nact);
      __syncthreads();                 // wsum is reused by the next token
    }

    /* the split fold (owner note 138): the f32 row output for ru_split_norm_kernel, which norms, gates, stores and
     * quantises it exactly as the fused path below does */
    if constexpr (CV && !NORM) {
      if (half == 0) cv.onf[((size_t)tok * H + hv) * RU_V + row] = on;
    } else {
    const unsigned short ob = f2bf(on);
    if (half == 0) o[((size_t)tok * H + hv) * RU_V + row] = ob;

    // The out projection's quantiser, folded in (saves 48 launches per decode step, each with less
    // work than its launch gap). The quant group (128) is the row and NORM gives one workgroup per
    // (token, head), so the amax is a workgroup reduction and the scale index is (tok, hv).
    // Bit-identical to the standalone quantiser: it quantises the stored bf16 `ob`, not the f32
    // `on`, with r4d_fused_quant_fp8.hip's quant_group arithmetic (including the clamp that keeps
    // the group maximum from becoming NaN). NORM == 2 writes int8 codes instead (quant_act_i8g:
    // scale amax / 127, round to nearest-even).
    if (NORM && oq) {
      const float ov = bf2f(ob);
      float mv = (half == 0) ? fabsf(ov) : 0.0f;
      mv = fmaxf(mv, r4d_xor_lane<16>(mv)); mv = fmaxf(mv, r4d_xor_lane<8>(mv));
      mv = fmaxf(mv, r4d_xor_lane<4>(mv));  mv = fmaxf(mv, r4d_xor_lane<2>(mv));
      mv = fmaxf(mv, r4d_xor_lane<1>(mv));
      if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = mv;
      __syncthreads();
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) amax = fmaxf(amax, wsum[i]);
      const float qs = amax > 0.0f ? amax * (1.0f / q8_top<NORM == 2>()) : 1.0f;
      if (tid == 0) os[(size_t)tok * os_ld + hv] = qs;
      if (half == 0) {
        unsigned char code;
        if constexpr (NORM == 2) {
          code = (unsigned char)(int)rintf(fminf(fmaxf(ov * (1.0f / qs), -127.0f), 127.0f));
        } else {
          const float cv = fminf(fmaxf(ov * (1.0f / qs), -448.0f), 448.0f);
          code = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(cv, cv, 0, false) & 0xff);
        }
        oq[(size_t)tok * oq_pitch + (size_t)hv * RU_V + row] = code;
      }
      __syncthreads();               // wsum is reused by the next token
    }
    }

    if (!ANC) {
      const int so = sidx[(size_t)n * sidx_stride + t];
      // `>= 0`, not `> 0`: a negative slot is padding and 0 is a real slot (a single-sequence
      // decode's only slot), as for the read above.
      if (so >= 0) {
        ST* p = state + (size_t)so * st_slot + (size_t)hv * st_head
              + (size_t)row * RU_K + k0;
#pragma unroll
        for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
      }
    }
  }

  if constexpr (LAT & 4) {
    /* N1: the head's last arriving slice norms, gates, stores and quantises the head's T rows itself, so the norm
     * launch is skipped. Every lane's onf stores are issued before the barrier; the slice's device fence orders them
     * before its arrival, and the last arriver fences again before reading both slices' rows (the same code as
     * ru_split_norm_kernel: the same bits). THR / RU_V tokens a pass, a 128-row group each. */
    __syncthreads();
    if (tid == 0) {
      __threadfence();                                  /* release: this slice's onf rows before its arrival */
      const unsigned int wasn = __hip_atomic_fetch_add(cv.nctr + (size_t)n * H + hv, 1u, __ATOMIC_RELAXED,
                                                       __HIP_MEMORY_SCOPE_AGENT);
      nlast = wasn == (unsigned)(VS - 1);
      if (nlast) {
        __threadfence();                                /* acquire, in the arriving thread: the other slices' rows */
        __hip_atomic_store(cv.nctr + (size_t)n * H + hv, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
    if (nlast) {
      __threadfence();                                  /* every lane of the last arriver, before its reads */
      constexpr int NG = THR / RU_V;
      const int tg = tid / RU_V, nrow = tid - tg * RU_V;
      for (int t0 = 0; t0 < T; t0 += NG) {
        const int t = t0 + tg;
        ru_split_norm_body<2>(t < T, bos + t, hv, nrow, cv.onf, o, oq, os, oq_pitch, os_ld, zg, zg_pitch, nw, neps,
                              nact, H, ns2 + tg * RU_V, nws + tg * (RU_V / kSplitNormRowsAWave), nred + tg);
      }
    }
  }

  if constexpr (CV) {
    // The last arriving reader rewrites the key head's q and k channels. The count was taken
    // before the token loop, hiding its latency. Split: G x VS readers.
    if (G * VS > 1) {
      if (tid == 0) {
        cvlast = arrived == (unsigned)(G * VS - 1);
        if (cvlast) {
          __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          if constexpr (ANC && RU_ANC_MARK && VS > 1) stamp_group();   /* every slice has read its marker */
        }
      }
      __syncthreads();
      if (!cvlast) return;
    }
    if (!cvt) return;                           /* no channel of its own (KT > 2) */
    unsigned short rh[CS], rx[RU_CVT];
    rh[0] = 0;                                  /* never read: the oldest entry is shifted out */
#pragma unroll
    for (int i = 1; i < CS; ++i) rh[i] = cvh[(i - 1) * THR + tid];
#pragma unroll
    for (int t = 0; t < RU_CVT; ++t)
      rx[t] = t < T ? cvh[(CS - 1 + t) * THR + tid] : (unsigned short)0;
    ru_conv_rewrite(spa, rh, rx, T, cv.cs_tok);
  }
}

/* ============================================================================================================
 * Intel port, run 283 (owner note 224 item 2; briefs/2026-11-11-gdn-redesign-draft.md, logs/run281-gdn-stage1-plan.md):
 * the decode fold at VS 8 / 16 -- STAGE 1, the structure alone, no arithmetic moved.
 *
 * One workgroup is a SLICE-GROUP: the GC (= H / Hg = 3) value heads that share a key head, x NR = RU_V / VS rows each,
 * x KT = 8 lanes a row. Grid (N, Hg, VS): 64 workgroups at VS 8, 128 at VS 16 (the fused kernel: 24 of 32 cores).
 *   - the key head's q | k convolution is computed ONCE a workgroup (today: once in each of the G x VS workgroups),
 *     one channel a thread (two at VS 16), into cvq / cvk shared by the GC heads; the history parked by CHANNEL;
 *   - the v convolution for the workgroup's own rows of its GC heads;
 *   - the replay, the publish and the token loop are today's, per (row, half) lane, in today's order, with NORM off
 *     (the f32 row output to cv.onf for ru_split_norm_kernel, as the VS 2 / 4 split);
 *   - THE RUN-271 RACE FIX: no lane writes a 32-row group's shared factors (the group's k copy, decay / beta) inside
 *     the token loop -- at VS 8 two slices share a group and a late slice could replay this step's factors instead
 *     of the previous step's. The lanes of row 0 park the key head's scaled k (the same values every row computes)
 *     in skh[t], one lane a (head, token) parks (g, bt) in sgb; the LAST ARRIVER on the key head's counter (VS
 *     arrivals) writes every group's copy for every head of the key head after its own token loop, when every slice
 *     has finished reading the previous step's factors (the same read-before-rewrite argument as the conv state).
 *     Each lane still writes its own rows' v. The stored bytes are today's: (ST)kk, (ST)g, (ST)bt from the same
 *     expressions on the same inputs, each address written once a launch;
 *   - the shift marker: read at the top by every lane for its head (today's algebra); stamped only by the last
 *     arriver (run 113's stamp_group over the GC heads, the 1 / 2 / 3 order); the fold-only (T == 0) path arrives too.
 * Exactness: the same operations on the same operands per (row, half) lane; the row's 8 lanes stay consecutive in
 * one sub-group, so ru_row_sum<8> pairs the same halves in the same xor order. The gate is the hashes (gdn_split_bench
 * anchor-seq) against the fused kernel. GRF 256 in stage 1 (the fused kernel's); the zeinfo must show zero scratch.
 * Not for the CPU module (KT 8).
 * Run 284 (the coordinator's diagnostic sweep of stage 1): DG picks a TIMING-ONLY variant, compiled into the launcher
 * only under XE_GDN_SG_DIAG (the bench build; never an image) -- 1 the last arriver's tail skipped, 2 the skh / sgb
 * parks skipped, 3 the sbad barrier removed (a per-thread early return), 4 the per-token global stores skipped (the
 * factor v, its duplicates, onf), 5 q / k read straight from global x (no q | k convolution: wrong values), 6 GC 1
 * (one value head a workgroup; DG 6 changes nothing else). DG 0 is the kernel above, bit for bit. */
template <int NT, int ANC, int SDT, int VS, int GC, int DG = 0>
__global__ __launch_bounds__(GC * RU_V / VS * 8)
void ru_sg_update_kernel(const void* __restrict__ av, const void* __restrict__ bv, long ab_stride,
                         const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                         typename ru_state<SDT>::T* __restrict__ state, long st_slot, long st_head,
                         const int* __restrict__ cu, const int* __restrict__ sidx, long sidx_stride,
                         const int* __restrict__ naccept, int H, int Hg, float scale, float sp_thr,
                         const RuConv cv) {
  using ST = typename ru_state<SDT>::T;
  constexpr int KT = 8, NR = RU_V / VS, HK = RU_K / KT, HT = NR * KT, THR = GC * HT;
  constexpr int CW = R4D_GDN_CONV_W, CS = R4D_GDN_CONV_W - 1;
  constexpr int CPT = (2 * RU_K + THR - 1) / THR;       /* q | k channels a thread: 1 at VS 8, 2 at VS 16 */
  static_assert(VS == 8 || VS == 16, "the slice-group form is for VS 8 / 16; 2 / 4 take the split kernel");
  static_assert(HT % 32 == 0, "no head straddles a sub-group");
  static_assert(NR <= 32, "a slice lies inside one 32-row group");
  XE_SHARED(unsigned short, cvq, RU_CVT * RU_K); XE_SHARED(unsigned short, cvk, RU_CVT * RU_K);
  XE_SHARED(unsigned short, cvv, GC * RU_CVT * RU_K);               /* [head][token][row], the lambda's RU_K pitch */
  XE_SHARED(unsigned short, cvh, (CS - 1 + RU_CVT) * 2 * RU_K);    /* the key head's history by channel */
  XE_SHARED(float, sab, GC * 2 * RU_CVT);                           /* [head][a (T) | b (T)] */
  XE_SHARED(float, skh, RU_CVT * RU_K);                             /* the scaled k a token (the same every row) */
  XE_SHARED(float, sgb, GC * RU_CVT * 2);                           /* [head][token][g, bt] */
  XE_SHARED_VAR(int, cvlast);
  XE_SHARED_VAR(int, sbad);
  const int G = H / Hg;   /* == GC (the launcher checks); at GC 1 (the sweep's V6) blockIdx.y is the value head */
  const int n = blockIdx.x, hk = GC == 1 ? (int)blockIdx.y / G : (int)blockIdx.y, s = blockIdx.z;
  const int bos = cu[n], T = cu[n + 1] - bos;
  const bool empty = (T <= 0);
  if (empty && !ANC) return;
  const int tid = threadIdx.x;
  const int gi = tid / HT, ti = tid - gi * HT, rl = ti / KT, half = tid & (KT - 1), k0 = half * HK;
  const int hv = GC == 1 ? (int)blockIdx.y : hk * GC + gi, row = s * NR + rl;
  const unsigned last_arrival = (unsigned)(GC == 1 ? VS * G - 1 : VS - 1);   /* the key head's arrivals a launch */
  const float a_log_h = A_log[hv], dtb = dt_bias[hv];
  if constexpr (DG != 3) {
    if (tid == 0) sbad = 0;
    __syncthreads();                                      /* sbad's zero is ordered before any lane's set */
  }

  /* ---- the convolution's inputs first (in flight during the replay) */
  float cwa[CPT][CW], cwb[CW], cba[CPT], cbb = 0.0f;
  unsigned short cha[CPT][CS], chb[CS], cxa[CPT][RU_CVT], cxb[RU_CVT];
  unsigned short* spa[CPT];
  unsigned short* spb = nullptr;
  unsigned int arrived = 0;
  float ca_b = 0.0f;
#pragma unroll
  for (int c = 0; c < CPT; ++c) { spa[c] = nullptr; cba[c] = 0.0f; }
  int cslot = 0;
  if (!empty) {
    cslot = cv.cidx[(size_t)n * cv.ci_stride];
    if (cslot < 0) return;                               /* padding: every workgroup of the sequence leaves */
    const int off = naccept ? naccept[n] - 1 : 0;
    const unsigned short* xr = cv.x + (size_t)bos * cv.xpitch;
    if constexpr (DG != 5) {
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
      const int ch = tid + c * THR;                       /* this thread's q | k channel(s) */
      if (ch < 2 * RU_K) {
        const int ca = ch < RU_K ? hk * RU_K + ch : (Hg + hk) * RU_K + (ch - RU_K);
#pragma unroll
        for (int t = 0; t < RU_CVT; ++t)
          if (t < T) cxa[c][t] = xr[(size_t)t * cv.xpitch + ca];
        const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)ca * CW);
        cwa[c][0] = bf2f(w4.x); cwa[c][1] = bf2f(w4.y); cwa[c][2] = bf2f(w4.z); cwa[c][3] = bf2f(w4.w);
        if (cv.b) cba[c] = bf2f(cv.b[ca]);
        spa[c] = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)ca * cv.cs_dim;
#pragma unroll
        for (int i = 0; i < CS; ++i) cha[c][i] = spa[c][(size_t)(off + i) * cv.cs_tok];
      }
    }
    }
    if (half == 0) {                                      /* one v channel a (head, row): this lane's own */
      const int cb = (2 * Hg + hv) * RU_V + row;
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t)
        if (t < T) cxb[t] = xr[(size_t)t * cv.xpitch + cb];
      const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)cb * CW);
      cwb[0] = bf2f(w4.x); cwb[1] = bf2f(w4.y); cwb[2] = bf2f(w4.z); cwb[3] = bf2f(w4.w);
      if (cv.b) cbb = bf2f(cv.b[cb]);
      spb = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)cb * cv.cs_dim;
#pragma unroll
      for (int i = 0; i < CS; ++i) chb[i] = spb[(size_t)(off + i) * cv.cs_tok];
    }
    if (ti < 2 * T)                                       /* a (ti < T) or b of this head's token ti */
      ca_b = ti < T ? ru_ab<1>(av, (size_t)(bos + ti) * ab_stride + hv)
                    : ru_ab<1>(bv, (size_t)(bos + ti - T) * ab_stride + hv);
  }

  /* ---- the anchor's columns (today's algebra, per sequence; the marker per head) */
  const int nacc = naccept ? naccept[n] : 1;
  const int sa0 = ANC ? sidx[(size_t)n * sidx_stride + 0] : 0;
  const int sa1 = ANC ? sidx[(size_t)n * sidx_stride + 1] : 0;
  const int sa2 = ANC ? sidx[(size_t)n * sidx_stride + 2] : 0;
  const int sf0 = ANC ? sidx[(size_t)n * sidx_stride + 3] : 0;
  const int sf  = ANC ? sidx[(size_t)n * sidx_stride + 4] : 0;
  const int sf2 = ANC ? sidx[(size_t)n * sidx_stride + 5] : 0;
  if (ANC && (sa0 < 0 || sa1 < 0 || sa2 < 0 || sf0 < 0 || sf < 0 || sf2 < 0)) return;   /* per sequence: uniform */
  ST* const fb  = ANC ? (state + (size_t)sf  * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb0 = ANC ? (state + (size_t)sf0 * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb2 = ANC ? (state + (size_t)sf2 * st_slot + (size_t)hv * st_head) : nullptr;
  const bool dup_f = ANC && RU_ANC_DUP && (sf0 != sf || sf2 != sf);
  int acol = RU_ANC_MARK ? 2 : 1;
  int mrk = 1;
  if (ANC && RU_ANC_MARK) {
    const int m = (int)((float)fb[RU_MARK_OFF] + 0.5f) - 1;
    mrk = (m >= 0 && m <= 2) ? m : 1;
    acol = 3 - mrk;
  }
  const ST* const fr = (ANC && RU_ANC_MARK && (RU_ANC_FRD || !RU_ANC_DUP))
      ? (const ST*)(state + (size_t)(mrk == 0 ? sf2 : (mrk == 1 ? sf : sf0)) * st_slot + (size_t)hv * st_head)
      : (const ST*)fb;
  /* run 283 (L1, folded as in the kernel above): the anchor column's slot from the registers, not a dependent reload */
  const int si = ANC ? (acol == 1 ? sa1 : (acol == 2 ? sa2 : sf0)) : sidx[(size_t)n * sidx_stride + (nacc - 1)];
  /* the anchor slot is per (sequence, head's marker): make a negative one uniform over the slice-group before any
   * barrier (consistent markers agree; a disagreement counts the whole sequence as padding, as the plan says) */
  if constexpr (DG == 3) {                                /* the sweep's V3: a per-thread early return, no barrier */
    if (si < 0) return;
  } else {
    if (si < 0) sbad = 1;
    __syncthreads();
    if (sbad) return;
  }
  const int grp = __builtin_amdgcn_readfirstlane(row >> 5);

  float h[HK];
  {
    const ST* p = state + (size_t)si * st_slot + (size_t)hv * st_head + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < HK; j += 8) ru_load8<NT, SDT>(&h[j], p + j);
  }
  if (ANC) {
    for (int t = 0; t < nacc; ++t) {
      const ST* f = fr + (size_t)t * RU_FSTR;
      float kk[HK];
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_load8<0, SDT>(&kk[j], f + grp * RU_K + k0 + j);
      const float vv = (float)f[4 * RU_K + row];
      const float eg = __expf((float)f[4 * RU_K + RU_V + 2 * grp]);
      const float bt = (float)f[4 * RU_K + RU_V + 2 * grp + 1];
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
      dot = ru_row_sum<KT>(dot);
      const float u = (vv - dot) * bt;
#pragma unroll
      for (int j = 0; j < HK; ++j) h[j] += u * kk[j];
    }
    const size_t rowoff = (size_t)hv * st_head + (size_t)row * RU_K + k0;
    ST* p  = state + (size_t)(RU_ANC_MARK ? sa2 : sa1) * st_slot + rowoff;
#pragma unroll
    for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
    if (!RU_ANC_MARK && RU_ANC_DUP && (sa0 != sa2 || sa1 != sa2)) {
      ST* p0 = state + (size_t)sa0 * st_slot + rowoff;
      ST* p2 = state + (size_t)sa2 * st_slot + rowoff;
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p0 + j, &h[j]);
#pragma unroll
      for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p2 + j, &h[j]);
    }
    __syncthreads();                                      /* every replay read is done before any factor write */
  }
  /* the stamp of every head of the key head, by the last arriver (run 113's order) */
  auto stamp_group = [&]() {
    for (int g2 = 0; g2 < G; ++g2) {
      const size_t ho = (size_t)(hk * G + g2) * st_head + RU_MARK_OFF;
      state[(size_t)sf0 * st_slot + ho] = (ST)1.0f;
      state[(size_t)sf  * st_slot + ho] = (ST)2.0f;
      state[(size_t)sf2 * st_slot + ho] = (ST)3.0f;
    }
  };
  if (empty) {
    if constexpr (ANC && RU_ANC_MARK) {
      if (tid == 0) {
        const unsigned int was = __hip_atomic_fetch_add(cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT);
        if (was == last_arrival) {
          __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          stamp_group();
        }
      }
    }
    return;
  }

  /* ---- the convolution into LDS (today's lambda), the history parked by channel */
  auto conv = [&](const float* wt, float bs, const unsigned short* hist, const unsigned short* xs, unsigned short* dst) {
    float win[CW];
#pragma unroll
    for (int i = 0; i < CS; ++i) win[i] = bf2f(hist[i]);
#pragma unroll
    for (int t = 0; t < RU_CVT; ++t) {
      if (t < T) {
        win[CW - 1] = bf2f(xs[t]);
        float acc = bs;
#pragma unroll
        for (int i = 0; i < CW; ++i) acc += wt[i] * win[i];
        dst[t * RU_K] = f2bf(cp_silu(acc));
#pragma unroll
        for (int i = 0; i < CW - 1; ++i) win[i] = win[i + 1];
      }
    }
  };
  if constexpr (DG != 5) {
#pragma unroll
  for (int c = 0; c < CPT; ++c) {
    const int ch = tid + c * THR;
    if (ch < 2 * RU_K) {
      conv(cwa[c], cba[c], cha[c], cxa[c], (ch < RU_K ? cvq : cvk) + (ch & (RU_K - 1)));
#pragma unroll
      for (int i = 1; i < CS; ++i) cvh[(i - 1) * (2 * RU_K) + ch] = cha[c][i];
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t)
        if (t < T) cvh[(CS - 1 + t) * (2 * RU_K) + ch] = cxa[c][t];
    }
  }
  }
  if (half == 0) {   /* this lane's (head, row) v channel: one reader, this workgroup */
    conv(cwb, cbb, chb, cxb, cvv + gi * RU_CVT * RU_K + rl);   /* dst[t * RU_K]: this head's [token][row] block */
    ru_conv_rewrite(spb, chb, cxb, T, cv.cs_tok);
  }
  if (ti < 2 * T) sab[gi * 2 * RU_CVT + ti] = ca_b;
  __syncthreads();
  if (tid == 0)
    arrived = __hip_atomic_fetch_add(cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  const float alog = __expf(a_log_h);
  float dsink = 0.0f;                                     /* the sweep's V4: keeps `on` alive without its store */
  for (int t = 0; t < T; ++t) {
    const int tok = bos + t;
    const unsigned short* qp;
    const unsigned short* kp;
    if constexpr (DG == 5) {                              /* the sweep's V5: unconvolved x from global (wrong values) */
      qp = cv.x + (size_t)tok * cv.xpitch + hk * RU_K + k0;
      kp = cv.x + (size_t)tok * cv.xpitch + (Hg + hk) * RU_K + k0;
    } else {
      qp = cvq + t * RU_K + k0;
      kp = cvk + t * RU_K + k0;
    }
    float qq[HK], kk[HK];
#pragma unroll
    for (int j = 0; j < HK; j += 8) {
      const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
      for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq[e]); kk[j + e] = bf2f(sk[e]); }
    }
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < HK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
    sq = ru_row_sum<KT>(sq);
    sk = ru_row_sum<KT>(sk);
    const float qs = __frsqrt_rn(sq + 1e-6f) * scale, ks = __frsqrt_rn(sk + 1e-6f);
#pragma unroll
    for (int j = 0; j < HK; ++j) { qq[j] *= qs; kk[j] *= ks; }
    const float ra  = sab[gi * 2 * RU_CVT + t];
    const float rb  = sab[gi * 2 * RU_CVT + T + t];
    const float g   = -alog * ru_softplus(ra + dtb, sp_thr);
    const float bt  = 1.0f / (1.0f + __expf(-rb));
    const float eg  = __expf(g);
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    dot = ru_row_sum<KT>(dot);
    const float vraw = bf2f(cvv[gi * RU_CVT * RU_K + t * RU_K + rl]);
    const float u = (vraw - dot) * bt;
    if (ANC) {
      /* this step's factors for the next step's replay: the key head's scaled k and this head's (g, bt) are parked
       * in LDS for the last arriver (the race fix); every lane writes its own row's v now */
      if constexpr (DG != 2) {                            /* the sweep's V2 skips the parks */
        if (tid < KT) {                                   /* row 0 of head 0: the key head's k, the same every row */
#pragma unroll
          for (int j = 0; j < HK; ++j) skh[t * RU_K + k0 + j] = kk[j];
        }
        if (ti == 0) { sgb[(gi * RU_CVT + t) * 2] = g; sgb[(gi * RU_CVT + t) * 2 + 1] = bt; }
      }
      if constexpr (DG != 4) {                            /* the sweep's V4 skips the per-token global stores */
        if (half == 0) {
          ST* f  = fb  + (size_t)t * RU_FSTR;
          f[4 * RU_K + row] = (ST)vraw;
          if (dup_f) {
            fb0[(size_t)t * RU_FSTR + 4 * RU_K + row] = (ST)vraw;
            fb2[(size_t)t * RU_FSTR + 4 * RU_K + row] = (ST)vraw;
          }
        }
      }
    }
    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
    on = ru_row_sum<KT>(on);
    if constexpr (DG != 4) {
      if (half == 0) cv.onf[((size_t)tok * H + hv) * RU_V + row] = on;
    } else {
      dsink += on;
    }
    if (!ANC) {
      const int so = sidx[(size_t)n * sidx_stride + t];
      if (so >= 0) {
        ST* p = state + (size_t)so * st_slot + (size_t)hv * st_head + (size_t)row * RU_K + k0;
#pragma unroll
        for (int j = 0; j < HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
      }
    }
  }

  /* ---- the tail: the last arriver writes the shared factors of every head of the key head, stamps, and rewrites
   * the key head's conv state; everyone else leaves. The parked values are complete: every lane's loop is done
   * before the barrier below, and the last arrival follows every slice's replay (each arrived after its own replay
   * and conv barrier). */
  if constexpr (DG == 4) { if (dsink == 1.2345e-37f) cv.onf[0] = dsink; }   /* the sink: never true in practice */
  if constexpr (DG == 1) return;                          /* the sweep's V1: no tail (the counter stays unreset) */
  __syncthreads();
  if (tid == 0) {
    cvlast = arrived == last_arrival;
    if (cvlast) __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (!cvlast) return;
  if (ANC) {
    /* G heads x T tokens x 4 groups: the k copy (RU_K floats) and (g, bt), into fb and the duplicates */
    for (int idx = tid; idx < G * T * 4 * (RU_K / 8); idx += THR) {
      const int blk8 = idx % (RU_K / 8), g4 = (idx / (RU_K / 8)) % 4, t = (idx / (RU_K / 8) / 4) % T, g2 = idx / (RU_K / 8) / 4 / T;
      const int hv2 = hk * G + g2;
      float kn[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) kn[e] = skh[t * RU_K + blk8 * 8 + e];
      ST* f = state + (size_t)sf * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + g4 * RU_K + blk8 * 8;
      ru_store8<0, SDT>(f, kn);
      if (dup_f) {
        ru_store8<0, SDT>(state + (size_t)sf0 * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + g4 * RU_K + blk8 * 8, kn);
        ru_store8<0, SDT>(state + (size_t)sf2 * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + g4 * RU_K + blk8 * 8, kn);
      }
    }
    for (int idx = tid; idx < G * T * 4; idx += THR) {
      const int g4 = idx % 4, t = (idx / 4) % T, g2 = idx / 4 / T;
      const int hv2 = hk * G + g2;
      const float g = sgb[(g2 * RU_CVT + t) * 2], bt = sgb[(g2 * RU_CVT + t) * 2 + 1];
      ST* f = state + (size_t)sf * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + 4 * RU_K + RU_V + 2 * g4;
      f[0] = (ST)g; f[1] = (ST)bt;
      if (dup_f) {
        ST* f0 = state + (size_t)sf0 * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + 4 * RU_K + RU_V + 2 * g4;
        ST* f2 = state + (size_t)sf2 * st_slot + (size_t)hv2 * st_head + (size_t)t * RU_FSTR + 4 * RU_K + RU_V + 2 * g4;
        f0[0] = (ST)g; f0[1] = (ST)bt; f2[0] = (ST)g; f2[1] = (ST)bt;
      }
    }
    if constexpr (RU_ANC_MARK) { if (tid == 0) stamp_group(); }
  }
  /* the key head's conv state, one channel a thread from the parked history */
  if constexpr (DG != 5) {
#pragma unroll
  for (int c = 0; c < CPT; ++c) {
    const int ch = tid + c * THR;
    if (ch < 2 * RU_K) {
      unsigned short rh[CS], rx[RU_CVT];
      rh[0] = 0;
#pragma unroll
      for (int i = 1; i < CS; ++i) rh[i] = cvh[(i - 1) * (2 * RU_K) + ch];
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t) rx[t] = t < T ? cvh[(CS - 1 + t) * (2 * RU_K) + ch] : (unsigned short)0;
      ru_conv_rewrite(spa[c], rh, rx, T, cv.cs_tok);
    }
  }
  }
}
/* ============================================================================================================ */

/* Intel port, run 112 (owner note 138): the split fold's norm, one workgroup a (token, value head), a thread a row.
 * It is the fused path's arithmetic in the fused path's order at KT 8: on^2 a row; a 32-lane wave there holds four
 * rows (lanes 0 / 8 / 16 / 24) and its xor tree gives (r0^2 + r2^2) + (r1^2 + r3^2) (the other lanes add zeros); the
 * 32 wave sums are added in order from 0; then the gate, the bf16 store and the quantiser as there. The same bits.
 * Tokens past cu's end and padding sequences (no conv slot) are left alone, as the fused fold leaves them.
 * Run 284: the body is ru_split_norm_body (above the update kernel), shared with the split kernel's N1 fold. */
template <int NQ>
__global__ __launch_bounds__(RU_V)
void ru_split_norm_kernel(const float* __restrict__ onf, unsigned short* __restrict__ o,
                          unsigned char* __restrict__ oq, float* __restrict__ os, long oq_pitch, long os_ld,
                          const unsigned short* __restrict__ zg, long zg_pitch, const float* __restrict__ nw,
                          float neps, int nact, int H, const int* __restrict__ cu, int N,
                          const int* __restrict__ cidx, long ci_stride) {
  XE_SHARED(float, s2, RU_V);
  XE_SHARED(float, ws, RU_V / kSplitNormRowsAWave);
  XE_SHARED_VAR(float, red);
  const int tok = blockIdx.x, hv = blockIdx.y, row = threadIdx.x;
  if (tok >= cu[N]) return;
  int lo = 0, hi = N;                        /* the token's sequence: the last n with cu[n] <= tok */
  while (hi - lo > 1) { const int mid = (lo + hi) / 2; if (cu[mid] <= tok) lo = mid; else hi = mid; }
  if (cidx[(size_t)lo * ci_stride] < 0) return;
  ru_split_norm_body<NQ>(true, tok, hv, row, onf, o, oq, os, oq_pitch, os_ld, zg, zg_pitch, nw, neps, nact, H,
                         s2, ws, &red);
}

// Non-temporal hint on the state load and store (TH_LOAD_NT / TH_STORE_NT): no result changes;
// chosen per launch because its cost and saving land in different kernels. In isolation it makes
// this kernel 43.8% slower (DRAM-write bound), but no state survives the 64 MB last-level cache
// within a step, so the hint only decides who pays for the write-back, and it saves the following
// GEMMs more (about -140 us on the tiled GEMM vs +25 us here under anchor state).
//
// Anchor state: one state per sequence with the accepted prefix replayed onto it. The caller's
// prefill path must zero state slot sidx[n][1] for every sequence it writes. Off (per_candidate),
// every candidate writes its own slot, matching vLLM's Triton path. A radiance caller declares the
// form as the op's `state_form` (ru_state_form); the raw C entry, which vLLM's radiance_gdn.py
// calls, has no op parameters and reads RADIANCE_GDN_ANCHOR, as that caller does.
/* RADIANCE_XE_GDN_NT=0|1 (run 101, owner note 108 item 1): the state's non-temporal hint forced off or on instead of
 * chosen by the flood size (RU_NT_MB). The hint changes no result. GB3 found the per-token state store holding the
 * decode chain (~60 us a token of ~97); the hint's lowering on Xe is the first suspect. -1: the shipped choice. */
static int ru_nt_force() {
  static const int v = [] { const char* e = getenv("RADIANCE_XE_GDN_NT"); return e && (e[0] == '0' || e[0] == '1') ? e[0] - '0' : -1; }();
  return v;
}

static bool ru_anchor() {
  static const bool v = [] {
    const char* e = getenv("RADIANCE_GDN_ANCHOR");
    return e && e[0] == '1';
  }();
  return v;
}

// STG: the q/k l2 scales are computed once per head and shared by the 128 v rows (bit-identical to
// per row). Folding the scales into the reductions instead is slower and moves about one output in
// 600 by a bf16 ULP.
//
// Flood size (MB of state a launch writes) up to which the non-temporal hint is used. It wins when
// the state would otherwise stay in the last-level cache and become the next GEMM's write-back
// (6.3 MB at one sequence: ~3% of a step) and loses when nothing survives anyway (50 MB at eight:
// ~2% slower). The threshold leaves cache room for the streaming weights. Under anchor state the
// hint nets about -100 us a step; under per-candidate state the flood is ~8x larger.
#define RU_NT_MB 16

// `oq_i8` picks the folded quantiser's codes: E4M3 (0) or int8 (1); see the kernel's epilogue.
static int ru_update_launch(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
    long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
    long oq_pitch, long os_ld, const void* cu,
    const void* ssm_state_indices, long indices_stride, const void* num_accepted,
    const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
    int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
    void* stream, int oq_i8, bool anc)
{
  if (K != RU_K || V != RU_V) return -1;
  if (H % Hg) return -2;
  const int nm0 = (z_gate && norm_weight) ? 1 : 0;
  // Fill the GPU: split a head's v rows until there are at least WGTARGET workgroups, at most
  // four ways (64 threads, two waves).
  int vs = 1;
  if (!nm0) { const int want = (WGTARGET + N * H - 1) / (N * H);
              vs = want >= 4 ? 4 : (want >= 2 ? 2 : 1); }
  // The shift marker forbids splitting a head: with vs > 1 one workgroup could stamp this step's
  // marker before another read the previous one, and they would disagree on the page shift. The
  // wave count is the same either way.
  if (anc && RU_ANC_MARK) vs = 1;
  dim3 grid(N, H, vs), blk(RU_V / vs * 2);
  // Every switch is a template parameter; the nested pick macros enumerate the instantiations.
#define RU_LAUNCH(AB, NM, VSC, NTC, ANCC, STGC, SDTC)                                         \
  XE_LAUNCH_GGL_GRF256((r4d_gdn_recurrent_update_kernel<AB, NM, VSC, NTC, ANCC, STGC, SDTC>),   \
                     grid, blk, 0, (hipStream_t)stream,                                       \
                     (const unsigned short*)q, (const unsigned short*)k,                      \
                     (const unsigned short*)v, a, b, ab_stride,                               \
                     (const float*)A_log, (const float*)dt_bias,                              \
                     (ru_state<SDTC>::T*)state, state_slot_stride, state_head_stride,         \
                     (unsigned short*)o, (unsigned char*)oq, (float*)os, oq_pitch, os_ld,    \
                     (const int*)cu, (const int*)ssm_state_indices,                          \
                     indices_stride, (const int*)num_accepted,                                \
                     (const unsigned short*)z_gate, z_pitch, (const float*)norm_weight, norm_eps,      \
                     norm_act, H, Hg, scale, softplus_thr, RuConv{})
#define RU_PICKV(AB, NTC, ANCC, SDTC)                                                         \
  do { if (vs == 4)      RU_LAUNCH(AB, 0, 4, NTC, ANCC, 0, SDTC);                             \
       else if (vs == 2) RU_LAUNCH(AB, 0, 2, NTC, ANCC, 0, SDTC);                             \
       else              RU_LAUNCH(AB, 0, 1, NTC, ANCC, 0, SDTC); } while (0)
#define RU_PICKNM(AB, NTC, ANCC, SDTC)                                                        \
  do { if (oq_i8) RU_LAUNCH(AB, 2, 1, NTC, ANCC, 0, SDTC);                                    \
       else       RU_LAUNCH(AB, 1, 1, NTC, ANCC, 0, SDTC); } while (0)
#define RU_PICKAB(NTC, ANCC, SDTC)                                                            \
  do { if (ab_is_bf16) { if (nm0) RU_PICKNM(1, NTC, ANCC, SDTC);                              \
                         else     RU_PICKV(1, NTC, ANCC, SDTC); }                             \
       else            { if (nm0) RU_PICKNM(0, NTC, ANCC, SDTC);                              \
                         else     RU_PICKV(0, NTC, ANCC, SDTC); } } while (0)
#define RU_PICKNT(ANCC, SDTC)                                                                 \
  do { if (nt) RU_PICKAB(1, ANCC, SDTC); else RU_PICKAB(0, ANCC, SDTC); } while (0)
#define RU_PICKANC(SDTC)                                                                      \
  do { if (anc) RU_PICKNT(1, SDTC); else RU_PICKNT(0, SDTC); } while (0)

  // The staged path (STG) is instantiated only for the decode shape: bf16 a/b, no fused row norm,
  // VS 1. Everything else uses the per-row path (STG 0).
  const int sdt = state_fp16 ? 1 : 0;
  // Bytes of state this launch writes: one slot per sequence under the anchor, one per candidate
  // without it.
  const long slots_written = anc ? 1 : (long)indices_stride;
  const long flood = (long)N * slots_written * state_slot_stride * (sdt ? 2 : 4);
  const bool nt = ru_nt_force() >= 0 ? ru_nt_force() == 1 : flood <= (long)RU_NT_MB * 1024 * 1024;
  // Anchor bounds: six columns are addressed (three anchor, three factor), and the factor block
  // (indices_stride tokens of RU_FSTR) must fit in one head's slice of a slot. Keep these before
  // the staged path, which returns early and is the shipped anchor path.
  if (anc && indices_stride < 6) return -4;
  if (anc && (long)indices_stride * RU_FSTR > state_head_stride) return -3;
  if (ab_is_bf16 && !nm0 && vs == 1 && indices_stride <= RU_TMAX) {
#define RU_PICKSTG2(ANCC, SDTC)                                                               \
  do { if (nt) RU_LAUNCH(1, 0, 1, 1, ANCC, 1, SDTC);                                          \
       else    RU_LAUNCH(1, 0, 1, 0, ANCC, 1, SDTC); } while (0)
#define RU_PICKSTG(SDTC) do { if (anc) RU_PICKSTG2(1, SDTC); else RU_PICKSTG2(0, SDTC); } while (0)
    if (sdt) RU_PICKSTG(1); else RU_PICKSTG(0);
#undef RU_PICKSTG
#undef RU_PICKSTG2
    return (int)hipGetLastError();
  }
  if (sdt) RU_PICKANC(1); else RU_PICKANC(0);
#undef RU_PICKANC
#undef RU_PICKNT
#undef RU_PICKAB
#undef RU_PICKNM
#undef RU_PICKV
#undef RU_LAUNCH
  return (int)hipGetLastError();
}

extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
    long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
    long oq_pitch, long os_ld, const void* cu,
    const void* ssm_state_indices, long indices_stride, const void* num_accepted,
    const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
    int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
    void* stream)
{
  return ru_update_launch(q, k, v, a, b, ab_stride, ab_is_bf16, A_log, dt_bias, state,
                          state_slot_stride, state_head_stride, o, oq, os, oq_pitch, os_ld, cu,
                          ssm_state_indices, indices_stride, num_accepted, z_gate, z_pitch,
                          norm_weight, norm_eps, norm_act, N, H, Hg, K, V, scale, softplus_thr,
                          state_fp16, stream, 0, ru_anchor());
}

// ============================================================== the folded convolution

// Arrival counters for the folded convolution: one block per (stream, output), zeroed once and
// reset by the last arriver of each group. As with split_ctr in r4d_gemm_fp8a8.hip, launches
// sharing a block must not overlap: different streams get different blocks, and launches on one
// stream sharing an output are ordered by their dependency.
constexpr unsigned kMaxConvCtr = 4096;

/* Owner note 138 (run 112): the split fold's f32 row outputs, [token][H][RU_V], for up to 64 tokens x 64 heads a
 * launch (2 MiB): a decode or verify step's tokens; a larger launch keeps the fused form. Allocated once per
 * (stream, output) before any recording, as the counter block below. */
constexpr long kSplitOnfTokHeads = 64 * 64;
static float* ru_split_onf(RadStream s, const void* out) {
  static std::mutex m;
  static std::map<std::pair<RadStream, const void*>, float*> blocks;
  std::lock_guard<std::mutex> lk(m);
  const auto key = std::make_pair(s, out);
  if (auto it = blocks.find(key); it != blocks.end()) return it->second;
  sycl::queue& q = *(sycl::queue*)s;
  float* p = nullptr;
  try {
    p = sycl::malloc_device<float>((size_t)kSplitOnfTokHeads * RU_V, q);
  } catch (const sycl::exception&) {
    return nullptr;
  }
  if (p) blocks.emplace(key, p);
  return p;
}

static unsigned int* ru_conv_ctr(RadStream s, const void* out) {
  static std::mutex m;
  static std::map<std::pair<RadStream, const void*>, unsigned int*> blocks;
  std::lock_guard<std::mutex> lk(m);
  const auto key = std::make_pair(s, out);
  if (auto it = blocks.find(key); it != blocks.end()) return it->second;
  // libxe: device memory on the stream's own queue (its context is the backend's), zeroed and
  // waited for before any launch reads it -- the hipMalloc / hipMemset / sync of the original.
  sycl::queue& q = *(sycl::queue*)s;
  void* p = nullptr;
  try {
    p = sycl::malloc_device((size_t)kMaxConvCtr * sizeof(unsigned int), q);
    if (!p) return nullptr;
    /* zeroed on a private queue: the stream's own may be recording a SYCL graph (a tape), where
     * a wait is refused (card1 run 10: the first recorded decode pass failed here) */
    sycl::queue(q.get_context(), q.get_device()).memset(p, 0, (size_t)kMaxConvCtr * sizeof(unsigned int)).wait();
  } catch (const sycl::exception&) {
    if (p) sycl::free(p, q);
    return nullptr;
  }
  blocks.emplace(key, (unsigned int*)p);
  return (unsigned int*)p;
}

/* Run 284 (the gdn-split-latency brief's N1): the per-(sequence, value head) arrival counters of the split fold's own
 * norm, kMaxConvCtr of them, allocated and zeroed as ru_conv_ctr's block (one per (stream, output)); the launcher
 * declines N1 when N * H exceeds the block. */
static unsigned int* ru_split_ctr(RadStream s, const void* out) {
  static std::mutex m;
  static std::map<std::pair<RadStream, const void*>, unsigned int*> blocks;
  std::lock_guard<std::mutex> lk(m);
  const auto key = std::make_pair(s, out);
  if (auto it = blocks.find(key); it != blocks.end()) return it->second;
  sycl::queue& q = *(sycl::queue*)s;
  void* p = nullptr;
  try {
    p = sycl::malloc_device((size_t)kMaxConvCtr * sizeof(unsigned int), q);
    if (!p) return nullptr;
    sycl::queue(q.get_context(), q.get_device()).memset(p, 0, (size_t)kMaxConvCtr * sizeof(unsigned int)).wait();
  } catch (const sycl::exception&) {
    if (p) sycl::free(p, q);
    return nullptr;
  }
  blocks.emplace(key, (unsigned int*)p);
  return (unsigned int*)p;
}

// The recurrent update with gdn_conv_update folded in, decode shape only (bf16 a/b, fused row
// norm, one workgroup per (sequence, value head)). State contract, non-temporal choice and
// factor-block bounds are as in ru_update_launch.
static int ru_conv_recurrent_launch(const RuConv& cv, const void* a, const void* b, long ab_stride,
                                    const void* A_log, const void* dt_bias, void* state,
                                    long st_slot, long st_head, void* o, void* oq, void* os,
                                    long oq_pitch, long os_ld, const void* cu, const void* sidx,
                                    long sidx_stride, const void* nacc, const void* z,
                                    long z_pitch, const void* nw, float neps, int nact, int N,
                                    int H, int Hg, float scale, float thr, int state_fp16,
                                    RadStream stream, int oq_i8, bool anc, int ntok = 0) {
  if (anc && sidx_stride < 6) return RAD_E_SHAPE;
  if (anc && sidx_stride * RU_FSTR > st_head) return RAD_E_SHAPE;
  const int sdt = state_fp16 ? 1 : 0;
  const long slots_written = anc ? 1 : sidx_stride;
  const long flood = (long)N * slots_written * st_slot * (sdt ? 2 : 4);
  const bool nt = ru_nt_force() >= 0 ? ru_nt_force() == 1 : flood <= (long)RU_NT_MB * 1024 * 1024;
  /* KT = 4 threads a v row (milestone 3, kernel 4): 32 state + 32 key + 32 query floats a lane fit
   * the large register file; the shipped two-thread layout spilled 32 KB a thread on Xe */
  /* 8 on the card (run 26: 16 state + 16 key + 16 query floats a lane; KT = 4 still spilled 7.5 KB at SIMD32); the
   * CPU device's JIT crashes on the 1,024-thread instantiation (run 24), so its module keeps the shipped 2 */
#if defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__)
  constexpr int RC_KT = 8;
#else
  constexpr int RC_KT = 2;
#endif
  const dim3 grid(N, H, 1), blk(RU_V * RC_KT);
#if defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__)
  /* Owner note 138 (run 112, RADIANCE_XE_GDN_SPLIT=2|4): a head's v rows over 2 or 4 workgroups without the fused
   * norm, then ru_split_norm_kernel -- the same bits (see both kernels). KT 8 only (the norm kernel mirrors that
   * layout's order). Run 113: the anchor form too (MTP on), its shift marker stamped by the key head's last
   * arriving workgroup. */
  static const int split_vs = [] {
    const char* e = std::getenv("RADIANCE_XE_GDN_SPLIT");
    return e && (e[0] == '2' || e[0] == '4') ? e[0] - '0' : 0;
  }();
  /* Run 283 (owner note 224 item 2): RADIANCE_XE_GDN_SPLIT=8|16 -> the slice-group kernel (ru_sg_update_kernel: the key
   * head's GC = 3 value heads x RU_V / VS rows a workgroup, the shared factors by the last arriver), then the split
   * norm kernel as the 2 / 4 split. Needs H / Hg == 3 (Flash-Next's 24 / 8) and the split's other conditions. */
  static const int split_sg = [] {
    const char* e = std::getenv("RADIANCE_XE_GDN_SPLIT");
    const int v = e ? std::atoi(e) : 0;
    return (v == 8 || v == 16) ? v : 0;
  }();
  if (split_sg && cv.onf && ntok > 0 && (long)ntok * H <= kSplitOnfTokHeads && (!anc || RU_ANC_MARK) && H % Hg == 0
      && H / Hg == 3) {
    dim3 ggrid(N, Hg, split_sg), gblk(3 * RU_V / split_sg * RC_KT);
#define RC_SG_A(VSC, NTC, ANCC, SDTC, GCC, DGC)                                                                 \
  XE_LAUNCH_GGL_GRF256((ru_sg_update_kernel<NTC, ANCC, SDTC, VSC, GCC, DGC>), ggrid, gblk, 0, (hipStream_t)stream, \
                     a, b, ab_stride, (const float*)A_log, (const float*)dt_bias, (ru_state<SDTC>::T*)state,     \
                     st_slot, st_head, (const int*)cu, (const int*)sidx, sidx_stride, (const int*)nacc, H, Hg,  \
                     scale, thr, cv)
#define RC_SG(VSC, NTC, SDTC) do { if (anc) RC_SG_A(VSC, NTC, 1, SDTC, 3, 0); else RC_SG_A(VSC, NTC, 0, SDTC, 3, 0); } while (0)
#define RC_SG_NT(VSC, SDTC) do { if (nt) RC_SG(VSC, 1, SDTC); else RC_SG(VSC, 0, SDTC); } while (0)
#define RC_SG_VS(SDTC) do { if (split_sg == 16) RC_SG_NT(16, SDTC); else RC_SG_NT(8, SDTC); } while (0)
#ifdef XE_GDN_SG_DIAG
    /* Run 284 (the coordinator's diagnostic sweep): RADIANCE_XE_GDN_SG_DIAG=1..6 picks one of the kernel's TIMING-ONLY
     * variants (DG above; the hashes may differ) -- the bench build alone (-DXE_GDN_SG_DIAG=ON), never an image
     * (make-rt-neo2.sh refuses such a tree). The anchor form at NT 0 and VS 8 only, as the sweep measures; elsewhere
     * the knob is ignored. V6 is GC 1: grid (N, H, 8), one value head a workgroup of RU_V / 8 x 8 lanes. */
    static const int sg_diag = [] {
      const char* e = std::getenv("RADIANCE_XE_GDN_SG_DIAG");
      const int v = e ? std::atoi(e) : 0;
      return (v >= 1 && v <= 6) ? v : 0;
    }();
    if (sg_diag && anc && !nt && split_sg == 8) {
      if (sg_diag == 6) { ggrid.y = H; gblk.x = RU_V / 8 * RC_KT; }
#define RC_SG_D(DGC, GCC) do { if (sdt) RC_SG_A(8, 0, 1, 1, GCC, DGC); else RC_SG_A(8, 0, 1, 0, GCC, DGC); } while (0)
      switch (sg_diag) {
        case 1: RC_SG_D(1, 3); break;
        case 2: RC_SG_D(2, 3); break;
        case 3: RC_SG_D(3, 3); break;
        case 4: RC_SG_D(4, 3); break;
        case 5: RC_SG_D(5, 3); break;
        default: RC_SG_D(6, 1); break;
      }
#undef RC_SG_D
    } else
#endif
    { if (sdt) RC_SG_VS(1); else RC_SG_VS(0); }
#undef RC_SG_VS
#undef RC_SG_NT
#undef RC_SG
#undef RC_SG_A
    const dim3 ngrid((unsigned)ntok, (unsigned)H), nblk(RU_V);
#define RC_NORM(NQ) hipLaunchKernelGGL((ru_split_norm_kernel<NQ>), ngrid, nblk, 0, (hipStream_t)stream,          \
                                       (const float*)cv.onf, (unsigned short*)o, (unsigned char*)oq, (float*)os, \
                                       oq_pitch, os_ld, (const unsigned short*)z, z_pitch, (const float*)nw,    \
                                       neps, nact, H, (const int*)cu, N, cv.cidx, cv.ci_stride)
    if (!oq) RC_NORM(0); else if (oq_i8) RC_NORM(2); else RC_NORM(1);
#undef RC_NORM
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
  }
  if (split_vs && cv.onf && ntok > 0 && (long)ntok * H <= kSplitOnfTokHeads && (!anc || RU_ANC_MARK)) {
    const dim3 sgrid(N, H, split_vs), sblk(RU_V / split_vs * RC_KT);
    /* Run 283 (briefs/2026-11-11-gdn-split-latency.md, KG17): RADIANCE_XE_GDN_LAT=7 = N1, the exact opt-in RECORD (the
     * folded norm; L1 is folded into every kernel, L2 dropped), instantiated for the VS 2 split at NT 0 with the f32
     * state and the int8 quantiser (the served form) when its counter block holds N * H; elsewhere ignored. With N1
     * the norm launch below is skipped. */
    static const int split_lat = [] {
      const char* e = std::getenv("RADIANCE_XE_GDN_LAT");
      return (e && std::atoi(e) == 7) ? 7 : 0;
    }();
    int lat = (split_vs == 2 && !nt && !sdt) ? split_lat : 0;
    if (lat && !(oq && oq_i8 && cv.nctr && (long)N * H <= (long)kMaxConvCtr)) lat = 0;
#define RC_SPLIT_A(VSC, NTC, ANCC, SDTC, LATC)                                                 \
  XE_LAUNCH_GGL_GRF256((r4d_gdn_recurrent_update_kernel<1, 0, VSC, NTC, ANCC, 0, SDTC, 1, RC_KT, LATC>), \
                     sgrid, sblk, 0, (hipStream_t)stream, nullptr, nullptr, nullptr, a, b,     \
                     ab_stride, (const float*)A_log, (const float*)dt_bias,                    \
                     (ru_state<SDTC>::T*)state, st_slot, st_head, (unsigned short*)o,          \
                     (unsigned char*)oq, (float*)os, oq_pitch, os_ld, (const int*)cu,          \
                     (const int*)sidx, sidx_stride, (const int*)nacc,                          \
                     (const unsigned short*)z, z_pitch, (const float*)nw, neps, nact, H, Hg,   \
                     scale, thr, cv)
#define RC_SPLIT(VSC, NTC, SDTC) do { if (anc) RC_SPLIT_A(VSC, NTC, 1, SDTC, 0); else RC_SPLIT_A(VSC, NTC, 0, SDTC, 0); } while (0)
#define RC_SPLIT_NT(VSC, SDTC) do { if (nt) RC_SPLIT(VSC, 1, SDTC); else RC_SPLIT(VSC, 0, SDTC); } while (0)
#define RC_SPLIT_VS(SDTC) do { if (split_vs == 4) RC_SPLIT_NT(4, SDTC); else RC_SPLIT_NT(2, SDTC); } while (0)
#define RC_SPLIT_L(LATC) do { if (anc) RC_SPLIT_A(2, 0, 1, 0, LATC); else RC_SPLIT_A(2, 0, 0, 0, LATC); } while (0)
    if (lat == 7) RC_SPLIT_L(4);
    else { if (sdt) RC_SPLIT_VS(1); else RC_SPLIT_VS(0); }
#undef RC_SPLIT_L
#undef RC_SPLIT_VS
#undef RC_SPLIT_NT
#undef RC_SPLIT
#undef RC_SPLIT_A
    if (lat != 7) {
      const dim3 ngrid((unsigned)ntok, (unsigned)H), nblk(RU_V);
#define RC_NORM(NQ) hipLaunchKernelGGL((ru_split_norm_kernel<NQ>), ngrid, nblk, 0, (hipStream_t)stream,          \
                                       (const float*)cv.onf, (unsigned short*)o, (unsigned char*)oq, (float*)os, \
                                       oq_pitch, os_ld, (const unsigned short*)z, z_pitch, (const float*)nw,    \
                                       neps, nact, H, (const int*)cu, N, cv.cidx, cv.ci_stride)
      if (!oq) RC_NORM(0); else if (oq_i8) RC_NORM(2); else RC_NORM(1);
#undef RC_NORM
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
  }
#endif
#define RC_LAUNCH(NTC, ANCC, SDTC)                                                            \
  do { if (oq_i8) RC_LAUNCH_NM(2, NTC, ANCC, SDTC); else RC_LAUNCH_NM(1, NTC, ANCC, SDTC); }    \
  while (0)
#define RC_LAUNCH_NM(NM, NTC, ANCC, SDTC)                                                     \
  XE_LAUNCH_GGL_GRF256((r4d_gdn_recurrent_update_kernel<1, NM, 1, NTC, ANCC, 0, SDTC, 1, RC_KT>), \
                     grid, blk, 0, (hipStream_t)stream, nullptr, nullptr, nullptr, a, b,       \
                     ab_stride, (const float*)A_log, (const float*)dt_bias,                    \
                     (ru_state<SDTC>::T*)state, st_slot, st_head, (unsigned short*)o,          \
                     (unsigned char*)oq, (float*)os, oq_pitch, os_ld, (const int*)cu,          \
                     (const int*)sidx, sidx_stride, (const int*)nacc,                          \
                     (const unsigned short*)z, z_pitch, (const float*)nw, neps, nact, H, Hg,   \
                     scale, thr, cv)
#define RC_PICKNT(ANCC, SDTC) \
  do { if (nt) RC_LAUNCH(1, ANCC, SDTC); else RC_LAUNCH(0, ANCC, SDTC); } while (0)
#define RC_PICKANC(SDTC) \
  do { if (anc) RC_PICKNT(1, SDTC); else RC_PICKNT(0, SDTC); } while (0)
  if (sdt) RC_PICKANC(1); else RC_PICKANC(0);
#undef RC_PICKANC
#undef RC_PICKNT
#undef RC_LAUNCH
#undef RC_LAUNCH_NM
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h. Operand enums use an OPD_ prefix because RU_K and RU_V
// are kernel constants.
/* ------------------------------------------------------------------ recurrent update */

enum { OPD_RU_Q = 0, OPD_RU_K, OPD_RU_V, OPD_RU_A, OPD_RU_B, OPD_RU_ALOG, OPD_RU_DTB, OPD_RU_STATE, OPD_RU_SIDX, OPD_RU_NACC, OPD_RU_CU,
       OPD_RU_Z, OPD_RU_NORMW, OPD_RU_O, OPD_RU_OQ, OPD_RU_OS };

/* The op's `state_form`: 1 for anchor, 0 for per_candidate (also the default), -1 for a form this
 * kernel does not implement. The pool behind `state` was sized for the declared form. */
static int ru_state_form(const RadArgs* a) {
    const char* sf = r4d_gets(a, "state_form");
    if (!sf || std::strcmp(sf, "per_candidate") == 0) return 0;
    return std::strcmp(sf, "anchor") == 0 ? 1 : -1;
}

extern "C" int r4d_rad_gdn_recurrent_update(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* q     = r4d_opt(a, OPD_RU_Q);
    const RadTensor* k     = r4d_opt(a, OPD_RU_K);
    const RadTensor* v     = r4d_opt(a, OPD_RU_V);
    const RadTensor* ta    = r4d_opt(a, OPD_RU_A);
    const RadTensor* tb    = r4d_opt(a, OPD_RU_B);
    const RadTensor* alog  = r4d_opt(a, OPD_RU_ALOG);
    const RadTensor* dtb   = r4d_opt(a, OPD_RU_DTB);
    const RadTensor* state = r4d_opt(a, OPD_RU_STATE);
    const RadTensor* sidx  = r4d_opt(a, OPD_RU_SIDX);
    const RadTensor* nacc  = r4d_opt(a, OPD_RU_NACC);
    const RadTensor* cu    = r4d_opt(a, OPD_RU_CU);
    const RadTensor* z     = r4d_opt(a, OPD_RU_Z);
    const RadTensor* nw    = r4d_opt(a, OPD_RU_NORMW);
    const RadTensor* o     = r4d_opt(a, OPD_RU_O);
    const RadTensor* oq    = r4d_opt(a, OPD_RU_OQ);
    const RadTensor* os    = r4d_opt(a, OPD_RU_OS);
    if (!q || !k || !v || !ta || !tb || !alog || !dtb || !state || !sidx || !cu || !o)
        return R4D_NO(RAD_E_INVAL);
    if (q->rank != 3 || k->rank != 3 || v->rank != 3 || o->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (state->rank < 2 || sidx->rank < 1 || ta->rank != 2 || tb->rank != 2) return R4D_NO(RAD_E_SHAPE);

    /* N comes from cu (the kernel reads cu[n + 1] for every n in the grid); state_idx is checked
     * against it. */
    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;
    if (sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);

    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0 || r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(o))
        return R4D_NO(RAD_E_STRIDE);

    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    const int anc = ru_state_form(a);
    if (anc < 0) return R4D_NO(RAD_E_UNSUPPORTED);

    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const float thr   = (float)r4d_getf(a, "softplus_thr", 20.0);
    /* Folded gated norm parameters. */
    const float neps  = (float)r4d_getf(a, "eps", 0.0);
    const int   nact  = r4d_act_code(a, "act", "norm_act", 0);
    if (nact < 0) return R4D_NO(RAD_E_INVAL);
    /* The l2 norm epsilon is compiled in, as in r4d_rad_gdn_conv_prep. */
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    /* State width from the cache's dtype, as in the chunk scan. */
    const int   sfp16 = state->dtype == RAD_F16;
    const int   ab_is_bf16 = (ta->dtype == RAD_BF16) ? 1 : 0;
    if (tb->dtype != ta->dtype) return R4D_NO(RAD_E_DTYPE);
    /* Optional n_head_v must agree with the value tensor. */
    const long long nhv = r4d_geti_or(a, "n_head_v", 0);
    if (nhv > 0 && nhv != H) return R4D_NO(RAD_E_SHAPE);

    /* bf16 q/k/v/out/z; f32 or f16 state; fp32 A_log, dt_bias and norm weight. */
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16); R4D_DT(o, RAD_BF16);
    R4D_DT(z, RAD_BF16);
    if (state->dtype != RAD_F32 && state->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(nw, RAD_F32);

    /* The folded quantiser needs oq and os together, and the fused norm: without it a head's rows
     * may be split across workgroups (VS > 1), and a group amax cannot span workgroups. */
    if ((oq != nullptr) != (os != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (oq) {
        if (!z || !nw) return R4D_NO(RAD_E_UNSUPPORTED);
        if (V != 128) return R4D_NO(RAD_E_SHAPE);
        if (oq->dtype != RAD_F8E4M3 && oq->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
        R4D_DT(os, RAD_F32);
        if (!r4d_contig(oq)) return R4D_NO(RAD_E_STRIDE);
    }

    /* z may be a column view of the fused in-projection (rad_block_gdn_fp8.h): tokens are z_row
     * elements apart, not H*V. */
    const long long z_row = z ? gn_row_pitch(z, H * V) : 0;
    if (z && z_row < 0) return R4D_NO(RAD_E_STRIDE);

    /* Slot and head strides come from the tensor, so a cache with padded slots is served too
     * (see the header); the engine's own state cache is dense ([n_states, H, V, K] per layer). */
    return R4D_RAW(ru_update_launch(
        q->data, k->data, v->data, ta->data, tb->data, (long)ab_stride, ab_is_bf16,
        alog->data, dtb->data, state->data,
        (long)state->stride[0], (long)state->stride[1],
        o->data, oq ? oq->data : nullptr, os ? os->data : nullptr,
        oq ? (long)r4d_row_pitch(oq) : 0, os ? (long)r4d_row_pitch(os) : 0,
        cu->data, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, z ? z->data : nullptr, (long)z_row,
        nw ? nw->data : nullptr,
        neps, nact, N, (int)H, (int)Hg, (int)K, (int)V, scale, thr, sfp16, s,
        oq && oq->dtype == RAD_I8 ? 1 : 0, anc != 0));
}


/* ------------------------------------------------------------------ conv + recurrent update */

/* gdn_conv_update's inputs, then gdn_recurrent_update's minus q, k and v (never materialised).
 * Inputs first, outputs last (docs/OPS.md). */
enum { OPD_CR_X = 0, OPD_CR_CW, OPD_CR_CB, OPD_CR_CSTATE, OPD_CR_CIDX, OPD_CR_A, OPD_CR_B,
       OPD_CR_ALOG, OPD_CR_DTB, OPD_CR_STATE, OPD_CR_SIDX, OPD_CR_NACC, OPD_CR_CU, OPD_CR_Z,
       OPD_CR_NORMW, OPD_CR_O, OPD_CR_OQ, OPD_CR_OS };

extern "C" int r4d_rad_gdn_conv_recurrent_update(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x     = r4d_opt(a, OPD_CR_X);
    const RadTensor* cw    = r4d_opt(a, OPD_CR_CW);
    const RadTensor* cb    = r4d_opt(a, OPD_CR_CB);
    const RadTensor* cst   = r4d_opt(a, OPD_CR_CSTATE);
    const RadTensor* cidx  = r4d_opt(a, OPD_CR_CIDX);
    const RadTensor* ta    = r4d_opt(a, OPD_CR_A);
    const RadTensor* tb    = r4d_opt(a, OPD_CR_B);
    const RadTensor* alog  = r4d_opt(a, OPD_CR_ALOG);
    const RadTensor* dtb   = r4d_opt(a, OPD_CR_DTB);
    const RadTensor* state = r4d_opt(a, OPD_CR_STATE);
    const RadTensor* sidx  = r4d_opt(a, OPD_CR_SIDX);
    const RadTensor* nacc  = r4d_opt(a, OPD_CR_NACC);
    const RadTensor* cu    = r4d_opt(a, OPD_CR_CU);
    const RadTensor* z     = r4d_opt(a, OPD_CR_Z);
    const RadTensor* nw    = r4d_opt(a, OPD_CR_NORMW);
    const RadTensor* o     = r4d_opt(a, OPD_CR_O);
    const RadTensor* oq    = r4d_opt(a, OPD_CR_OQ);
    const RadTensor* os    = r4d_opt(a, OPD_CR_OS);
    /* The row norm is required: the fold is the decode shape, one workgroup per (sequence, value
     * head). */
    if (!x || !cw || !cst || !cidx || !ta || !tb || !alog || !dtb || !state || !sidx || !cu ||
        !z || !nw || !o)
        return R4D_NO(RAD_E_INVAL);
    if ((oq != nullptr) != (os != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (x->rank != 2 || cst->rank != 3 || o->rank != 3 || ta->rank != 2 || tb->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    if (state->rank < 2 || sidx->rank < 1 || cidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;
    if (sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);

    /* Head geometry from o ([tokens, H, V]) and x ([q | k | v] columns, 2 Hg K + H V wide). */
    const long long T  = o->shape[0];
    const long long H  = o->shape[1];
    const long long V  = o->shape[2];
    const long long K  = r4d_geti_or(a, "head_k", 0);
    if (K != RU_K || V != RU_V || r4d_geti_or(a, "head_v", 0) != RU_V) return R4D_NO(RAD_E_SHAPE);
    if (x->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    const long long conv_dim = x->shape[1];
    if (H <= 0 || conv_dim <= H * V || (conv_dim - H * V) % (2 * K)) return R4D_NO(RAD_E_SHAPE);
    const long long Hg = (conv_dim - H * V) / (2 * K);
    if (H % Hg) return R4D_NO(RAD_E_SHAPE);
    const long long nhv = r4d_geti_or(a, "n_head_v", 0);
    if (nhv > 0 && nhv != H) return R4D_NO(RAD_E_SHAPE);
    if ((long long)N * Hg > (long long)kMaxConvCtr) return R4D_NO(RAD_E_SHAPE);

    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(cw) || !r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);
    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0 || r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);

    R4D_DT(x, RAD_BF16); R4D_DT(cw, RAD_BF16); R4D_DT(cb, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(ta, RAD_BF16); R4D_DT(tb, RAD_BF16); R4D_DT(o, RAD_BF16); R4D_DT(z, RAD_BF16);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(nw, RAD_F32);
    if (state->dtype != RAD_F32 && state->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (oq) {
        if (oq->dtype != RAD_F8E4M3 && oq->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
        R4D_DT(os, RAD_F32);
        if (!r4d_contig(oq)) return R4D_NO(RAD_E_STRIDE);
    }

    /* The window as in gdn_conv_update: max_query_len = state_len_max - (width - 2), and it must
     * fit RU_CVT, since the fold holds the tokens in registers. */
    if (r4d_geti_or(a, "conv_width", 0) != R4D_GDN_CONV_W) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long mql = cst->shape[2] - (R4D_GDN_CONV_W - 2);
    if (mql < 1) return R4D_NO(RAD_E_SHAPE);
    if (mql > RU_CVT) return R4D_NO(RAD_E_UNSUPPORTED);

    /* The recurrent half's checks, as in r4d_rad_gdn_recurrent_update. */
    const int anc = ru_state_form(a);
    if (anc < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const float thr   = (float)r4d_getf(a, "softplus_thr", 20.0);
    const float neps  = (float)r4d_getf(a, "eps", 0.0);
    const int   nact  = r4d_act_code(a, "act", "norm_act", 0);
    if (nact < 0) return R4D_NO(RAD_E_INVAL);
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long z_row = gn_row_pitch(z, H * V);
    if (z_row < 0) return R4D_NO(RAD_E_STRIDE);

    RuConv cv{};
    cv.x = (const unsigned short*)x->data;  cv.xpitch = (long)xpitch;
    cv.w = (const unsigned short*)cw->data; cv.b = cb ? (const unsigned short*)cb->data : nullptr;
    cv.cs = (unsigned short*)cst->data;
    cv.cs_seq = (long)cst->stride[0]; cv.cs_dim = (long)cst->stride[1];
    cv.cs_tok = (long)cst->stride[2];
    cv.cidx = (const int*)cidx->data; cv.ci_stride = (long)cidx->stride[0];
    cv.ctr = ru_conv_ctr(s, o->data);
    if (!cv.ctr) return R4D_NO(RAD_E_NOMEM);
    /* owner note 138: the split fold's scratch, only when RADIANCE_XE_GDN_SPLIT asks for it */
    static const bool split_asked = [] { const char* e = std::getenv("RADIANCE_XE_GDN_SPLIT"); const int v = e ? std::atoi(e) : 0; return v == 2 || v == 4 || v == 8 || v == 16; }();
    cv.onf = split_asked ? ru_split_onf(s, o->data) : nullptr;
    cv.nctr = split_asked ? ru_split_ctr(s, o->data) : nullptr;   /* run 283, the brief's N1 (RADIANCE_XE_GDN_LAT=7) */

    return ru_conv_recurrent_launch(
        cv, ta->data, tb->data, (long)ab_stride, alog->data, dtb->data, state->data,
        (long)state->stride[0], (long)state->stride[1], o->data, oq ? oq->data : nullptr,
        os ? os->data : nullptr, oq ? (long)r4d_row_pitch(oq) : 0,
        os ? (long)r4d_row_pitch(os) : 0, cu->data, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, z->data, (long)z_row, nw->data, neps, nact, N, (int)H,
        (int)Hg, scale, thr, state->dtype == RAD_F16 ? 1 : 0, s,
        oq && oq->dtype == RAD_I8 ? 1 : 0, anc != 0, (int)T);
}
