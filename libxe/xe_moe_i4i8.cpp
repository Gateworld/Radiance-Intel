/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * xe_moe_i4i8.cpp -- moe_gemm_q at w4g64i8h ("format B", owner notes 131 / 134 item 3): the routed experts as
 * two's-complement int4 codes with a bf16 scale per (row, 64 of K), rotated at convert (fwht128), against int8
 * activations with an f32 scale per (row, 128 of K) -- had_quant_act_i8g's codes, row-major -- on the card's
 * int8 path (DP4A) instead of the E4M3 decode the codebook forms need.
 *
 * The experts are read from their CANONICAL planes, as libref reads them: the codes view one i4 plane [N, K],
 * element 2i in the low nibble of byte i (value (code ^ 8) - 8), the scale view one bf16 plane [N, K/64]. Neither
 * is rearranged at load (no layout hook: a single-plane view is stored as it is).
 *
 * Arithmetic as libref's (ref_moe.cpp, quant_dot): per 64-K group the exact int32 sum of code x activation, then
 * acc += (sum * a_scale(row, k / 128)) * w_scale(row, group). The integer sum is taken on biased codes, u = code ^ 8
 * = value + 8 in 0..15, as dot(u, a) - 8 * sum(a), each a DP4A over four byte lanes (the run-111 probe's form,
 * i4i8_probe.cpp). Rows outside the routing and columns past a narrower class's width are zeros. Experts are two
 * classes by parity (entry e / 2 of class e % 2), as the declaration splits them at one or two ranks; the four-class
 * form (four ranks) is not served here. A sub-group of 32 computes one output, lanes over the weight groups -- the
 * first, plain form; the probe's faster mappings (several rows a team, a group a lane pair) come next.
 *
 * Format C (w4nl64i8h, owner note 147, run 116): the same run form over today's joined w4nl64a8h entries (read as
 * xe_moe_planes.cpp lays them out, its layout hooks on the row), each code an index into its entry's 16 levels
 * rounded to int8 -- see moe_i4i8_run_kernel's FC.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "rad_planes.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>   /* the matrix-unit form (run 164) */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

/* 1: this file compiled again as xe_moe_w4nl_i8.cpp, format C's own translation unit -- its own device image, so the
 * CPU device's JIT of one form never takes the other with it (run 116: the CPU JIT crashed on the joined image) */
#ifndef XE_I4I8_FC
#define XE_I4I8_FC 0
#endif

extern "C" int rad_l0k_device(int index, const sycl::device** dev, const sycl::context** ctx);

namespace {

/* The tail skip (owner note 162 item 1, run 160; Strata's rule, opt-in RADIANCE_XE_TAIL_SKIP=R): a routed expert
 * that is NOT in VRAM (its table entry points into the pinned host pool, read over the link) and sits at rank >= R
 * of the token's top_k is dropped -- its output rows are zeros, so the gather adds nothing for it -- instead of
 * being read over PCIe; a run whose live rows are all dropped skips the weight reads entirely. The weights are not
 * renormalised: with expert parallelism each rank sees only its own experts' residency, and a per-rank
 * renormalisation would make the two ranks' partial sums inconsistent (the owner's "or say why not").
 *
 * Residency on the device: a table entry's address. On this platform device USM allocations and host USM
 * allocations lie on different sides of 2^47 (the probe at init says which side is the device; the feature refuses
 * itself when they do not split). The stats (device memory, read once a step through xe_ar2_stalls): pairs by rank
 * off the device [0, 16), dropped [16, 32), live [32, 48); runs skipped whole [48], runs with an off-device expert
 * [49]. Both GEMMs of a layer count, so a layer's pairs appear twice. */
struct TskipInst {
    int                  rank = -1;
    uint64_t             cfg = 0;          /* 0 off; else R | (device memory above 2^47 ? 256 : 0) */
    uint32_t*            stats = nullptr;  /* 64 counters, device memory */
    const sycl::device*  dev = nullptr;
    const sycl::context* ctx = nullptr;
    uint32_t             last[64] = {};
    long                 calls = 0;
    const void*          keys[8] = {};     /* the all-reduce instances that report: calls / keys ~ steps */
    int                  nkeys = 0;
};
constexpr unsigned kTskipSplit = 47;

struct ICls {
    const uint8_t* base;     /* the stacked codes plane, or null */
    const void* const* tab;  /* the pointer table, or null */
    uint64_t stride;         /* bytes an entry of a stacked plane */
    const uint8_t* sbase;
    const void* const* stab;
    uint64_t sstride;
    uint32_t n, k;
    uint32_t w_ld, s_ld;     /* row pitches: bytes of codes, bf16 elements of scales (format C: E4M3 bytes) */
    /* format C: the joined entries' plane offsets (codes, table; scale, scale.1); 0 at format B */
    uint32_t c_off, t_off, s_off, s1_off;
};

inline const uint8_t* entry_of(const uint8_t* base, const void* const* tab, uint64_t stride, uint32_t e) {
    return tab ? (const uint8_t*)tab[e] : base + (uint64_t)e * stride;
}
inline float e4m3_to_f32(uint32_t v) {
    const uint32_t s = (v & 0x80u) << 24, ex = (v >> 3) & 15u, m = v & 7u;
    if (ex == 15u && m == 7u) return __uint_as_float(s | 0x7fc00000u);
    if (ex == 0u) return __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));   /* m x 2^-9 */
    return __uint_as_float(s | ((ex + 120u) << 23) | (m << 20));
}
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}

/* sum of the four signed byte products, plus acc: DP4A's arithmetic. sycl/ext/oneapi/dot_product.hpp defines its
 * dot_acc out of line, so a second translation unit including it (xe_gemv_i8.cpp does) fails to link; IGC
 * recognises this form. */
inline int32_t dp4(int32_t a, int32_t b, int32_t acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}

/* Eight activation bytes a0..a7 (lo = a0..a3, hi = a4..a7) as their even and odd lanes, to meet the codes of
 * elements 8d .. 8d + 7 (byte b: element 2b low nibble, 2b + 1 high). */
inline int32_t a_even(uint32_t lo, uint32_t hi) {
    return (int32_t)((lo & 0xFFu) | ((lo >> 8) & 0xFF00u) | ((hi & 0xFFu) << 16) | ((hi << 8) & 0xFF000000u));
}
inline int32_t a_odd(uint32_t lo, uint32_t hi) {
    return (int32_t)(((lo >> 8) & 0xFFu) | ((lo >> 16) & 0xFF00u) | ((hi << 8) & 0xFF0000u) | (hi & 0xFF000000u));
}

void moe_i4i8_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                     const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                     uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                     uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    const int lane = (int)threadIdx.x & 31;
    const uint32_t n = blockIdx.x * 8u + ((uint32_t)threadIdx.x >> 5);
    const uint32_t i = blockIdx.y;
    if (n >= N || i >= T) return;   /* whole sub-groups: no barrier below */
    const int32_t flat = sorted[i];
    int32_t e = -1;
    uint32_t tok = 0;
    if (flat >= 0) {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        tok = a_sorted ? i : (uint32_t)flat / top_k;
        if (tok < a_rows) e = (int32_t)lo;
    }
    const uint32_t fy = N / parts, pp = n / fy, j = n - pp * fy;
    const ICls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    float acc = 0.0f;
    if (e >= 0 && j < fc) {
        const uint32_t entry = (uint32_t)e >> 1, r = pp * fc + j;
        const uint8_t* wrow = entry_of(c.base, c.tab, c.stride, entry) + (size_t)r * c.w_ld;
        const uint16_t* srow = (const uint16_t*)entry_of(c.sbase, c.stab, c.sstride, entry) + (size_t)r * c.s_ld;
        const int8_t* arow = A + (size_t)tok * a_ld;
        for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
            const uint32_t* wd = (const uint32_t*)(wrow + (size_t)g * 32);   /* 8 dwords, 64 codes */
            const uint32_t* ad = (const uint32_t*)(arow + (size_t)g * 64);   /* 16 dwords, 64 activations */
            int32_t s = 0, sa = 0;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const uint32_t u = wd[d] ^ 0x88888888u, lo = ad[2 * d], hi = ad[2 * d + 1];
                s = dp4((int32_t)(u & 0x0F0F0F0Fu), a_even(lo, hi), s);
                s = dp4((int32_t)((u >> 4) & 0x0F0F0F0Fu), a_odd(lo, hi), s);
                sa = dp4((int32_t)lo, (int32_t)0x01010101, sa);
                sa = dp4((int32_t)hi, (int32_t)0x01010101, sa);
            }
            const int32_t isum = s - 8 * sa;
            const float fa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
            const float fb = __uint_as_float((uint32_t)srow[g] << 16);
            acc += ((float)isum * fa) * fb;
        }
        for (int x = 16; x >= 1; x >>= 1) acc += __shfl_xor(acc, x);
    }
    if (lane == 0) Y[(size_t)i * y_ld + n] = to_bf16(acc);
}

/* The run form (moe_planes_v2m's shape, xe_moe_planes.cpp, at format B; the probe's L16 / L8 mappings): the
 * work-group whose row starts a run of up to MT routed rows of one expert stages the run's activations and writes
 * them all, so each weight chunk is loaded once for the run; the run's other work-groups return, and a padding row
 * writes its zeros. A team of LPR lanes owns R output columns; lane l loads 16-byte chunks (32 codes) l + u * LPR.
 * The activations are staged as (even, odd) dword pairs a block of eight, so a canonical code dword (byte b: element
 * 2b low, 2b + 1 high) meets its two activation dwords directly; a group's two chunks sit on lanes l and l ^ 1 and
 * their exact sums join by one xor (as floats: |sum| < 2^24) before the group's -8 sum(a) and libref's scaling.
 * GL (a group a lane, the probe's GL2R2 for down): lane l owns groups l + u * LPR whole, both chunks, so no join. */
constexpr int kMT = 4, kWG = 512;
constexpr uint32_t kMaxK = 4096;
/* FC (format C, w4nl64i8h, owner note 147): the codes are indices into the entry's 16 f32 levels, which libref
 * (ref_moe.cpp) rounds to int8 integers at the entry's amax, nearbyint(level * 127 / amax), moving amax / 127 into
 * the weight scale: e4m3(scale) * (scale.1 * (amax / 127)). The work-group builds a 256-entry table, a code byte
 * (low code, high code) to its two int8 levels (the probe's C-lds), and the sums need no -8 sum(a) term. */
template <int UPL, int LPR, int R, bool GL, bool FC = false>
void moe_i4i8_run_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                         const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                         uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                         uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows,
                         uint64_t sk_cfg, uint32_t* __restrict__ sk_stats, uint32_t bound) {
    XE_SHARED(uint4, s_a, kMT * (kMaxK / 16));     /* (even, odd) dwords a block of eight; a chunk = two uint4 */
    /* each group's activation sum (exact); FC, which needs none: the 256-entry table, a byte -> two levels (bits).
     * (A separate FC-only array, declared and unused in format B's instantiation, crashed the CPU device: run 116.) */
    XE_SHARED(float, s_g, kMT * (kMaxK / 64));
    static_assert(kMT * (kMaxK / 64) >= 256, "FC's table lives in s_g");
    XE_SHARED(float, s_as, kMT * (kMaxK / 128));   /* the per-128 scales */
    const int tid = (int)threadIdx.x;
    const uint32_t i = blockIdx.y, nb = blockIdx.x;
    if (i >= T) return;
    const int32_t flat = sorted[i];
    int32_t e = -1;
    /* bound (run 202, RADIANCE_XE_I4I8_RUN_BOUND=1; run 108's MX_BOUND for this form): a row at or past off[n_expert]
     * is not this GEMM's -- the protected experts' live rows sort there, and unbounded such a row can lead a run of
     * expert n_expert - 1 with rn = off[n_expert] - i wrapped to kMT (the wrong weights, up to 3 rows past T); bounded
     * it is a padding row, zeroed, as the arch expects ("zeroed their sorted rows as past its live count") */
    if (flat >= 0 && !(bound && i >= (uint32_t)off[n_expert])) {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        e = (int32_t)lo;
    }
    uint32_t rn = 1;
    if (e >= 0) {
        const uint32_t first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % kMT != 0) return;
        rn = (last - i) < (uint32_t)kMT ? (last - i) : (uint32_t)kMT;
    }
    uint32_t tokr[kMT], rankr[kMT];
    bool liver[kMT], skipr[kMT];
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        tokr[q] = 0; liver[q] = false; rankr[q] = 0; skipr[q] = false;
        if (e >= 0 && (uint32_t)q < rn) {
            const int32_t fq = sorted[i + q];
            const uint32_t t = a_sorted ? i + q : (uint32_t)fq / top_k;
            tokr[q] = t;
            liver[q] = fq >= 0 && t < a_rows;
            rankr[q] = fq >= 0 ? (uint32_t)fq % top_k : 0u;   /* the slot's rank in its token's top_k */
        }
    }
    const ICls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
    /* The tail skip (TskipInst): the expert's table entry off the device, and a live row's rank at or past R, drops
     * the row; all of the run's live rows dropped skips the reads. Uniform over the work-group (one entry, one run). */
    bool all_skip = false;
    if (sk_cfg && e >= 0 && c.tab) {
        const uintptr_t wp = (uintptr_t)c.tab[(uint32_t)e >> 1];
        const bool high = (wp >> kTskipSplit) != 0, dev_high = (sk_cfg & 256u) != 0;
        const bool offdev = wp != 0 && high != dev_high;
        const uint32_t Rk = (uint32_t)(sk_cfg & 255u);
        bool any_live = false;
        all_skip = offdev;
#pragma unroll
        for (int q = 0; q < kMT; ++q) {
            if ((uint32_t)q >= rn || !liver[q]) continue;
            any_live = true;
            skipr[q] = offdev && rankr[q] >= Rk;
            if (!skipr[q]) all_skip = false;
        }
        if (!any_live) all_skip = false;   /* nothing to drop: the plain path writes the zeros */
        if (sk_stats && nb == 0 && tid == 0) {
#pragma unroll
            for (int q = 0; q < kMT; ++q) {
                if ((uint32_t)q >= rn || !liver[q]) continue;
                const uint32_t rk = rankr[q] < 16u ? rankr[q] : 15u;
                atomicAdd(&sk_stats[32 + rk], 1u);
                if (offdev) atomicAdd(&sk_stats[rk], 1u);
                if (skipr[q]) atomicAdd(&sk_stats[16 + rk], 1u);
            }
            if (any_live && offdev) atomicAdd(&sk_stats[49], 1u);
            if (all_skip) atomicAdd(&sk_stats[48], 1u);
        }
    }
    /* A work-group whose columns all fall past this class's width, in every part they touch, writes its zeros and
     * returns before staging (run 117; IP9, run 116: at the two-card split the smaller gate_up class's 512 of 768
     * columns left a third of its work-groups staging K 2560 for nothing). Uniform over the work-group: no barrier
     * is skipped by part of it. A run the tail skip drops whole takes the same exit. */
    if (e >= 0) {
        const uint32_t cpw = (uint32_t)((kWG / LPR) * R), n0 = nb * cpw, n1 = n0 + cpw < N ? n0 + cpw : N;
        const uint32_t a = fy ? n0 % fy : 0u;
        const bool live = n0 < N && (n1 - n0 >= fy || a < fc || n0 + (fy - a) < n1);
        if (!live || all_skip) {
            const int team = tid / LPR, l = tid % LPR;
            if (l == 0)
                for (int q = 0; q < kMT; ++q) {
                    if ((uint32_t)q >= rn) break;
                    for (int r = 0; r < R; ++r) {
                        const uint32_t n = n0 + (uint32_t)(team * R + r);
                        if (n < N) Y[(size_t)(i + q) * y_ld + n] = 0;
                    }
                }
            return;
        }
    }
    if (e >= 0) {
#pragma unroll
        for (int q = 0; q < kMT; ++q) {
            if ((uint32_t)q >= rn || !liver[q]) continue;
            const uint32_t* arow = (const uint32_t*)(A + (size_t)tokr[q] * a_ld);
            /* Each group's activation sum in the same pass: eight lanes hold a group's 16 dwords (K % 64 == 0, so
             * they are live together), joined by three xors. The trip count is the work-group's, so every lane
             * reaches the shuffles. (It was a second pass over global memory, 16 dependent loads a group, in every
             * column block's work-group: run 114.) */
            for (uint32_t b0 = 0; b0 < K / 8u; b0 += kWG) {
                const uint32_t b = b0 + (uint32_t)tid;
                const bool in = b < K / 8u;
                const uint32_t lo = in ? arow[2 * b] : 0u, hi = in ? arow[2 * b + 1] : 0u;
                if (in) {
                    uint4& d = s_a[q * (kMaxK / 16) + b / 2u];   /* dwords 2b, 2b + 1: half of a uint4 */
                    if (b & 1u) { d.z = (uint32_t)a_even(lo, hi); d.w = (uint32_t)a_odd(lo, hi); }
                    else        { d.x = (uint32_t)a_even(lo, hi); d.y = (uint32_t)a_odd(lo, hi); }
                }
                int32_t sa = dp4((int32_t)hi, (int32_t)0x01010101, dp4((int32_t)lo, (int32_t)0x01010101, 0));
                sa += __shfl_xor(sa, 4);
                sa += __shfl_xor(sa, 2);
                sa += __shfl_xor(sa, 1);
                if (!FC && in && (b & 7u) == 0) s_g[q * (kMaxK / 64) + b / 8u] = (float)sa;
            }
            for (uint32_t g = (uint32_t)tid; g < K / 128u; g += kWG)
                s_as[q * (kMaxK / 128) + g] = As[(size_t)tokr[q] * as_ld + g];
        }
        if constexpr (FC) {
            if (tid < 256) {
                const float* tab = (const float*)(entry_of(c.base, c.tab, c.stride, (uint32_t)e >> 1) + c.t_off);
                float amax = 0.0f;
#pragma unroll
                for (int t = 0; t < 16; ++t) amax = sycl::fmax(amax, sycl::fabs(tab[t]));
                const float l0 = tab[tid & 15], l1 = tab[tid >> 4];
                const int i0 = amax > 0.0f ? (int)sycl::rint(l0 * 127.0f / amax) : (int)l0;
                const int i1 = amax > 0.0f ? (int)sycl::rint(l1 * 127.0f / amax) : (int)l1;
                /* 0x40000000: a normal float's bits, so no denormal flush can touch them (reads take the low 16) */
                s_g[tid] = __uint_as_float(0x40000000u | ((uint32_t)i0 & 255u) | (((uint32_t)i1 & 255u) << 8));
            }
        }
    }
    __syncthreads();
    const int team = tid / LPR, l = tid % LPR;
    float acc[kMT][R];
    uint32_t rowr[R];
    bool rowok[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[r] = e >= 0 && n < N && j < fc;
        rowr[r] = pp * fc + j;
#pragma unroll
        for (int q = 0; q < kMT; ++q) acc[q][r] = 0.0f;
    }
    if (e >= 0) {
        const uint32_t CPR = K / 32u;
        const uint32_t entry = (uint32_t)e >> 1;
        const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry) + c.c_off;
        const uint8_t* se8 = entry_of(c.sbase, c.stab, c.sstride, entry) + c.s_off;
        const uint16_t* se = (const uint16_t*)se8;
        float smul = 1.0f;   /* FC: scale.1 x amax / 127, as libref folds it */
        if constexpr (FC) {
            const float* tab = (const float*)(entry_of(c.base, c.tab, c.stride, entry) + c.t_off);
            float amax = 0.0f;
#pragma unroll
            for (int t = 0; t < 16; ++t) amax = sycl::fmax(amax, sycl::fabs(tab[t]));
            const float s1 = *(const float*)(se8 - c.s_off + c.s1_off);
            smul = amax > 0.0f ? s1 * (amax / 127.0f) : s1;
        }
        constexpr int CH = GL ? 2 : 1;   /* chunks a unit */
        uint4 w[UPL][R][CH];
#pragma unroll
        for (int u = 0; u < UPL; ++u)
#pragma unroll
            for (int ch = 0; ch < CH; ++ch) {
                const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) + (uint32_t)ch : (uint32_t)l + (uint32_t)u * LPR;
#pragma unroll
                for (int r = 0; r < R; ++r)
                    w[u][r][ch] = (jc < CPR && rowok[r]) ? *(const uint4*)(we + (size_t)rowr[r] * c.w_ld + jc * 16u)
                                                         : uint4{0, 0, 0, 0};
            }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            /* the unit's first chunk; GL: its group's first */
            const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) : (uint32_t)l + (uint32_t)u * LPR;
            /* No break: the join below is a sub-group shuffle every lane must reach. A chunk past the row (both of a
             * group's, CPR being even) computes on zero codes and adds nothing; its scales are not read. */
            const bool live = jc < CPR;
            const uint32_t g = live ? jc / 2u : 0u;
            float sbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if constexpr (FC)
                    sbr[r] = rowok[r] && live ? e4m3_to_f32(se8[(size_t)rowr[r] * c.s_ld + g]) * smul : 0.0f;
                else
                    sbr[r] = rowok[r] && live ? __uint_as_float((uint32_t)se[(size_t)rowr[r] * c.s_ld + g] << 16) : 0.0f;
            }
            /* FC: the unit's codes as their int8 levels once, for all its rows (run 117: the lookups, LDS gathers,
             * are the cost -- IC2's wider u64 tables were slower -- and the verify had repeated them a row) */
            uint32_t lv[FC ? R : 1][FC ? CH : 1][8];
            if constexpr (FC) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int ch = 0; ch < CH; ++ch) {
                        const uint32_t d[4] = { w[u][r][ch].x, w[u][r][ch].y, w[u][r][ch].z, w[u][r][ch].w };
#pragma unroll
                        for (int x = 0; x < 4; ++x) {   /* a code byte (low code, high code) to its two levels */
                            uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                            for (int bb = 0; bb < 4; ++bb) {
                                const uint32_t ev = __float_as_uint(s_g[(d[x] >> (8 * bb)) & 255u]);
                                lo8 |= (ev & 255u) << (8 * bb);
                                hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                            }
                            lv[r][ch][2 * x] = lo8;
                            lv[r][ch][2 * x + 1] = hi8;
                        }
                    }
            }
#pragma unroll
            for (int q = 0; q < kMT; ++q) {
                if ((uint32_t)q >= rn) break;   /* uniform over the work-group */
                const float asc = s_as[q * (kMaxK / 128) + g / 2u], gs = s_g[q * (kMaxK / 64) + g];
                /* 16-byte loads, unpredicated: a chunk past the row reads inside s_a (jc < kMaxK / 32 for every
                 * instance: 16 lanes x 8 units, or GL's 2 x 4 groups) and its sum is never accumulated */
                uint32_t a[8 * CH];
#pragma unroll
                for (int v = 0; v < 2 * CH; ++v) {
                    const uint4 x = s_a[q * (kMaxK / 16) + jc * 2u + (uint32_t)v];
                    a[4 * v] = x.x; a[4 * v + 1] = x.y; a[4 * v + 2] = x.z; a[4 * v + 3] = x.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int32_t s = 0;
#pragma unroll
                    for (int ch = 0; ch < CH; ++ch) {
                        const uint32_t d[4] = { w[u][r][ch].x, w[u][r][ch].y, w[u][r][ch].z, w[u][r][ch].w };
#pragma unroll
                        for (int x = 0; x < 4; ++x) {
                            if constexpr (FC) {
                                s = dp4((int32_t)lv[r][ch][2 * x], (int32_t)a[8 * ch + 2 * x], s);
                                s = dp4((int32_t)lv[r][ch][2 * x + 1], (int32_t)a[8 * ch + 2 * x + 1], s);
                            } else {
                                const uint32_t uu = d[x] ^ 0x88888888u;
                                s = dp4((int32_t)(uu & 0x0F0F0F0Fu), (int32_t)a[8 * ch + 2 * x], s);
                                s = dp4((int32_t)((uu >> 4) & 0x0F0F0F0Fu), (int32_t)a[8 * ch + 2 * x + 1], s);
                            }
                        }
                    }
                    float fs = (float)s;
                    if constexpr (!GL) fs += __shfl_xor(fs, 1);   /* the group's other chunk */
                    if constexpr (!FC) fs -= 8.0f * gs;
                    if ((GL || (l & 1) == 0) && rowok[r] && live) acc[q][r] += (fs * asc) * sbr[r];
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        if ((uint32_t)q >= rn) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[q][r];   /* not GL: the even lanes' sums (the odd lanes hold 0) */
            if constexpr (LPR == 16) v += __shfl_xor(v, 8);
            if constexpr (LPR >= 8) v += __shfl_xor(v, 4);
            if constexpr (LPR >= 4) v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] && !skipr[q] ? v : 0.0f);
        }
    }
}

#if XE_I4I8_FC
/* The matrix-unit form at prefill (owner note 168 item 2, run 164; RADIANCE_XE_I4I8_DPAS=1, format C only): the
 * run form's 4-row runs read every expert's weights once per 4 rows, which at a 4,096-row chunk is the prefill's
 * largest cost (B63: 5.5 s a rank of the 7K; B64: the f16 matrix form 2.8 s faster). Here a work-group of kDWaves
 * sub-groups (SG16) owns one expert's run of up to kDRows sorted rows (8 a sub-group) and kDCols output columns
 * (4 tiles of 16). Per 64-K group it stages the run's activations (8 rows x 64 int8 a sub-group, zeros past the
 * run and for a dead row) and the columns' codes as their int8 levels in the packed B layout (the trunk's
 * gemm_i8a8_dpas_kernel's), runs the two 32-deep joint_matrix steps into exact int32 tiles, then scales each tile's
 * (m, n) by the row's activation scale (per 128) and the column's E4M3 scale x smul (per 64), accumulating in fp32
 * in ascending g: the run kernel's exact per-group sums and its (sum x asc) x sbr order, so the same bits. Rows
 * past the run write nothing; a padding row (sorted < 0) writes zeros, as the run form does. */
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int kDWaves = 8, kDThreads = 16 * kDWaves, kDRows = 8 * kDWaves, kDCols = 64;
constexpr int kDTileB = 8 * 16 * 4;   /* one packed B tile: 32 k x 16 n int8 */
constexpr uint32_t kMapTailI4 = 0xffffffffu;   /* a block map entry of padding rows (owner note 195) */

/* Run 171: the work-group's shape is a template -- W sub-groups of RT rows each. <8, 8> is the form since run 164
 * (64 rows); <8, 16> (RADIANCE_XE_I4I8_DPAS_ROWS=16, launched with the large register file) gives a sub-group two
 * 8-row A tiles, so each B tile load feeds two MMAs and the staging, decode and barriers are paid once per 128
 * rows. (W = 16, the 256-thread work-group with the B staging split in halves, ran +15 % and not byte-exact in
 * B119 and is not instantiated; its code path stays generic in W.) */
/* Run 175 (owner note 176 item 3, RADIANCE_XE_I4I8_DPAS_PRE=1): `pre`, when not null, holds every routed expert's B
 * words already decoded by moe_i4i8_dec_kernel below -- word (step, tile, k4 % 8, n % 16) of (expert, 64-column block,
 * 64-K group) at pre[((e * pnb + nb) * png + g) * 1024 + ...], the same words this kernel's staging computes -- and
 * the staging copies them instead of loading and decoding the codes: the shared tiles hold the same bytes, so the
 * bits are this form's. */
template <int W, int RT>
void moe_i4i8_dpas_kernel_t(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                            const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                            uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                            uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows, uint32_t slmc,
                            const uint32_t* __restrict__ pre, uint32_t pnb, uint32_t png,
                            const uint32_t* __restrict__ map) {
    constexpr int kW = W, kT = 16 * W, kR = RT * W;  /* sub-groups, threads, rows a work-group */
    constexpr int kJ = 64 / W;                         /* B words a sub-group a group: 8 at W = 8 */
    constexpr int kRT = RT / 8;                        /* 8-row A tiles a sub-group */
    constexpr int kSw = kR * 64 / 16 / kT;             /* A sweeps a group: 16-byte pieces a thread */
    XE_SHARED(int8_t, sB, 2 * 4 * kDTileB);      /* [step 2][tile 4][packed 32 x 16]: one 64-K group */
    XE_SHARED(int8_t, sA, kR * 64);              /* the run's rows x this group's 64 K */
    XE_SHARED(int32_t, sC, kW * 8 * 16);         /* one 8 x 16 int32 tile a sub-group */
    const int tid = (int)threadIdx.x, wave = tid / 16, lane = tid % 16;
    const uint32_t nb = blockIdx.x, n0 = nb * (uint32_t)kDCols;
    if (n0 >= N) return;
    uint32_t i = 0, rn = 0;
    int32_t e = -1;
    if (map) {
        /* owner note 195 (run 202, RADIANCE_XE_I4I8_DPAS_MAP=1): blockIdx.y is a map entry -- an (expert, block start)
         * from moe_i4i8_map_kernel, or a tail block of padding rows -- so only the blocks that compute are launched,
         * not one work-group a sorted row that looks itself up and leaves unless it starts a block (63 in 64 did) */
        const uint32_t y = blockIdx.y;
        if (y >= map[0]) return;
        i = map[2 + 2 * y];
        if (map[1 + 2 * y] == kMapTailI4) {   /* padding rows i .. i + kR: their zeros, as the run form writes them */
            for (uint32_t r = i; r < i + (uint32_t)kR && r < T; ++r)
                for (uint32_t n = n0 + (uint32_t)tid; n < n0 + (uint32_t)kDCols && n < N; n += (uint32_t)kT)
                    Y[(size_t)r * y_ld + n] = 0;
            return;
        }
        e = (int32_t)map[1 + 2 * y];
        const uint32_t last = (uint32_t)off[e + 1];
        rn = (last - i) < (uint32_t)kR ? (last - i) : (uint32_t)kR;
    } else {
        i = blockIdx.y;
        if (i >= T) return;
        const int32_t flat0 = sorted[i];
        /* slmc == 6 (run 202, a test): a row past this call's live count is a padding row, as the map treats it. Without
         * it a row there whose slot is real (the protected experts' rows, which follow the regular ones in the sort)
         * searches to the last regular expert, and rn = last - i wraps to kR: a block of the wrong expert's weights,
         * written up to kR - 1 rows past it */
        if (slmc == 6u && i >= (uint32_t)off[n_expert]) {
            for (uint32_t n = n0 + (uint32_t)tid; n < n0 + (uint32_t)kDCols && n < N; n += (uint32_t)kT)
                Y[(size_t)i * y_ld + n] = 0;
            return;
        }
        if (flat0 >= 0) {
            uint32_t lo = 0, hi = n_expert;
            while (lo + 1 < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
            }
            e = (int32_t)lo;
        }
        if (e < 0) {   /* a padding row: its zeros over this work-group's columns, as the run form writes them */
            for (uint32_t n = n0 + (uint32_t)tid; n < n0 + (uint32_t)kDCols && n < N; n += (uint32_t)kT)
                Y[(size_t)i * y_ld + n] = 0;
            return;
        }
        const uint32_t first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % (uint32_t)kR != 0) return;
        rn = (last - i) < (uint32_t)kR ? (last - i) : (uint32_t)kR;
    }
    const ICls& c = (e & 1) ? c1 : c0;
    const uint32_t fy = N / parts, fcw = c.n / parts, K = c.k;
    const uint32_t entry = (uint32_t)e >> 1;
    const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry) + c.c_off;
    const uint8_t* se8 = entry_of(c.sbase, c.stab, c.sstride, entry) + c.s_off;
    /* the level table and smul, as the run form forms them. Run 166 (owner note 172 a): the 16 int8 levels live
     * one per lane (level `lane`) and a code's level is a sub-group shuffle, not an LDS table read -- B90 counted
     * 3.3e10 bank conflicts a 7K on this kernel, the random-code table reads and the strided B stores among them;
     * the levels are the same integers, so the same bits. */
    const float* tab = (const float*)(entry_of(c.base, c.tab, c.stride, entry) + c.t_off);
    float amax = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; ++t) amax = sycl::fmax(amax, sycl::fabs(tab[t]));
    const float lv = tab[lane];
    const uint32_t lvi = (uint32_t)(amax > 0.0f ? (int)sycl::rint(lv * 127.0f / amax) : (int)lv) & 255u;
    /* the 16 levels gathered once into four byte-packed words (B92: a per-code shuffle is lowered through the
     * load/store path -- 7.4e10 cache accesses, +20 % a launch -- so the per-code lookup is a select chain) */
    sycl::sub_group sg = xe::sg();
    uint32_t pk[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j) pk[k] |= sycl::select_from_group(sg, lvi, 4 * k + j) << (8 * j);
    /* slmc == 2 (run 170, test only -- WRONG numbers): the level decode skipped (lvl(c) = c), to time its share;
     * == 3: the scaling's loads skipped (acc += C); == 4: the B staging skipped (stale shared memory); == 5: the
     * MMAs and their tile loads skipped (C stays 0) -- the pipeline's shares by ablation, every one wrong by design */
    const bool nodec = slmc == 2u, noscale = slmc == 3u, nostage = slmc == 4u, nomma = slmc == 5u;
    auto lvl = [&](uint32_t cc) -> uint32_t {
        if (nodec) return cc;
        const uint32_t d = (cc & 8u) ? ((cc & 4u) ? pk[3] : pk[2]) : ((cc & 4u) ? pk[1] : pk[0]);
        return (d >> (8u * (cc & 3u))) & 255u;
    };
    const float s1 = *(const float*)(se8 - c.s_off + c.s1_off);
    const float smul = amax > 0.0f ? s1 * (amax / 127.0f) : s1;
    /* this sub-group's RT rows: the activation row and whether it is live */
    uint32_t tokr[RT];
    bool liver[RT];
#pragma unroll
    for (int q = 0; q < RT; ++q) {
        const uint32_t qq = (uint32_t)(wave * RT + q);
        tokr[q] = 0; liver[q] = false;
        if (qq < rn) {
            const int32_t fq = sorted[i + qq];
            const uint32_t t = a_sorted ? i + qq : (uint32_t)fq / top_k;
            tokr[q] = t;
            liver[q] = fq >= 0 && t < a_rows;
        }
    }
    /* this lane's column in each tile: the weight row and whether the column is live (the parts split) */
    uint32_t rowr[4];
    bool rowok[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t n = n0 + (uint32_t)(t * 16 + lane);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[t] = n < N && j < fcw;
        rowr[t] = pp * fcw + j;
    }
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> Am[kRT];
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> Bt;
    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C[kRT][4];
    /* The float order: every group's exact int32 sum, scaled as the run form scales it, into one accumulator in
     * ascending g. The run form's team of 16 lanes holds eight partial sums by g mod 8 joined by an xor tree, so
     * the last bits of a sum can differ (B66: the 7K's first-token logprob 0.006 off, the text the same); a sweep
     * in the tree's order with three partial banks (B68) cost 60 % more time and a fold bug, so the form is judged
     * by note 149's bar (the KL replay), as the owner's note says, not by the bits. */
    float acc[kRT][4][8];
#pragma unroll
    for (int r = 0; r < kRT; ++r)
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int q = 0; q < 8; ++q) acc[r][t][q] = 0.0f;
    uint32_t* sBw = (uint32_t*)sB;
    int32_t* sCw = sC + wave * 128;
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
    const uint32_t ng = K / 64u;
    /* Run 171: the group's global loads issued together. Run 170's ablation table (B111 / B113-B116: no decode
     * −0.59 s of the 7K, no scale loads −0.36, no B staging −1.15, no MMAs −0.53, a ~1.3 s remainder) reads as a
     * latency chain a group -- global load, decode, store, barrier, MMAs, global scale loads, FMAs, barrier --
     * that occupancy does not hide. B117 (image 200) double-buffered the next group's loads across the barrier:
     * exact but +5 % (the barrier's fence covered global memory, so the loads were waited for at it anyway, and
     * the copies cost registers). Here the two A pieces, the code bytes, the 4 scale bytes and the 8 activation
     * scales of THIS group are loaded back to back at the group's top before any use (one exposed latency a
     * group instead of three in series), the barriers fence shared memory only (xe::syncthreads_local: the
     * threads exchange nothing but shared memory between them), and the A rows' sorted lookups move out of the
     * loop. The same loads, the same arithmetic in the same order: the bits are the run-170 form's (B111). */
    /* the A sweeps' rows: this thread's piece of row (sw * kT / 4 + tid / 4) -- the token row, or ~0u when dead */
    uint32_t trow[kSw];
#pragma unroll
    for (int sw = 0; sw < kSw; ++sw) {
        const int q = sw * (kT / 4) + tid / 4;
        const uint32_t qw = (uint32_t)q;
        trow[sw] = ~0u;
        if (qw < rn) {
            const int32_t fq = sorted[i + qw];
            const uint32_t t = a_sorted ? i + qw : (uint32_t)fq / top_k;
            if (fq >= 0 && t < a_rows) trow[sw] = t;
        }
    }
    const size_t apiece = (size_t)(tid % 4) * 16u;
    /* this sub-group's B tile, step and word range: W = 8 -> (w / 2, w % 2, words 0-7); W = 16 -> sub-groups w and
     * w + 8 share (tile, step) and take words 0-3 / 4-7 (code bytes 0-7 / 8-15 of the step's 16) */
    const int bt = (wave & 7) >> 1, bstep = wave & 1, jbase = (wave >> 3) * kJ;
    const uint8_t* cbase = (rowok[bt] && !nostage && !pre)
        ? we + (size_t)rowr[bt] * c.w_ld + (size_t)bstep * 16u + (size_t)jbase * 2u : nullptr;
    /* the pre-decoded words of this expert and column block (run 175), and this lane's first word in a group's 1024 */
    const uint32_t* pblk = pre ? pre + ((size_t)e * pnb + nb) * (size_t)png * 1024u : nullptr;
    const uint32_t pw0 = (uint32_t)((bstep * 4 + bt) * 128 + jbase * 16 + lane);
    for (uint32_t g = 0; g < ng; ++g) {
        __syncthreads();                                  /* the previous group's tiles are consumed */
        /* the group's loads, all in flight together (B118: exact, neutral) */
        uint4 v[kSw];
        uint32_t cw[4];
        uint8_t sr[4];
#pragma unroll
        for (int sw = 0; sw < kSw; ++sw) {
            uint4 z = {0, 0, 0, 0};
            v[sw] = trow[sw] != ~0u ? *(const uint4*)(A + (size_t)trow[sw] * a_ld + (size_t)g * 64u + apiece) : z;
        }
        cw[0] = 0u; cw[1] = 0u; cw[2] = 0u; cw[3] = 0u;
        if (cbase) __builtin_memcpy(cw, cbase + (size_t)g * 32u, kJ * 2);
#pragma unroll
        for (int t = 0; t < 4; ++t) sr[t] = rowok[t] ? se8[(size_t)rowr[t] * c.s_ld + g] : (uint8_t)0;
        /* the A tile: kR rows x 64 bytes as kSw sweeps, a thread's 16 bytes at (sweep * kT * 16 + tid * 16): a
         * sub-group's 16 stores are one 256-byte run (run 166: the old row / half split put lanes 32 bytes apart,
         * two to a bank) */
#pragma unroll
        for (int sw = 0; sw < kSw; ++sw) {
            const int q = sw * (kT / 4) + tid / 4, piece = tid % 4;
            *(uint4*)(sA + q * 64 + piece * 16) = v[sw];
        }
        /* the B tiles: 64 columns x 16 words of 4 levels; word (step, tile, k4 % 8, n % 16) as the trunk's layout.
         * A lane is one column: its row's code bytes for its words decode to kJ words, stored as 16 consecutive
         * words a sub-group (the old lanes were 64 bytes apart, four to a bank). */
        if (pblk && !nostage) {
            /* run 175: the words decoded once a call (moe_i4i8_dec_kernel), copied at the same positions */
            const uint32_t* pg = pblk + (size_t)g * 1024u + pw0;
#pragma unroll
            for (int j = 0; j < kJ; ++j) sBw[pw0 + j * 16] = pg[j * 16];
        } else if (!nostage) {
#pragma unroll
            for (int j = 0; j < kJ; ++j) {
                const uint32_t b2 = (cw[j >> 1] >> (16 * (j & 1))) & 0xffffu;   /* code bytes (4 k4 + 0,1 | 2,3) */
                const uint32_t b0 = b2 & 255u, b1 = b2 >> 8;
                const uint32_t l0 = lvl(b0 & 15u), l1 = lvl(b0 >> 4), l2 = lvl(b1 & 15u), l3 = lvl(b1 >> 4);
                const uint32_t w = rowok[bt] ? (l0 | (l1 << 8) | (l2 << 16) | (l3 << 24)) : 0u;
                sBw[(bstep * 4 + bt) * 128 + (jbase + j) * 16 + lane] = w;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kRT; ++r)
#pragma unroll
            for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, C[r][t], 0);
        if (!nomma) {
#pragma unroll
          for (int step = 0; step < 2; ++step) {
#pragma unroll
            for (int r = 0; r < kRT; ++r)
                jm::joint_matrix_load(sg, Am[r], lptr(sA + (wave * RT + r * 8) * 64 + step * 32), 64);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bt, lptr(sB + (step * 4 + t) * kDTileB), 64);
#pragma unroll
                for (int r = 0; r < kRT; ++r) jm::joint_matrix_mad(sg, C[r][t], Am[r], Bt, C[r][t]);
            }
          }
        }
        /* the group's exact sums, scaled as the run form scales them: lane = column, 8 rows a lane. Run 170 (the
         * expert kernel's rework, cut 1): B110's counters put the kernel on the SLM port (~20 GB of SLM traffic a
         * 10.8 ms launch, bank conflicts 0), and this tile's store-and-read-back through sC was as much of it as the
         * B tiles' reads (4 x 512 B written and read a sub-group a group); the accumulator's own work-item data is
         * the same tile already laid out lane = column, element q = row q (what joint_matrix_store wrote and
         * sCw[q * 16 + lane] read), so it is scaled in registers (B111: the bits byte-equal; B112: the SLM bytes
         * −42 %, the time flat). The sC path stays behind RADIANCE_XE_I4I8_DPAS_SLMC=1 (`slmc`). The scale bytes and
         * the activation scales were loaded at the group's top (sr / ar): the same values. */
#pragma unroll
        for (int r = 0; r < kRT; ++r) {
            float ar[8];   /* this row tile's activation scales (per 128 K) */
#pragma unroll
            for (int q = 0; q < 8; ++q)
                ar[q] = (!noscale && liver[r * 8 + q]) ? As[(size_t)tokr[r * 8 + q] * as_ld + g / 2u] : (noscale ? 1.0f : 0.0f);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float sb = noscale ? 1.0f : (rowok[t] ? e4m3_to_f32(sr[t]) * smul : 0.0f);
                if (slmc == 1u) {
                    jm::joint_matrix_store(sg, C[r][t], lptr(sCw), 16, jm::layout::row_major);
                    xe::wave_barrier();
#pragma unroll
                    for (int q = 0; q < 8; ++q) acc[r][t][q] += ((float)sCw[q * 16 + lane] * ar[q]) * sb;
                    xe::wave_barrier();
                } else {
                    auto wd = sycl::ext::oneapi::detail::get_wi_data(sg, C[r][t]);
#pragma unroll
                    for (int q = 0; q < 8; ++q) acc[r][t][q] += ((float)(int32_t)wd[q] * ar[q]) * sb;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t n = n0 + (uint32_t)(t * 16 + lane);
        if (n >= N) continue;
#pragma unroll
        for (int r = 0; r < kRT; ++r)
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const uint32_t qq = (uint32_t)(wave * RT + r * 8 + q);
                if (qq < rn)
                    Y[(size_t)(i + qq) * y_ld + n] = to_bf16(rowok[t] && liver[r * 8 + q] ? acc[r][t][q] : 0.0f);
            }
    }
}

/* the two shapes as plain kernels for the launch (the body is the template above, inlined) */
void moe_i4i8_dpas_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                          const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                          uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                          uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows, uint32_t slmc,
                          const uint32_t* __restrict__ pre, uint32_t pnb, uint32_t png, const uint32_t* __restrict__ map) {
    moe_i4i8_dpas_kernel_t<8, 8>(A, As, c0, c1, sorted, off, Y, T, N, n_expert, top_k, a_ld, as_ld, y_ld, a_sorted,
                                 parts, a_rows, slmc, pre, pnb, png, map);
}
void moe_i4i8_dpas_kernel_rt16(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                               const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                               uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                               uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows, uint32_t slmc) {
    moe_i4i8_dpas_kernel_t<8, 16>(A, As, c0, c1, sorted, off, Y, T, N, n_expert, top_k, a_ld, as_ld, y_ld, a_sorted,
                                  parts, a_rows, slmc, nullptr, 0u, 0u, nullptr);
}

/* Owner note 195 (run 202): the DPAS form's block map. B150's 7K profile put this GEMM at 42 % of the prefill's device
 * time, 7.6 ms a launch, on a grid of one work-group a sorted row and column block: at a 4,096-row chunk (top-10,
 * 40,960 sorted rows) down's grid is ~1.3 M work-groups, of which only the ~1,150 block starts a column compute -- the
 * rest read their row, binary-search the offsets and leave. One work-group writes the blocks instead, as r4d_moe.cpp's
 * moe_blockmap_kernel does for the bf16 form: map[0] = the count, then (expert, first row) a block of FM rows in
 * expert order, then the padding tail (kMapTailI4, first row) in blocks of FM. Every block the row grid computed is an
 * entry with the same first row and row count, so each output is the same arithmetic on the same operands: the bits
 * do not change. The cap is moe_i4i8_map_cap(): one block an expert plus one a FM rows, the tail's, and one. */
long long moe_i4i8_map_cap(long long T, long long n_expert, long long FM) { return n_expert + (T + FM - 1) / FM + 1; }
int64_t moe_i4i8_map_bytes(long long T, long long n_expert, long long FM) {
    return (int64_t)(1 + 2 * moe_i4i8_map_cap(T, n_expert, FM)) * (int64_t)sizeof(uint32_t);
}
constexpr int kMapThreads = 256;
void moe_i4i8_map_kernel(const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
                         uint32_t* __restrict__ map) {
    const uint32_t tid = threadIdx.x;
    const uint32_t live = (uint32_t)off[n_expert];
    uint32_t carry = 0;
    /* the experts in tiles of 256, an exclusive scan of their block counts a tile; each thread writes its expert's
     * entries from its base (integer sums: the order does not matter) */
    for (uint32_t b = 0; b < n_expert; b += (uint32_t)kMapThreads) {
        const uint32_t e = b + tid;
        const uint32_t lo = e < n_expert ? (uint32_t)off[e] : 0u;
        const uint32_t run = e < n_expert ? (uint32_t)off[e + 1] - lo : 0u;
        const uint32_t cnt = (run + FM - 1) / FM;
        const uint32_t base = carry + sycl::exclusive_scan_over_group(xe::wg(), cnt, sycl::plus<uint32_t>());
        uint32_t o = base;
        for (uint32_t r = 0; r < run && o < cap; r += FM, ++o) {
            map[1 + 2 * o] = e;
            map[2 + 2 * o] = lo + r;
        }
        carry += sycl::reduce_over_group(xe::wg(), cnt, sycl::plus<uint32_t>());
    }
    if (tid == 0) {
        uint32_t o = carry;
        for (uint32_t r = live; r < T && o < cap; r += FM, ++o) {
            map[1 + 2 * o] = kMapTailI4;
            map[2 + 2 * o] = r;
        }
        map[0] = o < cap ? o : cap;   /* the count, clamped to the grid */
    }
}

/* Run 175 (owner note 176 item 3): the decode once a call -- one work-group of the DPAS kernel's shape (8 sub-groups
 * of 16 lanes) a (64-column block, expert), each sub-group its (tile, step) as the GEMM's staging assigns it, every
 * 64-K group's words computed by the staging's own expressions (the level table, the select chain, the dead-column
 * zeros) and written to pre[((e * pnb + nb) * png + g) * 1024 + (step * 4 + tile) * 128 + k4 % 8 * 16 + n % 16]. An
 * expert without rows in this call is skipped (its words are never read). */
void moe_i4i8_dec_kernel(ICls c0, ICls c1, const int32_t* __restrict__ off, uint32_t N, uint32_t parts,
                         uint32_t* __restrict__ pre, uint32_t pnb, uint32_t png) {
    constexpr int kJ = 8;
    const int tid = (int)threadIdx.x, wave = tid / 16, lane = tid % 16;
    const uint32_t nb = blockIdx.x, e = blockIdx.y, n0 = nb * (uint32_t)kDCols;
    if (off[e + 1] <= off[e]) return;
    const ICls& c = (e & 1) ? c1 : c0;
    const uint32_t fy = N / parts, fcw = c.n / parts, K = c.k;
    const uint32_t entry = e >> 1;
    const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry) + c.c_off;
    const float* tab = (const float*)(entry_of(c.base, c.tab, c.stride, entry) + c.t_off);
    float amax = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; ++t) amax = sycl::fmax(amax, sycl::fabs(tab[t]));
    const float lv = tab[lane];
    const uint32_t lvi = (uint32_t)(amax > 0.0f ? (int)sycl::rint(lv * 127.0f / amax) : (int)lv) & 255u;
    sycl::sub_group sg = xe::sg();
    uint32_t pk[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j) pk[k] |= sycl::select_from_group(sg, lvi, 4 * k + j) << (8 * j);
    auto lvl = [&](uint32_t cc) -> uint32_t {
        const uint32_t d = (cc & 8u) ? ((cc & 4u) ? pk[3] : pk[2]) : ((cc & 4u) ? pk[1] : pk[0]);
        return (d >> (8u * (cc & 3u))) & 255u;
    };
    const int bt = (wave & 7) >> 1, bstep = wave & 1;
    const uint32_t n = n0 + (uint32_t)(bt * 16 + lane);
    const uint32_t pp = fy ? n / fy : 0u, jj = n - pp * fy;
    const bool ok = n < N && jj < fcw;
    const uint32_t row = pp * fcw + jj;
    const uint8_t* cbase = ok ? we + (size_t)row * c.w_ld + (size_t)bstep * 16u : nullptr;
    uint32_t* out = pre + ((size_t)e * pnb + nb) * (size_t)png * 1024u + (uint32_t)((bstep * 4 + bt) * 128 + lane);
    const uint32_t ng = K / 64u;
    for (uint32_t g = 0; g < ng; ++g) {
        uint32_t cw[4] = {0u, 0u, 0u, 0u};
        if (cbase) __builtin_memcpy(cw, cbase + (size_t)g * 32u, kJ * 2);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const uint32_t b2 = (cw[j >> 1] >> (16 * (j & 1))) & 0xffffu;
            const uint32_t b0 = b2 & 255u, b1 = b2 >> 8;
            const uint32_t l0 = lvl(b0 & 15u), l1 = lvl(b0 >> 4), l2 = lvl(b1 & 15u), l3 = lvl(b1 >> 4);
            out[(size_t)g * 1024u + j * 16] = ok ? (l0 | (l1 << 8) | (l2 << 16) | (l3 << 24)) : 0u;
        }
    }
}
/* RADIANCE_XE_I4I8_DPAS_ROWS=16 (run 171): 16 rows a sub-group (the 128-row work-group of 8 sub-groups) under the
 * large register file; 8 (64 rows) unless set */
int dpas_rows() {
    static const int v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS_ROWS"); return (s && *s == '1' && s[1] == '6') ? 16 : 8; }();
    return v;
}

/* RADIANCE_XE_I4I8_DPAS: the matrix-unit form for steps of RADIANCE_XE_I4I8_DPAS_MIN sorted rows or more (512 unless
 * set) -- the default since run 164 (B66: the 7K first token 13.7 -> 10.9 s); =0 the run form everywhere */
bool dpas_on() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS"); return !(s && *s == '0'); }();
    return v;
}
long long dpas_min_rows() {
    static const long long v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS_MIN"); const long long n = s ? std::strtoll(s, nullptr, 10) : 0; return n > 0 ? n : 512LL; }();
    return v;
}
/* RADIANCE_XE_I4I8_DPAS_SLMC=1 (run 170): the group's int32 tiles scaled through shared memory as before the rework's
 * cut 1 (the store-and-read-back path), for the A/B and as the fallback should the card's accumulator layout differ;
 * =2: TEST ONLY, the 4-bit level decode skipped (wrong numbers) to time the decode's share of the kernel; =3 / =4 / =5
 * likewise the scaling's loads / the B staging / the MMAs skipped (the pipeline's shares by ablation) */
uint32_t dpas_slmc() {
    static const uint32_t v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS_SLMC"); return s ? (uint32_t)std::strtoul(s, nullptr, 10) : 0u; }();
    return v;
}
/* RADIANCE_XE_I4I8_DPAS_PRE=1 (run 175, owner note 176 item 3): the codes decoded once a call into the op's declared
 * scratch (moe_i4i8_dec_kernel), the GEMM copying the words; opt-in -- the scratch, n_expert x N x K bytes, comes out
 * of the arena (the expert slab shrinks by it), and the hook declares none when the switch is off */
bool dpas_pre() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS_PRE"); return s && *s == '1'; }();
    return v;
}
/* the pre-decoded words' bytes for a call's geometry: every expert, every 64-column block, every 64-K group */
int64_t pre_bytes(long long n_expert, long long N, long long K) {
    return (int64_t)n_expert * ((N + kDCols - 1) / kDCols) * (K / 64) * 4096;
}
/* RADIANCE_XE_I4I8_DPAS_MAP (run 202, owner note 195): the DPAS form launched over its blocks (moe_i4i8_map_kernel)
 * instead of one work-group a sorted row -- THE DEFAULT (=0 the row grid, for comparisons). The map sits in the op's
 * scratch after the pre-decoded words (if any), at a 256-byte boundary; a scratch too small for it leaves the row grid
 * in place. B151 / B152 (image 241): the 7K first token 5.08 -> 4.57 s, the 20K 10.44 -> 9.59. Its bits are the row
 * grid's with the wrapped-leader blocks zeroed (B154 = B151, SLMC=6 below; KMAP1 at the floor over 54,722 positions),
 * as note 126 made the matrix form's fix (MX_COMPACT) the default for the same bug. */
bool dpas_map() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_DPAS_MAP"); return !(s && *s == '0'); }();
    return v;
}
int64_t map_offset(long long n_expert, long long N, long long K) {
    return dpas_pre() ? (pre_bytes(n_expert, N, K) + 255) / 256 * 256 : 0;
}
#endif

/* RADIANCE_XE_I4I8_RUN_BOUND (run 202): the run form's rows at or past the call's live count zeroed (the kernel's
 * `bound`) -- THE DEFAULT (=0 the unbounded rule, for comparisons); M283: the M-pair's texts identical */
bool run_bound() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_RUN_BOUND"); return !(s && *s == '0'); }();
    return v;
}
/* RADIANCE_XE_I4I8_PLAIN=1: the plain form only (a sub-group an output), for A/B and as the reference form */
bool plain_forced() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_PLAIN"); return s && *s == '1'; }();
    return v;
}

/* One class's table: a pointer table (stride 0) or a stacked plane. */
int table_of(const RadTensor* t, long long cnt, const uint8_t** base, const void* const** tab, uint64_t* stride) {
    if (!t || !t->data || t->rank < 2 || t->shape[0] < cnt) return R4D_NO(RAD_E_SHAPE);
    if (t->stride[0] == 0) { *tab = (const void* const*)t->data; *base = nullptr; *stride = 0; return RAD_OK; }
    const int64_t bits = rad_dtype_bits(t->dtype);
    if (bits <= 0 || (bits * t->stride[0]) % 8) return R4D_NO(RAD_E_STRIDE);
    *tab = nullptr;
    *base = (const uint8_t*)t->data;
    *stride = (uint64_t)(bits * t->stride[0] / 8);
    return RAD_OK;
}

/* moe_gemm_q (two classes): a, a_scale, w, w_scale, sorted, offsets, w_odd, w_scale_odd, y. fc: format C, the
 * joined w4nl64a8h entries (codes,table; scale,scale.1 -- xe_moe_planes.cpp's layout) against int8 activations. */
int moe_i4i8(const RadArgs* a, RadStream s, bool fc) {
    const RadTensor* Aq = r4d_opt(a, 0);
    const RadTensor* As = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off = r4d_opt(a, 5);
    const RadTensor* W[2] = { r4d_opt(a, 2), r4d_opt(a, 6) };
    const RadTensor* Ws[2] = { r4d_opt(a, 3), r4d_opt(a, 7) };
    const RadTensor* Y = r4d_opt(a, 8);
    if (!Aq || !As || !sorted || !off || !Y || !W[0] || !W[1] || !Ws[0] || !Ws[1]) return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const char* wdt = r4d_gets(a, "dtype");
    if (!wdt || std::strcmp(wdt, fc ? "w4nl64i8h" : "w4g64i8h")) return R4D_NO(RAD_E_UNSUPPORTED);
    long long Nc[2] = { r4d_geti_or(a, "N", 0), 0 }, Kc[2] = { r4d_geti_or(a, "K", 0), 0 };
    Nc[1] = r4d_geti_or(a, "N_odd", Nc[0]);
    Kc[1] = r4d_geti_or(a, "K_odd", Kc[0]);
    const long long parts = r4d_geti_or(a, "parts", 1), n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0), group = r4d_geti_or(a, "group", 128);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0 || group != 128) return R4D_NO(RAD_E_SHAPE);
    long long N = 0, K = 0;
    for (int q = 0; q < 2; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0 || Kc[q] % 128 || Nc[q] % parts) return R4D_NO(RAD_E_SHAPE);
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    if (Aq->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 || sorted->dtype != RAD_I32 ||
        off->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    const long long as_ld = r4d_rows_pitch(As, K / 128, &Ms);
    if (y_ld < 0 || T <= 0 || a_ld < 0 || M <= 0 || as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (a_ld % 4) return R4D_NO(RAD_E_STRIDE);   /* the activation is read a dword at a time */
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);
    ICls cl[2]{};
    if (fc != (XE_I4I8_FC != 0)) return R4D_NO(RAD_E_UNSUPPORTED);   /* each form is its own file's */
    if (fc && (K > (long long)kMaxK || plain_forced())) return R4D_NO(RAD_E_UNSUPPORTED);   /* the run form only */
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (n_expert + 1 - q) / 2;
        if (fc ? W[q]->dtype != RAD_U4 : (W[q]->dtype != RAD_I4 || Ws[q]->dtype != RAD_BF16))
            return R4D_NO(RAD_E_DTYPE);
        int rc = table_of(W[q], cnt, &cl[q].base, &cl[q].tab, &cl[q].stride);
        if (rc == RAD_OK) rc = table_of(Ws[q], cnt, &cl[q].sbase, &cl[q].stab, &cl[q].sstride);
        if (rc != RAD_OK) return rc;
        cl[q].n = (uint32_t)Nc[q];
        cl[q].k = (uint32_t)Kc[q];
        if (fc) {
            /* the joined entries as xe_moe_planes_relayout writes them (rad_planes.h): the planes after a header and
             * two entries, codes [N, K/2 bytes] then the 16 levels; E4M3 scales [N, K/64] then scale.1 */
            const uint32_t p0 = (uint32_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
            cl[q].w_ld = (uint32_t)(Kc[q] / 2);
            cl[q].s_ld = (uint32_t)(Kc[q] / 64);
            cl[q].c_off = p0;
            cl[q].t_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 2));
            cl[q].s_off = p0;
            cl[q].s1_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 64));
            continue;
        }
        /* the canonical planes: rows of K/2 bytes and K/64 scales (a stacked plane's own row stride if it has one) */
        const long long wr = W[q]->rank >= 3 && W[q]->stride[0] != 0 ? W[q]->stride[1] : Kc[q];
        const long long sr = Ws[q]->rank >= 3 && Ws[q]->stride[0] != 0 ? Ws[q]->stride[1] : Kc[q] / 64;
        if (wr < Kc[q] || wr % 8 || sr < Kc[q] / 64) return R4D_NO(RAD_E_STRIDE);
        cl[q].w_ld = (uint32_t)(wr / 2);
        cl[q].s_ld = (uint32_t)sr;
    }
    try {
        sycl::queue* qq = (sycl::queue*)s;
#if XE_I4I8_FC
        /* the matrix-unit form at prefill (owner note 168 item 2): the same bits as the run form, from 64-row tiles */
        if (fc && dpas_on() && T >= dpas_min_rows() && K % 64 == 0) {
            const bool rt16 = dpas_rows() == 16;
            const dim3 grid((unsigned)((N + kDCols - 1) / kDCols), (unsigned)T), block(kDThreads);
            /* run 175: the decode once a call (RADIANCE_XE_I4I8_DPAS_PRE=1), into the declared scratch */
            const uint32_t* pre = nullptr;
            const uint32_t pnb = (uint32_t)((N + kDCols - 1) / kDCols), png = (uint32_t)(K / 64);
            if (dpas_pre() && !rt16) {
                if (a->scratch && a->scratch_bytes >= pre_bytes(n_expert, N, K)) {
                    XE_LAUNCH_SG(16, qq, moe_i4i8_dec_kernel, dim3(pnb, (unsigned)n_expert), block, 0, cl[0], cl[1],
                                 (const int32_t*)off->data, (uint32_t)N, (uint32_t)parts, (uint32_t*)a->scratch, pnb, png);
                    pre = (const uint32_t*)a->scratch;
                } else {
                    static bool said = false;
                    if (!said) {
                        said = true;
                        std::fprintf(stderr, "[xe] moe_gemm_q: RADIANCE_XE_I4I8_DPAS_PRE=1 but the scratch is %lld bytes of "
                                     "the %lld it needs; decoding in the GEMM\n", (long long)a->scratch_bytes,
                                     (long long)pre_bytes(n_expert, N, K));
                    }
                }
            }
            /* run 202 (owner note 195): the blocks' map, then the GEMM over its entries (RADIANCE_XE_I4I8_DPAS_MAP=1) */
            const uint32_t* map = nullptr;
            dim3 mgrid = grid;
            if (dpas_map() && !rt16) {
                const long long moff = map_offset(n_expert, N, K);
                if (a->scratch && a->scratch_bytes >= moff + moe_i4i8_map_bytes(T, n_expert, kDRows)) {
                    uint32_t* m = (uint32_t*)((char*)a->scratch + moff);
                    const uint32_t cap = (uint32_t)moe_i4i8_map_cap(T, n_expert, kDRows);
                    XE_LAUNCH(qq, moe_i4i8_map_kernel, dim3(1), dim3(kMapThreads), 0, (const int32_t*)off->data,
                              (uint32_t)n_expert, (uint32_t)T, (uint32_t)kDRows, cap, m);
                    map = m;
                    mgrid = dim3(pnb, cap);
                } else {
                    static bool said = false;
                    if (!said) {
                        said = true;
                        std::fprintf(stderr, "[xe] moe_gemm_q: RADIANCE_XE_I4I8_DPAS_MAP=1 but the scratch is %lld bytes of "
                                     "the %lld it needs; the row grid\n", (long long)a->scratch_bytes,
                                     (long long)(moff + moe_i4i8_map_bytes(T, n_expert, kDRows)));
                    }
                }
            }
            if (rt16)
                xe::launch_as_grf256<moe_i4i8_dpas_kernel_rt16, 16>(
                    moe_i4i8_dpas_kernel_rt16, *qq, grid, block, (const int8_t*)Aq->data, (const float*)As->data,
                    cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, (uint16_t*)Y->data,
                    (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld,
                    (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M, dpas_slmc());
            else
                XE_LAUNCH_SG(16, qq, moe_i4i8_dpas_kernel, mgrid, block, 0, (const int8_t*)Aq->data, (const float*)As->data,
                             cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, (uint16_t*)Y->data,
                             (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld,
                             (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M, dpas_slmc(), pre, pnb, png,
                             map);
            return RAD_OK;
        }
#endif
        if (!plain_forced() && K <= (long long)kMaxK) {
            /* the run form: at K <= 512 (down, 2-8 groups a row) a group a lane, 2 lanes a column and two columns a
             * team (the probe's GL2R2: 33 us at the verify against L8R2's 42); else 16 lanes a column (gate_up) and
             * one column a team -- the probe's best gate_up mapping (L16R1), and at two columns IGC 2.41.5 spills
             * ~259 bytes of <5, 16, 2> (run 114) */
            const bool gl = K <= 512;
            const int lpr = gl ? 2 : 16, rpt = gl ? 2 : 1;
            const int upl = gl ? (int)((K / 64 + lpr - 1) / lpr) : (int)((K / 32 + lpr - 1) / lpr);
            const unsigned cpw = (unsigned)((kWG / lpr) * rpt);
            const dim3 grid((unsigned)((N + cpw - 1) / cpw), (unsigned)T), block(kWG);
            /* RADIANCE_XE_I4I8_TRACE=1 (run 116): each distinct launch geometry once, on stderr -- the engine's own
             * shapes, against the probe's (P53: the run form slower in the engine than in i4i8_probe) */
            static const bool trace = [] { const char* v = std::getenv("RADIANCE_XE_I4I8_TRACE"); return v && *v == '1'; }();
            if (trace) {
                static thread_local long long seen[64];
                static thread_local int nseen = 0;
                const long long key = (long long)T * 1000003LL + N * 131LL + K * 7LL + (a_sorted ? 1 : 0) + parts * 3LL;
                bool known = false;
                for (int x = 0; x < nseen; ++x) known = known || seen[x] == key;
                if (!known && nseen < 64) {
                    seen[nseen++] = key;
                    std::fprintf(stderr, "[xe] %s launch: T %lld N %lld K %lld parts %lld top_k %lld classes n %u / %u "
                                 "k %u / %u, a %s, M %lld, grid %u x %u of %d\n", wdt, (long long)T, (long long)N, (long long)K,
                                 (long long)parts, (long long)top_k, cl[0].n, cl[1].n, cl[0].k, cl[1].k,
                                 a_sorted ? "sorted" : "token", (long long)M, grid.x, grid.y, kWG);
                }
            }
            /* the tail skip's instance (format C's row has the init; format B's has none) */
            const TskipInst* ti = fc ? (const TskipInst*)a->instance : nullptr;
            const uint64_t sk_cfg = ti ? ti->cfg : 0u;
            uint32_t* const sk_stats = ti && ti->cfg ? ti->stats : nullptr;
#define XE_I4I8_RUN_F(U, L, R, G, F)                                                                         \
    XE_LAUNCH_SG(16, qq, (moe_i4i8_run_kernel<U, L, R, G, F>), grid, block, 0, (const int8_t*)Aq->data,      \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M, \
                 sk_cfg, sk_stats, run_bound() ? 1u : 0u)
#define XE_I4I8_RUN(U, L, R, G) XE_I4I8_RUN_F(U, L, R, G, (XE_I4I8_FC != 0))
            if (gl) switch (upl) {
                case 1: XE_I4I8_RUN(1, 2, 2, true); break;
                case 2: XE_I4I8_RUN(2, 2, 2, true); break;
                case 3: XE_I4I8_RUN(3, 2, 2, true); break;
                default: XE_I4I8_RUN(4, 2, 2, true); break;
            } else switch (upl) {
                case 1: XE_I4I8_RUN(1, 16, 1, false); break;
                case 2: XE_I4I8_RUN(2, 16, 1, false); break;
                case 3: XE_I4I8_RUN(3, 16, 1, false); break;
                case 4: XE_I4I8_RUN(4, 16, 1, false); break;
                case 5: XE_I4I8_RUN(5, 16, 1, false); break;
                case 6: XE_I4I8_RUN(6, 16, 1, false); break;
                case 7: XE_I4I8_RUN(7, 16, 1, false); break;
                default: XE_I4I8_RUN(8, 16, 1, false); break;
            }
#undef XE_I4I8_RUN
#undef XE_I4I8_RUN_F
            return RAD_OK;
        }
        XE_LAUNCH(qq, moe_i4i8_kernel, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const int8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

}  // namespace

#if XE_I4I8_FC
/* format C (owner note 147): w4nl64i8h, the w4nl64a8h planes against int8 activations */
extern "C" int xe_rad_moe_gemm_w4nl_i8(const RadArgs* a, RadStream s) { return moe_i4i8(a, s, true); }

/* The scratch hook (declare time, from the geometry; run 175): the pre-decoded words when RADIANCE_XE_I4I8_DPAS_PRE=1,
 * none otherwise -- n_expert x N x K bytes for the wider class */
extern "C" int64_t xe_rad_moe_gemm_w4nl_i8_scratch(const RadArgs* a) {
    if (!dpas_pre() && !dpas_map()) return 0;
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    long long N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    const long long N1 = r4d_geti_or(a, "N_odd", N), K1 = r4d_geti_or(a, "K_odd", K);
    if (N1 > N) N = N1;
    if (K1 > K) K = K1;
    if (n_expert <= 0 || N <= 0 || K <= 0) return 0;
    if (!dpas_map()) return pre_bytes(n_expert, N, K);
    /* run 202: the block map after the words, for the call's routed rows (M x top_k at the declared M, as r4d_moe.cpp's
     * hook counts them; the launcher checks what it was handed) */
    const long long T = r4d_geti_or(a, "M", 0) * r4d_geti_or(a, "top_k", 0);
    if (T <= 0) return dpas_pre() ? pre_bytes(n_expert, N, K) : 0;
    return map_offset(n_expert, N, K) + moe_i4i8_map_bytes(T, n_expert, kDRows);
}

/* The tail skip's instance, one a rank (every resolved instance of the row shares it): the flag read, the address
 * split probed on the rank's device, the stats zeroed. A refusal leaves the row serving with the skip off. */
namespace {
TskipInst* g_tskip[16] = {};
std::mutex g_tskip_mu;
}
extern "C" int xe_tskip_init(const RadParam*, int, int rank, int, void** out) {
    if (out) *out = nullptr;
    if (rank < 0 || rank >= 16 || !out) return RAD_OK;
    std::lock_guard<std::mutex> lk(g_tskip_mu);
    if (!g_tskip[rank]) {
        TskipInst* t = new TskipInst;
        t->rank = rank;
        const char* ev = std::getenv("RADIANCE_XE_TAIL_SKIP");
        const long Rk = ev ? std::atol(ev) : 0;
        if (Rk > 0 && Rk < 32 && rad_l0k_device(rank, &t->dev, &t->ctx) == 0 && t->dev && t->ctx) {
            void* dp = nullptr;
            void* hp = nullptr;
            try {
                dp = sycl::malloc_device(64, *t->dev, *t->ctx);
                hp = sycl::malloc_host(64, *t->ctx);
                const bool dh = dp && (((uintptr_t)dp >> kTskipSplit) != 0), hh = hp && (((uintptr_t)hp >> kTskipSplit) != 0);
                if (dp && hp && dh != hh) {
                    t->stats = sycl::aligned_alloc_device<uint32_t>(256, 64, *t->dev, *t->ctx);
                    sycl::queue q(*t->ctx, *t->dev, sycl::property::queue::in_order());
                    if (t->stats) q.memset(t->stats, 0, 64 * sizeof(uint32_t));
                    q.wait_and_throw();
                    t->cfg = (uint64_t)Rk | (dh ? 256u : 0u);
                    std::fprintf(stderr, "[xe] tail skip on rank %d (RADIANCE_XE_TAIL_SKIP=%ld): an expert off the device is dropped at "
                                 "ranks >= %ld; device memory %s 2^47 (probe: device %p, host %p)\n", rank, Rk, Rk,
                                 dh ? "above" : "below", dp, hp);
                } else {
                    std::fprintf(stderr, "[xe] tail skip refused on rank %d: device %p and host %p memory do not split at 2^47\n",
                                 rank, dp, hp);
                }
            } catch (const sycl::exception& x) {
                std::fprintf(stderr, "[xe] tail skip refused on rank %d: %s\n", rank, x.what());
                t->cfg = 0;
            }
            try { if (dp) sycl::free(dp, *t->ctx); if (hp) sycl::free(hp, *t->ctx); } catch (const sycl::exception&) {}
        }
        g_tskip[rank] = t;
    }
    *out = g_tskip[rank];
    return RAD_OK;
}
extern "C" void xe_tskip_fini(void*) {}   /* the instance lives for the process (shared by the row's instances) */

/* Once a step a rank (from xe_ar2_stalls, the engine's stall read, on the runtime queue): the stats every 200 calls
 * as deltas, with the calls a step estimated from the distinct reporters. */
extern "C" void xe_tskip_report(int rank, void* stream, const void* key) {
    TskipInst* t = (rank >= 0 && rank < 16) ? g_tskip[rank] : nullptr;
    if (!t || !t->cfg || !t->stats || !stream) return;
    bool known = false;
    for (int k = 0; k < t->nkeys; ++k) known = known || t->keys[k] == key;
    if (!known && t->nkeys < 8) t->keys[t->nkeys++] = key;
    if (++t->calls % 200 != 0) return;
    uint32_t cur[64];
    try { ((sycl::queue*)stream)->memcpy(cur, t->stats, sizeof cur).wait(); } catch (const sycl::exception&) { return; }
    const double steps = t->nkeys ? 200.0 / (double)t->nkeys : 200.0;
    char line[640];
    int n = std::snprintf(line, sizeof line, "[xe] tail skip rank %d, ~%.0f steps: pairs by rank live / off-device / dropped:",
                          rank, steps);
    for (int r = 0; r < 16 && n < (int)sizeof line - 48; ++r) {
        const uint32_t lv = cur[32 + r] - t->last[32 + r];
        if (!lv) continue;
        n += std::snprintf(line + n, sizeof line - (size_t)n, " r%d %u/%u/%u", r, lv, cur[r] - t->last[r], cur[16 + r] - t->last[16 + r]);
    }
    std::fprintf(stderr, "%s; runs with an off-device expert %u, skipped whole %u (both GEMMs of a layer count)\n", line,
                 cur[49] - t->last[49], cur[48] - t->last[48]);
    std::memcpy(t->last, cur, sizeof cur);
}
#else
extern "C" int xe_rad_moe_gemm_i4i8(const RadArgs* a, RadStream s) { return moe_i4i8(a, s, false); }
#endif
