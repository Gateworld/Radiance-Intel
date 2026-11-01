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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* 1: this file compiled again as xe_moe_w4nl_i8.cpp, format C's own translation unit -- its own device image, so the
 * CPU device's JIT of one form never takes the other with it (run 116: the CPU JIT crashed on the joined image) */
#ifndef XE_I4I8_FC
#define XE_I4I8_FC 0
#endif

namespace {

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
                         uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
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
    if (flat >= 0) {
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
    uint32_t tokr[kMT];
    bool liver[kMT];
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        tokr[q] = 0; liver[q] = false;
        if (e >= 0 && (uint32_t)q < rn) {
            const int32_t fq = sorted[i + q];
            const uint32_t t = a_sorted ? i + q : (uint32_t)fq / top_k;
            tokr[q] = t;
            liver[q] = fq >= 0 && t < a_rows;
        }
    }
    const ICls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
    /* A work-group whose columns all fall past this class's width, in every part they touch, writes its zeros and
     * returns before staging (run 117; IP9, run 116: at the two-card split the smaller gate_up class's 512 of 768
     * columns left a third of its work-groups staging K 2560 for nothing). Uniform over the work-group: no barrier
     * is skipped by part of it. */
    if (e >= 0) {
        const uint32_t cpw = (uint32_t)((kWG / LPR) * R), n0 = nb * cpw, n1 = n0 + cpw < N ? n0 + cpw : N;
        const uint32_t a = fy ? n0 % fy : 0u;
        const bool live = n0 < N && (n1 - n0 >= fy || a < fc || n0 + (fy - a) < n1);
        if (!live) {
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
                            if constexpr (FC) {   /* a code byte (low code, high code) to its two levels */
                                uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                                for (int bb = 0; bb < 4; ++bb) {
                                    const uint32_t ev = __float_as_uint(s_g[(d[x] >> (8 * bb)) & 255u]);
                                    lo8 |= (ev & 255u) << (8 * bb);
                                    hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                                }
                                s = dp4((int32_t)lo8, (int32_t)a[8 * ch + 2 * x], s);
                                s = dp4((int32_t)hi8, (int32_t)a[8 * ch + 2 * x + 1], s);
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
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] ? v : 0.0f);
        }
    }
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
#define XE_I4I8_RUN_F(U, L, R, G, F)                                                                         \
    XE_LAUNCH_SG(16, qq, (moe_i4i8_run_kernel<U, L, R, G, F>), grid, block, 0, (const int8_t*)Aq->data,      \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M)
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
#else
extern "C" int xe_rad_moe_gemm_i4i8(const RadArgs* a, RadStream s) { return moe_i4i8(a, s, false); }
#endif
