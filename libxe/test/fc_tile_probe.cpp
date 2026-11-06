/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 190, owner notes 188 / 189: the expert-layout probe, stage 1).
 *
 * fc_tile_probe.cpp -- format C's decode GEMM (moe_i4i8_run_kernel<.., FC>, libxe/xe_moe_i4i8.cpp, reached through
 * libxe's xe_rad_moe_gemm_w4nl_i8 as the engine reaches it) against the same arithmetic over a load-time repack of
 * the same entries (briefs/2026-11-05-expert-layout.md), on one card's decode shapes, in one process:
 *   gate_up  768 / 512 columns (the two parity classes) over K 2560, two parts, token rows;
 *   down     2560 columns over K 384 / 256, sorted rows;
 * at the verify mix (4 tokens x top_k 10 = 40 rows over 27 distinct experts) and at one token (10 rows).
 *
 * The repack (form T1) changes where the bytes are, never a value or the order of a float operation:
 *   - the codes: within each 8-element block of a row, byte b holds element b in its low nibble and element b + 4 in
 *     its high nibble (today: elements 2b and 2b + 1). The 256-entry pair table then yields the levels of elements
 *     0-3 and 4-7 of the block, which meet the activations' dwords in their natural order: no even / odd split, so
 *     the activations are read straight from global memory -- no LDS staging, no per-work-group copy of the rows;
 *   - each entry's pair table (the kernel's s_g words: 0x40000000 | level(low code) | level(high code) << 8, the same
 *     rint(level x 127 / amax) in f32) and its scale factor (scale.1 x amax / 127) precomputed beside the codes: the
 *     work-group copies 1 KB into LDS instead of two 16-level amax loops and 256 rints;
 *   - the scales stay where they are (abl4: 1-4 us).
 * Everything else is today's kernel line for line: the same lanes own the same chunks, each chunk's int32 sum is the
 * same exact sum of the same products, and the per-lane accumulation and the xor tree are the same, so the bf16
 * output must be the same bytes. The probe checks that (and 4 KiB canaries on both outputs) before it times anything.
 * RI_PAIR=1 (run 193, owner note 191 item 2): also the concurrency probe -- gate_up's GEMM beside an int8 GEMV, in one
 * in-order queue, one out-of-order queue and two in-order queues, outputs compared (PAIR lines).
 * T1b (run 191, down's form only): T1 with the codes also moved, so a 16-lane sub-group's load at one (unit, chunk,
 * row parity) reads 256 contiguous bytes instead of 16 pieces in 8 rows (t1b_off); the arithmetic is T1's.
 *
 *   xe_fc_tile_probe LIBXE.so [LAUNCHES=48] [RUNS=5]   LAUNCHES 0: the byte check only (the CPU device's ctest)
 * E1 / E1b (run 206, owner note 202 A.4, down's form only, GPU only): the same arithmetic as an ESIMD kernel over today's
 * codes (byte b: elements 2b, 2b + 1) and T1's table and scale factor. An ESIMD thread owns 16 columns as SIMD lanes and
 * up to 4 rows; the pair table sits in registers (256 x 16 bits, a code byte -> two levels by register-indirect select),
 * the int32 dot by dp4a, and each column keeps today's two partial sums (the even and the odd 64-K groups, in order),
 * added at the end. E1 read a group's codes by 16-lane gathers (P99: slower; dropped), E1b by one transposed 2D block load
 * (16 rows x 8 dwords); E2 = E1b with the cheaper decode (DEC2), timed at 8 / 16 / 32 threads a work-group.
 * T2 (run 206, gate_up only): T1 with the rows' activations staged in LDS once a work-group and CB = 1 / 2 / 4 / 8
 * column blocks a work-group (T2 and RESULT-T2 lines; the bytes on every device, the times on a card).
 * RESULT lines: per launch us (host wall over LAUNCHES back-to-back launches, the median of RUNS), today's and T1's.
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "xe_lanes.h"
#include "rad_abi.h"
#include "rad_planes.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

/* ---- today's kernel's helpers and constants, verbatim (xe_moe_i4i8.cpp) */
constexpr int kMT = 4, kWG = 512;
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
inline int32_t dp4(int32_t a, int32_t b, int32_t acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}

/* One class of the repacked entries. An entry: the codes [n, k/2] at 0, the 256 table words at tab_off, the f32 scale
 * factor at smul_off; the scales are today's entries (E4M3 [n, k/64] at s_off). */
struct TCls {
    const uint8_t* base;
    uint64_t stride;
    const uint8_t* sbase;
    uint64_t sstride;
    uint32_t n, k, w_ld, s_ld, s_off, tab_off, smul_off;
};

/* T1b (down's form only: LPR 2, R 2, GL): the same codes in sub-group order. Per 16-row block of a class, a 16-lane
 * sub-group's load at (unit u, chunk ch, row parity r) reads its 8 teams x 2 lanes' 16-byte pieces as 256 contiguous
 * bytes (row-major put them in 8 rows, half a line each). Byte offset of row `row`'s chunk `jc` in an entry: */
inline uint32_t t1b_off(uint32_t row, uint32_t jc, uint32_t cpr) {
    const uint32_t r = row & 1u, tl = (row & 15u) >> 1, ch = jc & 1u, wv = jc >> 1, l = wv & 1u, u = wv >> 1;
    return (row >> 4) * (16u * cpr * 16u) + ((((u * 2u + ch) * 2u + r) * 16u + tl * 2u + l) * 16u);
}

/* ---- T1: moe_i4i8_run_kernel<UPL, LPR, R, GL, FC = true> with the repacked reads (see the header); TB: T1b's code
 * order. The tail skip is left out: the probe's experts are all resident, as the skip's own off-device test would
 * find. */
/* T1c (run 207, owner yes on note 202's T1c): one 16-lane sub-group block read a (unit, chunk, row parity) piece --
 * lane l's 16 bytes as dwords l, 16 + l, 32 + l, 48 + l of the piece (the read's striped order; t1c_off) -- in place
 * of 16 lanes' own 16-byte loads (A.3: the weight loads' send queue, ~39 % of down's stalls, U2). */
inline uint4 sg_read4(const uint8_t* p) {
#ifdef __SYCL_DEVICE_ONLY__
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    auto gp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
        (const uint32_t*)p);
    const u32x4 v = __spirv_SubgroupBlockReadINTEL<u32x4>(gp.get_decorated());
    return uint4{ v.x, v.y, v.z, v.w };
#else
    (void)p;
    return uint4{ 0, 0, 0, 0 };
#endif
}
/* T1c's byte offset of dword k of row `row`'s chunk `jc`: T1b's piece, its 16 lanes' dwords striped */
inline uint32_t t1c_off(uint32_t row, uint32_t jc, uint32_t cpr, uint32_t k) {
    const uint32_t r = row & 1u, tl = (row & 15u) >> 1, ch = jc & 1u, wv = jc >> 1, l = wv & 1u, u = wv >> 1;
    return (row >> 4) * (16u * cpr * 16u) + (((u * 2u + ch) * 2u + r) * 256u) + (k * 16u + tl * 2u + l) * 4u;
}

template <int UPL, int LPR, int R, bool GL, bool TB = false, bool TC = false>
void t1_run_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, TCls c0, TCls c1,
                   const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                   uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                   uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    XE_SHARED(float, s_g, 256);   /* the entry's pair table */
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
    const TCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
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
    const uint8_t* ent = e >= 0 ? c.base + (uint64_t)((uint32_t)e >> 1) * c.stride : nullptr;
    if (e >= 0 && tid < 256) s_g[tid] = __uint_as_float(((const uint32_t*)(ent + c.tab_off))[tid]);
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
        const uint8_t* we = ent;
        const uint8_t* se8 = c.sbase + (uint64_t)((uint32_t)e >> 1) * c.sstride + c.s_off;
        const float smul = *(const float*)(ent + c.smul_off);
        constexpr int CH = GL ? 2 : 1;
        uint4 w[UPL][R][CH];
#pragma unroll
        for (int u = 0; u < UPL; ++u)
#pragma unroll
            for (int ch = 0; ch < CH; ++ch) {
                const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) + (uint32_t)ch : (uint32_t)l + (uint32_t)u * LPR;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if constexpr (TC) {
                        /* the piece is the sub-group's: its first lane's chunk decides the read for all (the probe's
                         * shapes keep a unit's liveness uniform across the two lanes of a team) */
                        const uint32_t jc0 = 2u * ((uint32_t)u * LPR) + (uint32_t)ch;
                        /* the piece's base: its row block and parity, lane 0's chunk, dword 0 */
                        const uint32_t brow = (rowr[r] & ~15u) | (rowr[r] & 1u);
                        const uint4 v = jc0 < CPR ? sg_read4(we + (size_t)t1c_off(brow, jc0, CPR, 0))
                                                  : uint4{ 0, 0, 0, 0 };
                        w[u][r][ch] = (jc < CPR && rowok[r]) ? v : uint4{ 0, 0, 0, 0 };
                    } else {
                        const size_t o = TB ? (size_t)t1b_off(rowr[r], jc, CPR) : (size_t)rowr[r] * c.w_ld + jc * 16u;
                        w[u][r][ch] = (jc < CPR && rowok[r]) ? *(const uint4*)(we + o) : uint4{0, 0, 0, 0};
                    }
                }
            }
        /* TC: every scale this lane reads, issued before the decode (more loads in flight; the same values) */
        float sbr_all[TC ? UPL : 1][R];
        if constexpr (TC) {
#pragma unroll
            for (int u = 0; u < UPL; ++u) {
                const uint32_t jc = 2u * ((uint32_t)l + (uint32_t)u * LPR);
                const bool live = jc < CPR;
                const uint32_t g = live ? jc / 2u : 0u;
#pragma unroll
                for (int r = 0; r < R; ++r)
                    sbr_all[u][r] = rowok[r] && live ? e4m3_to_f32(se8[(size_t)rowr[r] * c.s_ld + g]) * smul : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) : (uint32_t)l + (uint32_t)u * LPR;
            const bool live = jc < CPR;
            const uint32_t g = live ? jc / 2u : 0u;
            float sbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                sbr[r] = TC ? sbr_all[TC ? u : 0][r]
                            : (rowok[r] && live ? e4m3_to_f32(se8[(size_t)rowr[r] * c.s_ld + g]) * smul : 0.0f);
            uint32_t lv[R][CH][8];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int ch = 0; ch < CH; ++ch) {
                    const uint32_t d[4] = { w[u][r][ch].x, w[u][r][ch].y, w[u][r][ch].z, w[u][r][ch].w };
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                        for (int bb = 0; bb < 4; ++bb) {
                            const uint32_t ev = __float_as_uint(s_g[(d[x] >> (8 * bb)) & 255u]);
                            lo8 |= (ev & 255u) << (8 * bb);
                            hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                        }
                        lv[r][ch][2 * x] = lo8;        /* elements 8x' + 0..3 of the block */
                        lv[r][ch][2 * x + 1] = hi8;    /* elements 8x' + 4..7 */
                    }
                }
#pragma unroll
            for (int q = 0; q < kMT; ++q) {
                if ((uint32_t)q >= rn) break;   /* uniform over the work-group */
                /* the activations in their own order, from global memory (no staging): a chunk's 32 bytes = 2 uint4 */
                const bool rd = liver[q] && live;
                const float asc = rd ? As[(size_t)tokr[q] * as_ld + g / 2u] : 0.0f;
                const uint4* ap = (const uint4*)(A + (size_t)tokr[q] * a_ld);
                uint32_t a[8 * CH];
#pragma unroll
                for (int v = 0; v < 2 * CH; ++v) {
                    const uint4 x = rd ? ap[jc * 2u + (uint32_t)v] : uint4{0, 0, 0, 0};
                    a[4 * v] = x.x; a[4 * v + 1] = x.y; a[4 * v + 2] = x.z; a[4 * v + 3] = x.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int32_t s = 0;
#pragma unroll
                    for (int ch = 0; ch < CH; ++ch)
#pragma unroll
                        for (int x = 0; x < 4; ++x) {
                            s = dp4((int32_t)lv[r][ch][2 * x], (int32_t)a[8 * ch + 2 * x], s);
                            s = dp4((int32_t)lv[r][ch][2 * x + 1], (int32_t)a[8 * ch + 2 * x + 1], s);
                        }
                    float fs = (float)s;
                    if constexpr (!GL) fs += __shfl_xor(fs, 1);   /* the group's other chunk */
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
            float v = acc[q][r];
            if constexpr (LPR == 16) v += __shfl_xor(v, 8);
            if constexpr (LPR >= 8) v += __shfl_xor(v, 4);
            if constexpr (LPR >= 4) v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] ? v : 0.0f);
        }
    }
}

/* the launcher's choice of instance, as moe_i4i8()'s run form makes it */
void t1_launch(sycl::queue& q, const int8_t* A, const float* As, const TCls& c0, const TCls& c1, const int32_t* sorted,
               const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert, uint32_t top_k,
               uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, bool a_sorted, uint32_t parts, uint32_t M,
               bool tb = false, bool tc = false) {
    const bool gl = K <= 512;
    const int lpr = gl ? 2 : 16, rpt = gl ? 2 : 1;
    const int upl = gl ? (int)((K / 64 + lpr - 1) / lpr) : (int)((K / 32 + lpr - 1) / lpr);
    const unsigned cpw = (unsigned)((kWG / lpr) * rpt);
    const dim3 grid((N + cpw - 1) / cpw, T), block(kWG);
#define T1G(U, L, R, G, B, C)                                                                                  \
    XE_LAUNCH_SG(16, &q, (t1_run_kernel<U, L, R, G, B, C>), grid, block, 0, A, As, c0, c1, sorted, off, Y, T, N, \
                 n_expert, top_k, a_ld, as_ld, y_ld, a_sorted ? 1u : 0u, parts, M)
#define T1F(U, L, R, G, B) T1G(U, L, R, G, B, false)
#define T1(U, L, R, G) T1F(U, L, R, G, false)
    if (gl && tc) switch (upl) {   /* T1c (run 207) */
        case 2: T1G(2, 2, 2, true, false, true); break;
        default: T1G(3, 2, 2, true, false, true); break;
    } else if (gl && tb) switch (upl) {
        case 2: T1F(2, 2, 2, true, true); break;
        default: T1F(3, 2, 2, true, true); break;
    } else if (gl) switch (upl) {
        case 1: T1(1, 2, 2, true); break;
        case 2: T1(2, 2, 2, true); break;
        case 3: T1(3, 2, 2, true); break;
        default: T1(4, 2, 2, true); break;
    } else switch (upl) {
        case 4: T1(4, 16, 1, false); break;
        case 5: T1(5, 16, 1, false); break;
        case 6: T1(6, 16, 1, false); break;
        default: T1(8, 16, 1, false); break;
    }
#undef T1
#undef T1F
#undef T1G
}

/* ---- T2 (run 206, the supervisor's order: gate_up's activation staging reused): T1's arithmetic for gate_up's form
 * (LPR 16, R 1, chunks of 32, the lane pair's xor), with the work-group's rows of activations staged in LDS ONCE (their
 * natural order, 16-byte pieces) and CB column blocks a work-group -- one staging serves CB x 32 columns instead of 32.
 * The sums, their order and the reduction are T1's (bit-identical to today). */
template <int UPL, int CB>
void t2_run_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, TCls c0, TCls c1,
                   const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                   uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                   uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    constexpr int LPR = 16;
    XE_SHARED(float, s_g, 256);
    XE_SHARED(uint4, s_a, kMT * 2560 / 16);   /* rn rows x K bytes (K <= 2560) */
    const int tid = (int)threadIdx.x;
    const uint32_t i = blockIdx.y;
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
    const TCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
    const uint8_t* ent = e >= 0 ? c.base + (uint64_t)((uint32_t)e >> 1) * c.stride : nullptr;
    if (e >= 0 && tid < 256) s_g[tid] = __uint_as_float(((const uint32_t*)(ent + c.tab_off))[tid]);
    const uint32_t pr = K / 16u;   /* 16-byte pieces a row */
    if (e >= 0)
        for (uint32_t x = (uint32_t)tid; x < rn * pr; x += (uint32_t)kWG) {
            const uint32_t q = x / pr, p = x - q * pr;
            bool lq = false;
            uint32_t tq = 0;
#pragma unroll
            for (int z = 0; z < kMT; ++z)
                if ((uint32_t)z == q) { lq = liver[z]; tq = tokr[z]; }
            s_a[x] = lq ? ((const uint4*)(A + (size_t)tq * a_ld))[p] : uint4{0, 0, 0, 0};
        }
    __syncthreads();
    const int team = tid / LPR, l = tid % LPR;
    constexpr uint32_t cpw = (uint32_t)(kWG / LPR);
    for (int cb = 0; cb < CB; ++cb) {
        const uint32_t nb = blockIdx.x * (uint32_t)CB + (uint32_t)cb;
        if (e >= 0) {
            const uint32_t n0 = nb * cpw, n1 = n0 + cpw < N ? n0 + cpw : N;
            const uint32_t a = fy ? n0 % fy : 0u;
            const bool live = n0 < N && (n1 - n0 >= fy || a < fc || n0 + (fy - a) < n1);
            if (!live) {
                if (l == 0)
                    for (int q = 0; q < kMT; ++q) {
                        if ((uint32_t)q >= rn) break;
                        const uint32_t n = n0 + (uint32_t)team;
                        if (n < N) Y[(size_t)(i + q) * y_ld + n] = 0;
                    }
                continue;
            }
        }
        float acc[kMT];
        const uint32_t n = nb * cpw + (uint32_t)team;
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        const bool rowok = e >= 0 && n < N && j < fc;
        const uint32_t rowr = pp * fc + j;
#pragma unroll
        for (int q = 0; q < kMT; ++q) acc[q] = 0.0f;
        if (e >= 0) {
            const uint32_t CPR = K / 32u;
            const uint8_t* se8 = c.sbase + (uint64_t)((uint32_t)e >> 1) * c.sstride + c.s_off;
            const float smul = *(const float*)(ent + c.smul_off);
            uint4 w[UPL];
#pragma unroll
            for (int u = 0; u < UPL; ++u) {
                const uint32_t jc = (uint32_t)l + (uint32_t)u * LPR;
                w[u] = (jc < CPR && rowok) ? *(const uint4*)(ent + (size_t)rowr * c.w_ld + jc * 16u) : uint4{0, 0, 0, 0};
            }
#pragma unroll
            for (int u = 0; u < UPL; ++u) {
                const uint32_t jc = (uint32_t)l + (uint32_t)u * LPR;
                const bool live = jc < CPR;
                const uint32_t g = live ? jc / 2u : 0u;
                const float sbr = rowok && live ? e4m3_to_f32(se8[(size_t)rowr * c.s_ld + g]) * smul : 0.0f;
                uint32_t lv[8];
                const uint32_t d[4] = { w[u].x, w[u].y, w[u].z, w[u].w };
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const uint32_t ev = __float_as_uint(s_g[(d[x] >> (8 * bb)) & 255u]);
                        lo8 |= (ev & 255u) << (8 * bb);
                        hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                    }
                    lv[2 * x] = lo8;
                    lv[2 * x + 1] = hi8;
                }
#pragma unroll
                for (int q = 0; q < kMT; ++q) {
                    if ((uint32_t)q >= rn) break;
                    const bool rd = liver[q] && live;
                    const float asc = rd ? As[(size_t)tokr[q] * as_ld + g / 2u] : 0.0f;
                    uint32_t a[8];
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const uint4 x = rd ? s_a[(uint32_t)q * pr + jc * 2u + (uint32_t)v] : uint4{0, 0, 0, 0};
                        a[4 * v] = x.x; a[4 * v + 1] = x.y; a[4 * v + 2] = x.z; a[4 * v + 3] = x.w;
                    }
                    int32_t s = 0;
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        s = dp4((int32_t)lv[2 * x], (int32_t)a[2 * x], s);
                        s = dp4((int32_t)lv[2 * x + 1], (int32_t)a[2 * x + 1], s);
                    }
                    float fs = (float)s;
                    fs += __shfl_xor(fs, 1);   /* the group's other chunk */
                    if ((l & 1) == 0 && rowok && live) acc[q] += (fs * asc) * sbr;
                }
            }
        }
#pragma unroll
        for (int q = 0; q < kMT; ++q) {
            if ((uint32_t)q >= rn) break;
            float v = acc[q];
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok && liver[q] ? v : 0.0f);
        }
    }
}

/* gate_up's shape only (K 2560: UPL 5) */
void t2_launch(sycl::queue& q, int cb, const int8_t* A, const float* As, const TCls& c0, const TCls& c1,
               const int32_t* sorted, const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N, uint32_t n_expert,
               uint32_t top_k, uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, bool a_sorted, uint32_t parts, uint32_t M) {
    const unsigned cpw = (unsigned)(kWG / 16), nbx = (N + cpw - 1) / cpw;
#define T2(CB_)                                                                                                    \
    XE_LAUNCH_SG(16, &q, (t2_run_kernel<5, CB_>), dim3((nbx + CB_ - 1) / CB_, T), dim3(kWG), 0, A, As, c0, c1,    \
                 sorted, off, Y, T, N, n_expert, top_k, a_ld, as_ld, y_ld, a_sorted ? 1u : 0u, parts, M)
    switch (cb) {
        case 1: T2(1); break;
        case 2: T2(2); break;
        case 4: T2(4); break;
        default: T2(8); break;
    }
#undef T2
}

/* ---- E1 / E1b (see the header): one class's entries for the ESIMD form -- today's codes [n, k/2] at w_off of entry
 * e >> 1, T1's pair table and scale factor, today's scales */
namespace es = sycl::ext::intel::esimd;
struct ECls {
    const uint8_t* wbase;
    uint64_t wstride;
    const uint8_t* tbase;
    uint64_t tstride;
    const uint8_t* sbase;
    uint64_t sstride;
    uint32_t w_off, tab_off, smul_off, s_off, n, w_ld;
};

/* One thread's 16 columns n0.. of rows i .. i + rn - 1 over one class of G 64-K groups (s_ld = G). */
/* DEC2 (E2): the code bytes taken as indices by byte regions of the loaded dwords (no shifts / masks) and two pair
 * tables, the second pre-shifted by 16 (a level dword = one or of two selects). zero (always 0, a kernel argument):
 * each product goes through an integer xor with it before the add, so the backend cannot fuse the multiply and the
 * add into a mad (P99: E1's 1-2 differing outputs a case, all at near-cancelling sums). */
template <int G, bool B2D, bool DEC2>
ESIMD_INLINE void e1_cols(const ECls& c, uint32_t eh, uint32_t i, uint32_t rn, const uint32_t (&tokr)[kMT],
                          const bool (&liver)[kMT], uint32_t n0, const int8_t* A, const float* As, uint16_t* Y,
                          uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t zero) {
    const uint8_t* we = c.wbase + (uint64_t)eh * c.wstride + c.w_off;
    const uint8_t* te = c.tbase + (uint64_t)eh * c.tstride;
    /* the pair table: the low 16 bits of T1's words = level(low code) | level(high code) << 8 */
    es::simd<uint16_t, 256> P;
    es::simd<uint32_t, 256> P0, P1;   /* DEC2 */
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        es::simd<uint32_t, 64> t = es::block_load<uint32_t, 64>((const uint32_t*)(te + c.tab_off) + 64 * b);
        if constexpr (DEC2) {
            P0.select<64, 1>(64 * b) = t & 0xffffu;
            P1.select<64, 1>(64 * b) = t << 16;
        } else {
            P.select<64, 1>(64 * b) = t.bit_cast_view<uint16_t>().select<64, 2>(0);
        }
    }
    const float smul = *(const float*)(te + c.smul_off);
    /* the 16 rows' scales are 16 x G contiguous bytes: sbr[g] = e4m3_to_f32(scale) * smul, as the scalar form */
    es::simd<uint32_t, 4 * G> sw =
        es::block_load<uint32_t, 4 * G>((const uint32_t*)(c.sbase + (uint64_t)eh * c.sstride + c.s_off + (size_t)n0 * G));
    es::simd<float, 16> sbr[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const es::simd<uint8_t, 16> v8 = sw.template bit_cast_view<uint8_t>().template select<16, G>(g);
        const es::simd<uint32_t, 16> v = v8;
        const es::simd<uint32_t, 16> s = (v & 0x80u) << 24, ex = (v >> 3) & 15u, m = v & 7u;
        es::simd<float, 16> mf = es::simd<float, 16>(m) * 0.001953125f;
        es::simd<uint32_t, 16> r = s | ((ex + 120u) << 23) | (m << 20);
        r.merge(s | mf.template bit_cast_view<uint32_t>(), ex == 0u);
        r.merge(s | 0x7fc00000u, (v & 0x7fu) == 0x7fu);   /* ex 15 and m 7 */
        sbr[g] = r.template bit_cast_view<float>() * smul;
    }
    es::simd<float, 16> acc[2][kMT];
#pragma unroll
    for (int q = 0; q < kMT; ++q) { acc[0][q] = 0.0f; acc[1][q] = 0.0f; }
    es::simd<uint32_t, 16> rowo(0u, c.w_ld);
    rowo += n0 * c.w_ld;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        es::simd<uint32_t, 128> cw;   /* [dword w of the group's 32 code bytes][column] */
        if constexpr (B2D) {
            cw = es::load_2d<uint32_t, 8, 16, 1, true>((const uint32_t*)we, c.w_ld - 1u, c.n - 1u, c.w_ld - 1u, g * 8,
                                                       (int)n0);
        } else {
#pragma unroll
            for (int w = 0; w < 8; ++w)
                cw.select<16, 1>(16 * w) = es::gather<uint32_t, 16>((const uint32_t*)we, rowo + (uint32_t)(32 * g + 4 * w));
        }
        es::simd<int32_t, 16> lv[16];   /* levels of elements 4d .. 4d + 3 of the group, packed, a column a lane */
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if constexpr (DEC2) {
                auto cb = cw.bit_cast_view<uint8_t>();   /* byte b of dword (w, column) at 64 w + 4 column + b */
                const es::simd<uint16_t, 16> i0 = es::simd<uint8_t, 16>(cb.select<16, 4>(64 * w));
                const es::simd<uint16_t, 16> i1 = es::simd<uint8_t, 16>(cb.select<16, 4>(64 * w + 1));
                const es::simd<uint16_t, 16> i2 = es::simd<uint8_t, 16>(cb.select<16, 4>(64 * w + 2));
                const es::simd<uint16_t, 16> i3 = es::simd<uint8_t, 16>(cb.select<16, 4>(64 * w + 3));
                lv[2 * w] = es::simd<int32_t, 16>(P0.iselect(i0) | P1.iselect(i1));
                lv[2 * w + 1] = es::simd<int32_t, 16>(P0.iselect(i2) | P1.iselect(i3));
            } else {
                const es::simd<uint32_t, 16> d = cw.select<16, 1>(16 * w);
                const es::simd<uint32_t, 16> p0 = P.iselect(es::simd<uint16_t, 16>(d & 255u));
                const es::simd<uint32_t, 16> p1 = P.iselect(es::simd<uint16_t, 16>((d >> 8) & 255u));
                const es::simd<uint32_t, 16> p2 = P.iselect(es::simd<uint16_t, 16>((d >> 16) & 255u));
                const es::simd<uint32_t, 16> p3 = P.iselect(es::simd<uint16_t, 16>(d >> 24));
                lv[2 * w] = es::simd<int32_t, 16>(p0 | (p1 << 16));
                lv[2 * w + 1] = es::simd<int32_t, 16>(p2 | (p3 << 16));
            }
        }
#pragma unroll
        for (int q = 0; q < kMT; ++q) {
            if ((uint32_t)q >= rn) break;
            if (!liver[q]) continue;   /* written as 0 below, as the scalar form's dead rows */
            const float asc = As[(size_t)tokr[q] * as_ld + (uint32_t)g / 2u];
            const es::simd<int32_t, 16> av = es::block_load<int32_t, 16>((const int32_t*)(A + (size_t)tokr[q] * a_ld) + 16 * g);
            es::simd<int32_t, 16> s = 0;
#pragma unroll
            for (int x = 0; x < 16; ++x) s = es::dp4a<int32_t>(s, lv[x], es::simd<int32_t, 16>((int32_t)av[x]));
            const es::simd<float, 16> fs = s;
            const es::simd<float, 16> p = fs * asc;
            es::simd<float, 16> m = p * sbr[g];
            es::simd<uint32_t, 16> mb = m.bit_cast_view<uint32_t>();
            mb ^= zero;   /* no mad: the add takes the rounded product */
            const es::simd<float, 16> mr = mb.bit_cast_view<float>();
            acc[g & 1][q] += mr;
        }
    }
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        if ((uint32_t)q >= rn) break;
        es::simd<float, 16> v = acc[0][q] + acc[1][q];
        const es::simd<uint32_t, 16> u = v.bit_cast_view<uint32_t>();
        es::simd<uint32_t, 16> h = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
        h.merge((u >> 16) | 0x40u, (u & 0x7fffffffu) > 0x7f800000u);
        es::simd<uint16_t, 16> o = h;
        if (!liver[q]) o = 0;
        es::block_store<uint16_t, 16>(Y + (size_t)(i + q) * y_ld + n0, o);
    }
}

/* down's shape only (parts 1, both classes N columns, K = 64 G0 / 64 G1): grid T x N / 16 threads, LW a work-group */
template <int G0, int G1, bool B2D, bool DEC2>
void e1_launch(sycl::queue& q, const int8_t* A, const float* As, const ECls& c0, const ECls& c1, const int32_t* sorted,
               const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t a_ld, uint32_t as_ld,
               uint32_t y_ld, uint32_t a_rows, uint32_t lw, uint32_t zero) {
    q.parallel_for(sycl::nd_range<2>({ (size_t)T, (size_t)N / 16 }, { 1, (size_t)lw }),
                   [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const uint32_t i = (uint32_t)it.get_global_id(0), n0 = (uint32_t)it.get_global_id(1) * 16u;
        const int32_t flat = sorted[i];
        if (flat < 0) {
            es::block_store<uint16_t, 16>(Y + (size_t)i * y_ld + n0, es::simd<uint16_t, 16>(0));
            return;
        }
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        const uint32_t e = lo, first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % kMT != 0) return;
        const uint32_t rn = (last - i) < (uint32_t)kMT ? (last - i) : (uint32_t)kMT;
        uint32_t tokr[kMT];
        bool liver[kMT];
#pragma unroll
        for (int qq = 0; qq < kMT; ++qq) {
            tokr[qq] = i + (uint32_t)qq;   /* sorted rows */
            liver[qq] = (uint32_t)qq < rn && sorted[i + qq] >= 0 && i + (uint32_t)qq < a_rows;
        }
        if (e & 1u) e1_cols<G1, B2D, DEC2>(c1, e >> 1, i, rn, tokr, liver, n0, A, As, Y, a_ld, as_ld, y_ld, zero);
        else e1_cols<G0, B2D, DEC2>(c0, e >> 1, i, rn, tokr, liver, n0, A, As, Y, a_ld, as_ld, y_ld, zero);
    });
}

/* ---- inputs, as moe_i4i8_test.cpp's format C cases build them */
uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}
constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xA7;

struct Shape { const char* name; int N, N1, K, K1, parts; bool sorted_a; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

/* One shape at `tokens` tokens: today's kernel and T1 on the same inputs; the bytes compared; then (launches > 0) the
 * per-launch times. Returns false on a mismatch or a broken canary. */
LaunchFn pair_fn = nullptr;   /* RI_PAIR: libxe's xe_rad_gemm_i8a8_gemv */

bool run_shape(sycl::queue& q, LaunchFn fn, const Shape& sh, int tokens, int launches, int runs, uint64_t seed) {
    const int top_k = 10, n_expert = 512, T = tokens * top_k, AR = sh.sorted_a ? T : tokens;
    const int Ncl[2] = { sh.N, sh.N1 }, Kcl[2] = { sh.K, sh.K1 };
    const int Kmax = std::max(sh.K, sh.K1), Nmax = std::max(sh.N, sh.N1);
    const int ne[2] = { (n_expert + 1) / 2, n_expert / 2 };
    /* the routing: the verify mix's 27 distinct experts (or 10 at one token), distinct within a token */
    const int pool_n = tokens == 1 ? 10 : 27;
    std::vector<int> pool;
    for (uint64_t x = 0; (int)pool.size() < pool_n; ++x) {
        const int e = (int)(h32(seed, 1000 + x) % (uint32_t)n_expert);
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    std::vector<int> ex((size_t)T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0, k = 0; s < top_k; ++k) {
            const int e = pool[(size_t)(h32(seed + 50, (uint64_t)t * 1000 + k) % (uint32_t)pool_n)];
            bool dup = false;
            for (int z = 0; z < s; ++z) dup |= ex[(size_t)(t * top_k + z)] == e;
            if (!dup) ex[(size_t)(t * top_k + s++)] = e;
        }
    std::vector<int32_t> off((size_t)n_expert + 1, 0), sorted((size_t)T, -1);
    for (int i = 0; i < T; ++i) ++off[(size_t)ex[(size_t)i] + 1];
    for (int e = 0; e < n_expert; ++e) off[(size_t)e + 1] += off[(size_t)e];
    {
        std::vector<int32_t> cur(off.begin(), off.end() - 1);
        for (int i = 0; i < T; ++i) sorted[(size_t)cur[(size_t)ex[(size_t)i]]++] = i;
    }
    std::vector<int8_t> a((size_t)AR * Kmax);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(h32(seed + 1, i) % 255u) - 127);
    std::vector<float> as((size_t)AR * (Kmax / 128));
    for (size_t i = 0; i < as.size(); ++i) as[i] = 0.001f + (float)(h32(seed + 2, i) % 1000u) * 1e-5f;
    /* today's joined entries (xe_moe_planes_relayout's layout) and T1's repack of them */
    const size_t p0 = (size_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
    std::vector<uint8_t> w[2], wsb[2], tw[2];
    size_t went[2], sent[2], toff[2], s1off[2], tent[2], ttab[2], tsm[2];
    static const float kTab[8] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f };
    for (int c = 0; c < 2; ++c) {
        const size_t rb = (size_t)Kcl[c] / 2;
        toff[c] = (size_t)rad_planes_round((int64_t)(p0 + Ncl[c] * rb));
        s1off[c] = (size_t)rad_planes_round((int64_t)(p0 + (size_t)Ncl[c] * (Kcl[c] / 64)));
        went[c] = (size_t)rad_planes_round((int64_t)toff[c] + 64);
        sent[c] = (size_t)rad_planes_round((int64_t)s1off[c] + 4);
        ttab[c] = (size_t)rad_planes_round((int64_t)(Ncl[c] * rb));
        tsm[c] = ttab[c] + 1024;
        tent[c] = (size_t)rad_planes_round((int64_t)tsm[c] + 4);
        w[c].assign((size_t)ne[c] * went[c], 0);
        wsb[c].assign((size_t)ne[c] * sent[c], 0);
        tw[c].assign((size_t)ne[c] * tent[c], 0);
        for (int e = 0; e < ne[c]; ++e) {
            uint8_t* cw = &w[c][(size_t)e * went[c] + p0];
            for (size_t i = 0; i < (size_t)Ncl[c] * rb; ++i) cw[i] = (uint8_t)h32(seed + 3 + c, (size_t)e * Ncl[c] * rb + i);
            float tab[16];
            const float f = 0.5f + (float)(h32(seed + 9 + c, e) % 100u) * 0.01f;
            for (int t = 0; t < 16; ++t)
                tab[t] = e % 3 == 2 ? (float)((int)(h32(seed + 11 + c, e * 16 + t) % 4001u) - 2000) * 0.01f
                                    : (t < 8 ? kTab[t] : -kTab[t - 8]) * f;
            std::memcpy(&w[c][(size_t)e * went[c] + toff[c]], tab, sizeof tab);
            const float s1 = std::ldexp(1.0f + (float)(e % 5) * 0.125f, -13);
            std::memcpy(&wsb[c][(size_t)e * sent[c] + s1off[c]], &s1, 4);
            for (size_t i = 0; i < (size_t)Ncl[c] * (Kcl[c] / 64); ++i)
                wsb[c][(size_t)e * sent[c] + p0 + i] = (uint8_t)(0x20u + h32(seed + 5 + c, (size_t)e * 65536 + i) % 48u) |
                                                       (uint8_t)((h32(seed + 7 + c, i) & 1u) << 7);
            /* T1: the codes with each 8-element block's nibbles reordered (byte b: elements b, b + 4) */
            uint8_t* t1 = &tw[c][(size_t)e * tent[c]];
            for (int row = 0; row < Ncl[c]; ++row)
                for (int k0 = 0; k0 < Kcl[c]; k0 += 8) {
                    int cd[8];
                    for (int z = 0; z < 8; ++z) {
                        const uint8_t b = cw[(size_t)row * rb + (size_t)((k0 + z) / 2)];
                        cd[z] = ((k0 + z) & 1) ? (b >> 4) : (b & 15);
                    }
                    for (int b = 0; b < 4; ++b) t1[(size_t)row * rb + (size_t)(k0 / 2 + b)] = (uint8_t)(cd[b] | (cd[b + 4] << 4));
                }
            /* T1: the pair table and the scale factor, as the kernel computes them in f32 */
            float amax = 0.0f;
            for (int t = 0; t < 16; ++t) amax = std::fmax(amax, std::fabs(tab[t]));
            uint32_t* tt = (uint32_t*)(t1 + ttab[c]);
            for (int x = 0; x < 256; ++x) {
                const float l0 = tab[x & 15], l1 = tab[x >> 4];
                const int i0 = amax > 0.0f ? (int)std::nearbyint(l0 * 127.0f / amax) : (int)l0;
                const int i1 = amax > 0.0f ? (int)std::nearbyint(l1 * 127.0f / amax) : (int)l1;
                tt[x] = 0x40000000u | ((uint32_t)i0 & 255u) | (((uint32_t)i1 & 255u) << 8);
            }
            const float smul = amax > 0.0f ? s1 * (amax / 127.0f) : s1;
            std::memcpy(t1 + tsm[c], &smul, 4);
        }
    }
    /* T1b (down's form only): T1's entries with the codes moved to sub-group order (t1b_off); the table and the scale
     * factor where they were */
    const bool gl = Kmax <= 512;
    std::vector<uint8_t> tbw[2];
    if (gl)
        for (int c = 0; c < 2; ++c) {
            const uint32_t cpr = (uint32_t)Kcl[c] / 32u;
            if (cpr % 4u || Ncl[c] % 16) { std::printf("T1b: K %d / N %d do not tile\n", Kcl[c], Ncl[c]); return false; }
            tbw[c] = tw[c];
            const size_t rb = (size_t)Kcl[c] / 2;
            for (int e = 0; e < ne[c]; ++e) {
                const uint8_t* src = &tw[c][(size_t)e * tent[c]];
                uint8_t* dst = &tbw[c][(size_t)e * tent[c]];
                for (int row = 0; row < Ncl[c]; ++row)
                    for (uint32_t jc = 0; jc < cpr; ++jc)
                        std::memcpy(dst + t1b_off((uint32_t)row, jc, cpr), src + (size_t)row * rb + jc * 16u, 16);
            }
        }
    /* T1c (run 207): T1's entries with the codes in T1b's pieces, each piece's 16 lanes' dwords striped (t1c_off) */
    std::vector<uint8_t> tcw[2];
    if (gl)
        for (int c = 0; c < 2; ++c) {
            const uint32_t cpr = (uint32_t)Kcl[c] / 32u;
            tcw[c] = tw[c];
            const size_t rb = (size_t)Kcl[c] / 2;
            for (int e = 0; e < ne[c]; ++e) {
                const uint8_t* src = &tw[c][(size_t)e * tent[c]];
                uint8_t* dst = &tcw[c][(size_t)e * tent[c]];
                for (int row = 0; row < Ncl[c]; ++row)
                    for (uint32_t jc = 0; jc < cpr; ++jc)
                        for (uint32_t k = 0; k < 4; ++k)
                            std::memcpy(dst + t1c_off((uint32_t)row, jc, cpr, k), src + (size_t)row * rb + jc * 16u + k * 4u, 4);
            }
        }
    /* device buffers */
    const size_t ybytes = (size_t)T * Nmax * 2;
    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    uint8_t *dw[2], *dws[2], *dtw[2], *dtbw[2] = { nullptr, nullptr }, *dtcw[2] = { nullptr, nullptr };
    for (int c = 0; c < 2; ++c) {
        if (gl) {
            dtcw[c] = sycl::malloc_device<uint8_t>(tcw[c].size() + 64, q);
            q.memcpy(dtcw[c], tcw[c].data(), tcw[c].size());
        }
        dw[c] = sycl::malloc_device<uint8_t>(w[c].size() + 64, q);
        dws[c] = sycl::malloc_device<uint8_t>(wsb[c].size() + 64, q);
        dtw[c] = sycl::malloc_device<uint8_t>(tw[c].size() + 64, q);
        q.memcpy(dw[c], w[c].data(), w[c].size());
        q.memcpy(dws[c], wsb[c].data(), wsb[c].size());
        q.memcpy(dtw[c], tw[c].data(), tw[c].size());
        if (gl) {
            dtbw[c] = sycl::malloc_device<uint8_t>(tbw[c].size() + 64, q);
            q.memcpy(dtbw[c], tbw[c].data(), tbw[c].size());
        }
    }
    int32_t* dso = sycl::malloc_device<int32_t>((size_t)T, q);
    int32_t* doff = sycl::malloc_device<int32_t>(off.size(), q);
    uint8_t* dy0 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    uint8_t* dy1 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    uint8_t* dy2 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memset(dy2, kPat, ybytes + 2 * kCanary);
    uint8_t* dy3 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    uint8_t* dy4 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memset(dy3, kPat, ybytes + 2 * kCanary);
    q.memset(dy4, kPat, ybytes + 2 * kCanary);
    uint8_t* dy5 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);   /* T1c */
    q.memset(dy5, kPat, ybytes + 2 * kCanary);
    q.memcpy(da, a.data(), a.size());
    q.memcpy(das, as.data(), as.size() * 4);
    q.memcpy(dso, sorted.data(), sorted.size() * 4);
    q.memcpy(doff, off.data(), off.size() * 4);
    q.memset(dy0, kPat, ybytes + 2 * kCanary);
    q.memset(dy1, kPat, ybytes + 2 * kCanary);
    q.wait();
    /* today's call, as moe_i4i8_test.cpp makes it (stacked joined entries) */
    RadTensor t[9];
    t[0] = tn(da, RAD_I8, { AR, Kmax });
    t[1] = tn(das, RAD_F32, { AR, Kmax / 128 });
    for (int c = 0; c < 2; ++c) {
        RadTensor tw3 = tn(dw[c], RAD_U4, { ne[c], Ncl[c], Kcl[c] });
        RadTensor ts3 = tn(dws[c], RAD_F8E4M3, { ne[c], Ncl[c], Kcl[c] / 64 });
        tw3.stride[0] = (int64_t)went[c] * 2;
        ts3.stride[0] = (int64_t)sent[c];
        t[c == 0 ? 2 : 6] = tw3;
        t[c == 0 ? 3 : 7] = ts3;
    }
    t[4] = tn(dso, RAD_I32, { T });
    t[5] = tn(doff, RAD_I32, { n_expert + 1 });
    t[8] = tn(dy0 + kCanary, RAD_BF16, { T, Nmax });
    RadParam all[11]{};
    const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
    const long long iv[] = { AR, sh.N, sh.K, n_expert, top_k, 128, sh.N1, sh.K1, sh.parts };
    for (int i = 0; i < 9; ++i) { all[i].key = ik[i]; all[i].kind = RAD_P_INT; all[i].ival = iv[i]; }
    char ord[8], dt[16];
    std::snprintf(ord, sizeof ord, "%s", sh.sorted_a ? "sorted" : "token");
    std::snprintf(dt, sizeof dt, "w4nl64i8h");
    all[9].key = "a_order"; all[9].kind = RAD_P_STR; all[9].sval = ord;
    all[10].key = "dtype"; all[10].kind = RAD_P_STR; all[10].sval = dt;
    RadArgs args{};
    args.t = t; args.n_t = 9; args.p = all; args.n_p = 11;
    TCls tc[2];
    for (int c = 0; c < 2; ++c)
        tc[c] = TCls{ dtw[c], tent[c], dws[c], sent[c], (uint32_t)Ncl[c], (uint32_t)Kcl[c], (uint32_t)(Kcl[c] / 2),
                      (uint32_t)(Kcl[c] / 64), (uint32_t)p0, (uint32_t)ttab[c], (uint32_t)tsm[c] };
    TCls tcb[2] = { tc[0], tc[1] };
    for (int c = 0; c < 2; ++c) tcb[c].base = dtbw[c];
    auto today = [&]() { return fn(&args, (RadStream)&q); };
    TCls tcc[2] = { tc[0], tc[1] };   /* T1c */
    for (int c = 0; c < 2; ++c) tcc[c].base = dtcw[c];
    auto t1f = [&](const TCls* cls, uint8_t* dy, bool tb, bool tcf = false) {
        t1_launch(q, da, das, cls[0], cls[1], dso, doff, (uint16_t*)(dy + kCanary), (uint32_t)T, (uint32_t)Nmax,
                  (uint32_t)Kmax, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)Kmax, (uint32_t)(Kmax / 128),
                  (uint32_t)Nmax, sh.sorted_a, (uint32_t)sh.parts, (uint32_t)AR, tb, tcf);
        return 0;
    };
    auto t1 = [&]() { return t1f(tc, dy1, false); };
    auto t1b = [&]() { return t1f(tcb, dy2, true); };
    auto t1c = [&]() { return t1f(tcc, dy5, false, true); };
    /* E1 / E1b: down's shape on a GPU (the ESIMD form has no CPU device) */
    bool e1on = gl && q.get_device().is_gpu() && sh.parts == 1 && sh.N == sh.N1 && Kcl[0] == 384 && Kcl[1] == 256;
    const char* lwv = std::getenv("RI_E1_LW");
    const uint32_t e1lw = lwv ? (uint32_t)std::atoi(lwv) : 16u;
    ECls ec[2];
    for (int c = 0; c < 2; ++c)
        ec[c] = ECls{ dw[c], went[c], dtw[c], tent[c], dws[c], sent[c], (uint32_t)p0, (uint32_t)ttab[c], (uint32_t)tsm[c],
                      (uint32_t)p0, (uint32_t)Ncl[c], (uint32_t)(Kcl[c] / 2) };
    /* run 206 after P99: the gather form (E1) lost to the 2D load; dy3 now holds E2 (2D + DEC2), dy4 E1b */
    uint32_t lw_now = e1lw;
    auto e1f = [&](uint8_t* dy, bool dec2) {
        uint16_t* y = (uint16_t*)(dy + kCanary);
        if (dec2)
            e1_launch<6, 4, true, true>(q, da, das, ec[0], ec[1], dso, doff, y, (uint32_t)T, (uint32_t)Nmax,
                                        (uint32_t)n_expert, (uint32_t)Kmax, (uint32_t)(Kmax / 128), (uint32_t)Nmax,
                                        (uint32_t)AR, lw_now, 0u);
        else
            e1_launch<6, 4, true, false>(q, da, das, ec[0], ec[1], dso, doff, y, (uint32_t)T, (uint32_t)Nmax,
                                         (uint32_t)n_expert, (uint32_t)Kmax, (uint32_t)(Kmax / 128), (uint32_t)Nmax,
                                         (uint32_t)AR, lw_now, 0u);
        return 0;
    };
    auto e1 = [&]() { return e1f(dy3, true); };
    auto e1b = [&]() { return e1f(dy4, false); };
    const int rc = today();
    t1();
    if (gl) { t1b(); t1c(); }
    q.wait_and_throw();
    if (e1on) {
        try {
            e1();
            e1b();
            q.wait_and_throw();
        } catch (const std::exception& x) {
            std::printf("E1: %s\n", x.what());
            e1on = false;
        }
    }
    std::vector<uint8_t> h0(ybytes + 2 * kCanary), h1(ybytes + 2 * kCanary), h2(ybytes + 2 * kCanary);
    std::vector<uint8_t> h3(ybytes + 2 * kCanary), h4(ybytes + 2 * kCanary), h5(ybytes + 2 * kCanary);
    q.memcpy(h5.data(), dy5, h5.size());
    q.memcpy(h0.data(), dy0, h0.size());
    q.memcpy(h1.data(), dy1, h1.size());
    q.memcpy(h3.data(), dy3, h3.size());
    q.memcpy(h4.data(), dy4, h4.size());
    q.memcpy(h2.data(), dy2, h2.size()).wait();
    /* one form's output against today's: differing outputs, the first of them, broken canaries */
    auto cmp = [&](const std::vector<uint8_t>& hx, size_t* first) {
        size_t diff = 0;
        *first = (size_t)-1;
        for (size_t i = 0; i < ybytes; i += 2) {
            const bool d = h0[kCanary + i] != hx[kCanary + i] || h0[kCanary + i + 1] != hx[kCanary + i + 1];
            if (d && *first == (size_t)-1) *first = i / 2;
            diff += d;
        }
        return diff;
    };
    auto canary_ok = [&](const std::vector<uint8_t>& hx) {
        bool ok = true;
        for (size_t i = 0; i < kCanary; ++i) ok &= hx[i] == kPat && hx[kCanary + ybytes + i] == kPat;
        return ok;
    };
    size_t nonzero = 0;
    for (size_t i = 0; i < ybytes; i += 2) nonzero += h0[kCanary + i] != 0 || h0[kCanary + i + 1] != 0;
    size_t f1 = 0, f2 = 0, f5 = 0;
    const size_t diff = cmp(h1, &f1), diff2 = gl ? cmp(h2, &f2) : 0, diff5 = gl ? cmp(h5, &f5) : 0;
    const bool canary = canary_ok(h0) && canary_ok(h1) && (!gl || (canary_ok(h2) && canary_ok(h5)));
    const bool ok = rc == RAD_OK && canary && diff == 0 && diff2 == 0 && diff5 == 0 && nonzero > 0;
    if (diff5) std::printf("  T1c's first difference at output %zu (row %zu, column %zu)\n", f5, f5 / Nmax, f5 % Nmax);
    std::printf("%s %s tokens %d: T %d over %d experts, N %d/%d K %d/%d -- today rc %d; T1: %zu of %zu outputs differ%s"
                "%s; %zu nonzero; canaries %s\n", ok ? "PASS" : "FAIL", sh.name, tokens, T, pool_n, sh.N, sh.N1, sh.K,
                sh.K1, rc, diff, ybytes / 2, diff ? "" : " (bit-identical)",
                gl ? (diff2 ? "; T1b DIFFERS" : diff5 ? "; T1b bit-identical; T1c DIFFERS"
                                                      : "; T1b bit-identical; T1c bit-identical")
                   : "",
                nonzero, canary ? "intact" : "BROKEN");
    if (diff) std::printf("  T1's first difference at output %zu (row %zu, column %zu)\n", f1, f1 / Nmax, f1 % Nmax);
    if (diff2) std::printf("  T1b: %zu differ, the first at output %zu (row %zu, column %zu)\n", diff2, f2, f2 / Nmax, f2 % Nmax);
    bool e1bad = false;
    if (e1on) {
        size_t f3 = 0, f4 = 0;
        const size_t d3 = cmp(h3, &f3), d4 = cmp(h4, &f4);
        const bool c3 = canary_ok(h3), c4 = canary_ok(h4);
        std::printf("E1 %s tokens %d (LW %u): E2 %zu of %zu outputs differ%s, E1b %zu differ%s; canaries %s\n", sh.name, tokens,
                    e1lw, d3, ybytes / 2, d3 ? "" : " (bit-identical)", d4, d4 ? "" : " (bit-identical)",
                    c3 && c4 ? "intact" : "BROKEN");
        if (d3) std::printf("  E2's first difference at output %zu (row %zu, column %zu): %04x vs today %04x\n", f3, f3 / Nmax,
                            f3 % Nmax, ((const uint16_t*)(h3.data() + kCanary))[f3], ((const uint16_t*)(h0.data() + kCanary))[f3]);
        if (d4) std::printf("  E1b's first difference at output %zu (row %zu, column %zu): %04x vs today %04x\n", f4, f4 / Nmax,
                            f4 % Nmax, ((const uint16_t*)(h4.data() + kCanary))[f4], ((const uint16_t*)(h0.data() + kCanary))[f4]);
        e1bad = d3 || d4 || !c3 || !c4;
    }
    /* T2 (gate_up's shape): CB column blocks a work-group over one staging; bytes always, times on a card */
    bool t2bad = false;
    if (!gl && ok && Kcl[0] == 2560 && Kcl[1] == 2560)
        for (int cbv : { 1, 2, 4, 8 }) {
            q.memset(dy2, kPat, ybytes + 2 * kCanary).wait();
            auto t2 = [&]() {
                t2_launch(q, cbv, da, das, tc[0], tc[1], dso, doff, (uint16_t*)(dy2 + kCanary), (uint32_t)T, (uint32_t)Nmax,
                          (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)Kmax, (uint32_t)(Kmax / 128), (uint32_t)Nmax,
                          sh.sorted_a, (uint32_t)sh.parts, (uint32_t)AR);
                return 0;
            };
            t2();
            q.wait_and_throw();
            q.memcpy(h2.data(), dy2, h2.size()).wait();
            size_t f5 = 0;
            const size_t d5 = cmp(h2, &f5);
            const bool c5 = canary_ok(h2);
            t2bad |= d5 != 0 || !c5;
            std::printf("T2 %s tokens %d CB %d: %zu of %zu outputs differ%s; canaries %s\n", sh.name, tokens, cbv, d5,
                        ybytes / 2, d5 ? "" : " (bit-identical)", c5 ? "intact" : "BROKEN");
            if (d5) std::printf("  T2's first difference at output %zu (row %zu, column %zu)\n", f5, f5 / Nmax, f5 % Nmax);
            if (launches > 0 && !d5) {
                auto tm = [&](auto&& fn2) {
                    for (int z = 0; z < 3; ++z) fn2();
                    q.wait();
                    std::vector<double> per;
                    for (int r = 0; r < runs; ++r) {
                        const auto a0 = std::chrono::steady_clock::now();
                        for (int z = 0; z < launches; ++z) fn2();
                        q.wait();
                        per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0)
                                          .count() / launches);
                    }
                    return median(per);
                };
                const double ut = tm(today), u5 = tm(t2), u1b = tm(t1), ut2 = tm(today);
                const double b5 = std::min(ut, ut2);
                std::printf("RESULT-T2 %s tokens %d CB %d: today %.1f us (re-timed %.1f), T2 %.1f us -> %.0f %%, T1 %.1f us "
                            "-> %.0f %%\n", sh.name, tokens, cbv, ut, ut2, u5, 100.0 * u5 / b5, u1b, 100.0 * u1b / b5);
            }
        }
    bool pair_bad = false;   /* RI_PAIR: a form's outputs differed, or a call failed */
    if (ok && launches > 0) {
        auto time = [&](auto&& f) {
            for (int z = 0; z < 3; ++z) f();
            q.wait();
            std::vector<double> per;
            for (int r = 0; r < runs; ++r) {
                const auto a0 = std::chrono::steady_clock::now();
                for (int z = 0; z < launches; ++z) f();
                q.wait();
                per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count() /
                              launches);
            }
            return median(per);
        };
        const double u0 = time(today), u1 = time(t1), u2 = gl ? time(t1b) : 0.0, u5 = gl ? time(t1c) : 0.0,
                     u0b = time(today);
        const double base = std::min(u0, u0b);
        std::printf("RESULT %s tokens %d: today %.1f us (re-timed %.1f), T1 %.1f us a launch -> %.0f %% of today", sh.name,
                    tokens, u0, u0b, u1, 100.0 * u1 / base);
        if (gl) std::printf("; T1b %.1f us -> %.0f %%; T1c %.1f us -> %.0f %%", u2, 100.0 * u2 / base, u5, 100.0 * u5 / base);
        std::printf("\n");
        /* E1 timed even when it differs (its speed says whether the exact form is worth the work) */
        if (e1on) {
            for (uint32_t lw : { 8u, 16u, 32u }) {   /* the work-group size, 160 column blocks */
                lw_now = lw;
                const double u3 = time(e1), u4 = time(e1b), u0c = time(today);
                const double b2 = std::min(base, u0c);
                std::printf("RESULT-E1 %s tokens %d (LW %u): today %.1f us (re-timed %.1f), E2 %.1f us -> %.0f %%, E1b %.1f us "
                            "-> %.0f %%%s\n", sh.name, tokens, lw, base, u0c, u3, 100.0 * u3 / b2, u4, 100.0 * u4 / b2,
                            e1bad ? " (OUTPUTS DIFFER)" : "");
            }
            lw_now = e1lw;
        }
        /* RI_PAIR=1 (run 193, owner note 191 item 2: independent kernels side by side, briefs/2026-11-05-concurrency.md),
         * gate_up's shape at 4 tokens: today's routed-expert GEMM beside a shared-expert-shaped int8 GEMV (libxe's
         * xe_rad_gemm_i8a8_gemv, N 1280 x K 2560, the same 4 activation rows), timed as a pair in three forms -- one
         * in-order queue (today: back to back), one out-of-order queue (no barrier between them), two in-order queues
         * -- with every form's two outputs compared byte for byte against the first's. */
        if (std::getenv("RI_PAIR") && !sh.sorted_a && tokens == 4 && pair_fn) {
            const int GN = 1280, GK = Kmax;
            std::vector<int8_t> gb((size_t)GN * GK);
            for (size_t i = 0; i < gb.size(); ++i) gb[i] = (int8_t)((int)(h32(seed + 70, i) % 255u) - 127);
            std::vector<uint16_t> gbs((size_t)GN * (GK / 128));
            for (size_t i = 0; i < gbs.size(); ++i) {
                const float f = 0.0005f + (float)(h32(seed + 71, i) % 1000u) * 1e-6f;
                uint32_t u; std::memcpy(&u, &f, 4); gbs[i] = (uint16_t)(u >> 16);
            }
            int8_t* dgb = sycl::malloc_device<int8_t>(gb.size(), q);
            uint16_t* dgbs = sycl::malloc_device<uint16_t>(gbs.size(), q);
            uint16_t* dgy = sycl::malloc_device<uint16_t>((size_t)AR * GN, q);
            q.memcpy(dgb, gb.data(), gb.size());
            q.memcpy(dgbs, gbs.data(), gbs.size() * 2).wait();
            RadTensor gt[5];
            gt[0] = tn(da, RAD_I8, { AR, GK });
            gt[1] = tn(das, RAD_F32, { AR, GK / 128 });
            gt[2] = tn(dgb, RAD_I8, { GN, GK });
            gt[3] = tn(dgbs, RAD_BF16, { GN, GK / 128 });
            gt[4] = tn(dgy, RAD_BF16, { AR, GN });
            RadParam gp[3]{};
            gp[0].key = "N"; gp[0].kind = RAD_P_INT; gp[0].ival = GN;
            gp[1].key = "K"; gp[1].kind = RAD_P_INT; gp[1].ival = GK;
            gp[2].key = "group"; gp[2].kind = RAD_P_INT; gp[2].ival = 128;
            RadArgs ga{};
            ga.t = gt; ga.n_t = 5; ga.p = gp; ga.n_p = 3;
            sycl::queue q2(q.get_context(), q.get_device(), sycl::property::queue::in_order());
            sycl::queue qo(q.get_context(), q.get_device());   /* out of order */
            auto gemv = [&](sycl::queue& qq) { return pair_fn(&ga, (RadStream)&qq); };
            auto moe = [&](sycl::queue& qq) { return fn(&args, (RadStream)&qq); };
            /* the reference outputs: both on q */
            int prc = moe(q) | gemv(q);
            q.wait_and_throw();
            std::vector<uint8_t> r_moe(ybytes), r_g((size_t)AR * GN * 2), c_moe(ybytes), c_g((size_t)AR * GN * 2);
            q.memcpy(r_moe.data(), dy0 + kCanary, ybytes);
            q.memcpy(r_g.data(), dgy, r_g.size()).wait();
            auto form_ok = [&]() {
                q.memcpy(c_moe.data(), dy0 + kCanary, ybytes);
                q.memcpy(c_g.data(), dgy, c_g.size()).wait();
                return c_moe == r_moe && c_g == r_g;
            };
            auto pair_time = [&](auto&& f) {
                for (int z = 0; z < 3; ++z) f();
                q.wait(); q2.wait(); qo.wait();
                std::vector<double> per;
                for (int r = 0; r < runs; ++r) {
                    const auto a0 = std::chrono::steady_clock::now();
                    for (int z = 0; z < launches; ++z) f();
                    q.wait(); q2.wait(); qo.wait();
                    per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count() /
                                  launches);
                }
                return median(per);
            };
            const double t_m = pair_time([&]() { moe(q); }), t_g = pair_time([&]() { gemv(q); });
            const double f1 = pair_time([&]() { moe(q); gemv(q); });
            const bool ok1 = form_ok();
            const double f2 = pair_time([&]() { moe(qo); gemv(qo); });
            const bool ok2 = form_ok();
            const double f3 = pair_time([&]() { moe(q); gemv(q2); });
            const bool ok3 = form_ok();
            std::printf("PAIR gate_up tokens 4 + int8 GEMV %d x %d: rc %d; alone %.1f + %.1f us; one in-order queue %.1f us%s; "
                        "one out-of-order queue %.1f us (%.0f %%)%s; two in-order queues %.1f us (%.0f %%)%s\n",
                        GN, GK, prc, t_m, t_g, f1, ok1 ? "" : " (OUTPUTS DIFFER)", f2, 100.0 * f2 / f1,
                        ok2 ? "" : " (OUTPUTS DIFFER)", f3, 100.0 * f3 / f1, ok3 ? "" : " (OUTPUTS DIFFER)");
            pair_bad = prc != RAD_OK || !ok1 || !ok2 || !ok3;
            sycl::free(dgb, q); sycl::free(dgbs, q); sycl::free(dgy, q);
        }
    }
    for (int c = 0; c < 2; ++c) {
        sycl::free(dw[c], q); sycl::free(dws[c], q); sycl::free(dtw[c], q);
        if (dtbw[c]) sycl::free(dtbw[c], q);
        if (dtcw[c]) sycl::free(dtcw[c], q);
    }
    sycl::free(da, q); sycl::free(das, q); sycl::free(dso, q); sycl::free(doff, q); sycl::free(dy0, q); sycl::free(dy1, q);
    sycl::free(dy2, q); sycl::free(dy3, q); sycl::free(dy4, q); sycl::free(dy5, q);
    return ok && !pair_bad && !e1bad && !t2bad;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: xe_fc_tile_probe LIBXE.so [LAUNCHES=48] [RUNS=5]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_w4nl_i8");
    if (!fn) { std::printf("no xe_rad_moe_gemm_w4nl_i8\n"); return 2; }
    pair_fn = (LaunchFn)dlsym(h, "xe_rad_gemm_i8a8_gemv");
    const int launches = argc > 2 ? std::atoi(argv[2]) : 48, runs = argc > 3 ? std::atoi(argv[3]) : 5;
    const char* dv = std::getenv("RI_DEVICE");
    sycl::queue q = (dv && !std::strcmp(dv, "cpu")) ? sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order())
                                                    : sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("fc_tile_probe on %s, %d launches x %d runs\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                launches, runs);
    const Shape gate_up{ "gate_up", 768, 512, 2560, 2560, 2, false }, down{ "down", 2560, 2560, 384, 256, 1, true };
    int fails = 0;
    for (int tokens : { 4, 1 }) {
        fails += !run_shape(q, fn, gate_up, tokens, launches, runs, 101 + tokens);
        fails += !run_shape(q, fn, down, tokens, launches, runs, 201 + tokens);
    }
    std::printf("fc_tile_probe: %d of 4 cases failed\n", fails);
    return fails ? 1 : 0;
}
