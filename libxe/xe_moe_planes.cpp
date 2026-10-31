/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_moe_planes.cpp -- moe_gemm_q at w4nl64a8h over the experts' CANONICAL planes, joined as
 * abi/rad_planes.h defines (the form libref reads): the codes view "codes,table" (u4 codes, low
 * nibble first, indexing 16 f32 levels) and the scale view "scale,scale.1" (E4M3 per row per 64 of
 * K, times one f32). The layout hooks are rad_planes.h's, a copy, so the load does no relayout for
 * the experts (owner note 4: intel-host's CPU must stay cool at load); libr4d's packed row
 * (ports/r4d_moe.cpp, r4d_layout.cpp) rearranges every expert on the host at every load.
 *
 * Arithmetic as libref's (ref_moe.cpp, quant_dot): per 64-K group the sub-sum of level * activation,
 * then acc += (sub * a_scale(row, k / 128)) * (e4m3(scale) * scale.1); rows outside the routing and
 * columns past a narrower class's width are zeros. Experts are two classes by parity (entry e / 2 of
 * class e % 2), as the declaration splits them. A sub-group computes one output; lanes take the
 * weight groups. The planes' offsets are computed on the host from the shapes (rad_planes_plan's
 * arithmetic), so a launch reads nothing back from the card.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "rad_planes.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct PCls {
    const uint8_t* base;     /* the stacked plane, or null */
    const void* const* tab;  /* the pointer table, or null */
    uint64_t stride;         /* bytes an entry of a stacked plane */
    const uint8_t* sbase;
    const void* const* stab;
    uint64_t sstride;
    uint32_t n, k;
    uint32_t c_off, t_off;   /* codes and table inside a codes entry */
    uint32_t s_off, s1_off;  /* scale and scale.1 inside a scale entry */
    uint32_t w_ld, s_ld;     /* LOGICAL form only: row pitches in elements */
};

inline float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}
inline const uint8_t* entry_of(const uint8_t* base, const void* const* tab, uint64_t stride, uint32_t e) {
    return tab ? (const uint8_t*)tab[e] : base + (uint64_t)e * stride;
}

/* LOGICAL: the operands hold the values themselves, as rad-kbench and the oracle pass them
 * (libref's enc_selection_logical): an E4M3 level per element and a bf16 scale product per (row,
 * 64 of K). Otherwise the joined planes. */
template <bool LOGICAL>
void moe_planes_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                       const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                       uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                       uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                       uint32_t a_rows) {
    const int lane = (int)threadIdx.x & 31;
    const uint32_t n = blockIdx.x * 8u + ((uint32_t)threadIdx.x >> 5);
    const uint32_t i = blockIdx.y;
    if (n >= N || i >= T) return;
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
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    float acc = 0.0f;
    if (e >= 0 && j < fc) {
        const uint32_t entry = (uint32_t)e >> 1, r = pp * fc + j;
        const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry);
        const uint8_t* se = entry_of(c.sbase, c.stab, c.sstride, entry);
        const uint8_t* arow = A + (size_t)tok * a_ld;
        if constexpr (LOGICAL) {
            const uint8_t* wrow = we + (size_t)r * c.w_ld;
            const uint16_t* srow = (const uint16_t*)se + (size_t)r * c.s_ld;
            for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
                const uint8_t* wb = wrow + (size_t)g * 64;
                const uint8_t* ab = arow + (size_t)g * 64;
                float sub = 0.0f;
                for (int m = 0; m < 64; ++m) sub += e4m3(wb[m]) * e4m3(ab[m]);
                const float sa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
                const float sb = __uint_as_float((uint32_t)srow[g] << 16);
                acc += (sub * sa) * sb;
            }
        } else {
            const uint8_t* wrow = we + c.c_off + (size_t)r * (c.k / 2u);
            const float* lvp = (const float*)(we + c.t_off);
            const uint8_t* srow = se + c.s_off + (size_t)r * (c.k / 64u);
            const float s1 = *(const float*)(se + c.s1_off);
            float lv[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) lv[t] = lvp[t];
            for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
                const uint8_t* wb = wrow + (size_t)g * 32;   /* 64 codes, two a byte, low nibble first */
                const uint8_t* ab = arow + (size_t)g * 64;
                float sub = 0.0f;
                for (int m = 0; m < 32; ++m) {
                    const uint32_t b = wb[m];
                    sub += lv[b & 15u] * e4m3(ab[2 * m]);
                    sub += lv[b >> 4] * e4m3(ab[2 * m + 1]);
                }
                const float sa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
                const float sb = e4m3(srow[g]) * s1;
                acc += (sub * sa) * sb;
            }
        }
        for (int x = 16; x >= 1; x >>= 1) acc += __shfl_xor(acc, x);
    }
    if (lane == 0) Y[(size_t)i * y_ld + n] = to_bf16(acc);
}

/* The decode kernel (milestone 2, stage 1, kernel 1): round 2 M6's best configuration brought to the
 * planes form -- sub-group 16, 16 lanes a row along K with a 16-byte load each (32 codes), two rows
 * a team sharing the activation, 512 work-items a work-group (32 teams, 64 rows), one routed row a
 * work-group. Per work-group the activation row is staged once in work-group memory as f16 pairs
 * (exact for E4M3), with its per-128 scales, a 256-entry table byte -> the two levels (f32, from the
 * entry's 16-level table) and the 256 E4M3 values. Per code byte: one table read, one activation
 * read, two f32 FMAs (level * activation is exact in f32). Per 16-byte chunk the partial sum takes
 * the chunk's 64-group E4M3 scale and the activation's per-128 scale, as libref's quant_dot does per
 * 64-group ((sub * sa) * sb); the entry's scale.1 multiplies the row's total at the store. Rounding
 * differs from libref in the order of the sums (a half-group at a time, lanes meeting in a
 * butterfly): a tolerance check, as the old kernel was. Rows outside the routing and columns past a
 * narrower class's width write zeros, as before. K up to kV2MaxK; larger K and the LOGICAL form
 * take the kernel above. RADIANCE_XE_MOE_V1=1 forces the old kernel (the A/B on card1). */
constexpr int kV2WG = 512, kV2LPR = 16, kV2R = 2;
constexpr int kV2Teams = kV2WG / kV2LPR;        /* 32 */
constexpr int kV2RPW = kV2Teams * kV2R;         /* 64 rows a work-group */
constexpr uint32_t kV2MaxK = 4096;

inline float h2f(uint32_t bits16) { return (float)sycl::bit_cast<sycl::half>((uint16_t)bits16); }
inline uint32_t f2h(float f) { return (uint32_t)sycl::bit_cast<uint16_t>((sycl::half)f); }

/* UPL: 16-byte chunks a lane (K / 32 / 16, rounded up; the last may be partial): every lane's weight
 * loads of the row pair are issued together before the arithmetic, so the memory system sees
 * 2 x UPL loads a lane in flight (v2 issued one and waited: 144 us, 114 GB/s, 3x round 2 M6's 43 us
 * for this configuration). The level table is one packed f16 pair a byte and the products are f16,
 * exact (levels and activations are E4M3), summed in f32, as M6's v2 (DEC1). */
template <int UPL, int R = kV2R>
void moe_planes_v2_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                          const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                          uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                          uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                          uint32_t a_rows) {
    XE_SHARED(uint32_t, s_a, kV2MaxK / 2);      /* the activation row as f16 pairs */
    XE_SHARED(float, s_as, kV2MaxK / 128);      /* its per-128 scales */
    XE_SHARED(uint32_t, s_lut, 256);            /* byte -> (level of its low nibble, of its high) as f16 */
    XE_SHARED(float, s_e4, 256);                /* byte -> E4M3 value */
    const int tid = (int)threadIdx.x;
    const uint32_t i = blockIdx.y, nb = blockIdx.x;
    if (i >= T) return;
    /* the routed row's expert and token: uniform over the work-group */
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
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    const uint32_t K = e >= 0 ? c.k : 0u;
    const uint8_t* we = nullptr;
    const uint8_t* se = nullptr;
    float s1 = 0.0f;
    if (e >= 0) {
        const uint32_t entry = (uint32_t)e >> 1;
        we = entry_of(c.base, c.tab, c.stride, entry);
        se = entry_of(c.sbase, c.stab, c.sstride, entry);
        s1 = *(const float*)(se + c.s1_off);
        const float* lvp = (const float*)(we + c.t_off);
        const uint8_t* arow = A + (size_t)tok * a_ld;
        for (uint32_t b = (uint32_t)tid; b < K / 2u; b += kV2WG)
            s_a[b] = f2h(e4m3(arow[2 * b])) | (f2h(e4m3(arow[2 * b + 1])) << 16);
        for (uint32_t g = (uint32_t)tid; g < K / 128u; g += kV2WG) s_as[g] = As[(size_t)tok * as_ld + g];
        for (uint32_t t = (uint32_t)tid; t < 256u; t += kV2WG) {
            s_e4[t] = e4m3(t);
            s_lut[t] = f2h(lvp[t & 15u]) | (f2h(lvp[t >> 4]) << 16);
        }
    }
    __syncthreads();
    const int team = tid / kV2LPR, l = tid % kV2LPR;
    float acc[R];
    uint32_t rowr[R];
    bool rowok[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint32_t n = nb * (kV2Teams * R) + (uint32_t)(team * R + r);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[r] = e >= 0 && n < N && j < fc;
        rowr[r] = pp * fc + j;
        acc[r] = 0.0f;
    }
    if (e >= 0) {
        const uint32_t CPR = K / 32u;                   /* 16-byte chunks a row */
        const uint32_t rb = K / 2u, sb = K / 64u;       /* bytes a codes row, a scale row */
        const uint8_t* wrow0 = we + c.c_off;
        const uint8_t* srow0 = se + c.s_off;
        /* all weight loads of this lane first (chunk u of row r), then the arithmetic */
        uint4 w[UPL][R];
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
#pragma unroll
            for (int r = 0; r < R; ++r)
                w[u][r] = (jc < CPR && rowok[r]) ? *(const uint4*)(wrow0 + (size_t)rowr[r] * rb + jc * 16u)
                                                 : uint4{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
            if (jc >= CPR) break;                       /* uniform over the sub-group's tail */
            const uint32_t k0 = jc * 32u;               /* the chunk's first K index (32 codes) */
            const float asc = s_as[k0 / 128u];
            const uint32_t g = k0 / 64u;
            uint32_t a[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) a[t] = s_a[k0 / 2u + (uint32_t)t];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t d[4] = { w[u][r].x, w[u][r].y, w[u][r].z, w[u][r].w };
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const uint32_t byte = (d[x] >> (8 * b)) & 0xffu;
                        const uint32_t ab = a[x * 4 + b];
                        const uint32_t lv = s_lut[byte];
                        const sycl::half l0 = sycl::bit_cast<sycl::half>((uint16_t)(lv & 0xffffu));
                        const sycl::half l1 = sycl::bit_cast<sycl::half>((uint16_t)(lv >> 16));
                        const sycl::half a0 = sycl::bit_cast<sycl::half>((uint16_t)(ab & 0xffffu));
                        const sycl::half a1 = sycl::bit_cast<sycl::half>((uint16_t)(ab >> 16));
                        s += (float)(sycl::half)(l0 * a0);
                        s += (float)(sycl::half)(l1 * a1);
                    }
                if (rowok[r]) acc[r] += (s * asc) * s_e4[srow0[(size_t)rowr[r] * sb + g]];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
        v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 4);
        v += __shfl_xor(v, 2);
        v += __shfl_xor(v, 1);
        const uint32_t n = nb * (kV2Teams * R) + (uint32_t)(team * R + r);
        if (l == 0 && n < N) Y[(size_t)i * y_ld + n] = to_bf16(rowok[r] ? v * s1 : 0.0f);
    }
}

/* moe_planes_v2m (milestone 3): moe_planes_v2 over a run of up to MT routed rows of one expert. The rows
 * are sorted by expert, so a run is contiguous: the work-group whose row i starts a run ((i - off[e]) % MT
 * == 0) stages the run's activations, loads each weight chunk once, applies it to every row and writes
 * them all; the run's other work-groups return. A padding row (no expert) writes its zeros as before.
 * Per output the arithmetic and its order are v2's: bit for bit. */
constexpr int kV2MT = 4;
template <int UPL, int MT, int R = kV2R>
void moe_planes_v2m_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                           const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                           uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                           uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                           uint32_t a_rows) {
    XE_SHARED(uint32_t, s_a, MT * (kV2MaxK / 2));   /* the run's activation rows as f16 pairs */
    XE_SHARED(float, s_as, MT * (kV2MaxK / 128));   /* their per-128 scales */
    XE_SHARED(uint32_t, s_lut, 256);
    XE_SHARED(float, s_e4, 256);
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
    /* the run this row belongs to; a non-leader row is written by its run's leader */
    uint32_t r0 = i, rn = 1;
    if (e >= 0) {
        const uint32_t first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % MT != 0) return;
        r0 = i;
        rn = (last - i) < (uint32_t)MT ? (last - i) : (uint32_t)MT;
    }
    /* each row's token (a row past the activation rows has no expert: zeros, as in v2) */
    uint32_t tokr[MT];
    bool liver[MT];
#pragma unroll
    for (int q = 0; q < MT; ++q) {
        tokr[q] = 0; liver[q] = false;
        if (e >= 0 && (uint32_t)q < rn) {
            const int32_t fq = sorted[r0 + q];
            const uint32_t t = a_sorted ? r0 + q : (uint32_t)fq / top_k;
            tokr[q] = t;
            liver[q] = fq >= 0 && t < a_rows;
        }
    }
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    const uint32_t K = e >= 0 ? c.k : 0u;
    const uint8_t* we = nullptr;
    const uint8_t* se = nullptr;
    float s1 = 0.0f;
    if (e >= 0) {
        const uint32_t entry = (uint32_t)e >> 1;
        we = entry_of(c.base, c.tab, c.stride, entry);
        se = entry_of(c.sbase, c.stab, c.sstride, entry);
        s1 = *(const float*)(se + c.s1_off);
        const float* lvp = (const float*)(we + c.t_off);
#pragma unroll
        for (int q = 0; q < MT; ++q) {
            if ((uint32_t)q >= rn || !liver[q]) continue;
            const uint8_t* arow = A + (size_t)tokr[q] * a_ld;
            for (uint32_t b = (uint32_t)tid; b < K / 2u; b += kV2WG)
                s_a[q * (kV2MaxK / 2) + b] = f2h(e4m3(arow[2 * b])) | (f2h(e4m3(arow[2 * b + 1])) << 16);
            for (uint32_t g = (uint32_t)tid; g < K / 128u; g += kV2WG)
                s_as[q * (kV2MaxK / 128) + g] = As[(size_t)tokr[q] * as_ld + g];
        }
        for (uint32_t t = (uint32_t)tid; t < 256u; t += kV2WG) {
            s_e4[t] = e4m3(t);
            s_lut[t] = f2h(lvp[t & 15u]) | (f2h(lvp[t >> 4]) << 16);
        }
    }
    __syncthreads();
    const int team = tid / kV2LPR, l = tid % kV2LPR;
    float acc[MT][R];
    uint32_t rowr[R];
    bool rowok[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint32_t n = nb * (kV2Teams * R) + (uint32_t)(team * R + r);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[r] = e >= 0 && n < N && j < fc;
        rowr[r] = pp * fc + j;
#pragma unroll
        for (int q = 0; q < MT; ++q) acc[q][r] = 0.0f;
    }
    if (e >= 0) {
        const uint32_t CPR = K / 32u;
        const uint32_t rb = K / 2u, sb = K / 64u;
        const uint8_t* wrow0 = we + c.c_off;
        const uint8_t* srow0 = se + c.s_off;
        uint4 w[UPL][R];
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
#pragma unroll
            for (int r = 0; r < R; ++r)
                w[u][r] = (jc < CPR && rowok[r]) ? *(const uint4*)(wrow0 + (size_t)rowr[r] * rb + jc * 16u)
                                                 : uint4{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
            if (jc >= CPR) break;
            const uint32_t k0 = jc * 32u;
            const uint32_t g = k0 / 64u;
            float sbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r) sbr[r] = rowok[r] ? s_e4[srow0[(size_t)rowr[r] * sb + g]] : 0.0f;
#pragma unroll
            for (int q = 0; q < MT; ++q) {
                if ((uint32_t)q >= rn) break;                /* uniform over the work-group */
                const float asc = s_as[q * (kV2MaxK / 128) + k0 / 128u];
                uint32_t a[16];
#pragma unroll
                for (int t = 0; t < 16; ++t) a[t] = s_a[q * (kV2MaxK / 2) + k0 / 2u + (uint32_t)t];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t d[4] = { w[u][r].x, w[u][r].y, w[u][r].z, w[u][r].w };
                    float sacc = 0.0f;
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const uint32_t byte = (d[x] >> (8 * b)) & 0xffu;
                            const uint32_t ab = a[x * 4 + b];
                            const uint32_t lv = s_lut[byte];
                            const sycl::half l0 = sycl::bit_cast<sycl::half>((uint16_t)(lv & 0xffffu));
                            const sycl::half l1 = sycl::bit_cast<sycl::half>((uint16_t)(lv >> 16));
                            const sycl::half a0 = sycl::bit_cast<sycl::half>((uint16_t)(ab & 0xffffu));
                            const sycl::half a1 = sycl::bit_cast<sycl::half>((uint16_t)(ab >> 16));
                            sacc += (float)(sycl::half)(l0 * a0);
                            sacc += (float)(sycl::half)(l1 * a1);
                        }
                    if (rowok[r]) acc[q][r] += (sacc * asc) * sbr[r];
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < MT; ++q) {
        if ((uint32_t)q >= rn) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[q][r];
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            const uint32_t n = nb * (kV2Teams * R) + (uint32_t)(team * R + r);
            if (l == 0 && n < N)
                Y[(size_t)(r0 + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] ? v * s1 : 0.0f);
        }
    }
}

/* ---- the matrix-unit form for prefill rows (milestone 3, stage B) ----
 *
 * The rows are sorted by expert, so an expert's routed rows are one run (off[e] .. off[e+1]); the work-group whose
 * row starts a 64-row block of the run ((i - off[e]) % 64 == 0) computes the block's 64 rows by its 64 output
 * columns and the block's other work-groups return (v2m's leader rule at 64). Sub-group w owns rows 8w..8w+7 and
 * the four 16-column tiles. Per 64-K group: the block's activation rows staged as f16 (E4M3 -> f16 is exact; a
 * dead row -- past the run, padding, or past the activation rows -- stages zeros), the 64 weight rows' u4 codes
 * dequantised through the expert's 16-level table to f16 (v2m's rounding of the levels) straight into the packed B
 * layout (one code byte = two consecutive k = one packed word), four f16 joint_matrix_mad steps of K = 16 into f32
 * tiles (v2m's products without its per-product f16 rounding), then each tile stored
 * to the sub-group's LDS scratch and element (row, n) scaled by a_scale[row, g128] * e4m3(scale[n, g64]) into the
 * lane's accumulators (lane = column, 8 rows a lane): v2m's order per 64-K group.
 * y = acc * scale.1, one bf16 rounding; rows outside the run or dead, and columns past N or past a narrower
 * class's width, write zeros as v2m does. */
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int kMxSG = 16, kMxWaves = 8, kMxThreads = kMxSG * kMxWaves;   /* 128 */
constexpr int kMxRows = 8 * kMxWaves;                                    /* 64 rows a block */
constexpr int kMxCols = 64;                                              /* 4 tiles of 16 outputs */
constexpr int kMxAP = kMxCols + 8;                                       /* the A stage's row pitch in bf16 (64 + 8) */

void moe_planes_mx_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                          const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                          uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                          uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                          uint32_t a_rows, uint32_t compact) {
    XE_SHARED(uint16_t, s_a, kMxRows * kMxAP);        /* the block's activation rows, 64 k a group, f16 */
    XE_SHARED(uint32_t, s_b, 4 * 4 * 128);            /* [k step 4][n tile 4][packed 8 x 16 words]: 64 k x 64 n */
    XE_SHARED(float, s_c, kMxWaves * 8 * 16);         /* one 8 x 16 f32 tile a sub-group */
    XE_SHARED(float, s_as, kMxRows);                  /* the rows' a_scale of the current 128-K group */
    XE_SHARED(uint16_t, s_lv, 16);                    /* the expert's levels as f16 (v2m's rounding) */
    XE_SHARED(float, s_e4, 256);                      /* byte -> E4M3 value */
    /* Last, so the matrix tiles' buffers above keep their places (s_a at offset 0). */
    XE_SHARED(int32_t, s_blk, 2);                     /* compact: this work-group's expert and first row */
    const int tid = (int)threadIdx.x, wave = tid / kMxSG, lane = tid % kMxSG;
    const uint32_t nb = blockIdx.x;
    int32_t e = -1;
    uint32_t r0 = 0, rn = 1;
    if (compact == 1) {
        /* RADIANCE_XE_MOE_MX_COMPACT=1 (run 108): the grid is the 64-row blocks themselves -- every expert's run in
         * blocks, in expert order, then the rows past the runs (off[n_expert] .. T, written as zeros) -- instead of
         * one work-group a sorted row of which only a block's first does any work. Each work-group finds its block
         * from a group scan of the experts' block counts; the blocks, and so every product and store, are the
         * leader rule's. */
        const uint32_t b = blockIdx.y;
        const uint32_t per = (n_expert + (uint32_t)kMxThreads - 1) / (uint32_t)kMxThreads;
        const uint32_t e0 = sycl::min(n_expert, (uint32_t)tid * per), e1 = sycl::min(n_expert, e0 + per);
        uint32_t mine = 0;
        for (uint32_t x = e0; x < e1; ++x) mine += ((uint32_t)(off[x + 1] - off[x]) + kMxRows - 1) / kMxRows;
        const uint32_t before = sycl::exclusive_scan_over_group(xe::wg(), mine, sycl::plus<uint32_t>());
        const uint32_t total = sycl::reduce_over_group(xe::wg(), mine, sycl::plus<uint32_t>());
        if (tid == 0) s_blk[0] = -1;
        xe::syncthreads();
        if (b >= before && b < before + mine) {
            uint32_t k = b - before, x = e0;
            for (;; ++x) {
                const uint32_t nbx = ((uint32_t)(off[x + 1] - off[x]) + kMxRows - 1) / kMxRows;
                if (k < nbx) break;
                k -= nbx;
            }
            s_blk[0] = (int32_t)x;
            s_blk[1] = off[x] + (int32_t)(k * kMxRows);
        }
        xe::syncthreads();
        e = s_blk[0];
        if (e >= 0) {
            r0 = (uint32_t)s_blk[1];
            const uint32_t last = (uint32_t)off[e + 1];
            rn = (last - r0) < (uint32_t)kMxRows ? (last - r0) : (uint32_t)kMxRows;
        } else {
            r0 = (uint32_t)off[n_expert] + (b - total) * kMxRows;
            if (b < total || r0 >= T) return;
            rn = (T - r0) < (uint32_t)kMxRows ? (T - r0) : (uint32_t)kMxRows;
        }
    } else {
    const uint32_t i = blockIdx.y;
    if (i >= T) return;
    const int32_t flat = sorted[i];
    /* compact == 2 (RADIANCE_XE_MOE_MX_BOUND=1, run 108): a row at or past off[n_expert] is not this GEMM's. The
     * protected experts sort last and their rows are live (sorted >= 0); unbounded, such a row whose distance from
     * off[n_expert - 1] is a multiple of 64 becomes a "leader" of expert n_expert - 1, rn = off[n_expert] - i
     * wraps to 64, and the block computes 64 rows with the wrong expert -- in a full pass up to 63 of them past
     * Y's last row. Bounded, those rows are zeroed one by one, as v2m zeroes rows past its live count. */
    if (flat >= 0 && !(compact == 2 && i >= (uint32_t)off[n_expert])) {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        e = (int32_t)lo;
    }
    r0 = i;
    if (e >= 0) {
        const uint32_t first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % (uint32_t)kMxRows != 0) return;
        rn = (last - i) < (uint32_t)kMxRows ? (last - i) : (uint32_t)kMxRows;
    }
    }
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    const uint32_t K = e >= 0 ? c.k : 0u;
    const uint32_t n0 = nb * kMxCols;
    /* this lane's four output columns: the weight row and whether it exists */
    uint32_t rowr[4];
    bool rowok[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t n = n0 + (uint32_t)(t * 16 + lane);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[t] = e >= 0 && n < N && j < fc;
        rowr[t] = pp * fc + j;
    }
    /* this sub-group's 8 rows: their tokens and whether they are live */
    uint32_t tokr[8];
    bool liver[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const uint32_t r = (uint32_t)(wave * 8 + q);
        tokr[q] = 0; liver[q] = false;
        if (e >= 0 && r < rn) {
            const int32_t fq = sorted[r0 + r];
            const uint32_t tk = a_sorted ? r0 + r : (uint32_t)fq / top_k;
            tokr[q] = tk;
            liver[q] = fq >= 0 && tk < a_rows;
        }
    }
    const uint8_t* we = nullptr;
    const uint8_t* se = nullptr;
    float s1 = 0.0f;
    if (e >= 0) {
        const uint32_t entry = (uint32_t)e >> 1;
        we = entry_of(c.base, c.tab, c.stride, entry);
        se = entry_of(c.sbase, c.stab, c.sstride, entry);
        s1 = *(const float*)(se + c.s1_off);
        const float* lvp = (const float*)(we + c.t_off);
        if (tid < 16) s_lv[tid] = (uint16_t)f2h(lvp[tid]);
        for (uint32_t t = (uint32_t)tid; t < 256u; t += kMxThreads) s_e4[t] = e4m3(t);
    }
    float acc[4][8];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[t][q] = 0.0f;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> Am;
    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> Bm;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> Cm[4];
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
    float* s_cw = s_c + wave * 128;
    const uint8_t* wrow0 = e >= 0 ? we + c.c_off : nullptr;
    const uint8_t* srow0 = e >= 0 ? se + c.s_off : nullptr;
    const uint32_t rb = K / 2u, sb = K / 64u;
    if (e >= 0) {
        for (uint32_t g = 0; g < K / 64u; ++g) {
            const uint32_t k0 = g * 64u;
            xe::syncthreads();                                  /* the previous group's stages are consumed */
            /* the activation rows: 64 rows x 64 k, E4M3 -> bf16; a dead row stages zeros */
            for (int idx = tid; idx < kMxRows * 32; idx += kMxThreads) {   /* 2048 pairs: 16 a thread */
                const int r = idx / 32, kp = (idx % 32) * 2;
                uint32_t w = 0;
                const uint32_t rr = (uint32_t)r;
                if (rr < rn) {
                    const int32_t fq = sorted[r0 + rr];
                    const uint32_t tk = a_sorted ? r0 + rr : (uint32_t)fq / top_k;
                    if (fq >= 0 && tk < a_rows) {
                        const uint8_t* ar = A + (size_t)tk * a_ld + k0 + (uint32_t)kp;
                        w = f2h(e4m3(ar[0])) | (f2h(e4m3(ar[1])) << 16);
                    }
                }
                *(uint32_t*)(s_a + r * kMxAP + kp) = w;
            }
            if ((k0 & 127u) == 0u)                               /* a new 128-K group: the rows' activation scales */
                for (int r = tid; r < kMxRows; r += kMxThreads) {
                    float v = 0.0f;
                    const uint32_t rr = (uint32_t)r;
                    if (rr < rn) {
                        const int32_t fq = sorted[r0 + rr];
                        const uint32_t tk = a_sorted ? r0 + rr : (uint32_t)fq / top_k;
                        if (fq >= 0 && tk < a_rows) v = As[(size_t)tk * as_ld + k0 / 128u];
                    }
                    s_as[r] = v;
                }
            /* the weight rows: 64 n x 64 k of u4 codes, through the table to bf16, packed (one byte = one word) */
            for (int idx = tid; idx < kMxCols * 32; idx += kMxThreads) {   /* 2048 words: 16 a thread */
                const int n = idx / 32, byte = idx % 32;
                const int step = byte / 8, k2 = byte % 8;
                const uint32_t nn = n0 + (uint32_t)n;
                const uint32_t pp = fy ? nn / fy : 0u, j = nn - pp * fy;
                uint32_t w = 0;
                if (nn < N && j < fc) {
                    const uint8_t code = wrow0[(size_t)(pp * fc + j) * rb + k0 / 2u + (uint32_t)byte];
                    w = (uint32_t)s_lv[code & 15u] | ((uint32_t)s_lv[code >> 4] << 16);
                }
                s_b[((step * 4 + n / 16) * 8 + k2) * 16 + (n % 16)] = w;
            }
            xe::syncthreads();
#pragma unroll
            for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, Cm[t], 0.0f);
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                jm::joint_matrix_load(sg, Am, lptr((sycl::half*)(s_a + wave * 8 * kMxAP + step * 16)), (size_t)kMxAP);
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    jm::joint_matrix_load(sg, Bm, lptr((sycl::half*)(s_b + (step * 4 + t) * 128)), 32);
                    jm::joint_matrix_mad(sg, Cm[t], Am, Bm, Cm[t]);
                }
            }
            /* the group's sums, scaled per element: a_scale[row, g128] * e4m3(scale[n, g64]) */
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_store(sg, Cm[t], lptr(s_cw), 16, jm::layout::row_major);
                xe::wave_barrier();
                const float sbv = rowok[t] ? s_e4[srow0[(size_t)rowr[t] * sb + g]] : 0.0f;
#pragma unroll
                for (int q = 0; q < 8; ++q) acc[t][q] += (s_cw[q * 16 + lane] * s_as[wave * 8 + q]) * sbv;
                xe::wave_barrier();
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t n = n0 + (uint32_t)(t * 16 + lane);
        if (n >= N) continue;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint32_t r = (uint32_t)(wave * 8 + q);
            if (r >= rn) continue;
            Y[(size_t)(r0 + r) * y_ld + n] = to_bf16(rowok[t] && liver[q] ? acc[t][q] * s1 : 0.0f);
        }
    }
}

bool mx_on(sycl::queue* q) {   /* the matrix-unit form for prefill rows is the default (owner note 49, run 60: the
                                * exact arithmetic and the speed; the one-prompt near-tie against production is a
                                * recorded divergence); RADIANCE_XE_MOE_MX=0 keeps v2m -- the GEMV form one flag
                                * away; the CPU device has no matrix unit */
    static const bool want = [] { const char* s = std::getenv("RADIANCE_XE_MOE_MX"); return !(s && *s == '0'); }();
    static const bool gpu = q->get_device().is_gpu();
    return want && gpu;
}

uint32_t mx_compact() {   /* run 108, opt-in (moe_planes_mx_kernel's `compact`): RADIANCE_XE_MOE_MX_COMPACT=1 -> 1, the
                           * grid is the 64-row blocks of the regular runs plus the rows past them (zeroed);
                           * else RADIANCE_XE_MOE_MX_BOUND=1 -> 2, the per-row grid bounded at off[n_expert]; 0 */
    static const uint32_t v = [] {
        const char* s = std::getenv("RADIANCE_XE_MOE_MX_COMPACT");
        if (s && *s == '1') return 1u;
        const char* b = std::getenv("RADIANCE_XE_MOE_MX_BOUND");
        return b && *b == '1' ? 2u : 0u;
    }();
    return v;
}

bool multi_on() {   /* RADIANCE_XE_MOE_MULTI=0: one work-group a routed row (v2) */
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_MOE_MULTI"); return !(s && *s == '0'); }();
    return v;
}

int v2_rows() {   /* RADIANCE_XE_MOE_V2R=1: v2 / v2m with one row a team (32 rows a work-group); else the shipped 2 */
    static const int v = [] { const char* s = std::getenv("RADIANCE_XE_MOE_V2R"); return s && *s == '1' ? 1 : 2; }();
    return v;
}

/* Where the planes of an entry start: rad_planes_plan's arithmetic for two planes. */
uint32_t first_plane() {
    return (uint32_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
}

/* One class's table: a pointer table (stride 0) or a stacked plane; `bits` the operand dtype's. */
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

/* The two joined views this row reads: codes,table (u4 1x1 + 16 f32) and scale,scale.1 (E4M3 1x64
 * + one f32). */
bool view_ok(const RadEncoding* enc, const int* sel, const RadTensor* pl, int n) {
    /* RADIANCE_XE_NO_PLANES=1: decline, so libr4d's packed row serves (the load A/B of owner note 4) */
    static const bool off = [] { const char* v = std::getenv("RADIANCE_XE_NO_PLANES"); return v && *v == '1'; }();
    if (off || !enc || !sel || !pl) return false;
    if (std::getenv("RAD_R4D_TRACE")) {
        std::fprintf(stderr, "[xe] moe planes layout: %d plane(s)", n);
        for (int i = 0; i < n; ++i)
            std::fprintf(stderr, " [%s dtype %u kind %u block %lld,%lld extent %lld,%lld]", enc->plane[sel[i]].role,
                         (unsigned)pl[i].dtype, (unsigned)enc->plane[sel[i]].kind,
                         (long long)enc->plane[sel[i]].block[0], (long long)enc->plane[sel[i]].block[1],
                         (long long)enc->plane[sel[i]].extent[0], (long long)enc->plane[sel[i]].extent[1]);
        std::fprintf(stderr, "\n");
    }
    if (n != 2) return false;
    const RadEncPlane& a = enc->plane[sel[0]];
    const RadEncPlane& b = enc->plane[sel[1]];
    if (!std::strcmp(a.role, "codes") && !std::strcmp(b.role, "table"))
        return pl[0].dtype == RAD_U4 && a.block[0] == 1 && a.block[1] == 1 && pl[1].dtype == RAD_F32 &&
               b.kind == RAD_PLANE_TABLE && b.extent[0] * b.extent[1] == 16;
    if (!std::strcmp(a.role, "scale") && !std::strcmp(b.role, "scale.1"))
        return pl[0].dtype == RAD_F8E4M3 && a.block[0] == 1 && a.block[1] == 64 && pl[1].dtype == RAD_F32;
    return false;
}

bool v1_forced() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_MOE_V1"); return s && *s == '1'; }();
    return v;
}

}  // namespace

extern "C" int xe_moe_planes_layout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                    const RadTensor* pl, int n, RadLayout* out) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, pl, n, out);
}
extern "C" int xe_moe_planes_relayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_relayout(enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int xe_moe_planes_unrelayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                        const void* src, int64_t src_bytes, const RadTensor* pl, int n) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, pl, n);
}

/* moe_gemm_q (two classes): a, a_scale, w, w_scale, sorted, offsets, w_odd, w_scale_odd, y */
extern "C" int xe_rad_moe_gemm_w4p(const RadArgs* a, RadStream s) {
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
    if (!wdt || std::strcmp(wdt, "w4nl64a8h")) return R4D_NO(RAD_E_UNSUPPORTED);
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
    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 || sorted->dtype != RAD_I32 ||
        off->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    const long long as_ld = r4d_rows_pitch(As, K / 128, &Ms);
    if (y_ld < 0 || T <= 0 || a_ld < 0 || M <= 0 || as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);
    PCls cl[2]{};
    bool logical = false;
    const uint32_t p0 = first_plane();
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (n_expert + 1 - q) / 2;
        int rc = table_of(W[q], cnt, &cl[q].base, &cl[q].tab, &cl[q].stride);
        if (rc == RAD_OK) rc = table_of(Ws[q], cnt, &cl[q].sbase, &cl[q].stab, &cl[q].sstride);
        if (rc != RAD_OK) return rc;
        /* the joined codes view arrives as u4 (its first plane, as libref's cb_codes reads it); the
         * logical form as E4M3 levels with a bf16 scale product (rad-kbench, the oracle) */
        const bool lg = W[q]->dtype == RAD_F8E4M3 && Ws[q]->dtype == RAD_BF16;
        if (q == 0) logical = lg;
        if (lg != logical || (!lg && W[q]->dtype != RAD_U4)) return R4D_NO(RAD_E_DTYPE);
        if (lg) {
            if (W[q]->rank != 3 || Ws[q]->rank != 3 || W[q]->stride[2] != 1 || Ws[q]->stride[2] != 1)
                return R4D_NO(RAD_E_STRIDE);
            cl[q].w_ld = (uint32_t)W[q]->stride[1];
            cl[q].s_ld = (uint32_t)Ws[q]->stride[1];
        }
        cl[q].n = (uint32_t)Nc[q];
        cl[q].k = (uint32_t)Kc[q];
        cl[q].c_off = p0;
        cl[q].t_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 2));
        cl[q].s_off = p0;
        cl[q].s1_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 64));
    }
    try {
        sycl::queue* qq = (sycl::queue*)s;
        if (logical)
            XE_LAUNCH(qq, moe_planes_kernel<true>, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const uint8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
        else if (K <= (long long)kV2MaxK && !v1_forced()) {
            /* chunks a lane: K / 32 chunks a row over 16 lanes, rounded up (both classes' K are equal
             * in the shipped model; the larger decides, the smaller leaves lanes idle in the tail) */
            const int upl = (int)((K / 32 + kV2LPR - 1) / kV2LPR);
            /* rows a team (owner note 54 item 2, run 79): 2 = 64 rows a work-group (shipped); 1 = 32, twice the
             * work-groups, so a decode grid of 80-120 work-groups on 64 resident slots leaves a finer tail.
             * Per output the arithmetic and its order are unchanged: bit for bit. */
            const int rpt = v2_rows();
            const int rpw = kV2Teams * rpt;
            const dim3 grid((unsigned)((N + rpw - 1) / rpw), (unsigned)T), block(kV2WG);
            /* several tokens (T past one token's top_k rows): runs of one expert's rows share the weight
             * stream (milestone 3); a single token keeps v2 (its rows are distinct experts) */
            const bool multi = multi_on() && T > top_k;
            /* prefill-sized routing (64 rows or more) under RADIANCE_XE_MOE_MX=1: the matrix-unit form, 64-row
             * blocks of each expert's run (exact products; v2m rounds each to f16) */
            const bool mx = multi && T >= (long long)kMxRows && mx_on(qq);
#define XE_MOE_V2(U) do { if (rpt == 1) XE_MOE_V2R(U, 1); else XE_MOE_V2R(U, 2); } while (0)
#define XE_MOE_V2R(U, R)                                                                                    \
    if (mx)                                                                                                 \
    XE_LAUNCH_SG(kMxSG, qq, moe_planes_mx_kernel,                                                           \
                 dim3((unsigned)((N + kMxCols - 1) / kMxCols),                                              \
                      mx_compact() == 1 ? (unsigned)(T / kMxRows + n_expert + 2) : (unsigned)T),            \
                 dim3(kMxThreads), 0, (const uint8_t*)Aq->data, (const float*)As->data, cl[0], cl[1],       \
                 (const int32_t*)sorted->data, (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T,  \
                 (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld,          \
                 (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M, mx_compact());            \
    else if (multi)                                                                                         \
    XE_LAUNCH_SG(16, qq, (moe_planes_v2m_kernel<U, kV2MT, R>), grid, block, 0, (const uint8_t*)Aq->data,    \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts,       \
                 (uint32_t)M);                                                                              \
    else                                                                                                    \
    XE_LAUNCH_SG(16, qq, (moe_planes_v2_kernel<U, R>), grid, block, 0, (const uint8_t*)Aq->data,            \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts,       \
                 (uint32_t)M)
            switch (upl) {
                case 1: XE_MOE_V2(1); break;
                case 2: XE_MOE_V2(2); break;
                case 3: XE_MOE_V2(3); break;
                case 4: XE_MOE_V2(4); break;
                case 5: XE_MOE_V2(5); break;
                case 6: XE_MOE_V2(6); break;
                case 7: XE_MOE_V2(7); break;
                default: XE_MOE_V2(8); break;   /* K = 4096: 128 chunks, 8 a lane */
            }
#undef XE_MOE_V2
#undef XE_MOE_V2R
        } else
            XE_LAUNCH(qq, moe_planes_kernel<false>, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const uint8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
