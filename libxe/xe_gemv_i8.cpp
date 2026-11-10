/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_gemv_i8.cpp -- the trunk's W8A8 product (gemm_nt_q at dtype i8a8 / i8a8h) as a GEMV, written for
 * Xe2 rather than ported: libr4d's gemm_i8a8_nt_m16 is an iu8-WMMA kernel on a fragment-order weight.
 *
 *   y[m, n] = sum over 128-K groups g, ascending:  (isum(m, n, g) * a_scale[m, g]) * b_scale[n, g]
 *
 * isum is the exact int32 sum of 128 int8 products (it is exact in f32 too, |isum| < 2^24), so a
 * group's term is libref's (ref_gemm.h quant_dot) bit for bit; only the order the group terms are
 * added in differs (four groups at a time across the sub-group, then a butterfly). i8a8h is the same
 * product: the weight was rotated by a 128-wide Hadamard at convert and the activation by
 * had_quant_act_i8g, with the 1/128 folded into the weight's scale.
 *
 * Layout: the stored one (no relayout hook): b [N][K] int8 rows, b_scale [N][K/128] bf16 (or f32),
 * a [M][K] int8 rows (column order: i8a8 is not one of libref's step-order dtypes), a_scale
 * [M][K/128] f32, y [M][N] bf16. The weight streams once per launch for up to kMaxM activation rows.
 *
 * Mapping (round 2's M1 shape, sub-group 32): a sub-group computes one output row n; lane l takes
 * 16-byte chunk (l & 7) of group 4*j + (l >> 3); the 8 lanes of a group reduce their int sums exactly
 * (xor 1, 2, 4), the group's first lane scales and accumulates, and the 4 group lanes meet at the end
 * (xor 8, 16). A work-group is 8 sub-groups = 8 rows.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include <sycl/ext/oneapi/dot_product.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>
#include <atomic>   /* run 274: the paired GEMV ops' say-once fallback note */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "xe_dp4a.h"   /* run 207 (owner note 203 item 1): the GEMV's dot by IGC's dp4a builtin on the card */
#include "xe_lsc.h"    /* run 211 (owner note 206 item 2): the weight loads' cache-control hints */

namespace {

[[maybe_unused]] constexpr int kRowsPerWG = 8;   /* the default of xe::gemv_rows_wg() (run 207) */
constexpr int kMaxM = 4;   /* activation rows a weight pass serves; more rows loop the launch */

/* NX (RADIANCE_XE_GEMV_NX=1, run 102, owner note 109 item 2): the five cross-lane steps (the group's three int32
 * steps, the row's two float steps) as permute_group_by_xor instead of the emulated ds_bpermute / __shfl_xor (a
 * select_from_group over a lane computed a lane: ~5x the time of the xor form in LB1). The same partner lanes in the
 * same order, so the same bits. */
template <int MSK>
inline int gv_xor_i(int s) {
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), s, MSK);
}
template <int MSK>
inline float gv_xor_f(float v) {
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, MSK);
}

/* KC (run 209, owner note 206 item 1, RADIANCE_XE_GEMV_KC=1): the model's K as a compile-time constant (0: the
 * runtime K), so IGC knows the group count -- the group loop's trip count and its g < G tests fold. The same groups in
 * the same order, so the same bits. */
/* WCC (run 211, owner note 206 item 2, RADIANCE_XE_GEMV_WCC=2 / 5 / 6): the weight row's 16-byte loads with an LSC
 * cache-control hint (xe_lsc.h; the weights are read once a step); 0 the default policy. The same bytes. */
template <bool WS_BF16, int MB, bool NX = false, int KC = 0, int WCC = 0>
void gemv_i8a8_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                      const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                      uint16_t* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);   /* xe::gemv_rows_wg() */
    const int G = (KC ? KC : K) / 128;
    const int chunk = lane & 7, gsub = lane >> 3;
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const bool live = n < N;
    const int8_t* brow = b + (size_t)(live ? n : 0) * b_ld;
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int g = g0 + gsub;
        const bool gl = live && g < G;
        uint4 w = {0, 0, 0, 0};
        if (gl) w = xe_ld16<WCC>(brow + (size_t)g * 128 + chunk * 16);
        float sb = 0.0f;
        if (gl) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)n * bs_ld + g] << 16)
                             : ((const float*)bs)[(size_t)n * bs_ld + g];
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            int s = 0;
            if (gl && m < M) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16);
                const unsigned wv[4] = {w.x, w.y, w.z, w.w}, xv[4] = {x.x, x.y, x.z, x.w};
                /* the 4-way int8 dot (DP4A): the packed words whole, the same exact int32 sum (milestone 3) */
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    s = xe_dp4a((int32_t)wv[u], (int32_t)xv[u], s);
            }
            /* the group's exact int32 sum over its 8 lanes */
            if constexpr (NX) { s += gv_xor_i<1>(s); s += gv_xor_i<2>(s); s += gv_xor_i<4>(s); }
            else {
                s += __builtin_amdgcn_ds_bpermute((lane ^ 1) << 2, s);
                s += __builtin_amdgcn_ds_bpermute((lane ^ 2) << 2, s);
                s += __builtin_amdgcn_ds_bpermute((lane ^ 4) << 2, s);
            }
            if (gl && m < M && chunk == 0) acc[m] += ((float)s * as[(size_t)m * as_ld + g]) * sb;
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        if constexpr (NX) { v += gv_xor_f<8>(v); v += gv_xor_f<16>(v); }
        else { v += __shfl_xor(v, 8); v += __shfl_xor(v, 16); }
        if (live && lane == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);   /* round to nearest even, NaN kept quiet */
            y[(size_t)m * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                 : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* ------------------------------------------------------------- several projections of one activation (run 274)
 *
 * Pairs 4 and 5 of the decode step's adjacent launches (owner note 214 item 3): the gated attention's q|gate, k and v
 * projections, and the GDN block's in-projection with its bf16 a|b projection, are GEMVs over the same normed input
 * issued back to back; the small ones (one kv head a rank, 96 a|b rows) are launch and latency alone. One grid runs the
 * sets in ranges of work-groups, each row by a COPY of the standalone kernel's row (as gemv_bf16_pair_kernel does for
 * the router and the shared gate): the same lane mapping, loads, DP4A sums, cross-lane steps and rounding, so each
 * output row's bytes are its own launch's. The standalone kernels above are untouched, so their codegen is too (the
 * coordinator's condition, run 274). The forms copied are the trunk's: bf16 weight scales, the NX cross-lane steps, the
 * runtime K, WCC 5 or the default policy; the launcher takes any other form to the separate launches (gemv_multi_form).
 * The pointers ride in structs of 8-byte fields (no padding: the tape audit compares bytes). */
struct GvI8Set {
    const int8_t* b = nullptr;
    const void* bs = nullptr;
    uint16_t* y = nullptr;
    long b_ld = 0, bs_ld = 0, y_ld = 0, n = 0;
};
struct GvBfSet {
    const uint16_t* a = nullptr;   /* the activation's bf16 plane (the int8 sets read its codes) */
    const uint16_t* b = nullptr;
    uint16_t* y = nullptr;
    long a_ld = 0, b_ld = 0, y_ld = 0, n = 0;
};

/* gemv_i8a8_kernel<WS_BF16, MB, true, 0, WCC>'s row `n` of one set: its body from `live` on, verbatim. */
template <bool WS_BF16, int MB, int WCC>
inline void gemv_i8a8_set_row(int n, int lane, const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as,
                              long as_ld, const GvI8Set& t, int K, int M) {
    const int G = K / 128;
    const int chunk = lane & 7, gsub = lane >> 3;
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const bool live = n < (int)t.n;
    const int8_t* brow = t.b + (size_t)(live ? n : 0) * t.b_ld;
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int g = g0 + gsub;
        const bool gl = live && g < G;
        uint4 w = {0, 0, 0, 0};
        if (gl) w = xe_ld16<WCC>(brow + (size_t)g * 128 + chunk * 16);
        float sb = 0.0f;
        if (gl) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)t.bs)[(size_t)n * t.bs_ld + g] << 16)
                             : ((const float*)t.bs)[(size_t)n * t.bs_ld + g];
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            int s = 0;
            if (gl && m < M) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16);
                const unsigned wv[4] = {w.x, w.y, w.z, w.w}, xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    s = xe_dp4a((int32_t)wv[u], (int32_t)xv[u], s);
            }
            s += gv_xor_i<1>(s); s += gv_xor_i<2>(s); s += gv_xor_i<4>(s);
            if (gl && m < M && chunk == 0) acc[m] += ((float)s * as[(size_t)m * as_ld + g]) * sb;
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        v += gv_xor_f<8>(v); v += gv_xor_f<16>(v);
        if (live && lane == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);
            t.y[(size_t)m * t.y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                     : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* xe_gemv_bf16.cpp's gemv_bf16_kernel<false, false, MB, 0>'s row `n` of one set: its body from `live` on, verbatim
 * (ld_bf16 / to_bf16 written out). */
template <int MB>
inline void gemv_bf16_set_row(int n, int lane, const GvBfSet& t, int K, int M) {
    const bool live = n < (int)t.n;
    const uint16_t* brow = t.b + (size_t)(live ? n : 0) * t.b_ld;
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const int nchunk = K / 8;
    for (int c = lane; c < nchunk && live; c += 32) {
        const uint4 w = *(const uint4*)(brow + (size_t)c * 8);
        const unsigned wv[4] = {w.x, w.y, w.z, w.w};
        float wf[8];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            wf[2 * u] = __uint_as_float(wv[u] << 16);
            wf[2 * u + 1] = __uint_as_float(wv[u] & 0xffff0000u);
        }
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            float s = 0.0f;
            const uint4 x = *(const uint4*)(t.a + (size_t)m * t.a_ld + (size_t)c * 8);
            const unsigned xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned h = (e & 1) ? (xv[e >> 1] & 0xffff0000u) : (xv[e >> 1] << 16);
                s += __uint_as_float(h) * wf[e];
            }
            acc[m] += s;
        }
    }
    for (int k = nchunk * 8 + lane; k < K && live; k += 32)
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            const float av = __uint_as_float((uint32_t)t.a[(size_t)m * t.a_ld + k] << 16);
            acc[m] += av * __uint_as_float((uint32_t)brow[k] << 16);
        }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        for (int x = 16; x >= 1; x >>= 1) v += __shfl_xor(v, x);
        if (live && lane == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);
            t.y[(size_t)m * t.y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                     : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* Three int8 sets over one activation (pair 5: q|gate, k, v): work-groups [0, g1) set 1, [g1, g1 + g2) set 2, the rest
 * set 3; within a set the row is the standalone kernel's `blockIdx.x * rows_wg + sub-group`. */
template <bool WS_BF16, int MB, int WCC>
void gemv_i8a8_x3_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld, int K,
                         int M, long g1, long g2, GvI8Set s1, GvI8Set s2, GvI8Set s3) {
    const int lane = (int)threadIdx.x & 31, rows = (int)blockDim.x >> 5, sub = (int)threadIdx.x >> 5;
    const long blk = (long)blockIdx.x;
    if (blk < g1)
        gemv_i8a8_set_row<WS_BF16, MB, WCC>((int)blk * rows + sub, lane, a, a_ld, as, as_ld, s1, K, M);
    else if (blk < g1 + g2)
        gemv_i8a8_set_row<WS_BF16, MB, WCC>((int)(blk - g1) * rows + sub, lane, a, a_ld, as, as_ld, s2, K, M);
    else
        gemv_i8a8_set_row<WS_BF16, MB, WCC>((int)(blk - g1 - g2) * rows + sub, lane, a, a_ld, as, as_ld, s3, K, M);
}

/* An int8 set and a bf16 set over one activation's two planes (pair 4: the GDN in-projection and its a|b). */
template <bool WS_BF16, int MB, int WCC>
void gemv_i8a8_bf16_pair_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                                int K, int M, long g1, GvI8Set s1, GvBfSet s2) {
    const int lane = (int)threadIdx.x & 31, rows = (int)blockDim.x >> 5, sub = (int)threadIdx.x >> 5;
    const long blk = (long)blockIdx.x;
    if (blk < g1)
        gemv_i8a8_set_row<WS_BF16, MB, WCC>((int)blk * rows + sub, lane, a, a_ld, as, as_ld, s1, K, M);
    else
        gemv_bf16_set_row<MB>((int)(blk - g1) * rows + sub, lane, s2, K, M);
}

/* The forms the paired launches are built for (the trunk's: NX on, no KC instance for this K, WCC 5 or the default
 * policy, no GL / RT env form). Anything else returns false and the adapter issues the sets' separate launches. */
static bool gemv_multi_form(int K, int* wcc_out) {
    static const bool nx = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_NX"); return e && e[0] == '1'; }();
    static const bool kc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_KC"); return e && e[0] == '1'; }();
    static const int wcc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_WCC"); return e ? std::atoi(e) : 5; }();
    static const int glf = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_GL"); return e ? std::atoi(e) : 0; }();
    static const bool rt = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_RT"); return e && e[0] == '1'; }();
    *wcc_out = wcc;
    if (!nx || glf != 0 || rt) return false;
    if (kc && (K == 2560 || K == 6144 || K == 640)) return false;
    return wcc == 5 || wcc == 0;
}

template <int MB, int WCC>
static void launch_x3_mb(sycl::queue* q, const int8_t* am, long a_ld, const float* asm_, long as_ld, int K, int mb,
                         long g1, long g2, long g3, const GvI8Set& s1, const GvI8Set& s2, const GvI8Set& s3) {
    const int rw = xe::gemv_rows_wg();
    const dim3 grid((unsigned)(g1 + g2 + g3)), block(32 * rw);
    XE_LAUNCH(q, (gemv_i8a8_x3_kernel<true, MB, WCC>), grid, block, 0, am, a_ld, asm_, as_ld, K, mb, g1, g2, s1, s2, s3);
}
template <int MB, int WCC>
static void launch_pair_mb(sycl::queue* q, const int8_t* am, long a_ld, const float* asm_, long as_ld, int K, int mb,
                           long g1, long g2, const GvI8Set& s1, const GvBfSet& s2) {
    const int rw = xe::gemv_rows_wg();
    const dim3 grid((unsigned)(g1 + g2)), block(32 * rw);
    XE_LAUNCH(q, (gemv_i8a8_bf16_pair_kernel<true, MB, WCC>), grid, block, 0, am, a_ld, asm_, as_ld, K, mb, g1, s1, s2);
}

/* RT (RADIANCE_XE_GEMV_RT=1, the 4-row launches; owner note 129 item 3, run 110): the GEMV's mapping, loads and
 * float steps, with the four rows' group sums reduced together over the group's 8 lanes by a transpose-reduce --
 * xor 4 swaps two rows' halves (2 steps), xor 2 one row's (1), xor 1 the rest (1): 4 cross-lane steps where the GEMV
 * takes 12 -- after which chunk lane 2m holds row m's exact int32 sum (any order: integers). That lane scales and
 * accumulates row m as the GEMV's chunk-0 lane does, groups g = gsub, gsub + 4, ... in order, and the group lanes
 * meet by the same xor 8, 16: the same bits. */
template <bool WS_BF16>
void gemv_i8a8_rt_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                         const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                         uint16_t* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);   /* xe::gemv_rows_wg() */
    const int G = K / 128;
    const int chunk = lane & 7, gsub = lane >> 3;
    const int mine = ((chunk >> 2) & 1) * 2 + ((chunk >> 1) & 1);   /* the row this lane ends up holding */
    float acc = 0.0f;
    const bool live = n < N;
    const int8_t* brow = b + (size_t)(live ? n : 0) * b_ld;
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int g = g0 + gsub;
        const bool gl = live && g < G;
        uint4 w = {0, 0, 0, 0};
        if (gl) w = *(const uint4*)(brow + (size_t)g * 128 + chunk * 16);
        float sb = 0.0f;
        if (gl) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)n * bs_ld + g] << 16)
                             : ((const float*)bs)[(size_t)n * bs_ld + g];
        int s[kMaxM];
#pragma unroll
        for (int m = 0; m < kMaxM; ++m) {
            s[m] = 0;
            if (gl && m < M) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16);
                const unsigned wv[4] = {w.x, w.y, w.z, w.w}, xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) s[m] = xe_dp4a((int32_t)wv[u], (int32_t)xv[u], s[m]);
            }
        }
        /* xor 4: the lanes with chunk bit 2 clear keep rows 0, 1, the others rows 2, 3 */
        const bool hi4 = (chunk & 4) != 0;
        int t[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int keep = hi4 ? s[2 + j] : s[j], give = hi4 ? s[j] : s[2 + j];
            t[j] = keep + gv_xor_i<4>(give);
        }
        /* xor 2: chunk bit 1 picks one of the two */
        const bool hi2 = (chunk & 2) != 0;
        int r = (hi2 ? t[1] : t[0]) + gv_xor_i<2>(hi2 ? t[0] : t[1]);
        r += gv_xor_i<1>(r);
        if (gl && mine < M && (chunk & 1) == 0) acc += ((float)r * as[(size_t)mine * as_ld + g]) * sb;
    }
    float v = acc;
    v += gv_xor_f<8>(v);
    v += gv_xor_f<16>(v);
    if (live && gsub == 0 && (chunk & 1) == 0 && mine < M) {
        const uint32_t u = __float_as_uint(v);
        y[(size_t)mine * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
    }
}

/* GL (RADIANCE_XE_GEMV_GL=1, the multi-row launches; owner note 141 item 1, run 114): a group a lane. Four lanes own
 * a column, lane j the groups j, j + 4, ... ascending -- the GEMV's group lane j -- and each sums its whole 128-byte
 * group alone (the exact int32 group sum: no cross-lane step a group), scales and accumulates it exactly as the
 * GEMV's chunk-0 lane does; the four meet by xor 1 then xor 2, the GEMV's xor 8 then 16: (a0 + a1) + (a2 + a3). So
 * the same bits. A sub-group of 16 holds four columns, whose lanes read the same activation chunks together, so a
 * row's activation is read once a sub-group instead of once a column. Sub-group 16, 256 threads: 64 columns. */
constexpr int kGlCols = 64;
/* FM, the group term's arithmetic, probes (run 115): 0 acc + ((float)s * a) * b (the source's), 1 fma((float)s * a, b,
 * acc) (GL=3 / 4), 2 acc + (float)s * (a * b) (GL=5), 3 fma((float)s, a * b, acc) (GL=6). On card1 none of 1-3 gives
 * the GEMV's f32-scale bits either: IGC's ISA for both kernels has the source's term (mul, mul, add). Checked over
 * every output against the source order emulated on the host (gemv_i8_test RI_CHECK_ALL=1), GL (FM 0) is exact at both
 * scale types (0 of 49152 at 4 x 12288 x 2560); the GEMV strays in a few outputs a launch (2 of 49152 at f32 scales,
 * 1 of 12288-16384 at one row) -- so the GEMV's bits are not its own order's, and no form reproduces them. */
template <bool WS_BF16, int MB, int FM = 0>
void gemv_i8a8_gl_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                         const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                         uint16_t* __restrict__ y, long y_ld, int N, int K, int M) {
    const int gsub = (int)threadIdx.x & 3;
    const int n = (int)blockIdx.x * kGlCols + ((int)threadIdx.x >> 2);
    const int G = K / 128;
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const bool live = n < N;
    const int8_t* brow = b + (size_t)(live ? n : 0) * b_ld;
    /* no cross-lane step inside: a lane's own trip count */
    for (int g = gsub; live && g < G; g += 4) {
        uint4 w[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) w[c] = *(const uint4*)(brow + (size_t)g * 128 + c * 16);
        const float sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)n * bs_ld + g] << 16)
                                 : ((const float*)bs)[(size_t)n * bs_ld + g];
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            int s = 0;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + c * 16);
                const unsigned wv[4] = {w[c].x, w[c].y, w[c].z, w[c].w}, xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) s = sycl::ext::oneapi::dot_acc((int32_t)wv[u], (int32_t)xv[u], s);
            }
            const float sa = as[(size_t)m * as_ld + g];
            if constexpr (FM == 1) acc[m] = sycl::fma((float)s * sa, sb, acc[m]);
            else if constexpr (FM == 2) acc[m] += (float)s * (sa * sb);
            else if constexpr (FM == 3) acc[m] = sycl::fma((float)s, sa * sb, acc[m]);
            else acc[m] += ((float)s * sa) * sb;
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        v += gv_xor_f<1>(v);
        v += gv_xor_f<2>(v);
        if (live && gsub == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);   /* round to nearest even, NaN kept quiet */
            y[(size_t)m * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                 : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* ---- the matrix-unit form for the prefill shapes (milestone 3, stage B): M >= 64 on the GPU ----
 *
 * A work-group of kDpasWaves sub-groups (sub-group 16: the DPAS width, B1's query) owns kDpasRT activation rows
 * by kDpasNT output columns; sub-group w owns rows 8w..8w+7 and all four 16-column tiles. Per 128-K group the
 * work-group stages the weight block packed for the B operand (word (n, k4) = the 4 consecutive int8 of row n at
 * 4*k4: the stored [N][K] row read as uint32, scattered into the (K/4) x N word layout the matrix unit takes),
 * then each sub-group runs the four 32-wide steps of the group with joint_matrix_mad into exact int32 tiles,
 * stores each tile to its own LDS scratch and scales element (m, n) by a_scale[m, g] * b_scale[n, g] into float
 * accumulators (lane = column n of the tile, 8 rows a lane): isum exact as the GEMV's, the group terms added
 * in ascending g, one bf16 rounding at the end. The weight is read from memory once per kDpasRT rows instead of
 * once per 4. The launcher sends full 64-row blocks here and the tail rows to the GEMV. */
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int kDpasSG = 16, kDpasWaves = 8, kDpasThreads = kDpasSG * kDpasWaves;
constexpr int kDpasRT = 8 * kDpasWaves;   /* 64 activation rows a work-group */
constexpr int kDpasNT = 64;               /* 4 tiles of 16 output columns a work-group */
constexpr int kDpasTileB = 8 * 16 * 4;    /* bytes of one packed B tile: 32 k x 16 n int8 */

template <bool WS_BF16>
void gemm_i8a8_dpas_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                           const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                           uint16_t* __restrict__ y, long y_ld, int N, int K) {
    XE_SHARED(int8_t, sB, 4 * 4 * kDpasTileB);          /* [step 4][n tile 4][packed 32 x 16]: one 128-K group */
    const int tid = (int)threadIdx.x, wave = tid / kDpasSG, lane = tid % kDpasSG;
    const int m0 = (int)blockIdx.y * kDpasRT + wave * 8;   /* always a full 8-row tile: the launcher's blocks */
    const int n0 = (int)blockIdx.x * kDpasNT;
    const int G = K / 128;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> A;
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> Bt;
    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C[4];
    float acc[4][8];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;
    const int8_t* arow = a + (size_t)m0 * a_ld;
    const float* asg = as + (size_t)m0 * as_ld;
    uint32_t* sBw = (uint32_t*)sB;
    auto gptr = [](const int8_t* p) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
    };
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
    /* Run 266 (V3b, after the expert kernel's V3): the staging below was a rolled loop of 16 iterations a thread, each
     * one weight-word gather and the shared-memory store that waits for it -- sixteen global latencies in series a
     * group (the ISA: load.ugm.d32.a64 -> store.slm.d32.a32 {I@1} -> jmpi). The 16 words are this thread's rows
     * nb + 4 i at word k4 (compile-time shapes of that loop), so they load together and store together; a column past
     * N reads column N - 1 (valid memory) and stores the zero the guard produced. The same words at the same places. */
    const int k4 = tid % 32, nb = tid / 32;
    const int8_t* bk = b + (size_t)4 * k4;   /* this thread's word of every weight row */
    for (int g = 0; g < G; ++g) {
        xe::syncthreads();                                  /* the previous group's tiles are consumed */
        uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int nn = n0 + nb + 4 * i, nc = nn < N ? nn : N - 1;
            const uint32_t v = *(const uint32_t*)(bk + (size_t)nc * b_ld + (size_t)g * 128);
            w[i] = nn < N ? v : 0u;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int n = nb + 4 * i;
            sBw[((k4 / 8) * 4 + n / 16) * 128 + (k4 % 8) * 16 + (n % 16)] = w[i];
        }
        xe::syncthreads();
#pragma unroll
        for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, C[t], 0);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, A, gptr(arow + (size_t)g * 128 + step * 32), (size_t)a_ld);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bt, lptr(sB + (step * 4 + t) * kDpasTileB), 64);
                jm::joint_matrix_mad(sg, C[t], A, Bt, C[t]);   /* the toolchain's form: D = A * B + C */
            }
        }
        /* the group's exact sums, scaled: lane = column, 8 rows a lane. V3b: from the accumulator's own work-item data
         * (the tile laid out lane = column, element e = row e -- what joint_matrix_store wrote and sCw[e * 16 + lane]
         * read back; the expert kernel's run-170 cut), so no store, no read-back, no wave barriers; and the 8 row
         * scales once a group (the barriers had fenced them, so they were gathered again for every tile: 33 a group).
         * A column past N reads column N - 1 and takes the zero scale the guard gave it. The same values, the same
         * (sum x a_scale) x b_scale in the same t, e order. */
        float ar[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) ar[e] = asg[(size_t)e * as_ld + g];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int nn = n0 + t * 16 + lane, nc = nn < N ? nn : N - 1;
            const float sbv = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)nc * bs_ld + g] << 16)
                                      : ((const float*)bs)[(size_t)nc * bs_ld + g];
            const float sb = nn < N ? sbv : 0.0f;
            auto wd = sycl::ext::oneapi::detail::get_wi_data(sg, C[t]);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] += ((float)(int32_t)wd[e] * ar[e]) * sb;
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int nn = n0 + t * 16 + lane;
        if (nn >= N) continue;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t u = __float_as_uint(acc[t][e]);   /* round to nearest even, NaN kept quiet */
            y[(size_t)(m0 + e) * y_ld + nn] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                            : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* ---- run 270 (owner note 209 item 3): the DPAS form without the shared-memory repack -- the "T form",
 * RADIANCE_XE_GEMM_DPAS_T=1 (opt-in) ----
 *
 * gemm_i8a8_dpas_kernel stages every 128-K weight block through LDS, repacked for the B operand: 16 gathers, 16
 * scatters and two barriers a thread a group, in every one of the N / 64 work-groups (B90: the matrix pipe 5.8 %
 * busy, stalled 50 %). The matrix unit's B operand must be packed (the 4 consecutive k of one column in a word) and
 * both stored operands are [rows][K] -- the wrong major for a packed load. So this form swaps the roles: the WEIGHT
 * is the A operand (an 8 x 32 int8 row-major tile of [N][K], loaded straight from memory, as the activation is in
 * the form above) and the ACTIVATION is the B operand, packed ONCE per call by pack_a_i8_kernel into P[k4][m] (word
 * (k4, m) = the 4 consecutive int8 of row m at 4*k4 -- the same words the staging above scattered into LDS, written
 * once for all the work-groups). The tile is then C^T: row e = weight row n0 + 8w + e, column (lane) = activation
 * row m0 + 16t + lane; the inner loop is block loads and joint_matrix_mad only. The int32 group sums are the same
 * exact integers in any order; the fold keeps the form above's ((float)isum * a_scale[m, g]) * b_scale[n, g], g
 * ascending from 0.0f, per element, and the same bf16 rounding -- so the output is bit for bit the form above's /
 * the GEMV's. The launcher gives this form whole 64-row x 64-column blocks only: the column tail (N % 64) keeps the
 * form above and the row tail (M % 64) the GEMV, so no tile reads past N or M. The packed buffer is one device
 * allocation a queue, grown to the largest M x K seen (first at the startup warm-up, before any recording) and never
 * freed; a call the buffer cannot take (allocation failure) runs the form above. */
void pack_a_i8_kernel(const int8_t* __restrict__ a, long a_ld, uint32_t* __restrict__ p, int Mp) {
    XE_SHARED(uint32_t, tile, 32 * 33);          /* [m 32][k4 32], padded a word: 32 rows x one 128-K group */
    const int tid = (int)threadIdx.x;              /* 256 threads */
    const int k40 = (int)blockIdx.x * 32, m0 = (int)blockIdx.y * 32;
    {
        const int r = tid >> 3, c = tid & 7;       /* row m0 + r, its words 4c .. 4c + 3 of the group (16 bytes) */
        const uint4 v = *(const uint4*)(a + (size_t)(m0 + r) * a_ld + (size_t)k40 * 4 + (size_t)c * 16);
        tile[r * 33 + 4 * c + 0] = v.x;
        tile[r * 33 + 4 * c + 1] = v.y;
        tile[r * 33 + 4 * c + 2] = v.z;
        tile[r * 33 + 4 * c + 3] = v.w;
    }
    xe::syncthreads();
    {
        const int k4 = tid >> 3, mm = (tid & 7) * 4;   /* word row k40 + k4, activation rows m0 + mm .. + 3 */
        uint4 o;
        o.x = tile[(mm + 0) * 33 + k4];
        o.y = tile[(mm + 1) * 33 + k4];
        o.z = tile[(mm + 2) * 33 + k4];
        o.w = tile[(mm + 3) * 33 + k4];
        *(uint4*)(p + (size_t)(k40 + k4) * (size_t)Mp + (size_t)(m0 + mm)) = o;
    }
}

template <bool WS_BF16>
void gemm_i8a8_dpas_t_kernel(const uint32_t* __restrict__ p, long p_ld, const float* __restrict__ as, long as_ld,
                             const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                             uint16_t* __restrict__ y, long y_ld, int K) {
    const int tid = (int)threadIdx.x, wave = tid / kDpasSG, lane = tid % kDpasSG;
    const int n0 = (int)blockIdx.x * kDpasNT + wave * 8;   /* this sub-group's 8 weight rows (whole: the launcher's N64) */
    const int m0 = (int)blockIdx.y * kDpasRT;              /* the work-group's 64 activation rows: 4 tiles of 16 */
    const int G = K / 128;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> A;              /* weights */
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> Bt;     /* activations */
    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C[4];
    float acc[4][8];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;
    auto gptr = [](const int8_t* q) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(q);
    };
    const int8_t* brow = b + (size_t)n0 * b_ld;
    const int8_t* pbase = (const int8_t*)(p + (size_t)m0);         /* + k4 * p_ld words */
    const size_t p_stride = (size_t)p_ld * 4;                        /* int8 elements between packed rows */
    const float* asm_ = as + (size_t)(m0 + lane) * as_ld;            /* this lane's activation rows: m0 + 16t + lane */
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, C[t], 0);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, A, gptr(brow + (size_t)g * 128 + step * 32), (size_t)b_ld);
            const int8_t* pk = pbase + (size_t)(g * 32 + step * 8) * p_stride;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bt, gptr(pk + (size_t)t * 16 * 4), p_stride);
                jm::joint_matrix_mad(sg, C[t], A, Bt, C[t]);
            }
        }
        /* the fold, element (n = n0 + e, m = m0 + 16t + lane): the same ((float)isum * a_scale) * b_scale, g ascending */
        float sbe[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
            sbe[e] = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)(n0 + e) * bs_ld + g] << 16)
                             : ((const float*)bs)[(size_t)(n0 + e) * bs_ld + g];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float ar = asm_[(size_t)t * 16 * as_ld + g];
            auto wd = sycl::ext::oneapi::detail::get_wi_data(sg, C[t]);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] += ((float)(int32_t)wd[e] * ar) * sbe[e];
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        uint16_t* yr = y + (size_t)(m0 + t * 16 + lane) * y_ld + n0;   /* 8 consecutive columns of one row */
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t u = __float_as_uint(acc[t][e]);   /* round to nearest even, NaN kept quiet */
            yr[e] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

bool dpas_t_on() {
    static const bool want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_DPAS_T"); return v && v[0] == '1'; }();
    return want;
}
/* the packed-activation buffer of a queue: grown to the largest K/4 x Mp words asked for, never freed (a few tens of
 * MB at most: 4,096 rows x 8,192 K = 32 MiB); the growth waits for the queue's in-flight work first */
uint32_t* pack_buffer(sycl::queue* q, size_t words) {
    struct Buf { uint32_t* p; size_t words; };
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, Buf> bufs;
    std::lock_guard<std::mutex> lk(mu);
    Buf& b = bufs[q];
    if (b.words >= words && b.p) return b.p;
    if (b.p) { q->wait(); sycl::free(b.p, *q); b.p = nullptr; b.words = 0; }
    uint32_t* np = nullptr;
    try { np = sycl::malloc_device<uint32_t>(words, *q); } catch (const sycl::exception&) { np = nullptr; }
    if (!np) return nullptr;
    b.p = np; b.words = words;
    static bool said = false;
    if (!said) { said = true; std::fprintf(stderr, "[xe] gemm_i8a8: the T form (RADIANCE_XE_GEMM_DPAS_T=1) packs the activations into %.1f MiB a queue\n", (double)words * 4.0 / 1048576.0); }
    return np;
}

/* ---- the matrix-unit form for decode rows (run 99, owner note 105 item 1; RADIANCE_XE_GEMM_DPAS_ROWS=1) ----
 *
 * The MTP verify's 4 rows paid per-row work on the GEMV (P41: gemv_i8a8<4> 40.9 us against 19.7 at one row): a
 * DP4A chain and three cross-lane steps a row a group. Here a work-group of kRowsPh sub-groups (sub-group 16) owns
 * one 16-column tile for up to 8 rows (rows past M staged as zeros); sub-group p takes the groups g = p, p + 4, ...
 * -- the GEMV's lanes (l >> 3) partition, in the same ascending order. Per group it stages its 8 x 128 activation
 * block and the tile's 16 x 128 weight block (packed for the B operand, as gemm_i8a8_dpas_kernel) in its own LDS,
 * runs four 8 x 16 x 32 joint_matrix_mad steps into an exact int32 tile, and adds ((float)isum * a_scale) * b_scale
 * to its row's accumulator: the GEMV's expression. The four phases' sums meet as the GEMV's butterfly does,
 * ((p0 + p1) + (p2 + p3)), and one bf16 rounding. So the same bits as the GEMV, with the products on the matrix
 * unit and the weight read once for all the rows. The phases run in lock-step rounds behind work-group barriers. */
constexpr int kRowsSG = 16, kRowsPh = 4, kRowsThreads = kRowsSG * kRowsPh;   /* 64: one 16-column tile */
constexpr int kRowsMax = 8;                                                  /* rows a launch serves */

template <bool WS_BF16, int MODE = 1>
void gemm_i8a8_rows_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                           const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                           uint16_t* __restrict__ y, long y_ld, int N, int K, int M) {
    XE_SHARED(int8_t, sA, kRowsPh * kRowsMax * 128);   /* a phase's 8 rows x 128 k */
    XE_SHARED(int8_t, sB, kRowsPh * 4 * kDpasTileB);   /* a phase's group: [step 4][packed 32 k x 16 n] */
    XE_SHARED(int32_t, sC, kRowsPh * 8 * 16);          /* a phase's 8 x 16 int32 tile */
    XE_SHARED(float, sR, kRowsPh * 8 * 16);            /* the phases' sums, for the butterfly's order */
    const int tid = (int)threadIdx.x, ph = tid / kRowsSG, lane = tid % kRowsSG;
    const int n0 = (int)blockIdx.x * 16, nn = n0 + lane;
    const int G = K / 128;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> A;
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> Bt;
    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C;
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
    int8_t* pA = sA + ph * kRowsMax * 128;
    int8_t* pB = sB + ph * 4 * kDpasTileB;
    uint32_t* pBw = (uint32_t*)pB;
    int32_t* pC = sC + ph * 128;
    float acc[kRowsMax];
#pragma unroll
    for (int e = 0; e < kRowsMax; ++e) acc[e] = 0.0f;
    /* Every phase runs the same number of rounds (a phase past G stages zeros and adds nothing), so the stages can
     * be fenced by work-group barriers: a sub-group barrier between the lanes' local stores and the matrix unit's
     * block loads let a few results differ from the GEMV's in BK12 (run 99). */
    const int rounds = (G + kRowsPh - 1) / kRowsPh;
    for (int rd = 0; rd < rounds; ++rd) {
        const int g = rd * kRowsPh + ph;
        const bool gl = g < G;
        xe::syncthreads();                                   /* the previous round's stages are consumed */
        for (int i = lane; i < kRowsMax * 8; i += kRowsSG) {  /* 8 rows x 8 chunks of 16 bytes */
            const int r = i / 8, c = (i % 8) * 16;
            uint4 v = {0u, 0u, 0u, 0u};
            if (gl && r < M) v = *(const uint4*)(a + (size_t)r * a_ld + (size_t)g * 128 + c);
            *(uint4*)(pA + r * 128 + c) = v;
        }
        for (int i = lane; i < 16 * 8; i += kRowsSG) {        /* 16 weight rows x 8 chunks of 16 bytes */
            const int n = i / 8, c16 = i % 8;
            uint4 v = {0u, 0u, 0u, 0u};
            if (gl && n0 + n < N) v = *(const uint4*)(b + (size_t)(n0 + n) * b_ld + (size_t)g * 128 + c16 * 16);
            const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k4 = c16 * 4 + j;                   /* word (n, k4): row n's int8 at 4 k4 .. 4 k4 + 3 */
                pBw[(k4 / 8) * 128 + (k4 % 8) * 16 + n] = wv[j];
            }
        }
        xe::syncthreads();
        jm::joint_matrix_fill(sg, C, 0);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, A, lptr(pA + step * 32), (size_t)128);
            jm::joint_matrix_load(sg, Bt, lptr(pB + step * kDpasTileB), 64);
            jm::joint_matrix_mad(sg, C, A, Bt, C);
        }
        jm::joint_matrix_store(sg, C, lptr(pC), 16, jm::layout::row_major);
        xe::syncthreads();
        if (!gl) continue;
        float sb = 0.0f;
        if (nn < N) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)nn * bs_ld + g] << 16)
                                 : ((const float*)bs)[(size_t)nn * bs_ld + g];
#pragma unroll
        for (int e = 0; e < kRowsMax; ++e)
            if (e < M) {
                /* MODE (run 101): BK12-BK12e found this form a last bit away from the GEMV on a few outputs, which
                 * flips a near-tie in the M-pair's text (M105 / M108). 1 the GEMV's expression as written; 2 the
                 * add fused (fma); 3 the product kept apart from the add (through a volatile), to find which the
                 * GEMV's compiled code does. */
                const float t = (float)pC[e * 16 + lane] * as[(size_t)e * as_ld + g];
                if constexpr (MODE == 2) acc[e] = sycl::fma(t, sb, acc[e]);
                else if constexpr (MODE == 3) { volatile float u = t * sb; acc[e] = acc[e] + u; }
                else acc[e] += t * sb;
            }
    }
#pragma unroll
    for (int e = 0; e < kRowsMax; ++e) sR[(ph * 8 + e) * 16 + lane] = acc[e];
    xe::syncthreads();
    if (ph != 0 || nn >= N) return;
#pragma unroll
    for (int e = 0; e < kRowsMax; ++e) {
        if (e >= M) break;
        const float v = (sR[(0 * 8 + e) * 16 + lane] + sR[(1 * 8 + e) * 16 + lane])
                        + (sR[(2 * 8 + e) * 16 + lane] + sR[(3 * 8 + e) * 16 + lane]);
        const uint32_t u = __float_as_uint(v);   /* round to nearest even, NaN kept quiet */
        y[(size_t)e * y_ld + nn] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                               : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
    }
}

bool dpas_on(sycl::queue* q) {
    static const bool want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_DPAS"); return !(v && !std::strcmp(v, "0")); }();
    static const bool gpu = q->get_device().is_gpu();   /* the CPU device has no matrix unit for int8 */
    return want && gpu;
}
/* RADIANCE_XE_GEMM_DPAS_ROWS=1|2|3 (owner note 105 item 1): 2-8 rows on the matrix unit, with the kernel's MODE */
int dpas_rows_mode(sycl::queue* q) {
    static const int want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_DPAS_ROWS"); return v && v[0] >= '1' && v[0] <= '3' ? v[0] - '0' : 0; }();
    return dpas_on(q) ? want : 0;
}

template <bool WS_BF16>
void launch(sycl::queue* q, const int8_t* a, long a_ld, const float* as, long as_ld, const int8_t* b, long b_ld,
            const void* bs, long bs_ld, uint16_t* y, long y_ld, int N, int K, int M) {
    int m_start = 0;
    if (M >= kDpasRT && dpas_on(q)) {
        const int blocks = M / kDpasRT;
        const dim3 dblock(kDpasThreads);
        /* run 270 (owner note 209 item 3): the T form for the whole 64 x 64 blocks; the column tail (N % 64) keeps the
         * staged form at an offset, the row tail (M % 64) the GEMV below. a_ld % 16 == 0 and a 16-byte aligned (the
         * entry's checks) are what the pack kernel's 16-byte loads need. */
        const int N64 = (N / kDpasNT) * kDpasNT;
        bool done = false;
        if (N64 > 0 && dpas_t_on()) {
            const int Mp = blocks * kDpasRT;
            uint32_t* p = pack_buffer(q, (size_t)(K / 4) * (size_t)Mp);
            if (p) {
                const dim3 pgrid((unsigned)(K / 128), (unsigned)(Mp / 32)), pblock(256);
                XE_LAUNCH_SG(16, q, pack_a_i8_kernel, pgrid, pblock, 0, a, a_ld, p, Mp);
                const dim3 tgrid((unsigned)(N64 / kDpasNT), (unsigned)blocks);
                XE_LAUNCH_SG(kDpasSG, q, (gemm_i8a8_dpas_t_kernel<WS_BF16>), tgrid, dblock, 0, (const uint32_t*)p, (long)Mp, as, as_ld, b, b_ld, bs, bs_ld, y, y_ld, K);
                if (N64 < N) {
                    const dim3 dgrid(1u, (unsigned)blocks);
                    const void* bs_t = (const char*)bs + (size_t)N64 * (size_t)bs_ld * (WS_BF16 ? 2u : 4u);
                    XE_LAUNCH_SG(kDpasSG, q, (gemm_i8a8_dpas_kernel<WS_BF16>), dgrid, dblock, 0, a, a_ld, as, as_ld, b + (size_t)N64 * b_ld, b_ld, bs_t, bs_ld, y + N64, y_ld, N - N64, K);
                }
                done = true;
            }
        }
        if (!done) {
            const dim3 dgrid((unsigned)((N + kDpasNT - 1) / kDpasNT), (unsigned)blocks);
            XE_LAUNCH_SG(kDpasSG, q, (gemm_i8a8_dpas_kernel<WS_BF16>), dgrid, dblock, 0, a, a_ld, as, as_ld, b, b_ld, bs, bs_ld, y, y_ld, N, K);
        }
        m_start = blocks * kDpasRT;
        if (m_start >= M) return;
    }
    const int rmode = M - m_start >= 2 ? dpas_rows_mode(q) : 0;
    if (rmode) {   /* owner note 105 item 1: chunks of up to 8 rows, a lone row to the GEMV */
        const dim3 rgrid((unsigned)((N + 15) / 16)), rblock(kRowsThreads);
        for (; M - m_start >= 2; m_start += kRowsMax) {
            const int mb = M - m_start < kRowsMax ? M - m_start : kRowsMax;
#define XE_ROWS(MD) XE_LAUNCH_SG(kRowsSG, q, (gemm_i8a8_rows_kernel<WS_BF16, MD>), rgrid, rblock, 0,                \
                                 a + (size_t)m_start * a_ld, a_ld, as + (size_t)m_start * as_ld, as_ld, b, b_ld, bs,  \
                                 bs_ld, y + (size_t)m_start * y_ld, y_ld, N, K, mb)
            if (rmode == 2) XE_ROWS(2); else if (rmode == 3) XE_ROWS(3); else XE_ROWS(1);
#undef XE_ROWS
        }
        if (m_start >= M) return;
    }
    const int rw = xe::gemv_rows_wg();   /* run 207: RADIANCE_XE_GEMV_WG (note 203 item 2) */
    const dim3 grid((unsigned)((N + rw - 1) / rw)), block(32 * rw);
    static const bool rt =[] { const char* e = std::getenv("RADIANCE_XE_GEMV_RT"); return e && e[0] == '1'; }();
    for (int m0 = m_start; m0 < M; m0 += kMaxM) {
        const int mb = M - m0 < kMaxM ? M - m0 : kMaxM;
        const int8_t* am = a + (size_t)m0 * a_ld;
        const float* asm_ = as + (size_t)m0 * as_ld;
        uint16_t* ym = y + (size_t)m0 * y_ld;
        /* owner note 141 item 1: a group a lane; 1 = the multi-row launches, 2 = every launch; 3 / 4 the same with the
         * fused scale-accumulate (run 115) */
        static const int glf = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_GL"); return e ? std::atoi(e) : 0; }();
        if (glf == 2 || glf == 4 || ((glf == 1 || glf == 3 || glf == 5 || glf == 6) && mb > 1)) {
            const dim3 ggrid((unsigned)((N + kGlCols - 1) / kGlCols)), gblock(4 * kGlCols);
#define XE_GL(MB_, F_) XE_LAUNCH_SG(16, q, (gemv_i8a8_gl_kernel<WS_BF16, MB_, F_>), ggrid, gblock, 0, am, a_ld, asm_, \
                                    as_ld, b, b_ld, bs, bs_ld, ym, y_ld, N, K, mb)
            const int fm = glf >= 5 ? glf - 3 : glf >= 3 ? 1 : 0;
            if (mb == 1) { if (fm == 1) { XE_GL(1, 1); } else { XE_GL(1, 0); } }
            else if (fm == 1) { XE_GL(kMaxM, 1); }
            else if (fm == 2) { XE_GL(kMaxM, 2); }
            else if (fm == 3) { XE_GL(kMaxM, 3); }
            else { XE_GL(kMaxM, 0); }
#undef XE_GL
            continue;
        }
        if (rt && mb > 1) {   /* owner note 129 item 3: the same bits, 4 cross-lane steps a group instead of 12 */
            XE_LAUNCH(q, (gemv_i8a8_rt_kernel<WS_BF16>), grid, block, 0, am, a_ld, asm_, as_ld, b, b_ld, bs, bs_ld, ym,
                      y_ld, N, K, mb);
            continue;
        }
        static const bool nx = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_NX"); return e && e[0] == '1'; }();
        /* run 209 (owner note 206 item 1): the model's trunk K values (2560, 6144, 640) as compile-time instances of the
         * NX form under RADIANCE_XE_GEMV_KC=1; any other K, or KC off, the runtime-K instance */
        static const bool kc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_KC"); return e && e[0] == '1'; }();
        /* run 211 (owner note 206 item 2): the weight loads' LSC cache-control hint, RADIANCE_XE_GEMV_WCC = 2 (L1
         * uncached, L3 cached), 5 (L1 streaming, L3 uncached) or 6 (L1 streaming, L3 cached); else the default */
        /* 5 the default since run 212 (owner, H146 +2.6 % with the three GRF launches); RADIANCE_XE_GEMV_WCC=0 opts out */
        static const int wcc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_WCC"); return e ? std::atoi(e) : 5; }();
#define XE_GVK(MB_, NX_, KC_) XE_LAUNCH(q, (gemv_i8a8_kernel<WS_BF16, MB_, NX_, KC_>), grid, block, 0, am, a_ld, asm_,   \
                                        as_ld, b, b_ld, bs, bs_ld, ym, y_ld, N, K, mb)
#define XE_GVW(MB_, WCC_) XE_LAUNCH(q, (gemv_i8a8_kernel<WS_BF16, MB_, true, 0, WCC_>), grid, block, 0, am, a_ld, asm_, \
                                    as_ld, b, b_ld, bs, bs_ld, ym, y_ld, N, K, mb)
#define XE_GVX(MB_)                                                                                                 \
        do {                                                                                                        \
            if (kc && K == 2560) XE_GVK(MB_, true, 2560);                                                           \
            else if (kc && K == 6144) XE_GVK(MB_, true, 6144);                                                      \
            else if (kc && K == 640) XE_GVK(MB_, true, 640);                                                        \
            else if (wcc == 2) XE_GVW(MB_, 2);                                                                      \
            else if (wcc == 5) XE_GVW(MB_, 5);                                                                      \
            else if (wcc == 6) XE_GVW(MB_, 6);                                                                      \
            else XE_GVK(MB_, true, 0);                                                                              \
        } while (0)
        if (mb == 1) { if (nx) XE_GVX(1); else XE_GVK(1, false, 0); }
        else { if (nx) XE_GVX(kMaxM); else XE_GVK(kMaxM, false, 0); }
#undef XE_GVX
#undef XE_GVW
#undef XE_GVK
    }
}

}  // namespace

/* gemm_nt_q operands: a, a_scale, b, b_scale, y [, a_sum, b_ref]. */
extern "C" int xe_rad_gemm_i8a8_gemv(const RadArgs* args, RadStream s) {
    const RadTensor* A = r4d_opt(args, 0);
    const RadTensor* AS = r4d_opt(args, 1);
    const RadTensor* B = r4d_opt(args, 2);
    const RadTensor* BS = r4d_opt(args, 3);
    const RadTensor* Y = r4d_opt(args, 4);
    if (!A || !AS || !B || !BS || !Y || !A->data || !AS->data || !B->data || !BS->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);
    if (r4d_opt(args, 5) || r4d_opt(args, 6)) return R4D_NO(RAD_E_UNSUPPORTED);   /* no zero point, no e8m0 */
    if (r4d_geti_or(args, "pb1", 0) || r4d_geti_or(args, "pb2", 0) || r4d_geti_or(args, "split_n", 0))
        return R4D_NO(RAD_E_UNSUPPORTED);
    if (A->dtype != RAD_I8 || B->dtype != RAD_I8 || AS->dtype != RAD_F32 || Y->dtype != RAD_BF16 ||
        (BS->dtype != RAD_BF16 && BS->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long N = r4d_geti_or(args, "N", r4d_dim(B, (int)B->rank - 2));
    if (K <= 0 || N <= 0 || K % 128 || r4d_geti_or(args, "group", 128) != 128) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mas = 0, Nb = 0, Nbs = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &M);
    const long long a_ld = r4d_rows_pitch(A, K, &Ma);
    const long long as_ld = r4d_rows_pitch(AS, K / 128, &Mas);
    const long long b_ld = r4d_rows_pitch(B, K, &Nb);
    const long long bs_ld = r4d_rows_pitch(BS, K / 128, &Nbs);
    if (y_ld < 0 || a_ld < 0 || as_ld < 0 || b_ld < 0 || bs_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || Mas < M || Nb < N || Nbs < N) return R4D_NO(RAD_E_SHAPE);
    /* 16-byte loads */
    if (((uintptr_t)A->data | (uintptr_t)B->data) % 16 || a_ld % 16 || b_ld % 16) return R4D_NO(RAD_E_STRIDE);
    try {
        sycl::queue* q = (sycl::queue*)s;
        if (BS->dtype == RAD_BF16)
            launch<true>(q, (const int8_t*)A->data, (long)a_ld, (const float*)AS->data, (long)as_ld,
                         (const int8_t*)B->data, (long)b_ld, BS->data, (long)bs_ld, (uint16_t*)Y->data, (long)y_ld,
                         (int)N, (int)K, (int)M);
        else
            launch<false>(q, (const int8_t*)A->data, (long)a_ld, (const float*)AS->data, (long)as_ld,
                          (const int8_t*)B->data, (long)b_ld, BS->data, (long)bs_ld, (uint16_t*)Y->data, (long)y_ld,
                          (int)N, (int)K, (int)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- the paired GEMV ops' adapters (run 274) */
extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* args, RadStream s);   /* xe_gemv_bf16.cpp: the bf16 set's fallback */

namespace {

RadParam gm_pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam gm_ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

/* One int8 set of a paired op (b, b_scale, y against the shared a): the standalone adapter's checks. RAD_OK, or
 * RAD_E_UNSUPPORTED for a form the paired kernels are not built for (f32 scales), or an error. */
int gm_i8_set(const RadTensor* B, const RadTensor* BS, const RadTensor* Y, long long N, long long K, long long M,
              GvI8Set* o) {
    if (!B || !BS || !Y || !B->data || !BS->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    if (B->dtype != RAD_I8 || Y->dtype != RAD_BF16 || (BS->dtype != RAD_BF16 && BS->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    long long My = 0, Nb = 0, Nbs = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &My), b_ld = r4d_rows_pitch(B, K, &Nb),
                    bs_ld = r4d_rows_pitch(BS, K / 128, &Nbs);
    if (y_ld < 0 || b_ld < 0 || bs_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < M || Nb < N || Nbs < N) return R4D_NO(RAD_E_SHAPE);
    if ((uintptr_t)B->data % 16 || b_ld % 16) return R4D_NO(RAD_E_STRIDE);
    if (BS->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;
    o->b = (const int8_t*)B->data; o->bs = BS->data; o->y = (uint16_t*)Y->data;
    o->b_ld = (long)b_ld; o->bs_ld = (long)bs_ld; o->y_ld = (long)y_ld; o->n = (long)N;
    return RAD_OK;
}

/* The bf16 set (x, b2, y2): gemv's entry conditions (16-byte x, a_ld % 8). */
int gm_bf_set(const RadTensor* X, const RadTensor* B, const RadTensor* Y, long long N, long long K, long long M,
              GvBfSet* o) {
    if (!X || !B || !Y || !X->data || !B->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || B->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0, Nb = 0;
    const long long a_ld = r4d_rows_pitch(X, K, &Mx), y_ld = r4d_rows_pitch(Y, N, &My), b_ld = r4d_rows_pitch(B, K, &Nb);
    if (a_ld < 0 || y_ld < 0 || b_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx < M || My < M || Nb < N) return R4D_NO(RAD_E_SHAPE);
    if ((uintptr_t)X->data % 16 || a_ld % 8 || (uintptr_t)B->data % 16 || b_ld % 8) return RAD_E_UNSUPPORTED;
    o->a = (const uint16_t*)X->data; o->b = (const uint16_t*)B->data; o->y = (uint16_t*)Y->data;
    o->a_ld = (long)a_ld; o->b_ld = (long)b_ld; o->y_ld = (long)y_ld; o->n = (long)N;
    return RAD_OK;
}

/* The shared activation: a int8 [M][K], a_scale f32 [M][K / 128]. */
int gm_act(const RadArgs* args, long long K, const int8_t** a, long* a_ld, const float** as, long* as_ld, long long* M) {
    const RadTensor* A = r4d_opt(args, 0);
    const RadTensor* AS = r4d_opt(args, 1);
    if (!A || !AS || !A->data || !AS->data) return R4D_NO(RAD_E_INVAL);
    if (A->dtype != RAD_I8 || AS->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (K <= 0 || K % 128 || r4d_geti_or(args, "group", 128) != 128) return R4D_NO(RAD_E_SHAPE);
    long long Ma = 0, Mas = 0;
    const long long ald = r4d_rows_pitch(A, K, &Ma), asld = r4d_rows_pitch(AS, K / 128, &Mas);
    if (ald < 0 || asld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Ma <= 0 || Mas < Ma) return R4D_NO(RAD_E_SHAPE);
    if ((uintptr_t)A->data % 16 || ald % 16) return R4D_NO(RAD_E_STRIDE);
    *a = (const int8_t*)A->data; *a_ld = (long)ald; *as = (const float*)AS->data; *as_ld = (long)asld; *M = Ma;
    return RAD_OK;
}

/* The separate launches (today's kernels) for a form the paired kernels are not built for; said once. */
void gm_fallback_note(const char* op, const char* why) {
    static std::atomic<int> said{0};
    if (said.fetch_add(1) == 0)
        std::fprintf(stderr, "libxe: %s runs its sets as separate launches (%s); the paired form is not built for it\n",
                     op, why);
}
int gm_fallback_i8(const RadArgs* args, int ib, long long N, long long K, RadStream s) {
    RadTensor t[5] = { args->t[0], args->t[1], args->t[ib], args->t[ib + 1], args->t[ib + 2] };
    RadParam p[4] = { gm_pi("N", N), gm_pi("K", K), gm_pi("group", 128), gm_ps("dtype", "i8a8") };
    RadArgs sub = *args; sub.t = t; sub.n_t = 5; sub.p = p; sub.n_p = 4;
    return xe_rad_gemm_i8a8_gemv(&sub, s);
}
int gm_fallback_bf(const RadArgs* args, int ix, long long N, long long K, RadStream s) {
    RadTensor t[3] = { args->t[ix], args->t[ix + 1], args->t[ix + 2] };
    RadParam p[3] = { gm_pi("N", N), gm_pi("K", K), gm_ps("dtype", "bf16") };
    RadArgs sub = *args; sub.t = t; sub.n_t = 3; sub.p = p; sub.n_p = 3;
    return xe_rad_gemm_nt_bf16(&sub, s);
}

}  // namespace

/* gemm_nt_q_x3 (pair 5): a, a_scale, b1, b1_scale, y1, b2, b2_scale, y2, b3, b3_scale, y3; M, K, N1, N2, N3, group,
 * dtype i8a8. One launch over the three sets in the trunk's forms, else the three separate launches. */
extern "C" int xe_rad_gemm_nt_q_x3(const RadArgs* args, RadStream s) {
    if (!args || args->n_t < 11) return R4D_NO(RAD_E_INVAL);
    const long long K = r4d_geti_or(args, "K", 0);
    const long long N[3] = { r4d_geti_or(args, "N1", 0), r4d_geti_or(args, "N2", 0), r4d_geti_or(args, "N3", 0) };
    const int8_t* a = nullptr; const float* as = nullptr; long a_ld = 0, as_ld = 0; long long M = 0;
    int rc = gm_act(args, K, &a, &a_ld, &as, &as_ld, &M);
    if (rc != RAD_OK) return rc;
    GvI8Set st[3];
    bool fused = true;
    for (int i = 0; i < 3; ++i) {
        rc = gm_i8_set(r4d_opt(args, 2 + 3 * i), r4d_opt(args, 3 + 3 * i), r4d_opt(args, 4 + 3 * i), N[i], K, M, &st[i]);
        if (rc == RAD_E_UNSUPPORTED) fused = false;
        else if (rc != RAD_OK) return rc;
    }
    int wcc = 5;
    if (fused && !gemv_multi_form((int)K, &wcc)) fused = false;
    if (!fused) {
        gm_fallback_note("gemm_nt_q_x3", "a GEMV env form or f32 scales");
        for (int i = 0; i < 3; ++i)
            if ((rc = gm_fallback_i8(args, 2 + 3 * i, N[i], K, s)) != RAD_OK) return rc;
        return RAD_OK;
    }
    try {
        sycl::queue* q = (sycl::queue*)s;
        const int rw = xe::gemv_rows_wg();
        const long g1 = (st[0].n + rw - 1) / rw, g2 = (st[1].n + rw - 1) / rw, g3 = (st[2].n + rw - 1) / rw;
        for (long long m0 = 0; m0 < M; m0 += kMaxM) {
            const int mb = (int)(M - m0 < kMaxM ? M - m0 : kMaxM);
            GvI8Set t1 = st[0], t2 = st[1], t3 = st[2];
            t1.y += (size_t)m0 * t1.y_ld; t2.y += (size_t)m0 * t2.y_ld; t3.y += (size_t)m0 * t3.y_ld;
            const int8_t* am = a + (size_t)m0 * a_ld;
            const float* asm_ = as + (size_t)m0 * as_ld;
            if (mb == 1) { if (wcc == 5) launch_x3_mb<1, 5>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, g3, t1, t2, t3);
                           else launch_x3_mb<1, 0>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, g3, t1, t2, t3); }
            else { if (wcc == 5) launch_x3_mb<kMaxM, 5>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, g3, t1, t2, t3);
                   else launch_x3_mb<kMaxM, 0>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, g3, t1, t2, t3); }
        }
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* gemm_nt_q_bf16_pair (pair 4): a, a_scale, b1, b1_scale, y1, x, b2, y2; M, K, N1, N2, group, dtype i8a8 -- an int8
 * set over the activation's codes and a bf16 set over its bf16 plane in one launch, else the two separate launches. */
extern "C" int xe_rad_gemm_nt_q_bf16_pair(const RadArgs* args, RadStream s) {
    if (!args || args->n_t < 8) return R4D_NO(RAD_E_INVAL);
    const long long K = r4d_geti_or(args, "K", 0);
    const long long N1 = r4d_geti_or(args, "N1", 0), N2 = r4d_geti_or(args, "N2", 0);
    const int8_t* a = nullptr; const float* as = nullptr; long a_ld = 0, as_ld = 0; long long M = 0;
    int rc = gm_act(args, K, &a, &a_ld, &as, &as_ld, &M);
    if (rc != RAD_OK) return rc;
    GvI8Set s1; GvBfSet s2;
    bool fused = true;
    rc = gm_i8_set(r4d_opt(args, 2), r4d_opt(args, 3), r4d_opt(args, 4), N1, K, M, &s1);
    if (rc == RAD_E_UNSUPPORTED) fused = false; else if (rc != RAD_OK) return rc;
    rc = gm_bf_set(r4d_opt(args, 5), r4d_opt(args, 6), r4d_opt(args, 7), N2, K, M, &s2);
    if (rc == RAD_E_UNSUPPORTED) fused = false; else if (rc != RAD_OK) return rc;
    int wcc = 5;
    if (fused && !gemv_multi_form((int)K, &wcc)) fused = false;
    if (!fused) {
        gm_fallback_note("gemm_nt_q_bf16_pair", "a GEMV env form, f32 scales or an unaligned bf16 set");
        if ((rc = gm_fallback_i8(args, 2, N1, K, s)) != RAD_OK) return rc;
        return gm_fallback_bf(args, 5, N2, K, s);
    }
    try {
        sycl::queue* q = (sycl::queue*)s;
        const int rw = xe::gemv_rows_wg();
        const long g1 = (s1.n + rw - 1) / rw, g2 = (s2.n + rw - 1) / rw;
        for (long long m0 = 0; m0 < M; m0 += kMaxM) {
            const int mb = (int)(M - m0 < kMaxM ? M - m0 : kMaxM);
            GvI8Set t1 = s1; GvBfSet t2 = s2;
            t1.y += (size_t)m0 * t1.y_ld; t2.y += (size_t)m0 * t2.y_ld; t2.a += (size_t)m0 * t2.a_ld;
            const int8_t* am = a + (size_t)m0 * a_ld;
            const float* asm_ = as + (size_t)m0 * as_ld;
            if (mb == 1) { if (wcc == 5) launch_pair_mb<1, 5>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, t1, t2);
                           else launch_pair_mb<1, 0>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, t1, t2); }
            else { if (wcc == 5) launch_pair_mb<kMaxM, 5>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, t1, t2);
                   else launch_pair_mb<kMaxM, 0>(q, am, a_ld, asm_, as_ld, (int)K, mb, g1, g2, t1, t2); }
        }
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
