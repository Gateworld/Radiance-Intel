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
#include <cstdlib>
#include <cstring>

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
    XE_SHARED(int32_t, sC, kDpasWaves * 8 * 16);        /* one 8 x 16 int32 tile a sub-group */
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
    int32_t* sCw = sC + wave * 128;
    auto gptr = [](const int8_t* p) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
    };
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
    for (int g = 0; g < G; ++g) {
        xe::syncthreads();                                  /* the previous group's tiles are consumed */
        for (int idx = tid; idx < kDpasNT * 32; idx += kDpasThreads) {   /* 2048 words: 16 a thread */
            const int n = idx / 32, k4 = idx % 32, nn = n0 + n;
            uint32_t w = 0;
            if (nn < N) w = *(const uint32_t*)(b + (size_t)nn * b_ld + (size_t)g * 128 + 4 * k4);
            sBw[((k4 / 8) * 4 + n / 16) * 128 + (k4 % 8) * 16 + (n % 16)] = w;
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
        /* the group's exact sums, scaled: lane = column, 8 rows a lane */
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            jm::joint_matrix_store(sg, C[t], lptr(sCw), 16, jm::layout::row_major);
            xe::wave_barrier();
            const int nn = n0 + t * 16 + lane;
            float sb = 0.0f;
            if (nn < N) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)nn * bs_ld + g] << 16)
                                     : ((const float*)bs)[(size_t)nn * bs_ld + g];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] += ((float)sCw[e * 16 + lane] * asg[(size_t)e * as_ld + g]) * sb;
            xe::wave_barrier();
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

bool dpas_on(sycl::queue* q) {
    static const bool want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_DPAS"); return !(v && !std::strcmp(v, "0")); }();
    static const bool gpu = q->get_device().is_gpu();   /* the CPU device has no matrix unit for int8 */
    return want && gpu;
}

template <bool WS_BF16>
void launch(sycl::queue* q, const int8_t* a, long a_ld, const float* as, long as_ld, const int8_t* b, long b_ld,
            const void* bs, long bs_ld, uint16_t* y, long y_ld, int N, int K, int M) {
    int m_start = 0;
    if (M >= kDpasRT && dpas_on(q)) {
        const int blocks = M / kDpasRT;
        const dim3 dgrid((unsigned)((N + kDpasNT - 1) / kDpasNT), (unsigned)blocks), dblock(kDpasThreads);
        XE_LAUNCH_SG(kDpasSG, q, (gemm_i8a8_dpas_kernel<WS_BF16>), dgrid, dblock, 0, a, a_ld, as, as_ld, b, b_ld, bs, bs_ld, y, y_ld, N, K);
        m_start = blocks * kDpasRT;
        if (m_start >= M) return;
    }
    const int rw = xe::gemv_rows_wg();   /* run 207: RADIANCE_XE_GEMV_WG (note 203 item 2) */
    const dim3 grid((unsigned)((N + rw - 1) / rw)), block(32 * rw);
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
        static const bool nx = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_NX"); return e && e[0] == '1'; }();
        /* run 209 (owner note 206 item 1): the model's trunk K values (2560, 6144, 640) as compile-time instances of the
         * NX form under RADIANCE_XE_GEMV_KC=1; any other K, or KC off, the runtime-K instance */
        static const bool kc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_KC"); return e && e[0] == '1'; }();
        /* run 211 (owner note 206 item 2): the weight loads' LSC cache-control hint, RADIANCE_XE_GEMV_WCC = 2 (L1
         * uncached, L3 cached), 5 (L1 streaming, L3 uncached) or 6 (L1 streaming, L3 cached); else the default */
        static const int wcc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_WCC"); return e ? std::atoi(e) : 0; }();
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
