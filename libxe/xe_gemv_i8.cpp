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

#include <cstdint>
#include <cstring>

namespace {

constexpr int kRowsPerWG = 8;
constexpr int kMaxM = 4;   /* activation rows a weight pass serves; more rows loop the launch */

template <bool WS_BF16, int MB>
void gemv_i8a8_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as, long as_ld,
                      const int8_t* __restrict__ b, long b_ld, const void* __restrict__ bs, long bs_ld,
                      uint16_t* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5);
    const int G = K / 128;
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
        if (gl) w = *(const uint4*)(brow + (size_t)g * 128 + chunk * 16);
        float sb = 0.0f;
        if (gl) sb = WS_BF16 ? __uint_as_float((uint32_t)((const uint16_t*)bs)[(size_t)n * bs_ld + g] << 16)
                             : ((const float*)bs)[(size_t)n * bs_ld + g];
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            int s = 0;
            if (gl && m < M) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16);
                const unsigned wv[4] = {w.x, w.y, w.z, w.w}, xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        s += (int)(int8_t)(wv[u] >> (8 * k)) * (int)(int8_t)(xv[u] >> (8 * k));
            }
            /* the group's exact int32 sum over its 8 lanes */
            s += __builtin_amdgcn_ds_bpermute((lane ^ 1) << 2, s);
            s += __builtin_amdgcn_ds_bpermute((lane ^ 2) << 2, s);
            s += __builtin_amdgcn_ds_bpermute((lane ^ 4) << 2, s);
            if (gl && m < M && chunk == 0) acc[m] += ((float)s * as[(size_t)m * as_ld + g]) * sb;
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 16);
        if (live && lane == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);   /* round to nearest even, NaN kept quiet */
            y[(size_t)m * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                 : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

template <bool WS_BF16>
void launch(sycl::queue* q, const int8_t* a, long a_ld, const float* as, long as_ld, const int8_t* b, long b_ld,
            const void* bs, long bs_ld, uint16_t* y, long y_ld, int N, int K, int M) {
    const dim3 grid((unsigned)((N + kRowsPerWG - 1) / kRowsPerWG)), block(32 * kRowsPerWG);
    for (int m0 = 0; m0 < M; m0 += kMaxM) {
        const int mb = M - m0 < kMaxM ? M - m0 : kMaxM;
        const int8_t* am = a + (size_t)m0 * a_ld;
        const float* asm_ = as + (size_t)m0 * as_ld;
        uint16_t* ym = y + (size_t)m0 * y_ld;
        if (mb == 1)
            XE_LAUNCH(q, (gemv_i8a8_kernel<WS_BF16, 1>), grid, block, 0, am, a_ld, asm_, as_ld, b, b_ld, bs, bs_ld, ym, y_ld, N, K, mb);
        else
            XE_LAUNCH(q, (gemv_i8a8_kernel<WS_BF16, kMaxM>), grid, block, 0, am, a_ld, asm_, as_ld, b, b_ld, bs, bs_ld, ym, y_ld, N, K, mb);
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
