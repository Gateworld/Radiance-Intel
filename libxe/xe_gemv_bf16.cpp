/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_gemv_bf16.cpp -- the bf16 products a qwen4exp decode step issues besides the int8 trunk:
 * gemm_nt (GDN a|b, the shared expert's gate column, the QSA indexer's q/k projection) and the router
 * GEMM in front of router_topk_scatter (router_gemm_topk_scatter). libr4d does these on bf16 WMMA;
 * here a sub-group GEMV: y[m, n] = sum_k a[m, k] * b[n, k], f32 accumulate, one rounding to y's
 * dtype. Lane l reads 16-byte chunks l, l + 32, ... of a weight row (8 bf16 each); the partial sums
 * meet in a butterfly, so the summation order differs from libref's ascending k (a tolerance check).
 * Up to 4 activation rows share one pass over the weight; more rows loop.
 */
#include "xe_lanes.h"
#include "r4d_args.h"

#include <cstdint>

extern "C" int r4d_rad_router_topk_scatter(const RadArgs* a, RadStream s);   /* ports/r4d_moe.cpp */

namespace {

constexpr int kRowsPerWG = 8;   /* one output row a sub-group */
constexpr int kMaxM = 4;

inline float ld_bf16(const uint16_t* p, long i) { return __uint_as_float((uint32_t)p[i] << 16); }
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}

template <bool A_F32, bool Y_F32, int MB>
void gemv_bf16_kernel(const void* __restrict__ a, long a_ld, const uint16_t* __restrict__ b, long b_ld,
                      void* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5);
    const bool live = n < N;
    const uint16_t* brow = b + (size_t)(live ? n : 0) * b_ld;
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
            if constexpr (A_F32) {
                const float* ar = (const float*)a + (size_t)m * a_ld + (size_t)c * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) s += ar[e] * wf[e];
            } else {
                const uint16_t* ar = (const uint16_t*)a + (size_t)m * a_ld + (size_t)c * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) s += ld_bf16(ar, e) * wf[e];
            }
            acc[m] += s;
        }
    }
    /* the K % 8 tail, bytewise, on lane 0 */
    for (int k = nchunk * 8 + lane; k < K && live; k += 32)
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            const float av = A_F32 ? ((const float*)a)[(size_t)m * a_ld + k] : ld_bf16((const uint16_t*)a, (long)((size_t)m * a_ld + k));
            acc[m] += av * ld_bf16(brow, k);
        }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        for (int x = 16; x >= 1; x >>= 1) v += __shfl_xor(v, x);
        if (live && lane == 0 && m < M) {
            if constexpr (Y_F32) ((float*)y)[(size_t)m * y_ld + n] = v;
            else ((uint16_t*)y)[(size_t)m * y_ld + n] = to_bf16(v);
        }
    }
}

template <bool A_F32, bool Y_F32>
void launch(sycl::queue* q, const void* a, long a_ld, const uint16_t* b, long b_ld, void* y, long y_ld, int N, int K, int M) {
    const dim3 grid((unsigned)((N + kRowsPerWG - 1) / kRowsPerWG)), block(32 * kRowsPerWG);
    const size_t ae = A_F32 ? 4 : 2, ye = Y_F32 ? 4 : 2;
    for (int m0 = 0; m0 < M; m0 += kMaxM) {
        const int mb = M - m0 < kMaxM ? M - m0 : kMaxM;
        const void* am = (const char*)a + (size_t)m0 * a_ld * ae;
        void* ym = (char*)y + (size_t)m0 * y_ld * ye;
        if (mb == 1) XE_LAUNCH(q, (gemv_bf16_kernel<A_F32, Y_F32, 1>), grid, block, 0, am, a_ld, b, b_ld, ym, y_ld, N, K, mb);
        else XE_LAUNCH(q, (gemv_bf16_kernel<A_F32, Y_F32, kMaxM>), grid, block, 0, am, a_ld, b, b_ld, ym, y_ld, N, K, mb);
    }
}

/* y = a @ b^T for a [M, K] (bf16 or f32), b [N, K] bf16, y [M, N] (bf16 or f32). */
int gemv(const RadTensor* A, const RadTensor* B, const RadTensor* Y, long long N, long long K, RadStream s) {
    if (!A || !B || !Y || !A->data || !B->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    if (B->dtype != RAD_BF16 || (A->dtype != RAD_BF16 && A->dtype != RAD_F32) ||
        (Y->dtype != RAD_BF16 && Y->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (N <= 0 || K <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Nb = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &M);
    const long long a_ld = r4d_rows_pitch(A, K, &Ma);
    const long long b_ld = r4d_rows_pitch(B, K, &Nb);
    if (y_ld < 0 || a_ld < 0 || b_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || Nb < N) return R4D_NO(RAD_E_SHAPE);
    /* 16-byte loads of b and a (an f32 a reads 32 bytes a chunk, still 16-byte aligned) */
    if (((uintptr_t)B->data | (uintptr_t)A->data) % 16 || b_ld % 8 || a_ld % 8) return R4D_NO(RAD_E_STRIDE);
    try {
        sycl::queue* q = (sycl::queue*)s;
        const bool af = A->dtype == RAD_F32, yf = Y->dtype == RAD_F32;
        const uint16_t* b = (const uint16_t*)B->data;
        if (af && yf) launch<true, true>(q, A->data, (long)a_ld, b, (long)b_ld, Y->data, (long)y_ld, (int)N, (int)K, (int)M);
        else if (af) launch<true, false>(q, A->data, (long)a_ld, b, (long)b_ld, Y->data, (long)y_ld, (int)N, (int)K, (int)M);
        else if (yf) launch<false, true>(q, A->data, (long)a_ld, b, (long)b_ld, Y->data, (long)y_ld, (int)N, (int)K, (int)M);
        else launch<false, false>(q, A->data, (long)a_ld, b, (long)b_ld, Y->data, (long)y_ld, (int)N, (int)K, (int)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

}  // namespace

/* gemm_nt: a, b, y */
extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* args, RadStream s) {
    const RadTensor* B = r4d_opt(args, 1);
    if (!B) return R4D_NO(RAD_E_INVAL);
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long N = r4d_geti_or(args, "N", r4d_dim(B, (int)B->rank - 2));
    return gemv(r4d_opt(args, 0), B, r4d_opt(args, 2), N, K, s);
}

/* router_gemm_topk_scatter: a, b, gate_b?, logits, expert_ids, expert_w, sorted_tok, expert_offset,
 * expert_count, gate? -- the reference's own launches (libref: "the reference IS the launches"):
 * logits = a @ b^T, the shared expert's gate = a @ gate_b^T when both are present, then
 * router_topk_scatter (the ported libr4d kernel) over the logits and the five outputs after it. */
extern "C" int xe_rad_router_gemm_topk_scatter(const RadArgs* args, RadStream s) {
    if (!args || args->n_t < 9) return R4D_NO(RAD_E_INVAL);
    const RadTensor* A = r4d_opt(args, 0);
    const RadTensor* B = r4d_opt(args, 1);
    const RadTensor* GB = r4d_opt(args, 2);
    const RadTensor* G = args->n_t > 9 ? r4d_opt(args, 9) : nullptr;
    if (!A || !B) return R4D_NO(RAD_E_INVAL);
    if ((GB && GB->data) != (G && G->data)) return R4D_NO(RAD_E_INVAL);   /* together or not at all */
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long E = r4d_geti_or(args, "n_expert", r4d_dim(B, (int)B->rank - 2));
    int rc = gemv(A, B, r4d_opt(args, 3), E, K, s);
    if (rc != RAD_OK) return rc;
    if (GB && GB->data && (rc = gemv(A, GB, G, 1, K, s)) != RAD_OK) return rc;
    RadArgs sub = *args;
    sub.t = args->t + 3;
    sub.n_t = 6;
    return r4d_rad_router_topk_scatter(&sub, s);
}

/* logits_gemm at dtype bf16: x, lm_head, logits. (Production's int8 lm_head is a joined code + scale
 * plane, r4d_fp8_layout.cpp's i8 logits layout: not served here yet.) */
extern "C" int xe_rad_logits_gemm_bf16(const RadArgs* args, RadStream s) {
    const RadTensor* B = r4d_opt(args, 1);
    if (!B) return R4D_NO(RAD_E_INVAL);
    const long long V = r4d_geti_or(args, "n_vocab", r4d_dim(B, (int)B->rank - 2));
    const long long E = r4d_geti_or(args, "n_embd", r4d_dim(B, (int)B->rank - 1));
    return gemv(r4d_opt(args, 0), B, r4d_opt(args, 2), V, E, s);
}
