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
#include <sycl/ext/oneapi/matrix/matrix.hpp>   /* the prefill form on the matrix unit (run 58) */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

extern "C" int r4d_rad_router_topk_scatter(const RadArgs* a, RadStream s);   /* ports/r4d_moe.cpp */

namespace {

constexpr int kRowsPerWG = 8;   /* one output row a sub-group */
constexpr int kMaxM = 4;

inline float ld_bf16(const uint16_t* p, long i) { return __uint_as_float((uint32_t)p[i] << 16); }
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}

/* KC (run 209, owner note 206 item 1, RADIANCE_XE_GEMV_KC=1): the model's K (2560) as a compile-time constant (0: the
 * runtime K) -- the chunk loop's trip count known, the K % 8 tail gone. The same chunks in the same order. */
template <bool A_F32, bool Y_F32, int MB, int KC = 0>
void gemv_bf16_kernel(const void* __restrict__ a, long a_ld, const uint16_t* __restrict__ b, long b_ld,
                      void* __restrict__ y, long y_ld, int N, int Kr, int M) {
    const int K = KC ? KC : Kr;
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);   /* xe::gemv_rows_wg() */
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
                /* run 207 (owner note 203 item 1): the chunk's 8 activations in one 16-byte load (the entry checks
                 * a % 16, a_ld % 8), split as the weights are; IGC 2.42.6 issues the eight 16-bit loads of
                 * ld_bf16(ar, e) one by one where 2.28.4 paired them. The same values in the same order. */
                const uint4 x = *(const uint4*)((const uint16_t*)a + (size_t)m * a_ld + (size_t)c * 8);
                const unsigned xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned h = (e & 1) ? (xv[e >> 1] & 0xffff0000u) : (xv[e >> 1] << 16);
                    s += __uint_as_float(h) * wf[e];
                }
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

/* RADIANCE_XE_ROUTER_PAIR=1 (run 102, owner note 109 item 3): the router's GEMV and the shared expert's gate column
 * (same activation, same K) in one launch -- rows [0, N) of b into y, rows [N, N + N2) of b2 into y2 -- instead of
 * two (48 launches a decode step fewer). Each row is gemv_bf16_kernel's row, the same arithmetic in the same order,
 * so the same bits. The pointers ride in a struct of 8-byte fields (no padding: the tape audit compares bytes). */
struct GvPair {
    const uint16_t* b2 = nullptr;
    void* y2 = nullptr;
    long b2_ld = 0, y2_ld = 0, n2 = 0;
};
template <bool A_F32, bool Y_F32, int MB>
void gemv_bf16_pair_kernel(const void* __restrict__ a, long a_ld, const uint16_t* __restrict__ b, long b_ld,
                           void* __restrict__ y, long y_ld, int N, int K, int M, GvPair p) {
    const int lane = (int)threadIdx.x & 31;
    const int nn = (int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5);   /* 8 rows: no CPU row test (run 207) */
    const bool second = nn >= N;
    const int n = second ? nn - N : nn;
    const bool live = second ? n < (int)p.n2 : true;
    const uint16_t* brow = (second ? p.b2 + (size_t)(live ? n : 0) * p.b2_ld : b + (size_t)n * b_ld);
    void* yy = second ? p.y2 : y;
    const long yld = second ? p.y2_ld : y_ld;
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
                /* run 207: one 16-byte load, as gemv_bf16_kernel's */
                const uint4 x = *(const uint4*)((const uint16_t*)a + (size_t)m * a_ld + (size_t)c * 8);
                const unsigned xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned h = (e & 1) ? (xv[e >> 1] & 0xffff0000u) : (xv[e >> 1] << 16);
                    s += __uint_as_float(h) * wf[e];
                }
            }
            acc[m] += s;
        }
    }
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
            if constexpr (Y_F32) ((float*)yy)[(size_t)m * yld + n] = v;
            else ((uint16_t*)yy)[(size_t)m * yld + n] = to_bf16(v);
        }
    }
}

/* ---- the matrix-unit form for prefill rows (milestone 3, stage B, run 58) ----
 *
 * The 7K prefill's profile (B10) showed gemm_nt's GEMV launched once a token a layer -- 204,546 launches of 19 us
 * on a rank, 12 % of the first token -- for products at M = a chunk's rows. For M >= 64 a work-group of 8
 * sub-groups (sub-group 16, the DPAS width) owns 64 activation rows by 64 output columns: per 64-K group it stages
 * the rows' activations as bf16 (an f32 activation rounded to nearest even; the GEMV reads bf16 exactly and f32 as
 * it is -- the products of bf16 pairs are exact in f32 either way) and the 64 weight rows in the packed B layout
 * (word (k2, n) = b[n][2 k2] | b[n][2 k2 + 1] << 16), runs four bf16 8 x 16 x 16 joint_matrix_mad steps into f32
 * tiles that persist over the whole K (no per-group scaling here), and writes y with one rounding to its dtype.
 * The summation order differs from the GEMV's butterfly as that differs from libref's ascending k: f32 sums of
 * exact products, a tolerance check. K's tail past a multiple of 64 is staged as zeros. The launcher sends full
 * 64-row blocks here (RADIANCE_XE_GEMM_BF16_DPAS=0 keeps the GEMV) and the tail rows to the GEMV. */
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int kBfSG = 16, kBfWaves = 8, kBfThreads = kBfSG * kBfWaves;   /* 128 */
constexpr int kBfRT = 8 * kBfWaves;                                      /* 64 rows a work-group */
constexpr int kBfNT = 64;                                                /* 4 tiles of 16 columns */
constexpr int kBfAP = 64 + 8;                                            /* the A stage's row pitch in bf16 */

template <bool A_F32, bool Y_F32>
void gemm_bf16_dpas_kernel(const void* __restrict__ a, long a_ld, const uint16_t* __restrict__ b, long b_ld,
                           void* __restrict__ y, long y_ld, int N, int K, int M) {
    XE_SHARED(uint16_t, sA, kBfRT * kBfAP);   /* 64 rows x 64 k bf16, pitch 72 */
    XE_SHARED(uint32_t, sB, 4 * 4 * 128);     /* [k step 4][n tile 4][packed 8 x 16 words]: 64 k x 64 n */
    XE_SHARED(float, sC, kBfWaves * 8 * 16);  /* one 8 x 16 f32 tile a sub-group */
    const int tid = (int)threadIdx.x, wave = tid / kBfSG, lane = tid % kBfSG;
    const int m0 = (int)blockIdx.y * kBfRT;   /* a full 64-row block: the launcher's choice */
    const int n0 = (int)blockIdx.x * kBfNT;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::a, 8, 16, jm::layout::row_major> Am;
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> Bm;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> Cm[4];
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
#pragma unroll
    for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, Cm[t], 0.0f);
    const int G = (K + 63) / 64;
    /* Run 266 (V3b, with gemm_i8a8_dpas_kernel's): the two stages below were rolled loops of 16 iterations a thread,
     * each a gather and the shared-memory store that waits for it -- 32 global latencies in series a group (the ISA).
     * A thread's 16 words of each stage are rows (columns) sr + 4 i at one k pair, compile-time shapes of those
     * loops: the 32 words load together, then store together. A row past M, a column past N or a k past K reads the
     * last valid one (valid memory) and keeps the zero the guard produced. The same words at the same places. */
    const int sr = tid / 32, kp = (tid % 32) * 2;   /* this thread's A words: rows sr + 4 i at the k pair kp */
    const int k2 = tid % 32, bstep = k2 / 8, bkw = k2 % 8;   /* its B words: columns sr + 4 i at the k pair 2 k2 */
    for (int g = 0; g < G; ++g) {
        const int k0 = g * 64;
        xe::syncthreads();                                  /* the previous group's stages are consumed */
        /* the activation rows: 64 rows x 64 k, two k a word (zeros past K or past M) */
        uint32_t wa[16];
        {
            const int kk = k0 + kp;
            const int kc0 = kk < K ? kk : K - 1, kc1 = kk + 1 < K ? kk + 1 : K - 1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int mm = m0 + sr + 4 * i, mc = mm < M ? mm : M - 1;
                uint16_t lo, hi;
                if constexpr (A_F32) {
                    const float* ar = (const float*)a + (size_t)mc * a_ld;
                    const float f0 = ar[kc0], f1 = ar[kc1];
                    lo = kk < K ? to_bf16(f0) : (uint16_t)0;
                    hi = kk + 1 < K ? to_bf16(f1) : (uint16_t)0;
                } else {
                    const uint16_t* ar = (const uint16_t*)a + (size_t)mc * a_ld;
                    const uint16_t v0 = ar[kc0], v1 = ar[kc1];
                    lo = kk < K ? v0 : (uint16_t)0;
                    hi = kk + 1 < K ? v1 : (uint16_t)0;
                }
                const uint32_t w = (uint32_t)lo | ((uint32_t)hi << 16);
                wa[i] = mm < M ? w : 0u;
            }
        }
        /* the weight rows: 64 n x 64 k, packed (word (k2, n)); zeros past N or K */
        uint32_t wb[16];
        {
            const int kk = k0 + 2 * k2;
            const int kc0 = kk < K ? kk : K - 1, kc1 = kk + 1 < K ? kk + 1 : K - 1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int nn = n0 + sr + 4 * i, nc = nn < N ? nn : N - 1;
                const uint16_t* br = b + (size_t)nc * b_ld;
                const uint16_t v0 = br[kc0], v1 = br[kc1];
                const uint16_t lo = kk < K ? v0 : (uint16_t)0, hi = kk + 1 < K ? v1 : (uint16_t)0;
                const uint32_t w = (uint32_t)lo | ((uint32_t)hi << 16);
                wb[i] = nn < N ? w : 0u;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) *(uint32_t*)(sA + (sr + 4 * i) * kBfAP + kp) = wa[i];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int n = sr + 4 * i;
            sB[((bstep * 4 + n / 16) * 8 + bkw) * 16 + (n % 16)] = wb[i];
        }
        xe::syncthreads();
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, Am, lptr((sycl::ext::oneapi::bfloat16*)(sA + wave * 8 * kBfAP + step * 16)), (size_t)kBfAP);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bm, lptr((sycl::ext::oneapi::bfloat16*)(sB + (step * 4 + t) * 128)), 32);
                jm::joint_matrix_mad(sg, Cm[t], Am, Bm, Cm[t]);
            }
        }
    }
    /* the tiles out: lane = column, 8 rows a lane */
    float* sCw = sC + wave * 128;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        jm::joint_matrix_store(sg, Cm[t], lptr(sCw), 16, jm::layout::row_major);
        xe::wave_barrier();
        const int nn = n0 + t * 16 + lane;
        if (nn < N) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int mm = m0 + wave * 8 + e;
                if (mm >= M) continue;
                const float v = sCw[e * 16 + lane];
                if constexpr (Y_F32) ((float*)y)[(size_t)mm * y_ld + nn] = v;
                else ((uint16_t*)y)[(size_t)mm * y_ld + nn] = to_bf16(v);
            }
        }
        xe::wave_barrier();
    }
}

/* ---- run 270 (owner note 209 item 3): the bf16 DPAS form without the shared-memory stages -- the "T form",
 * RADIANCE_XE_GEMM_DPAS_T=1 (opt-in, the same switch as xe_gemv_i8.cpp's), for K % 64 == 0 ----
 *
 * gemm_bf16_dpas_kernel stages both operands through LDS every 64-K group (32 gathers + 32 scatters + 2 barriers
 * a thread a group in each of the N / 64 work-groups) because the matrix unit's B operand must be packed and the
 * activation may be f32. Here the roles are swapped: the WEIGHT (bf16 [N][K], row-major) is the A operand loaded
 * straight from memory as 8 x 16 tiles, and the ACTIVATION is the B operand, packed ONCE per call by
 * pack_a_bf16_kernel into P[k2][m] (word (k2, m) = bf16(a[m][2 k2]) | bf16(a[m][2 k2 + 1]) << 16 -- the words the
 * stage above built, the f32 activation rounded to nearest even exactly as there) and read with packed block loads;
 * the inner loop is 1 A load + 4 B loads + 4 joint_matrix_mad a step, no LDS, no barriers; the f32 tiles persist over
 * K as above and the output is written from the accumulator's own work-item data (lane = activation row, 8 weight
 * rows a lane: 8 consecutive outputs of one y row). The same products in the same 16-k steps, so the sums are the
 * form above's (the hardware's k order inside a step does not depend on which operand is which); checked by the
 * 7K logprobs. Whole 64 x 64 blocks only (the column tail N % 64 keeps the form above at an offset, the row tail the
 * GEMV); K % 64 == 0 (the trunk's K are multiples of 64), so no tile reads past N, M or K. */
template <bool A_F32>
void pack_a_bf16_kernel(const void* __restrict__ a, long a_ld, uint32_t* __restrict__ p, int Mp) {
    XE_SHARED(uint32_t, tile, 32 * 33);          /* [m 32][k2 32], padded a word: 32 rows x one 64-K group */
    const int tid = (int)threadIdx.x;              /* 256 threads */
    const int k20 = (int)blockIdx.x * 32, m0 = (int)blockIdx.y * 32;
    {
        const int r = tid >> 3, c = tid & 7;       /* row m0 + r, its k pairs 4c .. 4c + 3 */
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long kk = (long)(k20 + 4 * c + j) * 2;
            uint16_t lo, hi;
            if constexpr (A_F32) {
                const float* ar = (const float*)a + (size_t)(m0 + r) * a_ld;
                lo = to_bf16(ar[kk]); hi = to_bf16(ar[kk + 1]);
            } else {
                const uint16_t* ar = (const uint16_t*)a + (size_t)(m0 + r) * a_ld;
                lo = ar[kk]; hi = ar[kk + 1];
            }
            tile[r * 33 + 4 * c + j] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
    }
    xe::syncthreads();
    {
        const int k2 = tid >> 3, mm = (tid & 7) * 4;   /* word row k20 + k2, activation rows m0 + mm .. + 3 */
        uint4 o;
        o.x = tile[(mm + 0) * 33 + k2];
        o.y = tile[(mm + 1) * 33 + k2];
        o.z = tile[(mm + 2) * 33 + k2];
        o.w = tile[(mm + 3) * 33 + k2];
        *(uint4*)(p + (size_t)(k20 + k2) * (size_t)Mp + (size_t)(m0 + mm)) = o;
    }
}

template <bool Y_F32>
void gemm_bf16_dpas_t_kernel(const uint32_t* __restrict__ p, long p_ld, const uint16_t* __restrict__ b, long b_ld,
                             void* __restrict__ y, long y_ld, int K) {
    const int tid = (int)threadIdx.x, wave = tid / kBfSG, lane = tid % kBfSG;
    const int n0 = (int)blockIdx.x * kBfNT + wave * 8;   /* this sub-group's 8 weight rows (whole: the launcher's N64) */
    const int m0 = (int)blockIdx.y * kBfRT;              /* the work-group's 64 activation rows: 4 tiles of 16 */
    const int G = K / 64;
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::a, 8, 16, jm::layout::row_major> Am;       /* weights */
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> Bm; /* activations */
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> Cm[4];
    auto gptr = [](const sycl::ext::oneapi::bfloat16* q) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(q);
    };
#pragma unroll
    for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, Cm[t], 0.0f);
    const sycl::ext::oneapi::bfloat16* brow = (const sycl::ext::oneapi::bfloat16*)(b + (size_t)n0 * b_ld);
    const sycl::ext::oneapi::bfloat16* pbase = (const sycl::ext::oneapi::bfloat16*)(p + (size_t)m0);   /* + k2 * p_ld words */
    const size_t p_stride = (size_t)p_ld * 2;                                                             /* bf16 elements between packed rows */
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, Am, gptr(brow + (size_t)g * 64 + step * 16), (size_t)b_ld);
            const sycl::ext::oneapi::bfloat16* pk = pbase + (size_t)(g * 32 + step * 8) * p_stride;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bm, gptr(pk + (size_t)t * 32), p_stride);
                jm::joint_matrix_mad(sg, Cm[t], Am, Bm, Cm[t]);
            }
        }
    }
    /* the tiles out from the work-item data: lane = activation row m0 + 16t + lane, element e = weight row n0 + e */
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        auto wd = sycl::ext::oneapi::detail::get_wi_data(sg, Cm[t]);
        const size_t row = (size_t)(m0 + t * 16 + lane) * y_ld + n0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float v = wd[e];
            if constexpr (Y_F32) ((float*)y)[row + e] = v;
            else ((uint16_t*)y)[row + e] = to_bf16(v);
        }
    }
}

bool bf16_dpas_t_on() {
    static const bool want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_DPAS_T"); return v && v[0] == '1'; }();
    return want;
}
/* the packed-activation buffer of a queue (its own, beside xe_gemv_i8.cpp's): grown to the largest K/2 x Mp words
 * asked for, never freed; the growth waits for the queue's in-flight work first */
uint32_t* bf16_pack_buffer(sycl::queue* q, size_t words) {
    struct Buf { uint32_t* p; size_t words; };
    static std::mutex mu;
    static std::map<sycl::queue*, Buf> bufs;
    std::lock_guard<std::mutex> lk(mu);
    Buf& bb = bufs[q];
    if (bb.words >= words && bb.p) return bb.p;
    if (bb.p) { q->wait(); sycl::free(bb.p, *q); bb.p = nullptr; bb.words = 0; }
    uint32_t* np = nullptr;
    try { np = sycl::malloc_device<uint32_t>(words, *q); } catch (const sycl::exception&) { np = nullptr; }
    if (!np) return nullptr;
    bb.p = np; bb.words = words;
    static bool said = false;
    if (!said) { said = true; std::fprintf(stderr, "[xe] gemm_bf16: the T form (RADIANCE_XE_GEMM_DPAS_T=1) packs the activations into %.1f MiB a queue\n", (double)words * 4.0 / 1048576.0); }
    return np;
}

bool bf16_dpas_on(sycl::queue* q) {   /* RADIANCE_XE_GEMM_BF16_DPAS=0: prefill rows keep the GEMV; the CPU device has no matrix unit */
    static const bool want = [] { const char* v = std::getenv("RADIANCE_XE_GEMM_BF16_DPAS"); return !(v && *v == '0'); }();
    static const bool gpu = q->get_device().is_gpu();
    return want && gpu;
}

template <bool A_F32, bool Y_F32>
void launch(sycl::queue* q, const void* a, long a_ld, const uint16_t* b, long b_ld, void* y, long y_ld, int N, int K, int M) {
    int m_start = 0;
    if (M >= kBfRT && bf16_dpas_on(q)) {
        const int blocks = M / kBfRT;
        const dim3 dblock(kBfThreads);
        /* run 270 (owner note 209 item 3): the T form for the whole 64 x 64 blocks at K % 64 == 0; the column tail
         * (N % 64) keeps the staged form at an offset, the row tail (M % 64) the GEMV below */
        const int N64 = (N / kBfNT) * kBfNT;
        bool done = false;
        if (N64 > 0 && K % 64 == 0 && bf16_dpas_t_on()) {
            const int Mp = blocks * kBfRT;
            uint32_t* p = bf16_pack_buffer(q, (size_t)(K / 2) * (size_t)Mp);
            if (p) {
                const dim3 pgrid((unsigned)(K / 64), (unsigned)(Mp / 32)), pblock(256);
                XE_LAUNCH_SG(16, q, (pack_a_bf16_kernel<A_F32>), pgrid, pblock, 0, a, a_ld, p, Mp);
                const dim3 tgrid((unsigned)(N64 / kBfNT), (unsigned)blocks);
                XE_LAUNCH_SG(kBfSG, q, (gemm_bf16_dpas_t_kernel<Y_F32>), tgrid, dblock, 0, (const uint32_t*)p, (long)Mp, b, b_ld, y, y_ld, K);
                if (N64 < N) {
                    const dim3 dgrid(1u, (unsigned)blocks);
                    void* y_t = (char*)y + (size_t)N64 * (Y_F32 ? 4u : 2u);
                    XE_LAUNCH_SG(kBfSG, q, (gemm_bf16_dpas_kernel<A_F32, Y_F32>), dgrid, dblock, 0, a, a_ld, b + (size_t)N64 * b_ld, b_ld, y_t, y_ld, N - N64, K, M);
                }
                done = true;
            }
        }
        if (!done) {
            const dim3 dgrid((unsigned)((N + kBfNT - 1) / kBfNT), (unsigned)blocks);
            XE_LAUNCH_SG(kBfSG, q, (gemm_bf16_dpas_kernel<A_F32, Y_F32>), dgrid, dblock, 0, a, a_ld, b, b_ld, y, y_ld, N, K, M);
        }
        m_start = blocks * kBfRT;
        if (m_start >= M) return;
    }
    const int rw = xe::gemv_rows_wg();   /* run 207: RADIANCE_XE_GEMV_WG (note 203 item 2) */
    const dim3 grid((unsigned)((N + rw - 1) / rw)), block(32 * rw);
    const size_t ae = A_F32 ? 4 : 2, ye = Y_F32 ? 4 : 2;
    for (int m0 = m_start; m0 < M; m0 += kMaxM) {
        const int mb = M - m0 < kMaxM ? M - m0 : kMaxM;
        const void* am = (const char*)a + (size_t)m0 * a_ld * ae;
        void* ym = (char*)y + (size_t)m0 * y_ld * ye;
        /* run 209 (owner note 206 item 1): K = 2560 as a compile-time instance under RADIANCE_XE_GEMV_KC=1 */
        static const bool kc = [] { const char* e = std::getenv("RADIANCE_XE_GEMV_KC"); return e && e[0] == '1'; }();
        if (kc && K == 2560) {
            if (mb == 1) XE_LAUNCH(q, (gemv_bf16_kernel<A_F32, Y_F32, 1, 2560>), grid, block, 0, am, a_ld, b, b_ld, ym, y_ld, N, K, mb);
            else XE_LAUNCH(q, (gemv_bf16_kernel<A_F32, Y_F32, kMaxM, 2560>), grid, block, 0, am, a_ld, b, b_ld, ym, y_ld, N, K, mb);
            continue;
        }
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

bool router_pair_on() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_XE_ROUTER_PAIR"); return e && e[0] == '1'; }();
    return v;
}

/* y = a @ b^T and y2 = a @ b2^T in one launch (gemv_bf16_pair_kernel): the decode / verify rows only (M <= kMaxM),
 * both outputs of one dtype. RAD_E_UNSUPPORTED for anything else, and the caller launches the two GEMVs. */
int gemv_pair(const RadTensor* A, const RadTensor* B, const RadTensor* Y, long long N, const RadTensor* B2,
              const RadTensor* Y2, long long N2, long long K, RadStream s) {
    if (!A || !B || !Y || !B2 || !Y2 || !A->data || !B->data || !Y->data || !B2->data || !Y2->data) return RAD_E_UNSUPPORTED;
    if (B->dtype != RAD_BF16 || B2->dtype != RAD_BF16 || (A->dtype != RAD_BF16 && A->dtype != RAD_F32) ||
        (Y->dtype != RAD_BF16 && Y->dtype != RAD_F32) || Y2->dtype != Y->dtype)
        return RAD_E_UNSUPPORTED;
    if (N <= 0 || N2 <= 0 || K <= 0) return RAD_E_UNSUPPORTED;
    long long M = 0, Ma = 0, Nb = 0, M2 = 0, Nb2 = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &M), y2_ld = r4d_rows_pitch(Y2, N2, &M2);
    const long long a_ld = r4d_rows_pitch(A, K, &Ma);
    const long long b_ld = r4d_rows_pitch(B, K, &Nb), b2_ld = r4d_rows_pitch(B2, K, &Nb2);
    if (y_ld < 0 || y2_ld < 0 || a_ld < 0 || b_ld < 0 || b2_ld < 0) return RAD_E_UNSUPPORTED;
    if (M <= 0 || M > kMaxM || M2 != M || Ma < M || Nb < N || Nb2 < N2) return RAD_E_UNSUPPORTED;
    if (((uintptr_t)B->data | (uintptr_t)B2->data | (uintptr_t)A->data) % 16 || b_ld % 8 || b2_ld % 8 || a_ld % 8)
        return RAD_E_UNSUPPORTED;
    try {
        sycl::queue* q = (sycl::queue*)s;
        GvPair p;
        p.b2 = (const uint16_t*)B2->data; p.y2 = Y2->data; p.b2_ld = (long)b2_ld; p.y2_ld = (long)y2_ld; p.n2 = (long)N2;
        /* run 207: pinned at 8 rows whatever RADIANCE_XE_GEMV_WG says (the pair has no bit-exact CPU row test) */
        const dim3 grid((unsigned)((N + N2 + kRowsPerWG - 1) / kRowsPerWG)), block(32 * kRowsPerWG);
        const bool af = A->dtype == RAD_F32, yf = Y->dtype == RAD_F32;
        const uint16_t* b = (const uint16_t*)B->data;
#define XE_GVP(AF, YF, MB_) XE_LAUNCH(q, (gemv_bf16_pair_kernel<AF, YF, MB_>), grid, block, 0, A->data, (long)a_ld, b, \
                                      (long)b_ld, Y->data, (long)y_ld, (int)N, (int)K, (int)M, p)
#define XE_GVP2(AF, YF) do { if (M == 1) XE_GVP(AF, YF, 1); else XE_GVP(AF, YF, kMaxM); } while (0)
        if (af && yf) XE_GVP2(true, true);
        else if (af) XE_GVP2(true, false);
        else if (yf) XE_GVP2(false, true);
        else XE_GVP2(false, false);
#undef XE_GVP2
#undef XE_GVP
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* The int8 lm_head (libr4d's logits_gemm_i8 stored row, r4d_fp8_layout.cpp): each row is K i8 codes,
 * then its K/128 bf16 scales, padded to w_ld bytes. y[m, n] = sum_g scale[n, g] * sum_{k in g}
 * x[m, k] * code[n, k], f32 out. Lane l reads 16-code chunks l, l + 32, ...; a chunk's partial sum
 * takes its group's scale before it joins the lane's sum. */
template <int MB>
void logits_i8_kernel(const uint16_t* __restrict__ x, long x_ld, const uint8_t* __restrict__ w, long w_ld,
                      float* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5);
    const bool live = n < N;
    const uint8_t* row = w + (size_t)(live ? n : 0) * w_ld;
    const uint16_t* srow = (const uint16_t*)(row + K);
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const int nchunk = K / 16;
    for (int c = lane; c < nchunk && live; c += 32) {
        const uint4 q = *(const uint4*)(row + (size_t)c * 16);
        const unsigned qv[4] = {q.x, q.y, q.z, q.w};
        float wf[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) wf[u] = (float)(int8_t)((qv[u >> 2] >> (8 * (u & 3))) & 0xffu);
        const float sc = ld_bf16(srow, c >> 3);
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            const uint16_t* xr = x + (size_t)m * x_ld + (size_t)c * 16;
            float s = 0.0f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s += ld_bf16(xr, e) * wf[e];
            acc[m] += s * sc;
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
        if (live && lane == 0 && m < M) y[(size_t)m * y_ld + n] = v;
    }
}

/* The rows form (owner note 151 C, run 123): R weight rows a sub-group. The 4-row form above spends its time on the
 * activations, not the weights -- per 16-code chunk a lane reads 16 bytes of codes but 4 x 32 bytes of bf16 rows, again
 * for every weight row -- and a lane's chunk indices are the same for every weight row. So a sub-group takes R weight
 * rows, loads each chunk's activation rows once (as packed bf16 in registers) and applies them to its R rows. Every
 * output keeps the 4-row form's arithmetic in the same order (the chunk's 16 products in e order, then the scale, the
 * chunks in the lane's order, the same butterfly), so the outputs are its bytes. */
template <int MB, int R>
void logits_i8_rows_kernel(const uint16_t* __restrict__ x, long x_ld, const uint8_t* __restrict__ w, long w_ld,
                           float* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n0 = ((int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5)) * R;
    float acc[R][MB];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    const int nchunk = K / 16;
    for (int c = lane; c < nchunk && n0 < N; c += 32) {
        uint4 xq[MB][2];
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            const uint4* xr = (const uint4*)(x + (size_t)(m < M ? m : 0) * x_ld + (size_t)c * 16);
            xq[m][0] = xr[0];
            xq[m][1] = xr[1];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r;
            if (n >= N) break;   /* uniform over the sub-group */
            const uint8_t* row = w + (size_t)n * w_ld;
            const uint16_t* srow = (const uint16_t*)(row + K);
            const uint4 q = *(const uint4*)(row + (size_t)c * 16);
            const unsigned qv[4] = {q.x, q.y, q.z, q.w};
            float wf[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) wf[u] = (float)(int8_t)((qv[u >> 2] >> (8 * (u & 3))) & 0xffu);
            const float sc = ld_bf16(srow, c >> 3);
#pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= M) break;
                const unsigned xv[8] = {xq[m][0].x, xq[m][0].y, xq[m][0].z, xq[m][0].w,
                                        xq[m][1].x, xq[m][1].y, xq[m][1].z, xq[m][1].w};
                float s = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const unsigned h = (e & 1) ? (xv[e >> 1] & 0xffff0000u) : (xv[e >> 1] << 16);
                    s += __uint_as_float(h) * wf[e];
                }
                acc[r][m] += s * sc;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            float v = acc[r][m];
            for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
            if (n < N && lane == 0 && m < M) y[(size_t)m * y_ld + n] = v;
        }
    }
}

/* The split rows form (run 219, owner note 207 item A2, RADIANCE_XE_LOGITS_R=12): the rows form held a chunk's whole 16
 * activation columns of MB rows (32 dwords) and each weight row's 16 decoded codes at once -- at SIMD32 past 128
 * registers (2 KB of spill a thread under IGC 2.42.6). Here the chunk goes in two halves of 8 columns: a half's
 * activation (16 dwords), then for each weight row its codes decoded as they are used. Every (row, activation row) sum
 * still runs over the chunk's 16 products in e order (the first half's 8, then the second's, in one running sum), then
 * takes the scale and joins the lane's sum where the rows form's does -- the same float operations in the same order,
 * so its bytes. */
template <int MB, int R>
void logits_i8_rows_split_kernel(const uint16_t* __restrict__ x, long x_ld, const uint8_t* __restrict__ w, long w_ld,
                                 float* __restrict__ y, long y_ld, int N, int K, int M) {
    const int lane = (int)threadIdx.x & 31;
    const int n0 = ((int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5)) * R;
    float acc[R][MB];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    const int nchunk = K / 16;
    for (int c = lane; c < nchunk && n0 < N; c += 32) {
        float s[R][MB];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < MB; ++m) s[r][m] = 0.0f;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            uint4 xq[MB];
#pragma unroll
            for (int m = 0; m < MB; ++m)
                xq[m] = ((const uint4*)(x + (size_t)(m < M ? m : 0) * x_ld + (size_t)c * 16))[hf];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int n = n0 + r;
                if (n >= N) break;   /* uniform over the sub-group */
                const uint8_t* row = w + (size_t)n * w_ld;
                const uint2 q = *(const uint2*)(row + (size_t)c * 16 + hf * 8);
                const unsigned qv[2] = {q.x, q.y};
#pragma unroll
                for (int m = 0; m < MB; ++m) {
                    if (m >= M) break;
                    const unsigned xv[4] = {xq[m].x, xq[m].y, xq[m].z, xq[m].w};
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float wf = (float)(int8_t)((qv[e >> 2] >> (8 * (e & 3))) & 0xffu);
                        const unsigned h = (e & 1) ? (xv[e >> 1] & 0xffff0000u) : (xv[e >> 1] << 16);
                        s[r][m] += __uint_as_float(h) * wf;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r;
            if (n >= N) break;
            const float sc = ld_bf16((const uint16_t*)(w + (size_t)n * w_ld + K), c >> 3);
#pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= M) break;
                acc[r][m] += s[r][m] * sc;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            float v = acc[r][m];
            for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
            if (n < N && lane == 0 && m < M) y[(size_t)m * y_ld + n] = v;
        }
    }
}

/* RADIANCE_XE_LOGITS_R: the rows form at more than one row (owner note 151 C, run 123) -- the default R=2 since owner
 * note 155 (bit-identical); 4 the wider form, 12 the split form above at R=2 (run 219, opt-in until timed), 0 the 4-row
 * form. Read at every call (one a step), so a test can compare the forms in one process. */
int logits_rows() {
    const char* e = std::getenv("RADIANCE_XE_LOGITS_R");
    const int v = e ? std::atoi(e) : 2;
    return v == 2 || v == 4 || v == 12 ? v : 0;
}

}  // namespace

/* logits_gemm over the int8 head: x (bf16), lm_head (the stored row above, declared [n_vocab,
 * n_embd] i8), logits (f32) -- libr4d's r4d_rad_logits_gemm_i8 operands and checks. */
extern "C" int xe_rad_logits_gemm_i8(const RadArgs* args, RadStream s) {
    const RadTensor* X = r4d_opt(args, 0);
    const RadTensor* W = r4d_opt(args, 1);
    const RadTensor* Y = r4d_opt(args, 2);
    if (!X || !W || !Y || !X->data || !W->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || W->dtype != RAD_I8 || Y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(W)) return R4D_NO(RAD_E_STRIDE);
    const long long V = r4d_geti_or(args, "n_vocab", r4d_dim(W, 0));
    const long long E = r4d_geti_or(args, "n_embd", 0);
    if (V <= 0 || E <= 0 || E % 128) return R4D_NO(RAD_E_SHAPE);
    if (W->rank != 2 || W->shape[0] != V || W->shape[1] != E) return R4D_NO(RAD_E_SHAPE);
    const long long w_ld = (E + 2 * (E / 128) + 15) & ~15LL;   /* r4d_fp8::lm_row_bytes */
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, E, &M);
    const long long y_ld = r4d_rows_pitch(Y, V, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (((uintptr_t)X->data | (uintptr_t)W->data) % 16 || x_ld % 8) return R4D_NO(RAD_E_ALIGN);
    try {
        sycl::queue* q = (sycl::queue*)s;
        const dim3 grid((unsigned)((V + kRowsPerWG - 1) / kRowsPerWG)), block(32 * kRowsPerWG);
        const int rr = logits_rows();
        for (long long m0 = 0; m0 < M; m0 += kMaxM) {
            const int mb = (int)(M - m0 < kMaxM ? M - m0 : kMaxM);
            const uint16_t* xm = (const uint16_t*)X->data + (size_t)m0 * x_ld;
            float* ym = (float*)Y->data + (size_t)m0 * y_ld;
            if (mb > 1 && rr > 0) {
                const int rw = rr == 12 ? 2 : rr;   /* 12: the split form at R=2 */
                const dim3 gr((unsigned)((V + kRowsPerWG * rw - 1) / (kRowsPerWG * rw)));
                if (rr == 12) XE_LAUNCH(q, (logits_i8_rows_split_kernel<kMaxM, 2>), gr, block, 0, xm, (long)x_ld, (const uint8_t*)W->data, (long)w_ld, ym, (long)y_ld, (int)V, (int)E, mb);
                else if (rr == 2) XE_LAUNCH(q, (logits_i8_rows_kernel<kMaxM, 2>), gr, block, 0, xm, (long)x_ld, (const uint8_t*)W->data, (long)w_ld, ym, (long)y_ld, (int)V, (int)E, mb);
                else XE_LAUNCH(q, (logits_i8_rows_kernel<kMaxM, 4>), gr, block, 0, xm, (long)x_ld, (const uint8_t*)W->data, (long)w_ld, ym, (long)y_ld, (int)V, (int)E, mb);
                continue;
            }
            if (mb == 1) XE_LAUNCH(q, (logits_i8_kernel<1>), grid, block, 0, xm, (long)x_ld, (const uint8_t*)W->data, (long)w_ld, ym, (long)y_ld, (int)V, (int)E, mb);
            else XE_LAUNCH(q, (logits_i8_kernel<kMaxM>), grid, block, 0, xm, (long)x_ld, (const uint8_t*)W->data, (long)w_ld, ym, (long)y_ld, (int)V, (int)E, mb);
        }
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* gemm_nt: a, b, y */
extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* args, RadStream s) {
    const RadTensor* B = r4d_opt(args, 1);
    if (!B) return R4D_NO(RAD_E_INVAL);
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long N = r4d_geti_or(args, "N", r4d_dim(B, (int)B->rank - 2));
    return gemv(r4d_opt(args, 0), B, r4d_opt(args, 2), N, K, s);
}

/* RADIANCE_XE_ROUTE_DUMP=<dir> (owner note 165 item 4, run 174): a measurement switch -- after the top-k of a
 * decode-sized call (M <= 16 rows) the router's rows go to <dir>/route-<queue>.bin: a 16-byte header ('RTD1',
 * M, n_expert, top_k as uint32), the bf16 logits (M x n_expert, de-pitched), the int32 ids (M x top_k) and
 * the bf16 weights (M x top_k); m3/route_union.py reads them (the distinct experts a step reads a layer and
 * the union under a routing bias). The copies wait for the queue, so the step's time is not a measurement;
 * the tapes must be off (RADIANCE_L0_TAPE=0): a replayed pass does not run this host code. One file a queue
 * (each rank's stream), so the two ranks' dumps are the same routing twice. */
static void route_dump(sycl::queue* q, const RadTensor* lg, const RadTensor* ids, const RadTensor* w,
                       long long E, long long top_k) {
    static const char* dir = [] { const char* e = std::getenv("RADIANCE_XE_ROUTE_DUMP"); return e && *e ? e : nullptr; }();
    if (!dir || !q || !lg || !ids || !w || !lg->data || !ids->data || !w->data || E <= 0 || top_k <= 0) return;
    long long M = 0;
    const long long ld = r4d_rows_pitch(lg, E, &M);
    if (ld < 0 || M <= 0 || M > 16) return;
    if (rad_tensor_numel(ids) < M * top_k || rad_tensor_numel(w) < M * top_k) return;
    std::vector<uint16_t> L((size_t)(M * E)), Wv((size_t)(M * top_k));
    std::vector<int32_t> I((size_t)(M * top_k));
    try {
        q->wait();
        for (long long m = 0; m < M; ++m)
            q->memcpy(L.data() + (size_t)m * E, (const uint16_t*)lg->data + (size_t)m * ld, (size_t)E * 2);
        q->memcpy(I.data(), ids->data, I.size() * 4);
        q->memcpy(Wv.data(), w->data, Wv.size() * 2);
        q->wait();
    } catch (const sycl::exception&) {
        return;
    }
    static std::mutex mu;
    static std::map<sycl::queue*, FILE*> files;
    std::lock_guard<std::mutex> g(mu);
    FILE*& f = files[q];
    if (!f) {
        char p[512];
        std::snprintf(p, sizeof p, "%s/route-%p.bin", dir, (void*)q);
        f = std::fopen(p, "ab");
        if (!f) return;
    }
    const uint32_t h[4] = { 0x31445452u, (uint32_t)M, (uint32_t)E, (uint32_t)top_k };
    std::fwrite(h, sizeof h[0], 4, f);
    std::fwrite(L.data(), 2, L.size(), f);
    std::fwrite(I.data(), 4, I.size(), f);
    std::fwrite(Wv.data(), 2, Wv.size(), f);
    std::fflush(f);
}

/* router_gemm_topk_scatter: a, b, gate_b?, logits, expert_ids, expert_w, sorted_tok, expert_offset,
 * expert_count, gate? -- the reference's own launches (libref: "the reference IS the launches"):
 * logits = a @ b^T, the shared expert's gate = a @ gate_b^T when both are present, then
 * router_topk_scatter (the ported libr4d kernel) over the logits and the five outputs after it. */
/* The GEMV part of the two fused router ops: logits = a @ b^T and the gate; `E` returned for the dump. */
static int router_gemm_part(const RadArgs* args, RadStream s, long long* E_out) {
    const RadTensor* A = r4d_opt(args, 0);
    const RadTensor* B = r4d_opt(args, 1);
    const RadTensor* GB = r4d_opt(args, 2);
    const RadTensor* G = args->n_t > 9 ? r4d_opt(args, 9) : nullptr;
    if (!A || !B) return R4D_NO(RAD_E_INVAL);
    if ((GB && GB->data) != (G && G->data)) return R4D_NO(RAD_E_INVAL);   /* together or not at all */
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long E = r4d_geti_or(args, "n_expert", r4d_dim(B, (int)B->rank - 2));
    *E_out = E;
    int rc;
    if (GB && GB->data && router_pair_on()) rc = gemv_pair(A, B, r4d_opt(args, 3), E, GB, G, 1, K, s);
    else rc = RAD_E_UNSUPPORTED;
    if (rc == RAD_E_UNSUPPORTED) {   /* the two launches (the pair form off, or a shape it does not take) */
        rc = gemv(A, B, r4d_opt(args, 3), E, K, s);
        if (rc != RAD_OK) return rc;
        if (GB && GB->data && (rc = gemv(A, GB, G, 1, K, s)) != RAD_OK) return rc;
    } else if (rc != RAD_OK) return rc;
    return RAD_OK;
}

extern "C" int xe_rad_router_gemm_topk_scatter(const RadArgs* args, RadStream s) {
    if (!args || args->n_t < 9) return R4D_NO(RAD_E_INVAL);
    long long E = 0;
    int rc = router_gemm_part(args, s, &E);
    if (rc != RAD_OK) return rc;
    RadArgs sub = *args;
    sub.t = args->t + 3;
    sub.n_t = 6;
    rc = r4d_rad_router_topk_scatter(&sub, s);
    if (rc == RAD_OK)
        route_dump((sycl::queue*)s, r4d_opt(args, 3), r4d_opt(args, 4), r4d_opt(args, 5), E,
                   r4d_geti_or(args, "top_k", 0));
    return rc;
}

/* router_gemm_topk_scatter_quant (run 273, the decode step's pair 2 on the path the step takes): the op above's ten
 * operands, then the block input `x`, its int8 codes `q` and scales `scale` (had_quant_act_i8g's) at 10 .. 12, with `n`
 * and `group`: the GEMV part as above, then the router's selection + sort and the quantiser's rows in ONE launch
 * (r4d_rad_router_topk_scatter_quant, ports/r4d_moe.cpp: the six router operands and the three of the quantiser,
 * contiguous as that adapter reads them). */
extern "C" int r4d_rad_router_topk_scatter_quant(const RadArgs* a, RadStream s);
/* The fused router op's tail (run 285; shared with router_gemm_topk_scatter_quant_sh in xe_gemv_i8.cpp, whose first 13
 * operands are this op's): the selection + sort and the quantiser as one launch over operands 3 .. 8 and 10 .. 12, then
 * the routing dump. */
extern "C" int xe_router_topk_quant_tail(const RadArgs* args, RadStream s, long long E) {
    RadTensor t9[9];
    for (int i = 0; i < 6; ++i) t9[i] = args->t[3 + i];
    for (int i = 0; i < 3; ++i) t9[6 + i] = args->t[10 + i];
    RadArgs sub = *args;
    sub.t = t9;
    sub.n_t = 9;
    const int rc = r4d_rad_router_topk_scatter_quant(&sub, s);
    if (rc == RAD_OK)
        route_dump((sycl::queue*)s, r4d_opt(args, 3), r4d_opt(args, 4), r4d_opt(args, 5), E,
                   r4d_geti_or(args, "top_k", 0));
    return rc;
}
extern "C" int xe_rad_router_gemm_topk_scatter_quant(const RadArgs* args, RadStream s) {
    if (!args || args->n_t < 13) return R4D_NO(RAD_E_INVAL);
    long long E = 0;
    const int rc = router_gemm_part(args, s, &E);
    if (rc != RAD_OK) return rc;
    return xe_router_topk_quant_tail(args, s, E);
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
