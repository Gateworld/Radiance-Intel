/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, run 267: owner lever #3).
 *
 * xe_moe_bf16_dpas.cpp -- the protected experts' bf16 prefill GEMM on the matrix units. ports/r4d_moe.cpp's
 * moe_gemm_bf16_kernel is the AMD wave32 WMMA kernel run through xe::wmma's scalar emulation (16 + 64 shuffles and
 * 128 scalar FMAs a lane per 16x16x16 tile): 5.6 ms a launch, 6.4 % of the prefill's device time (B187). This unit
 * does the same product on joint_matrix: gemm_bf16_dpas_kernel's tile (xe_gemv_bf16.cpp -- 8 sub-groups of 16, 64
 * rows x 64 columns, 64-K groups staged through shared memory, bf16 8x16x16 into fp32) over the expert kernel's row
 * mapping (xe_moe_i4i8.cpp's block map: one work-group per (expert, 64-row block, 64-column block); a tail block
 * zeroes the dropped slots past the live count). A bf16 plane has no group scales, so there is no fold: the matrix
 * unit accumulates across K.
 *
 * NOT bit-identical to the WMMA form: that one adds a tile's 16 products to the fp32 accumulator one k at a time in
 * ascending k; the matrix unit sums each 8x16x16 step in its own order and adds the step. Same bf16 operands (their
 * products are exact in fp32), a different association of the K sum: last-bit fp32 differences, an occasional one-ulp
 * bf16 output. So it is OPT-IN (RADIANCE_XE_MOE_BF16_DPAS=1) and the default follows note 149's KL-replay bar, not
 * the bit gate. With the switch unset nothing here runs and the dispatch is byte for byte the old one.
 *
 * Guards (the launcher returns 1 = not taken, and the caller runs the old kernel): K % 64, a call under 512 sorted
 * rows (the int8 form's cut-off), no or too small a scratch for the block map, a stride that is not a whole number of
 * 16-byte pieces (the caller already requires a_ld % 8 == 0, w_ld % 8 == 0, A and the planes 16-byte aligned).
 */
#include "xe_lanes.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int kSG = 16, kWaves = 8, kThreads = kSG * kWaves;   /* 128 */
constexpr int kRows = 8 * kWaves;                             /* 64 sorted rows a work-group */
constexpr int kCols = 64;                                     /* 4 tiles of 16 columns */
constexpr int kAP = 64 + 8;                                   /* the A stage's row pitch in bf16 (bank spread) */
constexpr uint32_t kTail = 0xffffffffu;                       /* a map entry of dropped slots */

/* four dwords at 4-byte alignment: a 16-byte piece of a bf16 row as one vector message */
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
typedef u32x4_t __attribute__((aligned(4))) u32x4a4;

inline uint16_t bf16_rne(float f) {   /* the WMMA form's f32_to_bf16: nearest-even, no NaN special case */
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* The block map: map[0] = the count, then (expert, first row) a 64-row block of every expert's run, then the
 * dropped slots past the live count (off[n_expert]) as (kTail, first row) blocks. One work-group of 256 threads;
 * the prefix over the experts' block counts is a group scan a tile of 256 experts. Integer sums: order-free. */
constexpr int kMapThreads = 256;
void moe_bf16_map_kernel(const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t cap,
                         uint32_t* __restrict__ map) {
    XE_SHARED(uint32_t, s_base, 1);
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid == 0) s_base[0] = 0;
    __syncthreads();
    for (uint32_t b = 0; b < n_expert; b += (uint32_t)kMapThreads) {
        const uint32_t e = b + tid;
        uint32_t nb = 0, first = 0;
        if (e < n_expert) {
            const uint32_t lo = (uint32_t)off[e], hi = (uint32_t)off[e + 1];
            nb = hi > lo ? (hi - lo + (uint32_t)kRows - 1) / (uint32_t)kRows : 0u;
            first = lo;
        }
        const uint32_t base = s_base[0];
        const uint32_t pref = sycl::exclusive_scan_over_group(xe::wg(), nb, sycl::plus<uint32_t>());
        const uint32_t total = sycl::reduce_over_group(xe::wg(), nb, sycl::plus<uint32_t>());
        for (uint32_t j = 0; j < nb; ++j) {
            const uint32_t idx = base + pref + j;
            if (idx < cap) { map[1 + 2 * idx] = e; map[2 + 2 * idx] = first + (uint32_t)kRows * j; }
        }
        __syncthreads();
        if (tid == 0) s_base[0] = base + total;
        __syncthreads();
    }
    if (tid == 0) {
        uint32_t cnt = s_base[0];
        const uint32_t live = (uint32_t)off[n_expert];
        for (uint32_t r = live; r < T; r += (uint32_t)kRows) {
            if (cnt < cap) { map[1 + 2 * cnt] = kTail; map[2 + 2 * cnt] = r; }
            ++cnt;
        }
        map[0] = cnt < cap ? cnt : cap;
    }
}

/* y[row, n] = sum_k A[token(row), k] * W_e[n, k] over the block's expert e, bf16 x bf16 -> fp32 -> bf16 (RNE). */
void moe_bf16_dpas_kernel(const uint16_t* __restrict__ A, const void* const* __restrict__ Wt, size_t w_s0,
                          const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                          uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                          uint32_t a_sorted, uint32_t a_ld, uint32_t w_ld, uint32_t y_ld,
                          const uint32_t* __restrict__ map) {
    XE_SHARED(uint16_t, sA, kRows * kAP);        /* 64 rows x 64 k bf16, pitch 72 */
    XE_SHARED(uint32_t, sB, 4 * 4 * 128);        /* [k step 4][n tile 4][packed 8 x 16 words]: 64 k x 64 n */
    XE_SHARED(int32_t, s_tok, kRows);            /* the block's token rows, -1 for a dead one */
    const int tid = (int)threadIdx.x, wave = tid / kSG, lane = tid % kSG;
    const uint32_t nb = blockIdx.x, n0 = nb * (uint32_t)kCols;
    if (n0 >= N) return;
    const uint32_t y = blockIdx.y;
    if (y >= map[0]) return;
    const uint32_t i = map[2 + 2 * y];
    const uint32_t e = map[1 + 2 * y];
    if (e == kTail) {   /* dropped slots i .. i + 64: zeros over this block's columns, as the WMMA form writes them */
        for (uint32_t r = i; r < i + (uint32_t)kRows && r < T; ++r)
            for (uint32_t n = n0 + (uint32_t)tid; n < n0 + (uint32_t)kCols && n < N; n += (uint32_t)kThreads)
                Y[(size_t)r * y_ld + n] = 0;
        return;
    }
    const uint32_t last = (uint32_t)off[e + 1];
    const uint32_t rn = (last - i) < (uint32_t)kRows ? (last - i) : (uint32_t)kRows;
    /* the block's rows: the token row of sorted row i + r, -1 past the run or for a dropped slot */
    for (int r = tid; r < kRows; r += kThreads) {
        int32_t t = -1;
        if ((uint32_t)r < rn) {
            const int32_t f = sorted[i + (uint32_t)r];
            if (f >= 0) t = a_sorted ? (int32_t)(i + (uint32_t)r) : f / (int32_t)top_k;
        }
        s_tok[r] = t;
    }
    const uint16_t* Wp = (w_s0 == 0) ? (const uint16_t*)Wt[e] : (const uint16_t*)((const uint8_t*)Wt + (size_t)e * w_s0);
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, bf16_t, jm::use::a, 8, 16, jm::layout::row_major> Am;
    jm::joint_matrix<sycl::sub_group, bf16_t, jm::use::b, 16, 16, jm::layout::ext_intel_packed> Bm;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> Cm[4];
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
#pragma unroll
    for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, Cm[t], 0.0f);
    __syncthreads();   /* s_tok */
    /* this thread's four 16-byte pieces of each stage: piece p = tid + 128 i is row p / 8, k 8 (p % 8) */
    const uint32_t G = K / 64u;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t k0 = g * 64u;
        __syncthreads();                                  /* the previous group's fragments are read */
        u32x4a4 va[4], vb[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {   /* the loads first, all in flight (V3b) */
            const int c = tid + p * kThreads, r = c >> 3, c8 = c & 7;
            const int32_t tok = s_tok[r];
            const u32x4a4 v = *(const u32x4a4*)(A + (size_t)(tok >= 0 ? tok : 0) * a_ld + k0 + 8u * (uint32_t)c8);
            va[p] = tok >= 0 ? v : (u32x4a4){0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int c = tid + p * kThreads, n = c >> 3, c8 = c & 7;
            const uint32_t nn = n0 + (uint32_t)n, nc = nn < N ? nn : N - 1u;
            const u32x4a4 v = *(const u32x4a4*)(Wp + (size_t)nc * w_ld + k0 + 8u * (uint32_t)c8);
            vb[p] = nn < N ? v : (u32x4a4){0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {   /* A: row-major, 16 bytes at (row, 8 c8) */
            const int c = tid + p * kThreads, r = c >> 3, c8 = c & 7;
            *(u32x4a4*)(sA + r * kAP + 8 * c8) = va[p];
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {   /* B: word (k2, n) = the k pair 2 k2 of column n; k2 = 4 c8 + j */
            const int c = tid + p * kThreads, n = c >> 3, c8 = c & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k2 = 4 * c8 + j, step = k2 >> 3, kw = k2 & 7;
                sB[((step * 4 + (n >> 4)) * 8 + kw) * 16 + (n & 15)] = vb[p][j];
            }
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, Am, lptr((bf16_t*)(sA + wave * 8 * kAP + step * 16)), (size_t)kAP);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bm, lptr((bf16_t*)(sB + (step * 4 + t) * 128)), 32);
                jm::joint_matrix_mad(sg, Cm[t], Am, Bm, Cm[t]);
            }
        }
    }
    /* the tiles out: lane = column, element q = row q of the sub-group's 8 (the accumulator's work-item data, as the
     * int8 expert kernel reads it); a dead row in the run writes zero, as the WMMA form does */
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t n = n0 + (uint32_t)(t * 16 + lane);
        if (n >= N) continue;
        auto wd = sycl::ext::oneapi::detail::get_wi_data(sg, Cm[t]);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint32_t qq = (uint32_t)(wave * 8 + q);
            if (qq < rn) Y[(size_t)(i + qq) * y_ld + n] = s_tok[qq] >= 0 ? bf16_rne((float)wd[q]) : (uint16_t)0;
        }
    }
}

}  // namespace

/* RADIANCE_XE_MOE_BF16_DPAS=1: the switch, read once */
extern "C" int xe_moe_bf16_dpas_on() {
    static const int v = [] { const char* s = std::getenv("RADIANCE_XE_MOE_BF16_DPAS"); return (s && *s == '1') ? 1 : 0; }();
    return v;
}

/* The launch for ports/r4d_moe.cpp's bf16 expert GEMM (its prefill forms' site). 0 = launched (the map in `scratch`,
 * then the GEMM over its entries); 1 = not taken, the caller runs its own kernel. T = the call's sorted rows. */
extern "C" int xe_moe_bf16_dpas_launch(void* stream, const uint16_t* A, const void* const* Wt, size_t w_s0,
                                       const int32_t* sorted, const int32_t* off, uint16_t* Y,
                                       uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert, uint32_t top_k,
                                       uint32_t a_sorted, uint32_t a_ld, uint32_t w_ld, uint32_t y_ld,
                                       void* scratch, size_t scratch_bytes) {
    if (!stream || !A || !Wt || !sorted || !off || !Y || !scratch) return 1;
    if (K == 0 || K % 64u != 0 || N == 0 || T < 512u || n_expert == 0 || top_k == 0) return 1;
    if (a_ld % 8u != 0 || w_ld % 8u != 0 || ((uintptr_t)A & 15u) != 0 || (w_s0 % 16u) != 0) return 1;
    const uint32_t cap = n_expert + (T + (uint32_t)kRows - 1) / (uint32_t)kRows + 1;
    const size_t bytes = (size_t)(1 + 2 * cap) * sizeof(uint32_t);
    if (scratch_bytes < bytes) return 1;
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[xe] moe_gemm_bf16: the matrix-unit form (RADIANCE_XE_MOE_BF16_DPAS=1; not bit-identical "
                             "to the WMMA form, note 149's bar): T %u N %u K %u experts %u\n", T, N, K, n_expert);
    }
    sycl::queue* qq = (sycl::queue*)stream;
    uint32_t* map = (uint32_t*)scratch;
    XE_LAUNCH(qq, moe_bf16_map_kernel, dim3(1), dim3(kMapThreads), 0, off, n_expert, T, cap, map);
    const dim3 grid((unsigned)((N + (uint32_t)kCols - 1) / (uint32_t)kCols), (unsigned)cap), block((unsigned)kThreads);
    XE_LAUNCH_SG(16, qq, moe_bf16_dpas_kernel, grid, block, 0, A, Wt, w_s0, sorted, off, Y, T, N, K, top_k, a_sorted,
                 a_ld, w_ld, y_ld, (const uint32_t*)map);
    return 0;
}
