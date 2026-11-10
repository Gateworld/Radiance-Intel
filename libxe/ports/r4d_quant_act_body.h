/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 273: the decode step's adjacent-launch pair 2).
 *
 * r4d_quant_act_body.h -- the per-row body of r4d_quant_act_fp8.cpp's activation quantiser (quant_act_fp8,
 * had_quant_act_fp8, quant_act_i8g, had_quant_act_i8g), in a header so that a kernel in another unit can run it for
 * its own work-groups: r4d_moe.cpp's router_topk_scatter_quant_kernel quantises the block input beside the router's
 * selection and sort. The standalone kernels call it with blockIdx.x as the row. The same instructions wherever it is
 * inlined; the bytes are compared (test/router_quant_test.cpp, on the CPU device and on the card). The comments below
 * are the original's. */
#ifndef R4D_QUANT_ACT_BODY_H
#define R4D_QUANT_ACT_BODY_H

#include "r4d.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include "r4d_chain.h"

namespace r4d_qa {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;

// One block per row; a wave owns one 128-column group, a lane four consecutive columns (one dword
// store, 128 contiguous bytes per wave, two v_cvt_pk_fp8_f32).
//
// ROT applies r4d_fwht_wave<128> in registers between the load and the absmax; the mapping (four
// values per lane) is the one it expects, so no LDS or barrier is needed, and the absmax sees the
// rotated values. The sqrt(128) factor of the unnormalised transform is not removed here:
// <Hw, Ha> = 128 <w, a>, and r4d_layout.cpp folds that into the stored weight scale at convert.
//
// I8 is the same quantiser at int8 codes (quant_act_i8g, had_quant_act_i8g): scale amax/127,
// round-to-nearest-even into -127..127, same row-major layout and per-(row, 128) scales.
//
// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); no block returns
// early. quant_act_fp8_kernel and r4dchain_quant_act_fp8 (r4d_quant_act_fp8.cpp) are the plain and
// chained forms; `row` is blockIdx.x there, blockIdx.x - 1 in the router's paired launch.
template <bool ROT, bool I8 = false, bool CHAIN = false>
__device__ __forceinline__ void quant_act_fp8_body(
    uint32_t row, const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t nkb, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    if (CHAIN) r4d_chain_wait();
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += kWaves) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4];
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = c0 + (uint32_t)j;
            v[j] = c < K ? __uint_as_float((uint32_t)x[xrow + c] << 16) : 0.0f;
        }
        // The rotated launcher requires K % 128 == 0: a partial block would rotate real values
        // with padding zeros, and the weight has no matching block.
        if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v, (int)lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = r4d_wave_max(m);
        constexpr float kTop = I8 ? 127.0f : 448.0f;
        const float sc = m > 0.0f ? m * (1.0f / kTop) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
        int pk = 0;
        if constexpr (I8) {
            uint32_t b = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = (int)rintf(fminf(fmaxf(v[j] * inv, -127.0f), 127.0f));
                b |= ((uint32_t)c & 0xFFu) << (8 * j);
            }
            pk = (int)b;
        } else {
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                                 fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk,
                                                 false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                                 fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk,
                                                 true);
        }
        // A ragged last group is written zero-padded; the GEMM's bounds guard keeps the padding out
        // of the accumulators. The caller must own K rounded up to 4 bytes of `q`.
        if (c0 < K) *(uint32_t*)(q + qrow + c0) = (uint32_t)pk;
    }
    if (CHAIN) r4d_chain_done();
}

}  // namespace r4d_qa

#endif
