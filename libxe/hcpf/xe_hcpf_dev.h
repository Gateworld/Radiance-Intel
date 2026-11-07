// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Deadcode and the radiance contributors
// Part of the Intel port of radiance (milestone 3, run 261; owner note 209, the supervisor's fix approved 21:20 11-07).
//
// The device helpers and the hcpf:: constants that r4d_hc_bf16.cpp and its prefill module (xe_hcpf.cpp) both use:
// one text for both units (the definitions r4d_hc_bf16.cpp had, unchanged). Included INSIDE each unit's own namespace
// (r4d_hc_bf16.cpp's anonymous one, xe_hcpf.cpp's xe_hcpf_mod), after <hip/hip_runtime.h>, r4d_common.h,
// r4d_gdn_wmma.h and <sycl/ext/oneapi/matrix/matrix.hpp>; no include guard on purpose (one inclusion a unit).

__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float siluf_(float v)    { return v * sigmoidf_(v); }

__device__ __forceinline__ float hc_h2f(uint32_t b) {
    return (float)sycl::bit_cast<sycl::half>((uint16_t)(b & 0xffffu));
}

/* Intel port, run 98 (owner notes 66 / 99, RADIANCE_XE_HC_V2=5): four E4M3 codes as their values times 2^-8, by
 * the hardware's FP16 conversion. A code's low seven bits shifted left by seven are an FP16 bit pattern whose value
 * is the code's times 2^-8 -- normals and subnormals alike, since FP16's exponent bias exceeds E4M3's by 8 and its
 * subnormals cover E4M3's -- and the sign goes to bit 15. About four operations a code against e4m3_to_f32's ~13
 * (xe_lanes.h emulates v_cvt_pk_f32_fp8). The caller multiplies the group scale by 2^8 instead: every product and
 * partial sum is then the original times an exact power of two, so the scaled dot gives the same bits. The NaN code
 * (S1111111) reads as +-480 here, not NaN; quantised weights never hold it. */
__device__ __forceinline__ void e4m3x4_s8(uint32_t w, float* f) {
    const uint32_t a = w & 0x00ff00ffu, b = (w >> 8) & 0x00ff00ffu;     /* bytes 0 / 2 and 1 / 3, one a half */
    const uint32_t ha = ((a & 0x007f007fu) << 7) | ((a & 0x00800080u) << 8);
    const uint32_t hb = ((b & 0x007f007fu) << 7) | ((b & 0x00800080u) << 8);
    f[0] = hc_h2f(ha); f[1] = hc_h2f(hb); f[2] = hc_h2f(ha >> 16); f[3] = hc_h2f(hb >> 16);
}
constexpr float kS8Scale = 256.0f;   /* the factor the S8 form leaves to the group scale */

/* hc_read on the matrix units (owner note 102): the tile constants the prefill kernels (xe_hcpf.cpp) and the
 * verify-row forms (r4d_hc_bf16.cpp's hc_sr_*) share */
namespace hcpf {
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int kSG = 16, kWv = 8, kThr = kSG * kWv;   /* 128 */
constexpr int kRT = 8 * kWv;                         /* 64 rows a work-group */
constexpr int kNT = 64;                              /* 64 columns: four 16-column tiles */
constexpr int kAP = 64 + 8;                          /* the A stage's row pitch in bf16 */
}  // namespace hcpf
constexpr long kPfMinRows = hcpf::kRT;
