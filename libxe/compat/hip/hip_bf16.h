/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * hip/hip_bf16.h for libxe: the part of HIP's bf16 API libr4d's units use -- an opaque 16-bit type
 * and its conversions (round to nearest even, NaN kept quiet, as HIP's __float2bfloat16). */
#pragma once
#include "xe_lanes.h"

struct __hip_bfloat16 { unsigned short data; };
inline float __bfloat162float(__hip_bfloat16 b) { return sycl::bit_cast<float>((uint32_t)b.data << 16); }
inline __hip_bfloat16 __float2bfloat16(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __hip_bfloat16{(unsigned short)((u >> 16) | 0x40u)};
    return __hip_bfloat16{(unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16)};
}
