/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * hip/hip_fp16.h for libxe: __half as clang's _Float16 (what HIP's is underneath) and the two
 * conversions libr4d's units call. */
#pragma once
#include "xe_lanes.h"

typedef _Float16 __half;
inline __half __float2half(float f) { return (__half)f; }
inline float __half2float(__half h) { return (float)h; }
