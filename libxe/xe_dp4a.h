/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 207, owner note 203 item 1).
 *
 * xe_dp4a.h -- the sum of the four signed byte products of two packed words plus an accumulator (DP4A's arithmetic),
 * for libxe's int8 kernels. IGC 2.42.6 still turns the portable form (the bytes shifted out and multiplied, or
 * sycl::ext::oneapi::dot_acc's char union) into dp4a, but rebuilds each packed operand byte by byte first: ~170 extra
 * movs in gemv_i8a8<1, 4, 1>, 24.6 -> 26.4 us a launch. On the card's AOT pass IGC's own builtin takes the packed words
 * as they are; the CPU device's JIT, which has no IGC builtins, keeps the portable form. Integer arithmetic: the same
 * exact sum either way. XE_DP4A_GPU comes from libxe's CMakeLists, given to the card target's frontend alone:
 * __SYCL_TARGET_INTEL_GPU_BMG_G31__ also reaches the spir64 pass, whose JIT then finds no __builtin_IB_dp4a_ss.
 */
#pragma once

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && defined(XE_DP4A_GPU)
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));
#define XE_DP4A_BUILTIN 1
#endif

inline int32_t xe_dp4a(int32_t a, int32_t b, int32_t acc) {
#ifdef XE_DP4A_BUILTIN
    return __builtin_IB_dp4a_ss(acc, a, b);
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
#endif
}
