/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 207, owner note 203 item 1).
 *
 * xe_dp4a.h -- the sum of the four signed byte products of two packed words plus an accumulator (DP4A's arithmetic),
 * for libxe's int8 kernels. IGC 2.42.6 still turns the portable form (the bytes shifted out and multiplied, or
 * sycl::ext::oneapi::dot_acc's char union) into dp4a, but rebuilds each packed operand byte by byte first: ~170 extra
 * movs in gemv_i8a8<1, 4, 1>, 24.6 -> 26.4 us a launch. On the card's AOT pass IGC's own builtin takes the packed words
 * as they are; the CPU device's JIT, which has no IGC builtins, keeps the portable form. Integer arithmetic: the same
 * exact sum either way. The spir64 pass is told apart by XE_DP4A_PORTABLE, which libxe's CMakeLists gives to that
 * triple's frontend alone (-Xsycl-target-frontend=spir64): in the two-target build the CPU device's image also sees
 * __SYCL_TARGET_INTEL_GPU_BMG_G31__ (its JIT then finds no __builtin_IB_dp4a_ss), and a define given to
 * -Xsycl-target-frontend=intel_gpu_bmg_g31 reaches neither pass (run 207's macro probe and ISA dumps). A unit that
 * includes this header and is missing from that list fails on the CPU device (every launch refused).
 */
#pragma once

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__) && !defined(XE_DP4A_PORTABLE)
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
