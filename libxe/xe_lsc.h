/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 211, owner note 206 item 2).
 *
 * xe_lsc.h -- a 16-byte global load with an LSC cache-control hint: xe_ld16<CC>(p) is *(const uint4*)p, with CC one
 * of IGC's LSC_LDCC values (0: the default policy, 2: L1 uncached / L3 cached, 5: L1 streaming / L3 uncached,
 * 6: L1 streaming / L3 cached ...). For data read once a step (the weights), so it need not displace the activations
 * and tables that are reused. The same bytes whatever the policy. On the card's AOT pass it is IGC's
 * __builtin_IB_lsc_load_global_uint4; the spir64 pass (the CPU device, XE_DP4A_PORTABLE: libxe's CMakeLists) and
 * CC 0 keep the plain load. Include after xe_lanes.h (uint4).
 */
#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__) && !defined(XE_DP4A_PORTABLE)
#define XE_LSC_BUILTIN 1
typedef unsigned int xe_lsc_v4u __attribute__((ext_vector_type(4)));
enum LSC_LDCC {
    LSC_LDCC_DEFAULT = 0, LSC_LDCC_L1UC_L3UC = 1, LSC_LDCC_L1UC_L3C = 2, LSC_LDCC_L1C_L3UC = 3, LSC_LDCC_L1C_L3C = 4,
    LSC_LDCC_L1S_L3UC = 5, LSC_LDCC_L1S_L3C = 6, LSC_LDCC_L1IAR_L3C = 7
};
extern "C" SYCL_EXTERNAL xe_lsc_v4u __builtin_IB_lsc_load_global_uint4(const __attribute__((opencl_global)) xe_lsc_v4u* base,
                                                                       int imm_elem_off, enum LSC_LDCC cache_opt);
#endif

template <int CC>
inline uint4 xe_ld16(const void* p) {
#ifdef XE_LSC_BUILTIN
    if constexpr (CC != 0) {
        const xe_lsc_v4u v = __builtin_IB_lsc_load_global_uint4((const __attribute__((opencl_global)) xe_lsc_v4u*)p, 0,
                                                                (enum LSC_LDCC)CC);
        return uint4{v.x, v.y, v.z, v.w};
    }
#endif
    return *(const uint4*)p;
}
