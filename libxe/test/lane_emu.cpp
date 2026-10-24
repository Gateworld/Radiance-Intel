/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * lane_emu.cpp -- the lane cases on tests/r4d_emu: the reference side of lane_test.
 *
 * The emulator header is the specification for the builtins it models. For the ones it does not
 * (it runs only the units its tests need), the definitions below are written from the ISA (gfx12
 * shader programming guide, HIP's documentation), independently of xe_lanes.h, and lane_test reports
 * them as "ISA reference" rather than "emulator".
 */
#include "hip/hip_runtime.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstring>

/* ---- builtins the emulator does not model */
typedef float    lane_v2f  __attribute__((ext_vector_type(2)));
typedef _Float16 lane_v2hf __attribute__((ext_vector_type(2)));
static lane_v2f emu_cvt_pk_f32_fp8(int v, bool hi) {
    const unsigned u = (unsigned)v;
    const int s = hi ? 16 : 0;
    return lane_v2f{rad_fp8e4m3_to_f32((uint8_t)(u >> s)), rad_fp8e4m3_to_f32((uint8_t)(u >> (s + 8)))};
}
static float emu_cvt_f32_fp8(int v, int sel) { return rad_fp8e4m3_to_f32((uint8_t)((unsigned)v >> (8 * sel))); }
/* toward zero: the nearest-even conversion, stepped back one f16 ulp when it rounded away */
static uint16_t f16_rtz(float f) {
    const _Float16 h = (_Float16)f;
    uint16_t b;
    std::memcpy(&b, &h, 2);
    if (std::isnan(f)) return b;
    if (std::fabs((float)h) > std::fabs(f)) --b;   /* same sign: magnitude down by one */
    return b;
}
static lane_v2hf emu_cvt_pkrtz(float a, float b) {
    const uint16_t r[2] = {f16_rtz(a), f16_rtz(b)};
    lane_v2hf h;
    std::memcpy(&h, r, 4);
    return h;
}
static float emu_fdot2(lane_v2hf a, lane_v2hf b, float c, bool) {
    return c + (float)a[0] * (float)b[0] + (float)a[1] * (float)b[1];
}
#define __builtin_amdgcn_cvt_pk_f32_fp8(v, hi) emu_cvt_pk_f32_fp8((v), (hi))
#define __builtin_amdgcn_cvt_f32_fp8(v, s) emu_cvt_f32_fp8((v), (s))
#define __builtin_amdgcn_cvt_pkrtz(a, b) emu_cvt_pkrtz((a), (b))
#define __builtin_amdgcn_fdot2(a, b, c, cl) emu_fdot2((a), (b), (c), (cl))
#define __builtin_amdgcn_exp2f(x) std::exp2((float)(x))
#define __builtin_amdgcn_rcpf(x) (1.0f / (float)(x))
/* HIP's __shfl_up: lane L - delta within its group of `width`, its own value below the group */
inline float __shfl_up(float v, unsigned delta, int width = 32) {
    const int lane = (int)threadIdx.x & 31, g = lane & (width - 1);
    return emu_shfl_from(v, g >= (int)delta ? lane - (int)delta : lane);
}
inline int __popc(unsigned v) { return __builtin_popcount(v); }
inline int __ffs(unsigned v) { return __builtin_ffs((int)v); }
inline int __ffsll(unsigned long long v) { return __builtin_ffsll((long long)v); }
/* the wave barrier: every wave of the block is in lockstep in the emulator only at cross-lane ops */
#define __builtin_amdgcn_wave_barrier() ((void)0)
#define __builtin_amdgcn_s_barrier_signal(id) ((void)0)
#define __builtin_amdgcn_s_barrier_wait(id) __syncthreads()
#define __builtin_amdgcn_sched_group_barrier(m, n, id) ((void)0)
#define __builtin_amdgcn_s_sleep(n) ((void)0)
inline void __threadfence_system() {}
#undef __HIP_MEMORY_SCOPE_AGENT
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#define __hip_atomic_load(p, order, scope) (*(p))
#define __hip_atomic_store(p, v, order, scope) ((void)(*(p) = (v)))

#define LANE_SHARED(type, name, n) __shared__ type name[n]
#include "lane_cases.h"

extern "C" void lane_emu_run(int c, const int* in, int* out, int* counter) {
    const dim3 g(LANE_GRID), b(LANE_BLOCK);
    switch (c) {
    case 0:  hipLaunchKernelGGL(lane_case_dpp, g, b, 0, nullptr, in, out); break;
    case 1:  hipLaunchKernelGGL(lane_case_xlane, g, b, 0, nullptr, in, out); break;
    case 2:  hipLaunchKernelGGL(lane_case_ballot, g, b, 0, nullptr, in, out); break;
    case 3:  hipLaunchKernelGGL(lane_case_shfl, g, b, 0, nullptr, in, out); break;
    case 4:  hipLaunchKernelGGL(lane_case_perm, g, b, 0, nullptr, in, out); break;
    case 5:  hipLaunchKernelGGL(lane_case_fp8enc, g, b, 0, nullptr, in, out); break;
    case 6:  hipLaunchKernelGGL(lane_case_fp8dec, g, b, 0, nullptr, in, out); break;
    case 7:  hipLaunchKernelGGL(lane_case_f16, g, b, 0, nullptr, in, out); break;
    case 8:  hipLaunchKernelGGL(lane_case_math, g, b, 0, nullptr, in, out); break;
    case 9:  hipLaunchKernelGGL(lane_case_barrier, g, b, 0, nullptr, in, out); break;
    case 10: hipLaunchKernelGGL(lane_case_atomic, g, b, 0, nullptr, in, out, counter); break;
    case 11: hipLaunchKernelGGL(lane_case_wmma, g, b, 0, nullptr, in, out); break;
    }
}
