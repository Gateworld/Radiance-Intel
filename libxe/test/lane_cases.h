/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * lane_cases.h -- test kernels for the lane shim, in libr4d's HIP dialect. Compiled twice: against
 * tests/r4d_emu's hip_runtime.h (lane_emu.cpp: the emulator's CPU semantics, one fiber a lane) and
 * against libxe/xe_lanes.h (lane_test.cpp: SYCL, sub-group 32). Both sides must agree bit for bit,
 * except case 8 (transcendentals), which is held to a few ulps.
 *
 * Every case: blocks of 64 (two waves), `in` one int a thread, `out` LANE_NOUT ints a thread. All
 * cross-lane builtins are reached by every lane. The includer defines LANE_SHARED(type, name, n).
 */
#pragma once

/* internal linkage: the cases are compiled into both lane_emu.cpp and lane_test.cpp */
#define LANE_KERNEL static __global__ void

#define LANE_NOUT 32
#define LANE_BLOCK 64
#define LANE_GRID 2

#define LANE_T() ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define LANE_O(j) out[LANE_T() * LANE_NOUT + (j)]

/* 0: DPP row_shl / row_shr / row_xmask, as libr4d's reductions issue them */
LANE_KERNEL lane_case_dpp(const int* in, int* out) {
    const int v = in[LANE_T()];
    LANE_O(0)  = __builtin_amdgcn_update_dpp(0, v, 0x101, 0xF, 0xF, true);
    LANE_O(1)  = __builtin_amdgcn_update_dpp(0, v, 0x102, 0xF, 0xF, true);
    LANE_O(2)  = __builtin_amdgcn_update_dpp(0, v, 0x104, 0xF, 0xF, true);
    LANE_O(3)  = __builtin_amdgcn_update_dpp(0, v, 0x108, 0xF, 0xF, true);
    LANE_O(4)  = __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, true);
    LANE_O(5)  = __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, true);
    LANE_O(6)  = __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, true);
    LANE_O(7)  = __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, true);
    LANE_O(8)  = __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, true);
    LANE_O(9)  = __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, true);
    LANE_O(10) = __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, true);
    LANE_O(11) = __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, true);
    LANE_O(12) = __builtin_amdgcn_update_dpp(0, v, 0x16f, 0xF, 0xF, true);
    LANE_O(13) = __builtin_amdgcn_update_dpp(-7, v, 0x101, 0xF, 0xF, false);
    LANE_O(14) = __builtin_amdgcn_update_dpp(-7, v, 0x113, 0xF, 0xF, false);
    /* a sum over each row of 16 by row_shr, the shape of r4d_common.h's row reductions */
    int s = v;
    s += __builtin_amdgcn_update_dpp(0, s, 0x111, 0xF, 0xF, true);
    s += __builtin_amdgcn_update_dpp(0, s, 0x112, 0xF, 0xF, true);
    s += __builtin_amdgcn_update_dpp(0, s, 0x114, 0xF, 0xF, true);
    s += __builtin_amdgcn_update_dpp(0, s, 0x118, 0xF, 0xF, true);
    LANE_O(15) = s;
}

/* 1: permlanex16 (the cross-row step), readlane, readfirstlane, ds_bpermute */
LANE_KERNEL lane_case_xlane(const int* in, int* out) {
    const int v = in[LANE_T()], l = (int)threadIdx.x & 31;
    LANE_O(0) = __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
    LANE_O(1) = __builtin_amdgcn_permlanex16(v, v, 0x01234567u, 0x89ABCDEFu, false, false);
    LANE_O(2) = __builtin_amdgcn_permlanex16(v, v, 0x13579BDFu, 0x02468ACEu, false, false);
    LANE_O(3) = __builtin_amdgcn_readlane(v, 0);
    LANE_O(4) = __builtin_amdgcn_readlane(v, 5);
    LANE_O(5) = __builtin_amdgcn_readlane(v, 31);
    LANE_O(6) = __builtin_amdgcn_readfirstlane(v);
    LANE_O(7) = __builtin_amdgcn_ds_bpermute((l ^ 1) << 2, v);
    LANE_O(8) = __builtin_amdgcn_ds_bpermute(((l * 7 + 3) & 31) << 2, v);
    LANE_O(9) = __builtin_amdgcn_ds_bpermute(v, v);   /* lane-varying, out-of-range addresses wrap */
}

/* 2: ballots and bit counts */
LANE_KERNEL lane_case_ballot(const int* in, int* out) {
    const int v = in[LANE_T()];
    LANE_O(0) = (int)__builtin_amdgcn_ballot_w32((v & 1) != 0);
    LANE_O(1) = (int)__builtin_amdgcn_ballot_w32(v % 3 == 0);
    const unsigned long long b = __ballot(v > 0);
    LANE_O(2) = (int)(unsigned)b;
    LANE_O(3) = (int)(unsigned)(b >> 32);
    LANE_O(4) = __popc((unsigned)v);
    LANE_O(5) = __popcll(((unsigned long long)(unsigned)v << 20) | (unsigned)v);
    LANE_O(6) = __ffsll((unsigned long long)(unsigned)(v & 0xfff000) << 7);
    LANE_O(7) = __ffs((unsigned)(v & 0x7f00));
    LANE_O(8) = __popcll(__ballot((v & 7) == 2));
}

/* 3: HIP's float shuffles at widths 32, 16 and 8 */
LANE_KERNEL lane_case_shfl(const int* in, int* out) {
    const int v = in[LANE_T()];
    const float f = (float)(v % 100000) * 0.37f;
    LANE_O(0)  = __float_as_int(__shfl_xor(f, 1));
    LANE_O(1)  = __float_as_int(__shfl_xor(f, 2));
    LANE_O(2)  = __float_as_int(__shfl_xor(f, 4));
    LANE_O(3)  = __float_as_int(__shfl_xor(f, 8));
    LANE_O(4)  = __float_as_int(__shfl_xor(f, 16));
    LANE_O(5)  = __float_as_int(__shfl_xor(f, 1, 16));
    LANE_O(6)  = __float_as_int(__shfl_xor(f, 4, 16));
    LANE_O(7)  = __float_as_int(__shfl_xor(f, 2, 8));
    LANE_O(8)  = __float_as_int(__shfl(f, 3));
    LANE_O(9)  = __float_as_int(__shfl(f, 7, 8));
    LANE_O(10) = __float_as_int(__shfl(f, v & 31));
    LANE_O(11) = __float_as_int(__shfl_down(f, 1));
    LANE_O(12) = __float_as_int(__shfl_down(f, 4));
    LANE_O(13) = __float_as_int(__shfl_down(f, 16));
    LANE_O(14) = __float_as_int(__shfl_down(f, 3, 8));
    LANE_O(15) = __float_as_int(__shfl_up(f, 1));
    LANE_O(16) = __float_as_int(__shfl_up(f, 4));
    LANE_O(17) = __float_as_int(__shfl_up(f, 2, 8));
    /* the xor-butterfly sum every libr4d reduction ends with */
    float s = f;
    for (int m = 16; m >= 1; m >>= 1) s += __shfl_xor(s, m);
    LANE_O(18) = __float_as_int(s);
}

/* 4: v_perm_b32 and __byte_perm, fixed and lane-varying selectors */
LANE_KERNEL lane_case_perm(const int* in, int* out) {
    const unsigned v = (unsigned)in[LANE_T()];
    const unsigned a = v * 2654435761u, b = ~a ^ (unsigned)LANE_T();
    const unsigned vs = (v & 0x0f0f0f0fu);   /* every byte 0..15: data, sign, zero and 0xff */
    LANE_O(0) = (int)__builtin_amdgcn_perm(a, b, 0x03020100u);
    LANE_O(1) = (int)__builtin_amdgcn_perm(a, b, 0x07060504u);
    LANE_O(2) = (int)__builtin_amdgcn_perm(a, b, 0x0c0d0e0fu);
    LANE_O(3) = (int)__builtin_amdgcn_perm(a, b, 0x08090a0bu);
    LANE_O(4) = (int)__builtin_amdgcn_perm(a, b, 0x00010203u);
    LANE_O(5) = (int)__builtin_amdgcn_perm(a, b, 0x05040706u);
    LANE_O(6) = (int)__builtin_amdgcn_perm(a, b, vs);
    LANE_O(7) = (int)__byte_perm(a, b, 0x3210u);
    LANE_O(8) = (int)__byte_perm(a, b, 0x7654u);
    LANE_O(9) = (int)__byte_perm(a, b, 0x0123u);
    LANE_O(10) = (int)__byte_perm(a, b, 0x5140u);
    LANE_O(11) = (int)__byte_perm(a, b, v);
}

/* 5: f32 -> E4M3 (normal, subnormal, ties, saturation, zero signs) into either half */
LANE_KERNEL lane_case_fp8enc(const int* in, int* out) {
    const unsigned v = (unsigned)in[LANE_T()];
    for (int j = 0; j < 14; ++j) {
        const unsigned m = (v * (unsigned)(2 * j + 1) * 977u) & 0x807fffffu;
        const unsigned e = 113u + (unsigned)((v >> (j & 7)) + (unsigned)j) % 24u;   /* 2^-14 .. 2^9 */
        float x = __uint_as_float(m | (e << 23));
        if (j == 12) x = (float)((int)(v % 1001) - 500);        /* integers, past +-448 */
        if (j == 13) x = __uint_as_float((m & 0x80000000u) | (((v >> 4) % 64u) << 18) | (117u << 23));
        const float y = -0.5f * x;
        LANE_O(j) = __builtin_amdgcn_cvt_pk_fp8_f32(x, y, 0x5a5a5a5a, (j & 1) != 0);
    }
    LANE_O(14) = __builtin_amdgcn_cvt_pk_fp8_f32(448.0f, -448.0f, 0, false);
    LANE_O(15) = __builtin_amdgcn_cvt_pk_fp8_f32(0.0f, -0.0f, 0, true);
    LANE_O(16) = __builtin_amdgcn_cvt_pk_fp8_f32(0.0009765625f, 0.0029296875f, 0, false);   /* ties at 2^-10, 3*2^-10 */
}

/* 6: E4M3 -> f32, every byte value across the threads */
LANE_KERNEL lane_case_fp8dec(const int* in, int* out) {
    const int v = in[LANE_T()];
    const int w = (int)((unsigned)LANE_T() * 0x01010101u + 0x00402080u);   /* bytes t, t+0x80, t+0x20, t+0x40 */
    for (int r = 0; r < 2; ++r) {
        const int x = r ? v : w;
        const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(x, false);
        const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(x, true);
        LANE_O(8 * r + 0) = __float_as_int(lo[0]);
        LANE_O(8 * r + 1) = __float_as_int(lo[1]);
        LANE_O(8 * r + 2) = __float_as_int(hi[0]);
        LANE_O(8 * r + 3) = __float_as_int(hi[1]);
        for (int s = 0; s < 4; ++s) LANE_O(8 * r + 4 + s) = __float_as_int(__builtin_amdgcn_cvt_f32_fp8(x, s));
    }
}

/* 7: f32 -> f16 toward zero (cvt_pkrtz) and the f16 dot (fdot2) */
LANE_KERNEL lane_case_f16(const int* in, int* out) {
    const unsigned v = (unsigned)in[LANE_T()];
    for (int j = 0; j < 8; ++j) {
        const unsigned m = (v * (unsigned)(2 * j + 3) * 7919u) & 0x807fffffu;
        const unsigned e = 100u + (v >> j) % 46u;                     /* 2^-27 .. 2^18: f16 subnormals, overflow */
        const float x = __uint_as_float(m | (e << 23)), y = -x * 0.75f;
        const auto h = __builtin_amdgcn_cvt_pkrtz(x, y);
        LANE_O(j) = __builtin_bit_cast(int, h);
        const auto g = __builtin_amdgcn_cvt_pkrtz(0.5f * y + 1.0f, x * 0.001f);
        LANE_O(8 + j) = __float_as_int(__builtin_amdgcn_fdot2(h, g, 0.25f * (float)j, false));
    }
}

/* 8: transcendentals (compared to a few ulps, not bitwise) */
LANE_KERNEL lane_case_math(const int* in, int* out) {
    const int v = in[LANE_T()];
    const float x = (float)(v % 20000) * 0.001f;   /* -20 .. 20 */
    LANE_O(0) = __float_as_int(__builtin_amdgcn_exp2f(x));
    LANE_O(1) = __float_as_int(__builtin_amdgcn_rcpf(x + 40.5f));
    LANE_O(2) = __float_as_int(__expf(x));
    LANE_O(3) = __float_as_int(rsqrtf(x * x + 0.01f));
}

/* 9: block barriers, LDS, __syncthreads_or, wave/split barriers, LDS atomics */
LANE_KERNEL lane_case_barrier(const int* in, int* out) {
    LANE_SHARED(int, sh, LANE_BLOCK);
    LANE_SHARED(int, acc, 1);
    const int t = (int)threadIdx.x, v = in[LANE_T()];
    if (t == 0) acc[0] = 0;
    sh[t] = v;
    __syncthreads();
    LANE_O(0) = sh[LANE_BLOCK - 1 - t] + sh[(t + 1) % LANE_BLOCK];
    LANE_O(1) = __syncthreads_or(v == in[blockIdx.x * blockDim.x + 17]);
    LANE_O(2) = __syncthreads_or(v == 0x7fffffff);
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_s_barrier();
    atomicAdd(&acc[0], v & 255);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    LANE_O(3) = acc[0];
    __syncthreads();
}

/* 10: HIP's scoped atomics and the fences / scheduling hints (which must compile and do nothing
 * observable beyond ordering) */
LANE_KERNEL lane_case_atomic(const int* in, int* out, int* counter) {
    const int v = in[LANE_T()];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
    __builtin_amdgcn_s_sleep(1);
    __hip_atomic_fetch_add(&counter[0], v & 1023, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_add(&counter[1], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    __threadfence();
    __threadfence_system();
    __hip_atomic_store(&LANE_O(0), v ^ 0x55, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    LANE_O(1) = __hip_atomic_load(&LANE_O(0), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

#define LANE_NCASES 11
