/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_lanes.h -- the lane shim: libr4d's HIP dialect (thread indices, AMD lane builtins, byte
 * permutes, E4M3 converts, fences, atomics) on SYCL sub-groups of exactly 32.
 *
 * libr4d's kernels assume a 32-lane wave in lockstep (r4d_moe.hip:61 kWave = 32: DPP row ops, the
 * two 16-lane halves of permlanex16, readlane by index, 32-bit ballot masks). Every kernel launched
 * through xe::launch below gets [[sycl::reqd_sub_group_size(32)]], and HIP's x dimension is SYCL's
 * fastest (dimension 2), so lane = threadIdx.x % 32 exactly as on the card.
 *
 * The semantics are those of tests/r4d_emu/emu/hip/hip_runtime.h (the emulator's CPU reference);
 * for the builtins the emulator does not model, the ISA's (gfx12 shader programming guide).
 * libxe/test/lane_test checks each one against them, on the CPU SYCL device and on the card.
 *
 * Cross-lane builtins must be reached by all 32 lanes (as in the emulator, which reports a
 * divergence otherwise): SYCL group algorithms are undefined under divergence.
 *
 * Not here, by design: the WMMA builtins (matrix kernels are rewritten, not shimmed),
 * kernarg_segment_ptr and the chain forms (AQL only), raw buffer loads, the gfx12 transpose load.
 * `__shared__` is not mapped either: a kernel declares its static LDS with XE_SHARED and its dynamic
 * LDS with XE_DYN_LDS.
 */
#pragma once
#include <sycl/sycl.hpp>

#include <cmath>     /* before the __expf macro below: glibc's math.h declares __expf */
#include <math.h>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>

#include "rad_plugin.h"   /* rad_f32_to_fp8e4m3 / rad_fp8e4m3_to_f32: the ABI's E4M3 */

namespace xe {
namespace sx  = sycl::ext::oneapi;
namespace sxe = sycl::ext::oneapi::experimental;

constexpr int kWave = 32;

/* ------------------------------------------------------------------ the work-item */
inline sycl::nd_item<3> item() { return sx::this_work_item::get_nd_item<3>(); }
inline sycl::sub_group  sg()   { return sx::this_work_item::get_sub_group(); }
inline sycl::group<3>   wg()   { return sx::this_work_item::get_work_group<3>(); }
inline int              lane() { return (int)sg().get_local_linear_id(); }

struct Idx3 { unsigned x, y, z; };
inline Idx3 thread_idx() { auto i = item(); return {(unsigned)i.get_local_id(2), (unsigned)i.get_local_id(1), (unsigned)i.get_local_id(0)}; }
inline Idx3 block_idx()  { auto i = item(); return {(unsigned)i.get_group(2), (unsigned)i.get_group(1), (unsigned)i.get_group(0)}; }
inline Idx3 block_dim()  { auto i = item(); return {(unsigned)i.get_local_range(2), (unsigned)i.get_local_range(1), (unsigned)i.get_local_range(0)}; }
inline Idx3 grid_dim()   { auto i = item(); return {(unsigned)i.get_group_range(2), (unsigned)i.get_group_range(1), (unsigned)i.get_group_range(0)}; }

/* ------------------------------------------------------------------ cross-lane */
/* DPP16 row_shl / row_shr / row_xmask with every row and bank enabled (all libr4d uses); a source
 * outside the 16-lane row reads 0 under bound_ctrl, else keeps `old`. Other controls read `old`. */
inline int update_dpp(int old, int src, int ctrl, int /*row_mask*/, int /*bank_mask*/, bool bc) {
    const int l = lane(), r = l & 15, k = ctrl & 15;
    int from;
    if (ctrl > 0x100 && ctrl < 0x110)       from = r + k < 16 ? l + k : -1;   /* row_shl */
    else if (ctrl > 0x110 && ctrl < 0x120)  from = r - k >= 0 ? l - k : -1;   /* row_shr */
    else if (ctrl >= 0x160 && ctrl < 0x170) from = (l & 16) | (r ^ k);         /* row_xmask */
    else                                    from = -1;
    const int v = sycl::select_from_group(sg(), src, from < 0 ? l : from);
    return from < 0 ? (bc ? 0 : old) : v;
}
/* v_permlanex16_b32: lane L reads, from the other row of sixteen, the lane L's selector nibble names */
inline int permlanex16(int /*old*/, int src, unsigned sel_lo, unsigned sel_hi, bool, bool) {
    const int l = lane(), r = l & 15;
    const unsigned sel = (r < 8 ? sel_lo >> (4 * r) : sel_hi >> (4 * (r - 8))) & 15u;
    return sycl::select_from_group(sg(), src, (16 - (l & 16)) | (int)sel);
}
inline int readlane(int v, int l)  { return sycl::select_from_group(sg(), v, l & 31); }
/* every lane active (as the emulator assumes), so the first lane is lane 0 */
inline int readfirstlane(int v)    { return sycl::group_broadcast(sg(), v, 0); }
inline int bpermute(int addr, int v) { return sycl::select_from_group(sg(), v, (addr >> 2) & 31); }
inline unsigned ballot(bool p) {
    uint32_t m = 0;
    sx::group_ballot(sg(), p).extract_bits(m);
    return m;
}

/* HIP's shuffles within groups of `width` lanes */
template <class T> inline T shfl_from(T v, int src) { return sycl::select_from_group(sg(), v, src); }
template <class T> inline T shfl_xor(T v, int mask, int width = 32) {
    const int l = lane();
    return shfl_from(v, (l & ~(width - 1)) | ((l ^ mask) & (width - 1)));
}
template <class T> inline T shfl(T v, int src, int width = 32) {
    const int l = lane();
    return shfl_from(v, (l & ~(width - 1)) | (src & (width - 1)));
}
template <class T> inline T shfl_down(T v, unsigned delta, int width = 32) {
    const int l = lane(), g = l & (width - 1);
    return shfl_from(v, g + (int)delta < width ? l + (int)delta : l);
}
template <class T> inline T shfl_up(T v, unsigned delta, int width = 32) {
    const int l = lane(), g = l & (width - 1);
    return shfl_from(v, g >= (int)delta ? l - (int)delta : l);
}

/* ------------------------------------------------------------------ byte permutes */
/* v_perm_b32: result byte i is the byte of {a, b} (b low) selector byte i names; 8-11 a sign byte
 * (bit 15 or 31 of b, then of a), 12 zero, 13 and up 0xff */
inline unsigned perm(unsigned a, unsigned b, unsigned sel) {
    const unsigned long long d = ((unsigned long long)a << 32) | b;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (8 * i)) & 0xffu;
        const unsigned v = s >= 13 ? 0xffu : s == 12 ? 0u
                         : s >= 8 ? (unsigned)((d >> (16 * (s - 8) + 15)) & 1u) * 0xffu
                                  : (unsigned)(d >> (8 * s)) & 0xffu;
        r |= v << (8 * i);
    }
    return r;
}
/* HIP's __byte_perm: result byte i is the byte of {y, x} (x low) the low three bits of nibble i name */
inline unsigned byte_perm(unsigned x, unsigned y, unsigned s) {
    const unsigned long long d = ((unsigned long long)y << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) r |= (unsigned)((d >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
}

/* ------------------------------------------------------------------ conversions */
typedef float    v2f  __attribute__((ext_vector_type(2)));
typedef _Float16 v2hf __attribute__((ext_vector_type(2)));
typedef short    v2s  __attribute__((ext_vector_type(2)));

/* v_cvt_pk_fp8_f32 (OCP E4M3, round to nearest even, saturating as rad_f32_to_fp8e4m3): a and b
 * into the low half of `old` (hi false) or its high half, keeping the other half */
inline int cvt_pk_fp8_f32(float a, float b, int old, bool hi) {
    const unsigned pk = (unsigned)rad_f32_to_fp8e4m3(a) | ((unsigned)rad_f32_to_fp8e4m3(b) << 8);
    const unsigned o = (unsigned)old;
    return (int)(hi ? ((o & 0xffffu) | (pk << 16)) : ((o & 0xffff0000u) | pk));
}
/* v_cvt_pk_f32_fp8: bytes 0,1 (hi false) or 2,3 of v to two floats; exact */
inline v2f cvt_pk_f32_fp8(int v, bool hi) {
    const unsigned u = (unsigned)v >> (hi ? 16 : 0);
    return v2f{rad_fp8e4m3_to_f32((uint8_t)(u & 0xffu)), rad_fp8e4m3_to_f32((uint8_t)((u >> 8) & 0xffu))};
}
/* v_cvt_f32_fp8: byte `sel` of v to a float; exact */
inline float cvt_f32_fp8(int v, int sel) {
    return rad_fp8e4m3_to_f32((uint8_t)(((unsigned)v >> (8 * (sel & 3))) & 0xffu));
}
/* f32 -> f16 rounding toward zero (v_cvt_pkrtz_f16_f32): no rounding up, overflow to the largest
 * finite, NaN kept quiet */
inline uint16_t f32_to_f16_rtz(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    const uint16_t s = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint16_t)(s | 0x7e00u | ((a >> 13) & 0x3ffu));
    if (a == 0x7f800000u) return (uint16_t)(s | 0x7c00u);
    const int e = (int)(a >> 23) - 127;
    if (e > 15) return (uint16_t)(s | 0x7bffu);
    if (e >= -14) return (uint16_t)(s | (uint16_t)((e + 15) << 10) | (uint16_t)((a >> 13) & 0x3ffu));
    if (e < -25) return s;
    const uint32_t m = (a & 0x7fffffu) | 0x800000u;   /* subnormal: value = m * 2^(e-23) */
    return (uint16_t)(s | (uint16_t)(m >> (-e - 14 + 13)));
}
inline v2hf cvt_pkrtz(float a, float b) {
    const v2s r{(short)f32_to_f16_rtz(a), (short)f32_to_f16_rtz(b)};
    return __builtin_bit_cast(v2hf, r);
}
/* v_dot2_f32_f16: c + a.x*b.x + a.y*b.y; f16 products are exact in f32 */
inline float fdot2(v2hf a, v2hf b, float c, bool /*clamp*/) {
    return c + (float)a.x * (float)b.x + (float)a.y * (float)b.y;
}

/* ------------------------------------------------------------------ fences, barriers, atomics */
constexpr bool streq(const char* a, const char* b) {
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}
constexpr sycl::memory_scope fence_scope(const char* s) {
    return streq(s, "") ? sycl::memory_scope::system
         : streq(s, "agent") ? sycl::memory_scope::device
         : streq(s, "workgroup") ? sycl::memory_scope::work_group
         : streq(s, "wavefront") ? sycl::memory_scope::sub_group
         : sycl::memory_scope::system;
}
constexpr sycl::memory_order mo(int o) {
    return o == __ATOMIC_RELAXED ? sycl::memory_order::relaxed
         : o == __ATOMIC_ACQUIRE ? sycl::memory_order::acquire
         : o == __ATOMIC_RELEASE ? sycl::memory_order::release
         : o == __ATOMIC_ACQ_REL ? sycl::memory_order::acq_rel
         : sycl::memory_order::seq_cst;
}
inline void fence(int order, sycl::memory_scope scope) { sycl::atomic_fence(mo(order), scope); }

/* HIP's scopes: singlethread 1, wavefront 2, workgroup 3, agent 4, system 5 */
template <class T, sycl::memory_scope S>
using aref = sycl::atomic_ref<T, sycl::memory_order::relaxed, S, sycl::access::address_space::generic_space>;
#define XE_SCOPE_SWITCH(scope, ...)                                                    \
    switch (scope) {                                                                   \
    case 1: case 2: { constexpr auto S_ = sycl::memory_scope::sub_group;  __VA_ARGS__; }       \
    case 3:         { constexpr auto S_ = sycl::memory_scope::work_group; __VA_ARGS__; }       \
    case 4:         { constexpr auto S_ = sycl::memory_scope::device;     __VA_ARGS__; }       \
    default:        { constexpr auto S_ = sycl::memory_scope::system;     __VA_ARGS__; }       \
    }
template <class T> inline T atomic_load(T* p, int order, int scope) {
    XE_SCOPE_SWITCH(scope, return aref<T, S_>(*p).load(mo(order)))
}
template <class T> inline void atomic_store(T* p, T v, int order, int scope) {
    XE_SCOPE_SWITCH(scope, aref<T, S_>(*p).store(v, mo(order)); return)
}
template <class T> inline T atomic_fetch_add(T* p, T v, int order, int scope) {
    XE_SCOPE_SWITCH(scope, return aref<T, S_>(*p).fetch_add(v, mo(order)))
}
template <class T> inline T atomic_add(T* p, T v) {
    return aref<T, sycl::memory_scope::device>(*p).fetch_add(v);
}
template <class T> inline T atomic_min(T* p, T v) {
    return aref<T, sycl::memory_scope::device>(*p).fetch_min(v);
}
template <class T> inline T atomic_max(T* p, T v) {
    return aref<T, sycl::memory_scope::device>(*p).fetch_max(v);
}
template <class T> inline T atomic_cas(T* p, T expect, T v) {
    aref<T, sycl::memory_scope::device>(*p).compare_exchange_strong(expect, v);
    return expect;
}

inline void syncthreads()        { sycl::group_barrier(wg()); }
inline int  syncthreads_or(int p) { return sycl::any_of_group(wg(), p != 0) ? 1 : 0; }
inline void wave_barrier()       { sycl::group_barrier(sg()); }

/* ------------------------------------------------------------------ launch */
struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};

/* The launch: every argument converted to its parameter's type (as r4d_emit does), HIP's
 * grid x block mapped to an nd_range<3> with x fastest, sub-group 32 (or SG), and `lds` bytes of
 * dynamic work-group memory for XE_DYN_LDS. */
template <auto K, int SG = 32, class... P, class... A>
sycl::event launch_as(void (*)(P...), sycl::queue& q, dim3 g, dim3 b, size_t lds, const A&... a) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    std::tuple<P...> v(static_cast<P>(a)...);
    const sycl::range<3> local(b.z, b.y, b.x);
    const sycl::range<3> global((size_t)g.z * b.z, (size_t)g.y * b.y, (size_t)g.x * b.x);
    sxe::launch_config cfg(sycl::nd_range<3>(global, local),
                           sxe::properties{sxe::work_group_scratch_size(lds)});
    /* the attribute, not a sub_group_size property: launch_config carries only the scratch size,
     * which clang warns about as if it might hold a sub-group size too */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
    return sxe::submit_with_event(q, [&](sycl::handler& h) {
        sxe::nd_launch(h, cfg, [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(SG)]] {
            std::apply([](auto... x) { K(x...); }, v);
        });
    });
#pragma clang diagnostic pop
}
inline void* dyn_lds() { return sxe::get_work_group_scratch_memory(); }

}  /* namespace xe */

/* ------------------------------------------------------------------ the HIP names */
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define amdgpu_waves_per_eu(lo, hi)
#define __align__(n) __attribute__((aligned(n)))

#define threadIdx (xe::thread_idx())
#define blockIdx  (xe::block_idx())
#define blockDim  (xe::block_dim())
#define gridDim   (xe::grid_dim())

#define __builtin_amdgcn_update_dpp(o, s, c, rm, bm, bc) xe::update_dpp((o), (s), (c), (rm), (bm), (bc))
#define __builtin_amdgcn_permlanex16(o, s, lo, hi, fi, bc) xe::permlanex16((o), (s), (lo), (hi), (fi), (bc))
#define __builtin_amdgcn_readlane(v, l) xe::readlane((v), (l))
#define __builtin_amdgcn_readfirstlane(v) xe::readfirstlane((v))
#define __builtin_amdgcn_ds_bpermute(a, v) xe::bpermute((a), (v))
#define __builtin_amdgcn_ballot_w32(p) xe::ballot((p))
inline unsigned long long __ballot(int p) { return xe::ballot(p != 0); }
#define __shfl(...)      xe::shfl(__VA_ARGS__)
#define __shfl_xor(...)  xe::shfl_xor(__VA_ARGS__)
#define __shfl_down(...) xe::shfl_down(__VA_ARGS__)
#define __shfl_up(...)   xe::shfl_up(__VA_ARGS__)
inline int __popc(unsigned v)            { return (int)sycl::popcount(v); }
inline int __popcll(unsigned long long v) { return (int)sycl::popcount(v); }
inline int __ffs(unsigned v)             { return v ? (int)sycl::ctz(v) + 1 : 0; }
inline int __ffsll(unsigned long long v)  { return v ? (int)sycl::ctz(v) + 1 : 0; }

#define __builtin_amdgcn_perm(a, b, s) xe::perm((a), (b), (s))
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) { return xe::byte_perm(x, y, s); }

#define __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi) xe::cvt_pk_fp8_f32((a), (b), (old), (hi))
#define __builtin_amdgcn_cvt_pk_f32_fp8(v, hi) xe::cvt_pk_f32_fp8((v), (hi))
#define __builtin_amdgcn_cvt_f32_fp8(v, sel) xe::cvt_f32_fp8((v), (sel))
#define __builtin_amdgcn_cvt_pkrtz(a, b) xe::cvt_pkrtz((a), (b))
#define __builtin_amdgcn_fdot2(a, b, c, cl) xe::fdot2((a), (b), (c), (cl))
#define __builtin_amdgcn_exp2f(x) sycl::exp2((float)(x))
#define __builtin_amdgcn_rcpf(x) (1.0f / (float)(x))
/* a macro: glibc declares __expf itself (math.h), with a different exception specification */
#define __expf(x) sycl::exp((float)(x))
inline float rsqrtf(float x) { return sycl::rsqrt(x); }

/* fences: (order, scope[, address space]); the address-space hint is dropped */
#define __builtin_amdgcn_fence(order, scope, ...) xe::fence((order), xe::fence_scope(scope))
#define __builtin_amdgcn_sched_barrier(m) ((void)0)
#define __builtin_amdgcn_sched_group_barrier(m, n, id) ((void)0)
#define __builtin_amdgcn_s_sleep(n) ((void)0)
#define __builtin_amdgcn_s_barrier() xe::syncthreads()
#define __builtin_amdgcn_s_barrier_signal(id) ((void)0)
#define __builtin_amdgcn_s_barrier_wait(id) xe::syncthreads()
#define __builtin_amdgcn_wave_barrier() xe::wave_barrier()
inline void __syncthreads() { xe::syncthreads(); }
inline int  __syncthreads_or(int p) { return xe::syncthreads_or(p); }
inline void __threadfence_block()  { sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group); }
inline void __threadfence()        { sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device); }
inline void __threadfence_system() { sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); }

#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#define __hip_atomic_load(p, order, scope) xe::atomic_load((p), (order), (scope))
#define __hip_atomic_store(p, v, order, scope) xe::atomic_store((p), (v), (order), (scope))
#define __hip_atomic_fetch_add(p, v, order, scope) xe::atomic_fetch_add((p), (v), (order), (scope))
#define atomicAdd(p, v) xe::atomic_add((p), (v))
#define atomicMin(p, v) xe::atomic_min((p), (v))
#define atomicMax(p, v) xe::atomic_max((p), (v))
#define atomicCAS(p, e, v) xe::atomic_cas((p), (e), (v))

inline float    __uint_as_float(unsigned u) { return sycl::bit_cast<float>(u); }
inline unsigned __float_as_uint(float f)    { return sycl::bit_cast<unsigned>(f); }
inline float    __int_as_float(int u)       { return sycl::bit_cast<float>(u); }
inline int      __float_as_int(float f)     { return sycl::bit_cast<int>(f); }

struct alignas(8)  uint2  { unsigned x, y; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(16) int4   { int x, y, z, w; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(4)  uchar4 { unsigned char x, y, z, w; };
inline uchar4 make_uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) { return uchar4{a, b, c, d}; }
inline uint4  make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
inline uint2  make_uint2(unsigned a, unsigned b) { return uint2{a, b}; }
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }
using xe::dim3;
typedef sycl::queue* hipStream_t;

/* Static LDS: `XE_SHARED(float, red, 32);` for `__shared__ float red[32];` -- one per work-group,
 * reached uniformly at kernel scope. Dynamic LDS: the launch's `lds` bytes. */
#define XE_SHARED(type, name, n) \
    auto& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<type[n]>(xe::wg())
#define XE_SHARED2(type, name, n, m) \
    auto& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<type[n][m]>(xe::wg())
#define XE_DYN_LDS(type, name) type* const name = (type*)xe::dyn_lds()

/* ------------------------------------------------------------------ the rest of HIP's surface a
 * ported unit names: errors, constants, and builtins that must compile but are never reached --
 * the WMMA, transpose-load and kernarg ones belong to kernels libxe rewrites or AQL chaining, whose
 * rows libxe does not register. Each poisons its result (NaN, or the sign bit), so a row that
 * reached one by mistake fails its libref check loudly instead of passing by luck. */
typedef int hipError_t;
enum { hipSuccess = 0 };
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipPeekAtLastError() { return hipSuccess; }
/* Declared, never defined: libr4d's r4d_emit() template names it, and R4D_EMIT above replaces every
 * use, so it is never instantiated (a link error says otherwise). */
hipError_t hipLaunchKernel(const void* f, xe::dim3 g, xe::dim3 b, void** args, size_t lds, hipStream_t s);
#define __constant__ const
namespace xe {
alignas(16) inline constexpr unsigned char kZeroKarg[1024] = {};
typedef short xe_v8s __attribute__((ext_vector_type(8)));
}
#define __builtin_amdgcn_kernarg_segment_ptr() ((const void*)xe::kZeroKarg)
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c) ((c) + __builtin_nanf(""))
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) ((c) + __builtin_nanf(""))
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) ((c) + __builtin_nanf(""))
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, cl) ((c) | (int)0x80000000)
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) ((xe::xe_v8s)(short)0x7fc0)
#define R4D_DYN_LDS(type, name) XE_DYN_LDS(type, name)

/* R4D_EMIT / R4D_KLAUNCH / hipLaunchKernelGGL on a queue (the RadStream) */
#define XE_LAUNCH(q, kernel, grid, block, lds, ...) \
    xe::launch_as<kernel>(kernel, *(q), (grid), (block), (size_t)(lds), __VA_ARGS__)
#define XE_LAUNCH_SG(sgsize, q, kernel, grid, block, lds, ...) \
    xe::launch_as<kernel, sgsize>(kernel, *(q), (grid), (block), (size_t)(lds), __VA_ARGS__)
#define R4D_EMIT(stream, kernel, grid, block, lds, ...) \
    ((void)XE_LAUNCH((hipStream_t)(stream), kernel, grid, block, lds, __VA_ARGS__), 0)
#define hipLaunchKernelGGL(kernel, grid, block, lds, stream, ...) \
    ((void)XE_LAUNCH((hipStream_t)(stream), kernel, grid, block, lds, __VA_ARGS__))

/* libr4d spells global loads with __attribute__((address_space(1))) casts (r4d_common.h ldg_u16,
 * load_tr_b128). Under SYCL a pointer is generic and such a cast is refused; the attribute is
 * dropped. Last in this header, after every SYCL header it includes. */
#define address_space(n)

/* glibc's sincosf has no device definition; SYCL's accurate sin and cos (not the native forms:
 * rope angles reach 262144, past a fast argument reduction). */
namespace xe {
inline void sincos_acc(float x, float* s, float* c) { *s = sycl::sin(x); *c = sycl::cos(x); }
}
#define sincosf(x, s, c) xe::sincos_acc((x), (s), (c))
