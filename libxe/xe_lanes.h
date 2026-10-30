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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>   /* grf_size<256>: launch_as_grf256 */

#include <cmath>     /* before the __expf macro below: glibc's math.h declares __expf */
#include <math.h>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <algorithm>   /* the launch profiler (RADIANCE_XE_LAUNCH_PROFILE): its table */
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "rad_plugin.h"   /* rad_f32_to_fp8e4m3 / rad_fp8e4m3_to_f32: the ABI's E4M3 */
#include "rad_l0_rec.h"   /* the native recorder (owner note 28): launch_as records into a command list */
#include <new>

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
/* __fp16, as HIP's half2 is and libr4d's r4d_dt16.h declares it (the vit attention port mixes the
 * two); the conversions above and fdot2 go through float either way */
typedef __fp16   v2hf __attribute__((ext_vector_type(2)));
typedef short    v2s  __attribute__((ext_vector_type(2)));

/* v_cvt_pk_fp8_f32 (OCP E4M3, round to nearest even, saturating as rad_f32_to_fp8e4m3): a and b
 * into the low half of `old` (hi false) or its high half, keeping the other half */
inline int cvt_pk_fp8_f32(float a, float b, int old, bool hi) {
    const unsigned pk = (unsigned)rad_f32_to_fp8e4m3(a) | ((unsigned)rad_f32_to_fp8e4m3(b) << 8);
    const unsigned o = (unsigned)old;
    return (int)(hi ? ((o & 0xffffu) | (pk << 16)) : ((o & 0xffff0000u) | pk));
}
/* One OCP E4M3 code (the low byte of b) to f32, branch-free and bit-identical to the ABI's
 * rad_fp8e4m3_to_f32 on all 256 codes (Intel port, milestone 3, owner note 52 item 2: the
 * HyperConnection mixes decode every E4M3 code through this, and the ABI's form branches and loops
 * on a subnormal code -- Xe2 has no FP8 conversion instruction). A normal code is the f32 with
 * exponent e + 120 and the three mantissa bits on top; a subnormal one (e = 0) is the same pattern
 * read at exponent 1 less 2^-6, which is exact (m * 2^-9) and never makes an f32 denormal, so a
 * flush-to-zero mode cannot touch it; zero comes out signed as the ABI's; S1111111 is the NaN. */
inline float e4m3_to_f32(unsigned b) {
    const unsigned m = b & 0x7u, e = (b >> 3) & 0xfu;
    const float nrm = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));
    const float sub = sycl::bit_cast<float>((121u << 23) | (m << 20)) - 0.015625f;
    unsigned u = sycl::bit_cast<unsigned>(e ? nrm : sub) | ((b & 0x80u) << 24);
    if ((b & 0x7fu) == 0x7fu) u = ((b & 0x80u) << 24) | 0x7fc00000u;
    return sycl::bit_cast<float>(u);
}
/* Intel port, run 98 (owner note 66): two E4M3 bytes (bits 0-7 and 16-23 of x) to floats through the hardware's
 * FP16 conversion. A byte's low seven bits shifted left by seven are an FP16 pattern of its value times 2^-8 --
 * normals and subnormals alike, as FP16's exponent bias exceeds E4M3's by 8 and its subnormals cover E4M3's -- and
 * the sign goes to bit 15; times 2^8 is then exact. About half e4m3_to_f32's operations. S1111111 keeps its
 * signed quiet NaN, so every one of the 256 codes gives e4m3_to_f32's bits (lane_cases.h case 6). */
inline void e4m3x2_f16(unsigned x, float& a, float& b) {
    const unsigned h = ((x & 0x007f007fu) << 7) | ((x & 0x00800080u) << 8);
    a = (float)sycl::bit_cast<sycl::half>((uint16_t)(h & 0xffffu)) * 256.0f;
    b = (float)sycl::bit_cast<sycl::half>((uint16_t)(h >> 16)) * 256.0f;
    if ((x & 0x7fu) == 0x7fu) a = sycl::bit_cast<float>(((x & 0x80u) << 24) | 0x7fc00000u);
    if ((x & 0x7f0000u) == 0x7f0000u) b = sycl::bit_cast<float>(((x & 0x800000u) << 8) | 0x7fc00000u);
}
/* v_cvt_pk_f32_fp8: bytes 0,1 (hi false) or 2,3 of v to two floats; exact */
inline v2f cvt_pk_f32_fp8(int v, bool hi) {
    const unsigned u = (unsigned)v >> (hi ? 16 : 0);
    float a, b;
    e4m3x2_f16((u & 0xffu) | ((u & 0xff00u) << 8), a, b);
    return v2f{a, b};
}
/* v_cvt_f32_fp8: byte `sel` of v to a float; exact */
inline float cvt_f32_fp8(int v, int sel) {
    float a, b;
    e4m3x2_f16(((unsigned)v >> (8 * (sel & 3))) & 0xffu, a, b);
    return a;
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
/* A load may name a const object (libr4d's flag reads); atomic_ref wants a mutable one. */
template <class T> inline std::remove_const_t<T> atomic_load(T* p, int order, int scope) {
    using U = std::remove_const_t<T>;
    U* q = const_cast<U*>(p);
    XE_SCOPE_SWITCH(scope, return aref<U, S_>(*q).load(mo(order)))
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

/* RADIANCE_XE_LAUNCH_PROFILE=1 (Intel port, milestone 3, run 57): every plain (not recorded) launch keeps its
 * event in a per-thread ring; the entries that have completed are drained at the next launch and their device
 * times -- the queue's own profiling, which the backend enables under RADIANCE_L0_QUEUE_PROFILING=1 -- accumulate
 * per kernel (the launch lambda's type name, which carries the kernel function's); the table goes to stderr every
 * 65,536 launches and at exit. The tape profiler (RADIANCE_L0_NTAPE_PROFILE) covers the replayed passes; this
 * covers the live ones, which is what a prefill is. Off, it costs one branch a launch. */
inline bool launch_profile_on() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_XE_LAUNCH_PROFILE"); return e && *e == '1'; }();
    return v;
}
struct LaunchProfAcc { double ns = 0.0; uint64_t count = 0; };
inline std::mutex& lp_mutex() { static std::mutex m; return m; }
inline std::unordered_map<std::string, LaunchProfAcc>& lp_acc() {
    static auto* m = new std::unordered_map<std::string, LaunchProfAcc>;   /* process lifetime: threads drain at exit */
    return *m;
}
inline std::atomic<uint64_t>& lp_count() { static std::atomic<uint64_t> n{0}; return n; }
/* The table: device ms per kernel, launches, us a launch; the kernel's name after launch_as's own prefix. */
inline void lp_report(const char* when) {
    std::lock_guard<std::mutex> lk(lp_mutex());
    std::vector<std::pair<std::string, LaunchProfAcc>> rows(lp_acc().begin(), lp_acc().end());
    std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.second.ns > y.second.ns; });
    double tot = 0.0;
    uint64_t n = 0;
    for (const auto& r : rows) { tot += r.second.ns; n += r.second.count; }
    std::fprintf(stderr, "xe launch profile (%s): %llu launches timed, %zu kernels, %.1f ms of device time in all; the top 40:\n",
                 when, (unsigned long long)n, rows.size(), tot * 1e-6);
    for (size_t i = 0; i < rows.size() && i < 40; ++i) {
        std::string nm = rows[i].first;
        const size_t z = nm.find("_Z");
        if (z != std::string::npos && z + 2 < nm.size() && nm.find("_Z", z + 2) != std::string::npos) nm = nm.substr(nm.find("_Z", z + 2));
        if (nm.size() > 110) nm.resize(110);
        const LaunchProfAcc& a = rows[i].second;
        std::fprintf(stderr, "  %6.1f%%  %9.1f ms  %8llu x  %8.1f us  %s\n", 100.0 * a.ns / (tot > 0 ? tot : 1.0), a.ns * 1e-6,
                     (unsigned long long)a.count, a.count ? a.ns * 1e-3 / (double)a.count : 0.0, nm.c_str());
    }
    std::fflush(stderr);
}
struct LaunchProfRing {
    struct Entry { sycl::event ev; const char* name; };
    std::deque<Entry> q;
    ~LaunchProfRing() { drain(true); }
    void drain(bool all) {
        while (!q.empty()) {
            Entry& e = q.front();
            try {
                if (!all && e.ev.get_info<sycl::info::event::command_execution_status>() !=
                                sycl::info::event_command_status::complete)
                    break;
                if (all) e.ev.wait();
                const uint64_t s = e.ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                const uint64_t t = e.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                std::lock_guard<std::mutex> lk(lp_mutex());
                LaunchProfAcc& a = lp_acc()[e.name];
                a.ns += (double)(t >= s ? t - s : 0);
                a.count += 1;
            } catch (const sycl::exception&) {   /* a queue without profiling: nothing to count */
            }
            q.pop_front();
        }
    }
    void push(sycl::event ev, const char* name) {
        drain(false);
        q.push_back(Entry{std::move(ev), name});
        if (q.size() > 4096) { q.front().ev.wait(); drain(false); }
        if ((lp_count().fetch_add(1, std::memory_order_relaxed) + 1) % 65536 == 0) lp_report("running");
    }
};
inline LaunchProfRing& lp_ring() {
    static thread_local LaunchProfRing r;
    static std::once_flag once;
    std::call_once(once, [] { std::atexit([] { lp_report("exit"); }); });
    return r;
}

/* The launch: every argument converted to its parameter's type (as r4d_emit does), HIP's
 * grid x block mapped to an nd_range<3> with x fastest, sub-group 32 (or SG), and `lds` bytes of
 * dynamic work-group memory for XE_DYN_LDS. */
/* The native recording (abi/rad_l0_rec.h, owner note 28): while the calling thread records the
 * queue into a Level Zero command list, the launch is appended there instead of submitted -- the
 * kernel's sycl::kernel (once per device), the shape, the tuple as its one by-value parameter and
 * the dynamic work-group memory. The recorder's refusal (-1) falls through to the plain submit. */
template <class KN>
inline bool launch_native(sycl::queue& q, dim3 g, dim3 b, size_t lds, const void* arg0, size_t arg0_size) {
    if (!rad_l0k_rec_device || !rad_l0k_rec_launch) return false;   /* weak: no librad_l0k in this process */
    const int rd = rad_l0k_rec_device(&q);
    if (rd < 0) return false;
    const RadL0RecKernel* k = rad_l0_rec::kernel_of<KN>(q, rd);
    if (!k) return false;
    const uint32_t grid[3] = {g.x, g.y, g.z}, block[3] = {b.x, b.y, b.z};
    return rad_l0k_rec_launch(&q, k, grid, block, arg0, arg0_size, lds) == 0;
}

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
    auto kfn = [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(SG)]] {
        std::apply([](auto... x) { K(x...); }, v);
    };
    if (rad_l0k_rec_device && rad_l0k_rec_device(&q) >= 0) {
        /* the native argument in zeroed storage: a tuple's padding bytes are otherwise whatever the
         * stack held, and the tape audit compares the argument bytes (M2: a false difference at a
         * padding byte of hc_up_mix_kernel's tuple) */
        alignas(std::tuple<P...>) unsigned char vz[sizeof v] = {};
        new (vz) std::tuple<P...>(v);
        if (launch_native<decltype(kfn)>(q, g, b, lds, vz, sizeof v)) return sycl::event{};
    }
    sycl::event ev = sxe::submit_with_event(q, [&](sycl::handler& h) { sxe::nd_launch(h, cfg, kfn); });
    if (launch_profile_on()) lp_ring().push(ev, typeid(kfn).name());
    return ev;
#pragma clang diagnostic pop
}
inline void* dyn_lds() { return sxe::get_work_group_scratch_memory(); }

/* The same launch with the large register file (Intel's grf_size kernel property, 256 GRF a thread
 * instead of 128: half the threads a core, twice the registers a lane). For a ported kernel whose
 * per-lane state fits AMD's 256 VGPRs but not Xe's default file -- libr4d's GDN recurrent update
 * holds 64 state and 64 key floats a lane (207 VGPRs on gfx12, "no spill") and ran 20x slower than
 * its bandwidth on the B70 (milestone 2 P1b: 303 us a layer). No dynamic work-group memory here:
 * the property set is the kernel's, and such kernels declare their LDS statically (XE_SHARED). */
template <auto K, int SG = 32, class... P, class... A>
sycl::event launch_as_grf256(void (*)(P...), sycl::queue& q, dim3 g, dim3 b, const A&... a) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    /* The property is Intel-GPU-only: the OpenCL CPU device throws at submit (the oracle run of
     * milestone 2 run 2 aborted on it), so elsewhere this is the plain launch. */
    static const bool gpu = q.get_device().is_gpu();
    if (!gpu) return launch_as<K, SG>((void (*)(P...))nullptr, q, g, b, 0, a...);
    std::tuple<P...> v(static_cast<P>(a)...);
    const sycl::range<3> local(b.z, b.y, b.x);
    const sycl::range<3> global((size_t)g.z * b.z, (size_t)g.y * b.y, (size_t)g.x * b.x);
    auto kfn = [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(SG)]] {
        std::apply([](auto... x) { K(x...); }, v);
    };
    if (rad_l0k_rec_device && rad_l0k_rec_device(&q) >= 0) {
        alignas(std::tuple<P...>) unsigned char vz[sizeof v] = {};   /* zeroed padding, as in launch_as */
        new (vz) std::tuple<P...>(v);
        if (launch_native<decltype(kfn)>(q, g, b, 0, vz, sizeof v)) return sycl::event{};
    }
    sycl::event ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>(global, local),
                       sxe::properties{sycl::ext::intel::experimental::grf_size<256>}, kfn);
    });
    if (launch_profile_on()) lp_ring().push(ev, typeid(kfn).name());
    return ev;
}

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
/* HIP's other fast-math forms, the same way (glibc declares __logf and friends too); the accurate
 * SYCL functions, so a port is never less accurate than the original */
#define __logf(x)     sycl::log((float)(x))
#define __log2f(x)    sycl::log2((float)(x))
#define __log10f(x)   sycl::log10((float)(x))
#define __sinf(x)     sycl::sin((float)(x))
#define __cosf(x)     sycl::cos((float)(x))
#define __tanf(x)     sycl::tan((float)(x))
#define __powf(x, y)  sycl::pow((float)(x), (float)(y))
#define __exp10f(x)   sycl::exp10((float)(x))
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
namespace xe {
template <class T> inline T atomic_or(T* p, T v) { return aref<T, sycl::memory_scope::device>(*p).fetch_or(v); }
template <class T> inline T atomic_and(T* p, T v) { return aref<T, sycl::memory_scope::device>(*p).fetch_and(v); }
template <class T> inline T atomic_exch(T* p, T v) { return aref<T, sycl::memory_scope::device>(*p).exchange(v); }
}
#define atomicOr(p, v) xe::atomic_or((p), (v))
#define atomicAnd(p, v) xe::atomic_and((p), (v))
#define atomicExch(p, v) xe::atomic_exch((p), (v))

/* HIP's numeric min/max overloads (device and host) */
#define XE_MINMAX(T) \
    inline T min(T a, T b) { return a < b ? a : b; } \
    inline T max(T a, T b) { return a > b ? a : b; }
XE_MINMAX(int) XE_MINMAX(unsigned) XE_MINMAX(long) XE_MINMAX(unsigned long) XE_MINMAX(long long)
XE_MINMAX(unsigned long long)
inline float  min(float a, float b)   { return sycl::fmin(a, b); }
inline float  max(float a, float b)   { return sycl::fmax(a, b); }
inline double min(double a, double b) { return sycl::fmin(a, b); }
inline double max(double a, double b) { return sycl::fmax(a, b); }
#undef XE_MINMAX
inline float __frsqrt_rn(float x) { return sycl::rsqrt(x); }
/* HIP's float -> int conversions: _rn rounds to nearest even (v_cvt_rpi is not used: HIP's _rn is
 * rint), _rz truncates; out-of-range inputs are the kernels' to clamp first. */
inline int      __float2int_rn(float x)  { return (int)sycl::rint(x); }
inline int      __float2int_rz(float x)  { return (int)x; }
inline unsigned __float2uint_rn(float x) { return (unsigned)sycl::rint(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __frcp_rn(float x) { return 1.0f / x; }

/* The device queries a launcher sizes LDS by: one device, the B70's 64 KiB a work-group as libr4d
 * plans for the R9700 (both 64 KiB under libr4d's own cap). */
struct hipDeviceProp_t { size_t sharedMemPerBlock; int warpSize; int multiProcessorCount; };
inline int hipGetDevice(int* d) { *d = 0; return 0; }
inline int hipGetDeviceProperties(hipDeviceProp_t* p, int) { *p = hipDeviceProp_t{64 * 1024, 32, 32}; return 0; }
inline int hipDeviceGetAttribute(int* v, int, int) { *v = 0; return 0; }
inline int hipDeviceSynchronize() { return 0; }

inline float    __uint_as_float(unsigned u) { return sycl::bit_cast<float>(u); }
inline unsigned __float_as_uint(float f)    { return sycl::bit_cast<unsigned>(f); }
inline float    __int_as_float(int u)       { return sycl::bit_cast<float>(u); }
inline int      __float_as_int(float f)     { return sycl::bit_cast<int>(f); }

struct alignas(8)  uint2  { unsigned x, y; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(16) int4   { int x, y, z, w; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(4)  uchar4 { unsigned char x, y, z, w; };
struct alignas(2)  uchar2 { unsigned char x, y; };
struct alignas(4)  ushort2 { unsigned short x, y; };
struct alignas(8)  ushort4 { unsigned short x, y, z, w; };
struct alignas(4)  short2 { short x, y; };
struct alignas(8)  short4 { short x, y, z, w; };
struct alignas(8)  int2 { int x, y; };
struct alignas(8)  float2 { float x, y; };
inline ushort2 make_ushort2(unsigned short a, unsigned short b) { return ushort2{a, b}; }
inline ushort4 make_ushort4(unsigned short a, unsigned short b, unsigned short c, unsigned short d) { return ushort4{a, b, c, d}; }
inline short4  make_short4(short a, short b, short c, short d) { return short4{a, b, c, d}; }
inline int2    make_int2(int a, int b) { return int2{a, b}; }
inline int4    make_int4(int a, int b, int c, int d) { return int4{a, b, c, d}; }
inline float2  make_float2(float a, float b) { return float2{a, b}; }
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
#define XE_SHARED_VAR(type, name) \
    auto& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<type>(xe::wg())
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
/* The gfx12 wave32 16x16x16 WMMAs, from sub-group shuffles in tests/r4d_emu's layout (emu_wmma):
 * lane l holds 8 elements of A and of B; A(M, k) is element k & 7 of lane M + 16 * (k >> 3), and
 * B(k, N) element k & 7 of lane N + 16 * (k >> 3); the result lane l is column N = l & 15, rows
 * 8 * (l >> 4) + e. Each product is added to c in ascending k, as the emulator does. The operands
 * move a 32-bit word at a time (two bf16/f16 or four E4M3 elements). Correct, not fast: 16 + 64
 * shuffles a bf16 tile (8 + 32 for E4M3). */
namespace xe {
inline float wmma_e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;
    else if (e == 15 && m == 7) v = __builtin_bit_cast(float, 0x7fc00000u);
    else v = __builtin_bit_cast(float, ((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}
/* KIND 0 bf16, 1 f16 (four words), 2 E4M3 (two words) */
template <int KIND> inline float wmma_elem(const uint32_t* w, int j) {
    if constexpr (KIND == 2) return wmma_e4m3((w[j >> 2] >> (8 * (j & 3))) & 0xffu);
    const uint32_t h = (w[j >> 1] >> (16 * (j & 1))) & 0xffffu;
    if constexpr (KIND == 0) return __builtin_bit_cast(float, h << 16);
    else return (float)sycl::bit_cast<sycl::half>((uint16_t)h);
}
template <int KIND, class V, class C> inline C wmma(const V& a, const V& b, const C& c) {
    constexpr int NW = KIND == 2 ? 2 : 4;
    static_assert(sizeof(V) == 4 * NW && sizeof(C) == 32, "a gfx12 wave32 WMMA operand");
    uint32_t wa[NW], wb[NW];
    __builtin_memcpy(wa, &a, sizeof wa);
    __builtin_memcpy(wb, &b, sizeof wb);
    float d[8];
    __builtin_memcpy(d, &c, sizeof d);
    const int l = lane(), N = l & 15, h = l >> 4;
    float bcol[16];
    for (int kh = 0; kh < 2; ++kh) {
        uint32_t w[NW];
        for (int i = 0; i < NW; ++i) w[i] = shfl_from(wb[i], N + 16 * kh);
        for (int j = 0; j < 8; ++j) bcol[8 * kh + j] = wmma_elem<KIND>(w, j);
    }
    for (int e = 0; e < 8; ++e) {
        float acc = d[e];
        for (int kh = 0; kh < 2; ++kh) {
            uint32_t w[NW];
            for (int i = 0; i < NW; ++i) w[i] = shfl_from(wa[i], 8 * h + e + 16 * kh);
            for (int j = 0; j < 8; ++j) acc += wmma_elem<KIND>(w, j) * bcol[8 * kh + j];
        }
        d[e] = acc;
    }
    C r;
    __builtin_memcpy(&r, d, sizeof d);
    return r;
}
}  // namespace xe
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c) (xe::wmma<2>((a), (b), (c)))
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) (xe::wmma<0>((a), (b), (c)))
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) (xe::wmma<1>((a), (b), (c)))
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, cl) ((c) | (int)0x80000000)
/* global_load_tr_b128 (gfx12): an 8x8 transpose of 16-bit elements across each group of 8 lanes --
 * every lane loads 16 bytes from its own address, and lane j of a group receives element j of each
 * of the group's 8 lanes, in lane order (libr4d/r4d_common.h load_tr_b128). By shuffles: the
 * receiving lane's element index varies, so each source element is broadcast and the matching one
 * kept (64 shuffles; correct, not fast). */
namespace xe {
inline xe_v8s load_tr_b128(const void* p) {
    const xe_v8s v = *(const xe_v8s*)p;
    const int l = lane(), j = l & 7, g = l & ~7;
    xe_v8s r;
    for (int i = 0; i < 8; ++i) {
        short pick = 0;
        for (int e = 0; e < 8; ++e) {
            const short x = shfl_from((short)v[e], g + i);
            if (e == j) pick = x;
        }
        r[i] = pick;
    }
    return r;
}
}  // namespace xe
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) (xe::load_tr_b128((const void*)(p)))
#define R4D_DYN_LDS(type, name) XE_DYN_LDS(type, name)

/* R4D_EMIT / R4D_KLAUNCH / hipLaunchKernelGGL on a queue (the RadStream) */
#define XE_LAUNCH(q, kernel, grid, block, lds, ...) \
    xe::launch_as<kernel>(kernel, *(q), (grid), (block), (size_t)(lds), __VA_ARGS__)
#define XE_LAUNCH_SG(sgsize, q, kernel, grid, block, lds, ...) \
    xe::launch_as<kernel, sgsize>(kernel, *(q), (grid), (block), (size_t)(lds), __VA_ARGS__)
/* hipLaunchKernelGGL's form with the large register file (xe::launch_as_grf256; lds must be 0) */
#define XE_LAUNCH_GGL_GRF256(kernel, grid, block, lds, stream, ...) \
    ((void)xe::launch_as_grf256<kernel>(kernel, *((hipStream_t)(stream)), (grid), (block), __VA_ARGS__))
#define R4D_EMIT(stream, kernel, grid, block, lds, ...) \
    ((void)XE_LAUNCH((hipStream_t)(stream), kernel, grid, block, lds, __VA_ARGS__), 0)
#define hipLaunchKernelGGL(kernel, grid, block, lds, stream, ...) \
    ((void)XE_LAUNCH((hipStream_t)(stream), kernel, grid, block, lds, __VA_ARGS__))
/* r4d_launch.h's form (it leaves an R4D_KLAUNCH already defined alone) */
/* A launch a port keeps only so its unit compiles (a matrix kernel libxe rewrites): refused. */
#define XE_NOT_PORTED(...) R4D_NO(RAD_E_UNSUPPORTED)
#define R4D_KLAUNCH(kernel, grid, block, stream, ...) \
    ((void)XE_LAUNCH((hipStream_t)(stream), (kernel), grid, block, 0, __VA_ARGS__))

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

/* libr4d's only plain `asm(...)` is r4d_gdn_wmma.h's LDSOPAQUE, `asm("" : "+v"(x))`: an empty
 * statement that hides a value from the optimiser (AMDGPU's "v" register class), the identity in
 * effect. IGC parses inline asm as vISA and refuses it, so the form is dropped. `asm volatile(...)`
 * is not matched by this macro; libxe's ports replace their few by hand. */
#define asm(...) ((void)0)

/* libr4d spells its WMMA fragments as `__bf16` vectors (r4d_gdn_wmma.h, r4d_moe.hip, r4d_hc_bf16.hip);
 * the B70's SPIR-V target has no SPV_KHR_bfloat16, so any such type fails libxe's device link. In the
 * ports these vectors are only 16-bit containers handed to the poisoned WMMA stubs above (their rows
 * are not registered), so here `__bf16` is a 16-bit integer. A port that does bf16 arithmetic must use
 * __hip_bfloat16 / bf16_to_f32 (it would fail its libref check, never pass by luck). */
#define __bf16 short
