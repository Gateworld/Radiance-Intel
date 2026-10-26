/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * rad_l0_rec.h -- the native recorder channel (owner note 28: recorded passes as direct Level Zero
 * command lists). Three parties share it: the Level Zero backend (core/device/l0.cpp) records a pass
 * into regular command lists and installs the hooks; the core's device code (librad_l0k,
 * core/device/l0_kernels.sycl) exports them; a kernel library (libxe) asks them before every launch.
 *
 * A launch that goes through the SYCL queue while a native recording is open would run at once and
 * never be replayed, so a kernel library launches natively while `rad_l0k_rec_device` says the
 * calling thread records the queue it is about to use. The library gives the recorder its
 * sycl::kernel (the recorder creates its own Level Zero kernel object from the kernel's module, so
 * the runtime's argument state is never touched), the launch shape, the kernel's one by-value
 * parameter (xe::launch_as captures a single std::tuple of the arguments) and the dynamic
 * work-group memory size. The hooks return 0 when the command was appended and -1 when it was not:
 * the backend has then refused the tape, and the library submits the launch plainly, so the pass
 * still runs.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The three queries are weak: a kernel library or a test that is not linked with librad_l0k (a
 * standalone bench) sees them null and launches plainly. */
#define RAD_L0_REC_WEAK __attribute__((weak))

/* The device index this thread records natively on when `queue` (a RadStream, the backend's
 * sycl::queue) is one of the recording's streams; -1 otherwise (no recording, a paused one, a
 * stream outside it, the SYCL-graph tapes, the CPU device). */
RAD_L0_REC_WEAK int rad_l0k_rec_device(const void* queue);

/* A library kernel as the recorder launches it: its sycl::kernel (kept alive by the caller for the
 * process) and, from the runtime's kernel descriptor, the index of its one by-value parameter and
 * of its dynamic work-group memory parameter (-1 when it has none), so the recorder sets what the
 * compiler laid out and never guesses. */
typedef struct RadL0RecKernel {
    const void* kernel;      /* const sycl::kernel* */
    int32_t     i_arg0;      /* the by-value parameter's argument index */
    int32_t     i_lds;       /* the dynamic local memory's, or -1 */
    uint32_t    arg0_size;   /* the by-value parameter's size per the descriptor (0 = unknown) */
} RadL0RecKernel;

/* Appends a kernel launch to the open recording. `grid` is the work-group count and `block` the
 * work-group size per dimension; `arg0` / `arg0_size` the kernel's one by-value parameter; `lds`
 * the dynamic work-group memory in bytes. */
RAD_L0_REC_WEAK int rad_l0k_rec_launch(const void* queue, const RadL0RecKernel* kernel, const uint32_t grid[3],
                                       const uint32_t block[3], const void* arg0, size_t arg0_size, size_t lds);

/* Appends a pitched 2D copy (the copy engine's region copy). */
RAD_L0_REC_WEAK int rad_l0k_rec_copy2d(const void* queue, void* dst, size_t dpitch, const void* src, size_t spitch,
                                       size_t width, size_t height);

/* The backend's hooks (l0.cpp installs them at startup; null = no native recorder). */
typedef struct RadL0RecHooks {
    int (*device)(const void* queue);
    int (*launch)(const void* queue, const RadL0RecKernel* kernel, const uint32_t grid[3], const uint32_t block[3],
                  const void* arg0, size_t arg0_size, size_t lds);
    int (*copy2d)(const void* queue, void* dst, size_t dpitch, const void* src, size_t spitch,
                  size_t width, size_t height);
} RadL0RecHooks;
void rad_l0k_set_rec(const RadL0RecHooks* hooks);

#ifdef __cplusplus
}  /* extern "C" */

#include <sycl/sycl.hpp>
#include <sycl/detail/kernel_desc.hpp>

#include <atomic>
#include <memory>
#include <mutex>

namespace rad_l0_rec {

/* The recorder's view of an unnamed-lambda kernel `KN`, built once per device and kept for the
 * process (the kernel bundle holds the module its Level Zero kernel object needs). intel/llvm names
 * an unnamed lambda's kernel through __builtin_sycl_unique_stable_name, so the lambda's type is its
 * kernel name, and the same descriptor the runtime sets arguments from (KernelInfo) gives the
 * parameter kinds, sizes and order. Null when the lookup fails (then the launch goes plainly). */
constexpr int kMaxRecDevices = 16;
template <class KN> const RadL0RecKernel* kernel_of(sycl::queue& q, int dev) {
    static std::atomic<const RadL0RecKernel*> k[kMaxRecDevices];
    static std::atomic<bool> failed[kMaxRecDevices];
    static std::mutex m;
    if (dev < 0 || dev >= kMaxRecDevices) return nullptr;
    if (const RadL0RecKernel* got = k[dev].load(std::memory_order_acquire)) return got;
    if (failed[dev].load(std::memory_order_relaxed)) return nullptr;
    std::lock_guard<std::mutex> lk(m);
    if (const RadL0RecKernel* got = k[dev].load(std::memory_order_acquire)) return got;
    try {
        using KI = sycl::detail::KernelInfo<KN>;
        int i_arg0 = -1, i_lds = -1;
        uint32_t size0 = 0;
        for (int i = 0; i < (int)KI::getNumParams(); ++i) {
            const sycl::detail::kernel_param_desc_t d = KI::getParamDesc(i);
            if (d.kind == sycl::detail::kernel_param_kind_t::kind_dynamic_work_group_memory) { i_lds = i; continue; }
            if ((d.kind == sycl::detail::kernel_param_kind_t::kind_std_layout ||
                 d.kind == sycl::detail::kernel_param_kind_t::kind_pointer) && i_arg0 < 0) {
                i_arg0 = i;
                size0 = (uint32_t)d.info;
                continue;
            }
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), "a kernel parameter the recorder cannot set");
        }
        if (i_arg0 < 0) throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), "no by-value parameter");
        sycl::kernel_id id = sycl::get_kernel_id<KN>();
        auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()}, {id});
        RadL0RecKernel* made = new RadL0RecKernel{new sycl::kernel(kb.get_kernel(id)), i_arg0, i_lds, size0};   /* process lifetime */
        k[dev].store(made, std::memory_order_release);
        return made;
    } catch (const sycl::exception&) {
        failed[dev].store(true, std::memory_order_relaxed);
        return nullptr;
    }
}

}  /* namespace rad_l0_rec */
#endif
