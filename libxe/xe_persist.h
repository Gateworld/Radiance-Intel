/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, owner note 186 item 3b: the megakernel's stage 1).
 *
 * xe_persist.h -- the pieces a persistent launch needs on Xe with SYCL, which has no supported grid-wide barrier
 * (briefs/2026-11-05-megakernel.md section 3):
 *   - persist_groups(q): how many 256-thread work-groups are surely resident at once -- one an Xe core;
 *   - persist_barrier(): a device-scope phase barrier, nd_range_barrier's pattern (each work-group's thread 0 adds
 *     one to a counter in global memory, then waits for phase x ngroups), with a BOUNDED spin: past `spin_max` reads
 *     it writes the error word and gives up, so a group that is not resident (or a peer that never comes) ends the
 *     kernel with an error the host reads, never a hang the GuC times out (11-05, GA2: a stalled job -> a CAT fault).
 * The counter only grows within a launch (phase p waits for (p + 1) x ngroups), so it needs no reset between phases;
 * the host zeroes it and the error word before each launch.
 */
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace xe {

/* The work-groups of `threads` lanes a persistent launch may use on q's device: one an Xe core on an Intel GPU (a
 * 256-lane group of SIMD32 is eight hardware threads, an Xe core holds 64, so one is resident on every core whatever
 * else runs). The Xe core count is the slices x subslices the device reports; without those, compute units / 8 (Xe2:
 * 8 vector engines a core). On another device (the CPU in the tests): its compute units, at most `cap_other`. */
inline unsigned persist_groups(const sycl::queue& q, unsigned cap_other = 4) {
    const sycl::device d = q.get_device();
    const unsigned cu = d.get_info<sycl::info::device::max_compute_units>();
    if (!d.is_gpu()) return std::max(1u, std::min(cu, cap_other));
    if (d.has(sycl::aspect::ext_intel_gpu_slices) && d.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
        const unsigned s = d.get_info<sycl::ext::intel::info::device::gpu_slices>();
        const unsigned ss = d.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        if (s * ss > 0) return s * ss;
    }
    return std::max(1u, cu / 8u);
}

/* The launch's synchronisation words in device (or shared) memory: [0] the arrival counter, [1] the error word
 * (0 = clean; 1 = a barrier spun out; the phase that spun out + 2 in [2]). */
constexpr int kPersistWords = 4;

/* Every thread of every work-group calls it at the end of phase `phase` (0, 1, ...). True when every group arrived;
 * false when the spin ran out (then [1] is set and the kernel should return: a later phase would read unfinished
 * work). `it` is the calling thread's nd_item. */
template <int D>
inline bool persist_barrier(const sycl::nd_item<D>& it, uint32_t* sync, uint32_t ngroups, uint32_t phase,
                            uint64_t spin_max) {
    using aref = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
    const auto g = it.get_group();
    /* the group's own work, all of it, before the group says it arrived */
    sycl::group_barrier(g, sycl::memory_scope::device);
    uint32_t bad = 0;
    if (it.get_local_linear_id() == 0) {
        aref count(sync[0]), err(sync[1]);
        count.fetch_add(1u);
        const uint32_t target = (phase + 1u) * ngroups;
        uint64_t spins = 0;
        while (count.load(sycl::memory_order::acquire) < target) {
            if (err.load(sycl::memory_order::relaxed) != 0u || ++spins > spin_max) {
                uint32_t zero = 0u;
                if (err.compare_exchange_strong(zero, 1u)) aref(sync[2]).store(phase + 2u);
                bad = 1u;
                break;
            }
        }
    }
    bad = sycl::group_broadcast(g, bad, 0);
    /* every other group's phase work, visible to this group's next phase */
    sycl::group_barrier(g, sycl::memory_scope::device);
    return bad == 0u;
}

/* The same for a kernel without an nd_item in hand (libxe's HIP-dialect ports): the work-item from SYCL's query. */
inline bool persist_barrier_here(uint32_t* sync, uint32_t ngroups, uint32_t phase, uint64_t spin_max) {
    return persist_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), sync, ngroups, phase, spin_max);
}

/* At the kernel's end, after its last barrier: each group's thread 0 counts out in [3], and the last one zeroes the
 * arrival counter and [3] -- every other group has passed its last barrier by then (it counted out after it), so no
 * one reads the counter any more, and the next launch on the stream (a recorded pass's replay too) starts from 0
 * without a host memset. After a spin-out nobody resets: the error stays for the host, and later launches fail fast. */
inline void persist_exit_here(uint32_t* sync, uint32_t ngroups) {
    using aref = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (it.get_local_linear_id() != 0) return;
    aref out(sync[3]);
    if (out.fetch_add(1u) == ngroups - 1u) {
        aref(sync[0]).store(0u);
        out.store(0u);
    }
}

}  /* namespace xe */
