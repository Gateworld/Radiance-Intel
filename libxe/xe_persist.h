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

/* ---- stage 1b (owner note 187): work by ticket and per-item ready flags, no device-wide barrier.
 * A group takes its next unit of work from a counter (persist_ticket), in a fixed order in which every unit's inputs
 * come from units of lower tickets. A unit waits only on the flags of the units it reads (persist_wait_flag). Each
 * flag's units were all ticketed before it -- by groups that are running, and that finish them without waiting on
 * anything later -- so a wait always ends, whether or not every group of the launch is resident at once: no
 * co-residency is needed for progress, only for speed, and the launch can be sized by occupancy. The spins stay
 * bounded (the error word, as above). */
using persist_aref = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>;

/* The group's next ticket from `counter` (the same value in every thread of the group). */
inline uint32_t persist_ticket(uint32_t* counter) {
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const auto g = it.get_group();
    uint32_t t = 0;
    if (it.get_local_linear_id() == 0) t = persist_aref(*counter).fetch_add(1u);
    return sycl::group_broadcast(g, t, 0);
}

/* No earlier spin-out on these words: thread 0's read, the same answer in every thread of the group (a per-thread read
 * could split a group around the ticket loop's collectives when another group sets the error meanwhile -- the CPU row
 * test hung so, run 189). */
inline bool persist_clean(uint32_t* sync) {
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t e = 0;
    if (it.get_local_linear_id() == 0) e = persist_aref(sync[1]).load(sycl::memory_order::relaxed);
    return sycl::group_broadcast(it.get_group(), e, 0) == 0u;
}

/* The group's unit is written: every thread's stores, then one count on `flag` (release, device scope). */
inline void persist_signal(uint32_t* flag) {
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sycl::group_barrier(it.get_group(), sycl::memory_scope::device);
    if (it.get_local_linear_id() == 0) persist_aref(*flag).fetch_add(1u, sycl::memory_order::release);
}

/* Until `flag` reaches `target` (thread 0 spins, acquire; the group then reads what the flag's units wrote). False when
 * the spin ran out or the error word was already set: [1] = 1, [2] = `what` + 2; the kernel should return. */
inline bool persist_wait_flag(uint32_t* flag, uint32_t target, uint32_t* sync, uint32_t what, uint64_t spin_max) {
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const auto g = it.get_group();
    uint32_t bad = 0;
    if (it.get_local_linear_id() == 0) {
        persist_aref f(*flag), err(sync[1]);
        uint64_t spins = 0;
        while (f.load(sycl::memory_order::acquire) < target) {
            if (err.load(sycl::memory_order::relaxed) != 0u || ++spins > spin_max) {
                uint32_t zero = 0u;
                if (err.compare_exchange_strong(zero, 1u)) persist_aref(sync[2]).store(what + 2u);
                bad = 1u;
                break;
            }
        }
    }
    bad = sycl::group_broadcast(g, bad, 0);
    sycl::group_barrier(g, sycl::memory_scope::device);
    return bad == 0u;
}

/* At the end of a ticketed launch: as persist_exit_here, and the last group out also zeroes the ticket counter [0]
 * and `nflags` flags from sync[kPersistWords], for the next launch. Every group passes here, including one that
 * stopped on an error (the error word itself stays set for the host). */
inline void persist_exit_flags(uint32_t* sync, uint32_t ngroups, uint32_t nflags) {
    const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (it.get_local_linear_id() != 0) return;
    persist_aref out(sync[3]);
    if (out.fetch_add(1u) == ngroups - 1u) {
        for (uint32_t i = 0; i < nflags; ++i) persist_aref(sync[kPersistWords + i]).store(0u);
        persist_aref(sync[0]).store(0u);
        out.store(0u);
    }
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
