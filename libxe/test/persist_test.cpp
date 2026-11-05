/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, owner note 186 item 3b: the megakernel's stage 1).
 *
 * persist_test.cpp -- xe_persist.h's phase barrier, on the CPU SYCL device in ctest (RI_DEVICE=cpu) and on a card
 * as is:
 *   1. the chain: persist_groups() work-groups run kPhases phases; in phase p each group writes
 *      v[p][g] = sum over k of v[p-1][k] + g + 1, which is right only if every group's phase p-1 write was visible --
 *      checked against the host's own sums, with canary padding on both sides of v (UNATTENDED's rule for a new
 *      kernel);
 *   2. the bound: the same kernel told to wait for one group more than it launched must come back with the error
 *      word set at phase 0 and no hang (a bounded spin, never the GuC's timeout).
 * On a device that cannot keep the groups resident at once the chain spins out too; that is reported as such
 * ("not resident", rc 2), never as a pass.
 */
#include "xe_persist.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kPhases = 8;
constexpr int kLocal = 256;
constexpr uint32_t kPattern = 0xC0FFEE11u;
constexpr int kCanary = 1024;   /* words a side: 4 KiB */

/* launches `groups` work-groups that believe `told` groups take part */
void run_chain(sycl::queue& q, uint32_t* v, uint32_t* sync, unsigned groups, unsigned told, uint64_t spin_max) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)groups * kLocal), sycl::range<1>(kLocal)),
                   [=](sycl::nd_item<1> it) {
        const uint32_t g = (uint32_t)it.get_group(0);
        for (uint32_t p = 1; p <= (uint32_t)kPhases; ++p) {
            if (it.get_local_linear_id() == 0) {
                uint32_t s = 0;
                for (uint32_t k = 0; k < groups; ++k) s += v[(p - 1) * groups + k];
                v[p * groups + g] = s + g + 1u;
            }
            if (!xe::persist_barrier(it, sync, told, p - 1, spin_max)) return;
        }
    }).wait_and_throw();
}

}  // namespace

int main() {
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::device d;
    try {
        d = cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception& e) {
        std::printf("persist_test: no %s device: %s\n", cpu ? "cpu" : "gpu", e.what());
        return 2;
    }
    sycl::queue q(d, sycl::property::queue::in_order());
    const unsigned G = xe::persist_groups(q);
    std::printf("persist_test: device %s, %u resident work-groups of %d\n",
                d.get_info<sycl::info::device::name>().c_str(), G, kLocal);

    const size_t nv = (size_t)(kPhases + 1) * G;
    uint32_t* buf = sycl::malloc_device<uint32_t>(nv + 2 * kCanary, q);
    uint32_t* sync = sycl::malloc_device<uint32_t>(xe::kPersistWords, q);
    uint32_t* v = buf + kCanary;
    std::vector<uint32_t> h(nv + 2 * kCanary);
    int fails = 0;

    /* 1. the chain */
    for (size_t i = 0; i < h.size(); ++i) h[i] = (i < (size_t)kCanary || i >= kCanary + nv) ? kPattern : 0u;
    for (unsigned g = 0; g < G; ++g) h[kCanary + g] = g + 1u;   /* v[0][g] */
    q.memcpy(buf, h.data(), h.size() * 4);
    q.memset(sync, 0, xe::kPersistWords * 4).wait();
    run_chain(q, v, sync, G, G, 1ull << 26);
    std::vector<uint32_t> out(h.size()), sw(xe::kPersistWords);
    q.memcpy(out.data(), buf, out.size() * 4);
    q.memcpy(sw.data(), sync, sw.size() * 4).wait();
    if (sw[1] != 0u) {
        std::printf("persist_test: the chain spun out at phase %u -- the %u groups are not resident at once here\n",
                    sw[2] - 2u, G);
        return 2;
    }
    std::vector<uint32_t> ref(nv);
    for (unsigned g = 0; g < G; ++g) ref[g] = g + 1u;
    for (int p = 1; p <= kPhases; ++p) {
        uint32_t s = 0;
        for (unsigned k = 0; k < G; ++k) s += ref[(size_t)(p - 1) * G + k];
        for (unsigned g = 0; g < G; ++g) ref[(size_t)p * G + g] = s + g + 1u;
    }
    int bad = 0, canary = 0;
    for (size_t i = 0; i < nv; ++i) bad += out[kCanary + i] != ref[i];
    for (int i = 0; i < kCanary; ++i) canary += (out[i] != kPattern) + (out[kCanary + nv + i] != kPattern);
    std::printf("persist_test: chain over %d phases: %d of %zu values wrong, %d canary words broken, counter %u (want %u)\n",
                kPhases, bad, nv, canary, sw[0], (unsigned)(kPhases * G));
    fails += bad != 0 || canary != 0 || sw[0] != (uint32_t)(kPhases * G);

    /* 2. the bound: one group more than launched never arrives */
    q.memset(sync, 0, xe::kPersistWords * 4).wait();
    run_chain(q, v, sync, G, G + 1, 1ull << 16);
    q.memcpy(sw.data(), sync, sw.size() * 4).wait();
    const bool bound_ok = sw[1] == 1u && sw[2] == 2u;
    std::printf("persist_test: the bound: error word %u at phase %d (want 1 at 0), the kernel returned\n", sw[1],
                (int)sw[2] - 2);
    fails += !bound_ok;

    sycl::free(buf, q);
    sycl::free(sync, q);
    std::printf("persist_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
