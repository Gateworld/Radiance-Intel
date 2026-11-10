/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 267: owner lever #2's step 0, "card test only").
 *
 * two_queue_probe.cpp -- do two in-order queues on one card run their kernels at the same time? The all-reduce overlap
 * (logs/run267-ar-overlap-design-v2.md) puts a collective on a second lane beside the expert GEMMs; it is worth
 * building only if the card executes the two lanes concurrently and the GEMMs' own time does not grow much while the
 * collective's spinning threads share the EUs. Three measurements on one GPU, every spin BOUNDED (an iteration cap)
 * so a serialised queue can never hang the card:
 *   1. the memory-bound kernel alone: R passes of y = a + b over 64 M floats (768 MB of traffic a pass);
 *   2. the spinner alone: G work-groups x 128 threads polling a flag that is never set, to the cap (how long the cap is);
 *   3. together: the spinner on queue 1 FIRST, then the passes on queue 0; the host waits for the passes, sets the flag,
 *      waits for the spinner. Reported: the passes' wall time beside the spinner against alone (the slowdown the overlap
 *      would impose on the GEMMs), the spinner's iteration count (large and under the cap = it spun while the passes
 *      ran; at the cap = it ran first and blocked them; near zero = it ran after them), and the combined wall time.
 * Verdict: CONCURRENT when the passes' time grows less than 2x and the spinner's count is between 1 % and 99 % of the
 * cap; SERIAL when the count is at the cap or the passes' time is about alone + the cap's time; else UNCLEAR.
 *
 *   two_queue_probe [passes=40] [spin groups=64] [spin cap=4000000]     (GPU only; RI_DEVICE=cpu: SKIP)
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t kN = 64u << 20;   /* 64 M floats a buffer: 256 MB, three buffers 768 MB */

void passes(sycl::queue& q, float* y, const float* a, const float* b, int R) {
    for (int r = 0; r < R; ++r)
        q.parallel_for(sycl::nd_range<1>(kN, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            y[i] = a[i] + b[i] * (float)(r + 1);
        });
}

/* every thread polls the flag (an atomic load, device scope) up to cap times; thread 0 of every group writes its count */
void spinner(sycl::queue& q, uint32_t* flag, uint32_t* counts, int groups, uint32_t cap) {
    q.parallel_for(sycl::nd_range<1>((size_t)groups * 128, 128), [=](sycl::nd_item<1> it) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> f(*flag);
        uint32_t n = 0;
        while (n < cap && f.load() == 0u) ++n;
        if (it.get_local_id(0) == 0) counts[it.get_group(0)] = n;
    });
}

}  // namespace

int main(int argc, char** argv) {
    const int R = argc > 1 ? std::atoi(argv[1]) : 40;
    const int G = argc > 2 ? std::atoi(argv[2]) : 64;
    const uint32_t cap = argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 4000000u;
    const char* dev = std::getenv("RI_DEVICE");
    if (dev && !std::strcmp(dev, "cpu")) { std::printf("SKIP two_queue_probe: a GPU probe\n"); return 0; }
    sycl::device d{sycl::gpu_selector_v};
    sycl::context ctx{d};
    sycl::queue q0{ctx, d, sycl::property::queue::in_order{}}, q1{ctx, d, sycl::property::queue::in_order{}};
    std::printf("two_queue_probe: %s | passes %d x 768 MB, spinner %d groups x 128 threads, cap %u\n",
                d.get_info<sycl::info::device::name>().c_str(), R, G, cap);
    /* the device's command-queue groups (the coordinator's addition): a SERIAL verdict means nothing without knowing
     * whether the card offers more than one compute queue, and the Level Zero adapter's engine choice for its queues
     * (UR_L0_USE_COMPUTE_ENGINE / SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE: the compute engine index, unset = the
     * adapter's default) -- the ordinal a sycl::queue landed on is not queryable through the queue handle itself */
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
        uint32_t n = 0;
        if (zeDeviceGetCommandQueueGroupProperties(zd, &n, nullptr) == ZE_RESULT_SUCCESS && n > 0 && n < 16) {
            std::vector<ze_command_queue_group_properties_t> p(n);
            for (auto& x : p) { x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES; x.pNext = nullptr; }
            if (zeDeviceGetCommandQueueGroupProperties(zd, &n, p.data()) == ZE_RESULT_SUCCESS)
                for (uint32_t i = 0; i < n; ++i)
                    std::printf("QUEUES ordinal %u: %u queue(s)%s%s%s\n", i, p[i].numQueues,
                                (p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ? " COMPUTE" : "",
                                (p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ? " COPY" : "",
                                (p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS) ? " COOPERATIVE" : "");
        }
        const char* e1 = std::getenv("UR_L0_USE_COMPUTE_ENGINE"), *e2 = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE");
        std::printf("QUEUES adapter compute-engine choice: UR_L0_USE_COMPUTE_ENGINE=%s SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE=%s\n",
                    e1 ? e1 : "(unset)", e2 ? e2 : "(unset)");
    }
    float* a = sycl::malloc_device<float>(kN, q0);
    float* b = sycl::malloc_device<float>(kN, q0);
    float* y = sycl::malloc_device<float>(kN, q0);
    uint32_t* flag = sycl::malloc_device<uint32_t>(1, q0);
    uint32_t* counts = sycl::malloc_device<uint32_t>((size_t)G, q0);
    std::vector<uint32_t> hc((size_t)G);
    /* Run 268 (TQ4: 1,010 GB/s "alone", above the card's 608 GB/s): constant fills compress on the way to memory
     * (Xe2's buffer compression), so the passes read and write pseudo-random words instead -- incompressible, the
     * kernel really memory-bound. */
    q0.parallel_for(sycl::range<1>(kN), [=](sycl::id<1> id) {
        const uint32_t i = (uint32_t)id[0];
        uint32_t h = i * 0x9e3779b9u; h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        a[i] = (float)(h & 0xffffu) * (1.0f / 65536.0f);
        b[i] = (float)((h >> 16) & 0xffffu) * (1.0f / 65536.0f);
        y[i] = 0.0f;
    });
    q0.memset(flag, 0, 4); q0.wait();
    /* 1. the passes alone. Run 268 (TQ1-TQ3): one warm-up pass left the "alone" figure at the card's idle clock (719,
     * 533 and 407 GB/s in three runs; TQ3's passes ran 0.67x "alone" beside the spinner), so warm up with R passes
     * and take the better of two timed R-pass runs. */
    passes(q0, y, a, b, R); q0.wait();
    double t0 = now_ms(); passes(q0, y, a, b, R); q0.wait(); double t_alone = now_ms() - t0;
    t0 = now_ms(); passes(q0, y, a, b, R); q0.wait(); const double t_alone2 = now_ms() - t0;
    if (t_alone2 < t_alone) t_alone = t_alone2;
    /* 2. the spinner alone, to the cap */
    t0 = now_ms(); spinner(q1, flag, counts, G, cap); q1.wait(); const double t_cap = now_ms() - t0;
    q0.memcpy(hc.data(), counts, (size_t)G * 4).wait();
    uint32_t c_alone = 0; for (int g = 0; g < G; ++g) c_alone = hc[(size_t)g] > c_alone ? hc[(size_t)g] : c_alone;
    /* 3. together: the spinner first on queue 1, the passes on queue 0; the flag after the passes */
    q0.memset(flag, 0, 4).wait();
    const double t_start = now_ms();
    spinner(q1, flag, counts, G, cap);
    passes(q0, y, a, b, R);
    q0.wait();
    const double t_pass_shared = now_ms() - t_start;
    const uint32_t one = 1u;
    q0.memcpy(flag, &one, 4).wait();
    q1.wait();
    const double t_both = now_ms() - t_start;
    q0.memcpy(hc.data(), counts, (size_t)G * 4).wait();
    uint32_t c_min = cap, c_max = 0;
    for (int g = 0; g < G; ++g) { c_min = hc[(size_t)g] < c_min ? hc[(size_t)g] : c_min; c_max = hc[(size_t)g] > c_max ? hc[(size_t)g] : c_max; }
    const double slow = t_pass_shared / t_alone;
    /* run 268 (TQ5: fast passes, the spinner at 90 % of its alone rate, yet "ran after" by the 1 %-of-cap rule): the
     * spinner's count is judged against what its alone rate would have reached in the passes' time */
    const double expected = t_pass_shared * ((double)cap / t_cap);
    const double share_min = expected > 0.0 ? (double)c_min / expected : 0.0, share_max = expected > 0.0 ? (double)c_max / expected : 0.0;
    const char* verdict = "UNCLEAR";
    if (c_max >= cap || t_pass_shared > t_alone + 0.8 * t_cap) verdict = "SERIAL (the spinner ran to its cap before the passes)";
    else if (share_max < 0.05) verdict = "SERIAL (the spinner ran after the passes)";
    else if (slow < 2.0 && share_min >= 0.3) verdict = "CONCURRENT";
    else if (slow >= 2.0 && share_min >= 0.3) verdict = "CONCURRENT, the passes slowed >= 2x";
    std::printf("RESULT passes alone %.1f ms (%.1f ms a pass, %.0f GB/s) | spinner alone to the cap %.1f ms (count %u)\n",
                t_alone, t_alone / R, 3.0 * kN * 4 * R / (t_alone * 1e6), t_cap, c_alone);
    std::printf("RESULT together: passes %.1f ms (%.0f GB/s) beside the spinner = %.2fx alone | spinner count %u..%u of cap %u "
                "= %.0f..%.0f %% of its alone rate over the passes' time | both %.1f ms\n",
                t_pass_shared, 3.0 * kN * 4 * R / (t_pass_shared * 1e6), slow, c_min, c_max, cap, 100.0 * share_min,
                100.0 * share_max, t_both);
    std::printf("VERDICT %s\n", verdict);
    sycl::free(a, q0); sycl::free(b, q0); sycl::free(y, q0); sycl::free(flag, q0); sycl::free(counts, q0);
    return 0;
}
