/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * copy_paths_probe.cpp -- owner note 137 (run 112): a second copy path for expert prefetch, measured. On one card:
 *   - host -> device copy throughput on the copy engine (queue group 1, bcs) alone, on a second immediate list in the
 *     compute group (group 0, ccs) alone, and on both at once (each into its own buffer);
 *   - a memory-bound SYCL kernel (a triad over device arrays) alone, while bcs copies run, and while the compute-group
 *     copies run -- whether a copy path slows our kernels.
 * Pinned host memory, 256 MiB a copy, every copy waited on through a host-visible event. No model.
 *
 *   xe_copy_paths_probe [copies=8]
 */
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Path {
    ze_command_list_handle_t list = nullptr;
    ze_event_handle_t ev = nullptr;
};

bool make_path(ze_context_handle_t ctx, ze_device_handle_t dev, uint32_t ordinal, ze_event_pool_handle_t pool,
               uint32_t ev_index, Path* p) {
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 0,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    if (zeCommandListCreateImmediate(ctx, dev, &qd, &p->list) != ZE_RESULT_SUCCESS) return false;
    ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, ev_index, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    return zeEventCreate(pool, &ed, &p->ev) == ZE_RESULT_SUCCESS;
}

/* `n` copies of `bytes` on one path, one after another, each waited on */
void copies(Path& p, void* dst, const void* src, size_t bytes, int n) {
    for (int i = 0; i < n; ++i) {
        zeEventHostReset(p.ev);
        zeCommandListAppendMemoryCopy(p.list, dst, src, bytes, p.ev, 0, nullptr);
        zeEventHostSynchronize(p.ev, UINT64_MAX);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 8;
    const size_t bytes = 256ull << 20;
    sycl::queue q{ sycl::device{ sycl::gpu_selector_v }, sycl::property_list{ sycl::property::queue::in_order{} } };
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("copy_paths_probe: %s, %d copies of %zu MiB a path\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), n, bytes >> 20);
    const auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    const auto dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 4};
    ze_event_pool_handle_t pool = nullptr;
    if (zeEventPoolCreate(ctx, &pd, 1, const_cast<ze_device_handle_t*>(&dev), &pool) != ZE_RESULT_SUCCESS) {
        std::printf("event pool failed\n"); return 2;
    }
    Path bcs, ccs;
    if (!make_path(ctx, dev, 1, pool, 0, &bcs) || !make_path(ctx, dev, 0, pool, 1, &ccs)) {
        std::printf("immediate lists refused\n"); return 2;
    }
    char* host = sycl::malloc_host<char>(bytes, q);
    char* d1 = sycl::malloc_device<char>(bytes, q);
    char* d2 = sycl::malloc_device<char>(bytes, q);
    if (!host || !d1 || !d2) { std::printf("alloc failed\n"); return 2; }
    for (size_t i = 0; i < bytes; i += 4096) host[i] = (char)i;
    copies(bcs, d1, host, bytes, 1);   /* warm both paths */
    copies(ccs, d2, host, bytes, 1);
    const double gb = (double)bytes * n / 1e9;
    {
        auto t0 = clk::now(); copies(bcs, d1, host, bytes, n); auto t1 = clk::now();
        std::printf("RESULT copy bcs alone: %.2f GB/s\n", gb / secs(t0, t1));
        t0 = clk::now(); copies(ccs, d2, host, bytes, n); t1 = clk::now();
        std::printf("RESULT copy compute-group list alone: %.2f GB/s\n", gb / secs(t0, t1));
        t0 = clk::now();
        std::thread a([&] { copies(bcs, d1, host, bytes, n); });
        std::thread b([&] { copies(ccs, d2, host, bytes, n); });
        a.join(); b.join();
        t1 = clk::now();
        std::printf("RESULT copy both at once: %.2f GB/s summed\n", 2 * gb / secs(t0, t1));
    }
    /* the compute side: a triad over 3 x 256 MiB of floats, repeated */
    const size_t nf = bytes / 4;
    float* x = sycl::malloc_device<float>(nf, q);
    float* y = sycl::malloc_device<float>(nf, q);
    float* z = sycl::malloc_device<float>(nf, q);
    q.fill(x, 1.0f, nf); q.fill(y, 2.0f, nf); q.wait();
    auto triad = [&](int reps) {
        for (int r = 0; r < reps; ++r)
            q.parallel_for(sycl::range<1>(nf), [=](sycl::id<1> i) { z[i] = x[i] + 0.5f * y[i]; });
        q.wait();
    };
    triad(2);
    const int reps = 40;
    auto t0 = clk::now(); triad(reps); auto t1 = clk::now();
    const double alone = secs(t0, t1);
    std::printf("RESULT compute alone: %.2f ms a triad (%.0f GB/s)\n", alone / reps * 1e3, 3.0 * bytes * reps / alone / 1e9);
    for (int which = 0; which < 2; ++which) {
        Path& p = which == 0 ? bcs : ccs;
        bool stop = false;
        std::thread c([&] { while (!stop) copies(p, which == 0 ? d1 : d2, host, bytes, 1); });
        t0 = clk::now(); triad(reps); t1 = clk::now();
        stop = true; c.join();
        const double with = secs(t0, t1);
        std::printf("RESULT compute with %s copies running: %.2f ms a triad (%+.1f %% against alone)\n",
                    which == 0 ? "bcs" : "compute-group", with / reps * 1e3, 100.0 * (with - alone) / alone);
    }
    std::printf("RESULT copy_paths_probe DONE\n");
    return 0;
}
