/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * tape_bench.cpp -- owner note 1's per-launch figure (step 5): N tiny kernels on one in-order stream,
 * launched plainly and replayed from a recorded pass (the Level Zero backend's tape, core/device/l0.cpp),
 * same kernels and same output. Each kernel k adds k + 1 to one counter after doubling it, so the
 * final value depends on every launch and on their order.
 *
 *   xe_tape_bench [N=2000] [reps=20] [host|dev]
 *
 * `dev` keeps the counter in device memory (set and read back by copies outside the timed span), so
 * the kernels do not touch host memory and the figure is the launch cost alone.
 *
 * Prints, per launch: host submit time (the calls returning) and wall time (submits + the wait), for
 * plain launches and for a play of the recorded pass; and whether every rep's result equals the
 * host's. Exit 0 when all agree. Runs on the card or on the CPU device (RADIANCE_L0_DEVICE=cpu).
 */
#include "device/device.h"
#include "rad_device.h"
#include "xe_lanes.h"   /* the launch path a kernel library takes (milestone 3: it records natively) */

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

uint32_t expect(int n, uint32_t v0) {
    uint32_t v = v0;
    for (int k = 0; k < n; ++k) v = v * 2u + (uint32_t)(k + 1);
    return v;
}

__global__ void bump_kernel(uint32_t* c, uint32_t add) { *c = *c * 2u + add; }
/* Through xe::launch_as, as every libxe kernel: a native tape records only what comes that way
 * (a direct queue submission inside a recording is refused, native_tape_test case 5). */
void launch_all(sycl::queue& q, uint32_t* c, int n) {
    for (int k = 0; k < n; ++k)
        XE_LAUNCH(&q, bump_kernel, xe::dim3(1), xe::dim3(1), 0, c, (uint32_t)(k + 1));
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    if (rad_dev_count() <= 0 || rad_dev_set(0) != RAD_OK) { std::printf("tape_bench: no device\n"); return 77; }
    RadStream s = nullptr;
    if (rad_stream_create(&s, 0) != RAD_OK) { std::printf("tape_bench: no stream\n"); return 1; }
    sycl::queue& q = *(sycl::queue*)s;
    const bool dev = argc > 3 && std::string(argv[3]) == "dev";
    const int kind = dev ? RAD_MEM_DEVICE : RAD_MEM_HOST_PINNED;
    uint32_t* c = (uint32_t*)rad_dev_alloc(4096, kind);
    if (!c) { std::printf("tape_bench: no memory\n"); return 1; }
    int bad = 0;
    uint32_t hv = 0;
    auto set = [&](uint32_t v) { hv = v; rad_memcpy_async(c, &hv, 4, s); rad_stream_sync(s); };
    auto get = [&]() { rad_memcpy_async(&hv, c, 4, s); rad_stream_sync(s); return hv; };

    /* plain launches */
    double p_sub = 0, p_wall = 0;
    for (int r = 0; r < reps + 1; ++r) {
        set((uint32_t)r);
        const auto t0 = Clock::now();
        launch_all(q, c, n);
        const auto t1 = Clock::now();
        rad_stream_sync(s);
        const auto t2 = Clock::now();
        if (get() != expect(n, (uint32_t)r)) ++bad;
        if (r) { p_sub += us(t0, t1); p_wall += us(t0, t2); }   /* rep 0 warms */
    }

    /* the recorded pass: one segment of n kernels, recorded while it runs (as a pass is) */
    set(7u);
    if (rad::tape_begin() < 0) { std::printf("tape_bench: tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
    launch_all(q, c, n);
    rad::tape_mark();
    rad::RadTape t = rad::tape_end(true);
    if (!t) { std::printf("tape_bench: tape_end refused: %s\n", rad_dev_last_error()); return 1; }
    rad_stream_sync(s);
    if (get() != expect(n, 7u)) ++bad;
    const int len = rad::tape_len(t);
    double t_sub = 0, t_wall = 0;
    for (int r = 0; r < reps + 1; ++r) {
        set((uint32_t)(100 + r));
        const auto t0 = Clock::now();
        rad::tape_play(t, 0, len);
        const auto t1 = Clock::now();
        rad_stream_sync(s);   /* also submits a held play */
        const auto t2 = Clock::now();
        if (get() != expect(n, (uint32_t)(100 + r))) ++bad;
        if (r) { t_sub += us(t0, t1); t_wall += us(t0, t2); }
    }
    rad::tape_destroy(t);
    std::printf("tape_bench: %d kernels, %d reps, tape of %d record(s), counter in %s memory\n", n, reps, len,
                dev ? "device" : "host-pinned");
    std::printf("  plain : submit %.3f us/launch, wall %.3f us/launch\n", p_sub / reps / n, p_wall / reps / n);
    std::printf("  replay: submit %.3f us/launch, wall %.3f us/launch\n", t_sub / reps / n, t_wall / reps / n);
    std::printf("  results: %s (%d of %d runs wrong)\n", bad ? "DIFFER" : "all equal the host's", bad, 2 * reps + 3);
    rad_dev_free(c, kind);
    rad_stream_destroy(s);
    return bad ? 1 : 0;
}
