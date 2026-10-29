/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 88, owner note 84: the two-card first step's bind).
 *
 * hostbind_probe.cpp -- the two-card first-step hang without the model. Run 88's prebind (R15) put it at the first
 * copy on card1 from a 16,991 MiB host allocation of the context both cards share: NEO binds the buffer for card1's
 * copy engine, and xe answers DRM_IOCTL_XE_VM_BIND with EBUSY without end (runs 87-88), while card0 binds the same
 * allocation in milliseconds. This makes the same copy the engine makes (sycl::aligned_alloc_host on a context over
 * every GPU, then 64 bytes from it with queue::memcpy on one card's in-order queue), so a size, an order or a split
 * that avoids it can be found in seconds a case:
 *   xe_hostbind_probe GIB ORDER [CHUNK_MIB=0] [FILL=1]
 *     GIB        host memory to allocate, in GiB (fractions allowed)
 *     ORDER      the cards that touch it, in order: "1" (card1 only), "01" (card0 then card1, the engine's), "10"
 *     CHUNK_MIB  0 = one allocation (the engine's pool); else allocations of this size covering GIB
 *     FILL       1 = write every page from the host first, as the engine's load fills its pool
 * Each touch is printed before it starts and timed. One that takes longer than RI_HB_STUCK_S (30) seconds prints
 * STUCK with what it was touching and the process exits 3 (the bind does not return; the step's timeout is the
 * backstop). Exit 0 when every touch completes. A card probe; no CPU test.
 */
#include <sycl/sycl.hpp>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;

std::atomic<long long> g_touch_start{0};   /* ns since t0 of the touch in progress, 0 = none */
std::string g_what;                         /* what it is touching (written before g_touch_start) */
const clk::time_point t0 = clk::now();

long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count(); }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xe_hostbind_probe GIB ORDER [CHUNK_MIB=0] [FILL=1]\n");
        return 2;
    }
    const double gib = std::strtod(argv[1], nullptr);
    const std::string order = argv[2];
    const long long chunk_mib = argc > 3 ? std::strtoll(argv[3], nullptr, 10) : 0;
    const bool fill = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    const char* st = std::getenv("RI_HB_STUCK_S");
    const double stuck_s = st ? std::strtod(st, nullptr) : 30.0;

    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 2) {
        std::fprintf(stderr, "hostbind: %zu GPU(s); the probe needs two\n", gpus.size());
        return 2;
    }
    sycl::context ctx(gpus);   /* one context over every card, as l0.cpp makes for two ranks */
    std::vector<sycl::queue> qs;
    for (auto& d : gpus) qs.emplace_back(ctx, d, sycl::property::queue::in_order{});
    for (size_t i = 0; i < gpus.size(); ++i)
        std::printf("hostbind: card %zu = %s\n", i, gpus[i].get_info<sycl::info::device::name>().c_str());

    const long long total = (long long)(gib * 1073741824.0);
    const long long piece = chunk_mib > 0 ? chunk_mib << 20 : total;
    std::vector<std::pair<char*, long long>> allocs;
    for (long long off = 0; off < total; off += piece) {
        const long long n = std::min(piece, total - off);
        char* p = sycl::aligned_alloc_host<char>(4096, (size_t)n, ctx);
        if (!p) {
            std::printf("hostbind: allocation of %.2f MiB failed after %zu\n", n / 1048576.0, allocs.size());
            return 1;
        }
        if (fill) for (long long b = 0; b < n; b += 4096) p[b] = (char)(b >> 12);
        allocs.emplace_back(p, n);
    }
    std::printf("hostbind: %zu host allocation(s), %.2f GiB in all%s, order %s\n", allocs.size(), total / 1073741824.0,
                fill ? ", every page written" : "", order.c_str());
    std::fflush(stdout);

    std::vector<char*> scratch;
    for (size_t i = 0; i < gpus.size(); ++i) scratch.push_back(sycl::malloc_device<char>(4096, gpus[i], ctx));

    std::thread watch([&] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const long long s = g_touch_start.load();
            if (s > 0 && (now_ns() - s) * 1e-9 > stuck_s) {
                std::printf("hostbind: STUCK %.0f s in %s\n", stuck_s, g_what.c_str());
                std::fflush(stdout);
                _exit(3);
            }
        }
    });
    watch.detach();

    for (char c : order) {
        const int d = c - '0';
        if (d < 0 || d >= (int)gpus.size()) { std::fprintf(stderr, "hostbind: no card %c\n", c); return 2; }
        for (size_t i = 0; i < allocs.size(); ++i) {
            char what[160];
            std::snprintf(what, sizeof what, "card %d, allocation %zu of %zu (%.2f MiB at %p)", d, i + 1, allocs.size(),
                          allocs[i].second / 1048576.0, (void*)allocs[i].first);
            g_what = what;
            std::printf("hostbind: touch %s\n", what);
            std::fflush(stdout);
            const long long t = now_ns();
            g_touch_start.store(t);
            qs[(size_t)d].memcpy(scratch[(size_t)d], allocs[i].first, 64).wait();
            g_touch_start.store(0);
            std::printf("hostbind:   done in %.1f ms\n", (now_ns() - t) * 1e-6);
        }
    }
    std::printf("hostbind: every touch completed\n");
    std::fflush(stdout);
    for (size_t i = 0; i < gpus.size(); ++i) sycl::free(scratch[i], ctx);
    for (auto& a : allocs) sycl::free(a.first, ctx);
    return 0;
}
