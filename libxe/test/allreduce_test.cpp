/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 2).
 *
 * allreduce_test.cpp -- libxe's two-rank all_reduce rows (xe_allreduce.cpp) through the plugin
 * entry points, two rank threads on one process as the engine runs them: init (the rendezvous), many
 * launches with changing payloads, in place and into y, bf16 and f32, several message sizes, both
 * forms (peer stores and the host mailbox). On the CPU SYCL device both ranks are the same device in
 * one context, so the peer form's "peer memory" is plain memory and both forms run there. Checks: every
 * rank's result equals f32(x0) + f32(x1) rounded to the dtype (ascending rank order), identical on both
 * ranks, canaries intact, no stalled handshake.
 *
 *   allreduce_test LIBXE.so [iters=50]    RI_DEVICE=cpu for the CPU SYCL device (the ctest); on a
 *   two-GPU host without RI_DEVICE it uses the first two Level Zero GPUs and times the exchange.
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" void rad_l0k_set_devices(int n, const sycl::device* (*dev_of)(int), const sycl::context* (*ctx_of)(int));

namespace {

typedef int (*InitFn)(const RadParam*, int, int, int, void**);
typedef void (*FiniFn)(void*);
typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef uint32_t (*StallFn)(void*, RadStream);

sycl::device g_dev[2];
sycl::context g_ctx;
const sycl::device* dev_of(int i) { return &g_dev[i & 1]; }
const sycl::context* ctx_of(int) { return &g_ctx; }

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xA7;

struct Case { int dt; int64_t numel; bool inplace; };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: allreduce_test LIBXE.so [iters]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    setvbuf(stdout, nullptr, _IONBF, 0);   /* progress survives a timeout kill */
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("allreduce_test: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    InitFn init[2] = { (InitFn)dlsym(h, "xe_ar2_init"), (InitFn)dlsym(h, "xe_ar2_host_init") };
    FiniFn fini = (FiniFn)dlsym(h, "xe_ar2_fini");
    LaunchFn launch[2] = { (LaunchFn)dlsym(h, "xe_rad_ar_oneshot_2rank_exact"), (LaunchFn)dlsym(h, "xe_rad_ar_oneshot_2rank_exact_host") };
    StallFn stalled = (StallFn)dlsym(h, "xe_ar2_stalled");
    if (!init[0] || !init[1] || !fini || !launch[0] || !launch[1] || !stalled) { std::printf("allreduce_test: symbols missing\n"); return 2; }

    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    if (cpu) {
        g_dev[0] = g_dev[1] = sycl::device(sycl::cpu_selector_v);
        g_ctx = sycl::context(g_dev[0]);
    } else {
        std::vector<sycl::device> gpus;
        for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
            if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
        if (gpus.size() < 2) { std::printf("allreduce_test: needs two Level Zero GPUs (or RI_DEVICE=cpu), found %zu\n", gpus.size()); return 2; }
        g_dev[0] = gpus[0]; g_dev[1] = gpus[1];
        g_ctx = sycl::context({ gpus[0], gpus[1] });
        if (g_dev[0].ext_oneapi_can_access_peer(g_dev[1]) && g_dev[1].ext_oneapi_can_access_peer(g_dev[0])) {
            g_dev[0].ext_oneapi_enable_peer_access(g_dev[1]);
            g_dev[1].ext_oneapi_enable_peer_access(g_dev[0]);
            std::printf("peer access enabled both ways\n");
        } else std::printf("no peer access: the peer-store form will be skipped\n");
    }
    rad_l0k_set_devices(2, dev_of, ctx_of);
    sycl::queue q[2] = { sycl::queue(g_ctx, g_dev[0], sycl::property::queue::in_order()),
                         sycl::queue(g_ctx, g_dev[1], sycl::property::queue::in_order()) };
    std::printf("device 0: %s\ndevice 1: %s\n", g_dev[0].get_info<sycl::info::device::name>().c_str(),
                g_dev[1].get_info<sycl::info::device::name>().c_str());

    /* AR_BENCH=N (run 102, owner note 109 item 1): the decode exchange's device time. Both ranks submit N bf16
     * all-reduces of 2560 x AR_ROWS (default 1) values back to back, bracketed by a one-item marker kernel each side on
     * a profiling queue; the time a launch is (start of the closing marker - end of the opening one) / N, so the
     * ranks' host submission skew washes out over N. Environment for the forms (RADIANCE_XE_AR_*) passes through. */
    const int64_t bound_x = std::getenv("AR_BOUND_X") ? std::atoll(std::getenv("AR_BOUND_X")) : 1;
    if (const char* eb = std::getenv("AR_BENCH"); eb && !cpu) {
        const int nb = std::atoi(eb);
        const int rows = std::getenv("AR_ROWS") ? std::atoi(std::getenv("AR_ROWS")) : 1;
        const int64_t numel = 2560LL * rows, bytes = numel * 2;
        sycl::queue pq[2] = { sycl::queue(g_ctx, g_dev[0], { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() }),
                              sycl::queue(g_ctx, g_dev[1], { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() }) };
        void* inst[2] = { nullptr, nullptr };
        int irc[2] = { 0, 0 };
        RadParam p[4]{};
        p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
        p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = numel * bound_x;
        p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = "bf16";
        p[3].key = "exact"; p[3].kind = RAD_P_INT; p[3].ival = 1;
        {
            std::thread t0([&] { irc[0] = init[0](p, 4, 0, 2, &inst[0]); });
            std::thread t1([&] { irc[1] = init[0](p, 4, 1, 2, &inst[1]); });
            t0.join(); t1.join();
        }
        if (irc[0] != RAD_OK || irc[1] != RAD_OK) { std::printf("RESULT ar_bench: init failed %d / %d\n", irc[0], irc[1]); return 1; }
        uint8_t* x[2];
        for (int r = 0; r < 2; ++r) { x[r] = sycl::malloc_device<uint8_t>(bytes, pq[r]); pq[r].memset(x[r], 0, bytes).wait(); }
        for (int rep = 0; rep < 3; ++rep) {
            double per[2] = {0, 0}, sub[2] = {0, 0};
            std::thread th[2];
            for (int r = 0; r < 2; ++r) th[r] = std::thread([&, r] {
                RadTensor t{};
                t.data = x[r]; t.dtype = RAD_BF16; t.rank = 1; t.shape[0] = numel; t.stride[0] = 1;
                RadArgs a{};
                a.t = &t; a.n_t = 1; a.p = p; a.n_p = 4; a.instance = inst[r]; a.rank = r; a.world_size = 2;
                /* the launches recorded once into a SYCL graph and replayed, as the engine's tapes replay a pass:
                 * a host-submitted loop is submission-bound (~23 us a launch, AB1), not the exchange */
                namespace sxg = sycl::ext::oneapi::experimental;
                sxg::command_graph g{pq[r].get_context(), pq[r].get_device()};
                const auto h0 = std::chrono::steady_clock::now();
                g.begin_recording(pq[r]);
                for (int i = 0; i < nb; ++i) launch[0](&a, (RadStream)&pq[r]);
                g.end_recording(pq[r]);
                auto ge = g.finalize();
                sub[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count() / nb;
                sycl::event e0 = pq[r].single_task([] {});
                pq[r].ext_oneapi_graph(ge);
                sycl::event e1 = pq[r].single_task([] {});
                e1.wait();
                const auto a0 = e0.get_profiling_info<sycl::info::event_profiling::command_end>();
                const auto a1 = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
                per[r] = (double)(a1 - a0) * 1e-3 / nb;
            });
            th[0].join(); th[1].join();
            const bool st = stalled(inst[0], (RadStream)&pq[0]) || stalled(inst[1], (RadStream)&pq[1]);
            std::printf("RESULT ar_bench rows %d, %d launches (a graph): device %.2f / %.2f us a launch (rank 0 / 1), recording "
                        "%.2f / %.2f us a launch%s\n", rows, nb, per[0], per[1], sub[0], sub[1], st ? ", STALLED" : "");
        }
        for (int r = 0; r < 2; ++r) { sycl::free(x[r], pq[r]); fini(inst[r]); }
        return 0;
    }

    const Case cases[] = { { 0, 2560, true }, { 0, 2560, false }, { 2, 2560, true }, { 0, 4104, false },
                           { 0, 1 << 18, true }, { 2, 1 << 16, false } };
    bool all_ok = true;
    int canary_bad = 0;
    for (int form = 0; form < 2; ++form) {
        if (!cpu && form == 0 && !(g_dev[0].ext_oneapi_can_access_peer(g_dev[1]))) continue;
        for (const Case& c : cases) {
            const int64_t eb = c.dt == 2 ? 4 : 2, bytes = c.numel * eb;
            std::printf("case %s dt%d numel %lld %s: init...\n", form ? "host" : "peer", c.dt, (long long)c.numel, c.inplace ? "in-place" : "into y");
            /* the instance per rank: init in two threads, as the engine's rank threads do */
            void* inst[2] = { nullptr, nullptr };
            int irc[2] = { 0, 0 };
            {
                RadParam p[4]{};
                p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
                /* AR_BOUND_X=k (run 102): the instance's bound k x the message, as the engine's band bound exceeds a
                 * decode message (the LL form, RADIANCE_XE_AR_PROTO=4, needs twice the message in a slot) */
                p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = c.numel * bound_x;
                p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = c.dt == 2 ? "fp32" : "bf16";
                p[3].key = "exact"; p[3].kind = RAD_P_INT; p[3].ival = 1;
                std::thread t0([&] { irc[0] = init[form](p, 4, 0, 2, &inst[0]); });
                std::thread t1([&] { irc[1] = init[form](p, 4, 1, 2, &inst[1]); });
                t0.join(); t1.join();
            }
            if (irc[0] != RAD_OK || irc[1] != RAD_OK || !inst[0] || !inst[1]) {
                std::printf("RESULT allreduce %s dt%d numel %lld: init failed rc %d / %d\n", form ? "host" : "peer", c.dt, (long long)c.numel, irc[0], irc[1]);
                all_ok = false; continue;
            }
            /* buffers per rank: x, y (with canaries), host copies */
            uint8_t *x[2], *yb[2];
            std::vector<uint8_t> hx[2];
            for (int r = 0; r < 2; ++r) {
                x[r] = sycl::malloc_device<uint8_t>(bytes, q[r]);
                yb[r] = sycl::malloc_device<uint8_t>(bytes + 2 * kCanary, q[r]);
                hx[r].resize(bytes);
            }
            std::printf("  init ok; launching\n");
            double worst = 0; bool ok = true; double us_sum = 0; int timed = 0;
            /* run 87 (owner notes 81-82): where a launch pair's time goes, per rank -- the thread's start after
             * t0, the launch call, the queue wait; summed over the timed iterations */
            double ph_start[2] = {0, 0}, ph_launch[2] = {0, 0}, ph_wait[2] = {0, 0};
            for (int it = 0; it < iters && ok; ++it) {
                for (int r = 0; r < 2; ++r) {
                    for (int64_t i = 0; i < c.numel; ++i) {
                        const float v = ((int)(h32(it * 2 + r + 1, i) % 20001) - 10000) * 0.001f;
                        if (c.dt == 2) std::memcpy(&hx[r][i * 4], &v, 4);
                        else { const uint16_t b = f2bf(v); std::memcpy(&hx[r][i * 2], &b, 2); }
                    }
                    q[r].memcpy(x[r], hx[r].data(), bytes).wait();
                    q[r].memset(yb[r], kPat, bytes + 2 * kCanary).wait();
                }
                /* both ranks launch, each on its own thread and queue */
                int lrc[2] = { 0, 0 };
                double ts[2][3] = {};
                auto t0 = std::chrono::steady_clock::now();
                std::thread th[2];
                for (int r = 0; r < 2; ++r) th[r] = std::thread([&, r] {
                    auto us = [&] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); };
                    ts[r][0] = us();
                    RadTensor t[2]{};
                    t[0].data = x[r]; t[0].dtype = c.dt == 2 ? RAD_F32 : RAD_BF16; t[0].rank = 1; t[0].shape[0] = c.numel; t[0].stride[0] = 1;
                    t[1] = t[0]; t[1].data = yb[r] + kCanary;
                    RadParam p[4]{};
                    p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
                    p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = c.numel;
                    p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = c.dt == 2 ? "fp32" : "bf16";
                    p[3].key = "exact"; p[3].kind = RAD_P_INT; p[3].ival = 1;
                    RadArgs a{};
                    a.t = t; a.n_t = c.inplace ? 1 : 2; a.p = p; a.n_p = 4; a.instance = inst[r]; a.rank = r; a.world_size = 2;
                    lrc[r] = launch[form](&a, (RadStream)&q[r]);
                    ts[r][1] = us();
                    q[r].wait();
                    ts[r][2] = us();
                });
                th[0].join(); th[1].join();
                auto t1 = std::chrono::steady_clock::now();
                if (it > 0) {
                    us_sum += std::chrono::duration<double, std::micro>(t1 - t0).count(); ++timed;
                    for (int r = 0; r < 2; ++r) { ph_start[r] += ts[r][0]; ph_launch[r] += ts[r][1] - ts[r][0]; ph_wait[r] += ts[r][2] - ts[r][1]; }
                }
                if (lrc[0] != RAD_OK || lrc[1] != RAD_OK) { std::printf("  launch rc %d / %d at iter %d\n", lrc[0], lrc[1], it); ok = false; break; }
                for (int r = 0; r < 2; ++r) if (stalled(inst[r], (RadStream)&q[r])) { std::printf("  rank %d stalled at iter %d\n", r, it); ok = false; }
                /* check both ranks against the host sum */
                std::vector<uint8_t> hy[2];
                for (int r = 0; r < 2; ++r) {
                    if (c.inplace) { hy[r].resize(bytes); q[r].memcpy(hy[r].data(), x[r], bytes).wait(); }
                    else {
                        hy[r].resize(bytes + 2 * kCanary); q[r].memcpy(hy[r].data(), yb[r], bytes + 2 * kCanary).wait();
                        for (size_t i = 0; i < kCanary; ++i) if (hy[r][i] != kPat || hy[r][kCanary + bytes + i] != kPat) { ++canary_bad; ok = false; }
                        hy[r].erase(hy[r].begin(), hy[r].begin() + kCanary); hy[r].resize(bytes);
                    }
                }
                for (int64_t i = 0; i < c.numel && ok; ++i) {
                    float a0, a1, want, got0, got1;
                    if (c.dt == 2) {
                        std::memcpy(&a0, &hx[0][i * 4], 4); std::memcpy(&a1, &hx[1][i * 4], 4);
                        want = a0 + a1;
                        std::memcpy(&got0, &hy[0][i * 4], 4); std::memcpy(&got1, &hy[1][i * 4], 4);
                    } else {
                        uint16_t b0, b1, g0, g1;
                        std::memcpy(&b0, &hx[0][i * 2], 2); std::memcpy(&b1, &hx[1][i * 2], 2);
                        want = bf2f(f2bf(bf2f(b0) + bf2f(b1)));
                        std::memcpy(&g0, &hy[0][i * 2], 2); std::memcpy(&g1, &hy[1][i * 2], 2);
                        got0 = bf2f(g0); got1 = bf2f(g1);
                    }
                    if (got0 != got1) { std::printf("  ranks differ at %lld: %g vs %g (iter %d)\n", (long long)i, got0, got1, it); ok = false; break; }
                    const double d = std::fabs((double)got0 - (double)want);
                    if (d > worst) worst = d;
                    if (d != 0.0) { std::printf("  wrong at %lld: got %g want %g (iter %d)\n", (long long)i, got0, want, it); ok = false; break; }
                }
            }
            std::printf("RESULT allreduce %-4s dt%d numel %7lld %-8s: %s, %d iters, worst |diff| %g%s\n", form ? "host" : "peer", c.dt,
                        (long long)c.numel, c.inplace ? "in-place" : "into y", ok ? "PASS" : "FAIL", iters, worst,
                        (!cpu && timed) ? (std::string(", ") + std::to_string(us_sum / timed) + " us a launch pair (host wall incl. waits)").c_str() : "");
            if (!cpu && timed)
                for (int r = 0; r < 2; ++r)
                    std::printf("  phases rank %d: thread start %.1f us, launch call %.1f us, queue wait %.1f us (means over %d)\n", r,
                                ph_start[r] / timed, ph_launch[r] / timed, ph_wait[r] / timed, timed);
            all_ok &= ok;
            for (int r = 0; r < 2; ++r) { sycl::free(x[r], q[r]); sycl::free(yb[r], q[r]); fini(inst[r]); }
        }
    }
    /* all_gather (xe_ar_gather_2rank): f32 words, plain and row-interleaved, both forms; y is [2 * numel]
     * with rank r's x at r * numel (plain) or at (i / row) * (2 row) + r * row + i % row. */
    LaunchFn glaunch[2] = { (LaunchFn)dlsym(h, "xe_rad_ar_gather_2rank"), (LaunchFn)dlsym(h, "xe_rad_ar_gather_2rank_host") };
    if (!glaunch[0] || !glaunch[1]) { std::printf("allreduce_test: gather symbols missing\n"); all_ok = false; }
    const struct { int64_t numel, row; } gcases[] = { { 128, 16 }, { 352, 44 }, { 16384, 2048 }, { 2560, 0 } };
    for (int form = 0; form < 2 && glaunch[0] && glaunch[1]; ++form) {
        if (!cpu && form == 0 && !(g_dev[0].ext_oneapi_can_access_peer(g_dev[1]))) continue;
        for (const auto& gc : gcases) {
            const int64_t bytes = gc.numel * 4;
            void* inst[2] = { nullptr, nullptr };
            int irc[2] = { 0, 0 };
            RadParam p[4]{};
            p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
            p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = gc.numel;
            p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = "f32";
            p[3].key = "row"; p[3].kind = RAD_P_INT; p[3].ival = gc.row;
            {
                std::thread t0([&] { irc[0] = init[form](p, 4, 0, 2, &inst[0]); });
                std::thread t1([&] { irc[1] = init[form](p, 4, 1, 2, &inst[1]); });
                t0.join(); t1.join();
            }
            if (irc[0] != RAD_OK || irc[1] != RAD_OK) { std::printf("RESULT allgather %s numel %lld: init failed\n", form ? "host" : "peer", (long long)gc.numel); all_ok = false; continue; }
            float *x[2], *y[2];
            std::vector<float> hx[2];
            for (int r = 0; r < 2; ++r) {
                x[r] = sycl::malloc_device<float>(gc.numel, q[r]); y[r] = sycl::malloc_device<float>(2 * gc.numel, q[r]);
                hx[r].resize(gc.numel);
                for (int64_t i = 0; i < gc.numel; ++i) hx[r][i] = (float)(r * 1000000 + i);
                q[r].memcpy(x[r], hx[r].data(), bytes).wait();
                q[r].memset(y[r], 0xff, 2 * bytes).wait();
            }
            int lrc[2] = { 0, 0 };
            std::thread th[2];
            for (int r = 0; r < 2; ++r) th[r] = std::thread([&, r] {
                RadTensor t[2]{};
                t[0].data = x[r]; t[0].dtype = RAD_F32; t[0].rank = 1; t[0].shape[0] = gc.numel; t[0].stride[0] = 1;
                t[1].data = y[r]; t[1].dtype = RAD_F32; t[1].rank = 1; t[1].shape[0] = 2 * gc.numel; t[1].stride[0] = 1;
                RadArgs a{};
                a.t = t; a.n_t = 2; a.p = p; a.n_p = 4; a.instance = inst[r]; a.rank = r; a.world_size = 2;
                lrc[r] = glaunch[form](&a, (RadStream)&q[r]);
                q[r].wait();
            });
            th[0].join(); th[1].join();
            bool ok = lrc[0] == RAD_OK && lrc[1] == RAD_OK;
            for (int r = 0; r < 2 && ok; ++r) {
                std::vector<float> hy(2 * gc.numel);
                q[r].memcpy(hy.data(), y[r], 2 * bytes).wait();
                for (int src = 0; src < 2 && ok; ++src)
                    for (int64_t i = 0; i < gc.numel; ++i) {
                        const int64_t o = (gc.row > 0 && gc.row < gc.numel) ? (i / gc.row) * (2 * gc.row) + src * gc.row + (i % gc.row) : src * gc.numel + i;
                        if (hy[o] != hx[src][i]) { std::printf("  gather rank %d: y[%lld] = %g, want %g (src %d i %lld)\n", r, (long long)o, hy[o], hx[src][i], src, (long long)i); ok = false; break; }
                    }
            }
            std::printf("RESULT allgather %-4s numel %6lld row %5lld: %s\n", form ? "host" : "peer", (long long)gc.numel, (long long)gc.row, ok ? "PASS" : "FAIL");
            all_ok &= ok;
            for (int r = 0; r < 2; ++r) { sycl::free(x[r], q[r]); sycl::free(y[r], q[r]); fini(inst[r]); }
        }
    }
    std::printf("CANARIES %d broken\nRESULT allreduce_test %s\n", canary_bad, all_ok && !canary_bad ? "PASS" : "FAIL");
    return all_ok && !canary_bad ? 0 : 1;
}
