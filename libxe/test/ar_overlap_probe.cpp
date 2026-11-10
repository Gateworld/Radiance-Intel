/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 271: owner lever #2's pricing probe, the coordinator's item 2 of
 * logs/run271-light-poll-design.md -- "card test only", no model).
 *
 * ar_overlap_probe.cpp -- what would overlapping prefill's all-reduce with the next half-chunk's GEMMs buy? On the two
 * cards, through libxe's own entries (dlopen, as allreduce_test): the engine's all-reduce of a 16 MiB bf16 message
 * (RADIANCE_XE_AR_PROTO default 6: the bandwidth form, 32 blocks, one polling thread a block) on queue A of each card,
 * and the trunk's int8 matrix-unit GEMM (xe_rad_gemm_i8a8_gemv at M = 4,096 rows, K 2,048, N 2,048: the DPAS form, as a
 * prefill chunk runs it) on queue B of the same card, each in-order. Three timings a card:
 *   1. the all-reduces alone: N_A launches back to back on queue A (both ranks in step), the wall / N_A;
 *   2. the GEMMs alone: N_G launches on queue B, the wall / N_G;
 *   3. together: N_G GEMMs queued on B and N_A all-reduces on A at once (A's batch sized to about B's time), each
 *      queue's own completion time and the pair's.
 * Reported: each kernel's slowdown beside the other, and the overlap's gain = (alone A + alone B) - together. Every
 * all-reduce wait is bounded by the library (RADIANCE_XE_AR_SPIN) and the whole run sits under the step's host
 * timeout; a stall is reported by xe_ar2_stalled. The GEMM's inputs are pseudo-random (incompressible); its outputs
 * are not checked (a timing probe; the kernels are the served ones).
 *
 *   ar_overlap_probe LIBXE.so [N_A=20] [N_G=20] [M=4096] [N=2048] [K=2048]        (two GPUs; RI_DEVICE=cpu: SKIP)
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

/* the device table the libxe entries resolve a rank's device / context from (librad_l0k; as allreduce_test) */
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

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* the GEMM's operands on one card: pseudo-random int8 A [M][K] and B [N][K], f32 row scales [M][K/128], bf16 weight
 * scales [N][K/128] (small positive), the bf16 output [M][N] */
struct Gemm {
    int8_t* a = nullptr; float* as = nullptr; int8_t* b = nullptr; uint16_t* bs = nullptr; uint16_t* y = nullptr;
    RadTensor t[5]{};
    RadParam p[3]{};
    RadArgs args{};
    void make(sycl::queue& q, int M, int N, int K, int rank) {
        a = sycl::malloc_device<int8_t>((size_t)M * K, q);
        as = sycl::malloc_device<float>((size_t)M * (K / 128), q);
        b = sycl::malloc_device<int8_t>((size_t)N * K, q);
        bs = sycl::malloc_device<uint16_t>((size_t)N * (K / 128), q);
        y = sycl::malloc_device<uint16_t>((size_t)M * N, q);
        int8_t* aa = a; int8_t* bb = b; float* ass = as; uint16_t* bss = bs;
        const uint32_t salt = 0x9e3779b9u * (uint32_t)(rank + 1);
        q.parallel_for(sycl::range<1>((size_t)M * K), [=](sycl::id<1> id) {
            uint32_t h = (uint32_t)id[0] * 0x85ebca6bu + salt; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
            aa[id[0]] = (int8_t)((h & 0xffu) - 128u);
        });
        q.parallel_for(sycl::range<1>((size_t)N * K), [=](sycl::id<1> id) {
            uint32_t h = (uint32_t)id[0] * 0xc2b2ae35u + salt; h ^= h >> 13; h *= 0x27d4eb2fu; h ^= h >> 15;
            bb[id[0]] = (int8_t)((h & 0xffu) - 128u);
        });
        q.parallel_for(sycl::range<1>((size_t)M * (K / 128)), [=](sycl::id<1> id) { ass[id[0]] = 0.0125f; });
        q.parallel_for(sycl::range<1>((size_t)N * (K / 128)), [=](sycl::id<1> id) { bss[id[0]] = (uint16_t)0x3c00u; /* bf16 0.0078125 */ });
        q.wait();
        auto set = [](RadTensor& t, void* d, int dtype, int64_t r0, int64_t r1) {
            t.data = d; t.dtype = dtype; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
        };
        set(t[0], a, RAD_I8, M, K);
        set(t[1], as, RAD_F32, M, K / 128);
        set(t[2], b, RAD_I8, N, K);
        set(t[3], bs, RAD_BF16, N, K / 128);
        set(t[4], y, RAD_BF16, M, N);
        p[0].key = "K"; p[0].kind = RAD_P_INT; p[0].ival = K;
        p[1].key = "N"; p[1].kind = RAD_P_INT; p[1].ival = N;
        p[2].key = "group"; p[2].kind = RAD_P_INT; p[2].ival = 128;
        args.t = t; args.n_t = 5; args.p = p; args.n_p = 3; args.rank = rank; args.world_size = 2;
    }
    void free(sycl::queue& q) { sycl::free(a, q); sycl::free(as, q); sycl::free(b, q); sycl::free(bs, q); sycl::free(y, q); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ar_overlap_probe LIBXE.so [N_A] [N_G] [M] [N] [K]\n"); return 2; }
    const int NA = argc > 2 ? std::atoi(argv[2]) : 20, NG = argc > 3 ? std::atoi(argv[3]) : 20;
    const int M = argc > 4 ? std::atoi(argv[4]) : 4096, N = argc > 5 ? std::atoi(argv[5]) : 2048, K = argc > 6 ? std::atoi(argv[6]) : 2048;
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dev = std::getenv("RI_DEVICE");
    if (dev && !std::strcmp(dev, "cpu")) { std::printf("SKIP ar_overlap_probe: a two-GPU probe\n"); return 0; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("ar_overlap_probe: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    InitFn init = (InitFn)dlsym(h, "xe_ar2_init");
    FiniFn fini = (FiniFn)dlsym(h, "xe_ar2_fini");
    LaunchFn ar = (LaunchFn)dlsym(h, "xe_rad_ar_oneshot_2rank_exact");
    LaunchFn gemm = (LaunchFn)dlsym(h, "xe_rad_gemm_i8a8_gemv");
    StallFn stalled = (StallFn)dlsym(h, "xe_ar2_stalled");
    if (!init || !fini || !ar || !gemm || !stalled) { std::printf("ar_overlap_probe: symbols missing\n"); return 2; }

    std::vector<sycl::device> gpus;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.size() < 2) { std::printf("ar_overlap_probe: needs two Level Zero GPUs, found %zu\n", gpus.size()); return 2; }
    g_dev[0] = gpus[0]; g_dev[1] = gpus[1];
    g_ctx = sycl::context({ gpus[0], gpus[1] });
    if (g_dev[0].ext_oneapi_can_access_peer(g_dev[1]) && g_dev[1].ext_oneapi_can_access_peer(g_dev[0])) {
        g_dev[0].ext_oneapi_enable_peer_access(g_dev[1]);
        g_dev[1].ext_oneapi_enable_peer_access(g_dev[0]);
    } else { std::printf("ar_overlap_probe: no peer access between the cards\n"); return 2; }
    rad_l0k_set_devices(2, dev_of, ctx_of);
    sycl::queue qa[2] = { sycl::queue(g_ctx, g_dev[0], sycl::property::queue::in_order()),
                          sycl::queue(g_ctx, g_dev[1], sycl::property::queue::in_order()) };
    sycl::queue qb[2] = { sycl::queue(g_ctx, g_dev[0], sycl::property::queue::in_order()),
                          sycl::queue(g_ctx, g_dev[1], sycl::property::queue::in_order()) };
    std::printf("ar_overlap_probe: %s | %s | all-reduce 16 MiB bf16 x %d on queue A, gemm_i8a8 M %d N %d K %d x %d on queue B\n",
                g_dev[0].get_info<sycl::info::device::name>().c_str(), g_dev[1].get_info<sycl::info::device::name>().c_str(),
                NA, M, N, K, NG);

    /* the all-reduce: 16 MiB bf16 a rank, the engine's exact two-rank form through its init / launch entries */
    const int64_t numel = 8LL << 20;
    RadParam p[4]{};
    p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
    p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = numel;
    p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = "bf16";
    p[3].key = "exact"; p[3].kind = RAD_P_INT; p[3].ival = 1;
    void* inst[2] = { nullptr, nullptr };
    int irc[2] = { 0, 0 };
    {
        std::thread t0([&] { irc[0] = init(p, 4, 0, 2, &inst[0]); });
        std::thread t1([&] { irc[1] = init(p, 4, 1, 2, &inst[1]); });
        t0.join(); t1.join();
    }
    if (irc[0] != RAD_OK || irc[1] != RAD_OK) { std::printf("RESULT ar init failed %d / %d\n", irc[0], irc[1]); return 1; }
    uint8_t* x[2];
    RadTensor xt[2]{};
    RadArgs xa[2]{};
    for (int r = 0; r < 2; ++r) {
        x[r] = sycl::malloc_device<uint8_t>((size_t)numel * 2, qa[r]);
        qa[r].memset(x[r], 0, (size_t)numel * 2).wait();
        xt[r].data = x[r]; xt[r].dtype = RAD_BF16; xt[r].rank = 1; xt[r].shape[0] = numel; xt[r].stride[0] = 1;
        xa[r].t = &xt[r]; xa[r].n_t = 1; xa[r].p = p; xa[r].n_p = 4; xa[r].instance = inst[r]; xa[r].rank = r; xa[r].world_size = 2;
    }
    Gemm g[2];
    for (int r = 0; r < 2; ++r) g[r].make(qb[r], M, N, K, r);

    auto run_ar = [&](int r, int n) { for (int i = 0; i < n; ++i) ar(&xa[r], (RadStream)&qa[r]); };
    auto run_gemm = [&](int r, int n) { for (int i = 0; i < n; ++i) gemm(&g[r].args, (RadStream)&qb[r]); };

    /* warm-up: one of each a card (the clock, the lists' first build) */
    { std::thread t0([&] { run_ar(0, 2); qa[0].wait(); }), t1([&] { run_ar(1, 2); qa[1].wait(); }); t0.join(); t1.join(); }
    for (int r = 0; r < 2; ++r) { run_gemm(r, 2); qb[r].wait(); }

    /* 1. the all-reduces alone (both ranks in step; the time a launch is the pair's) */
    double t_ar[2] = { 0, 0 };
    {
        const double t0 = now_ms();
        std::thread th0([&] { run_ar(0, NA); qa[0].wait(); t_ar[0] = now_ms() - t0; });
        std::thread th1([&] { run_ar(1, NA); qa[1].wait(); t_ar[1] = now_ms() - t0; });
        th0.join(); th1.join();
    }
    /* 2. the GEMMs alone, a card at a time (each card's own queue B) */
    double t_g[2] = { 0, 0 };
    for (int r = 0; r < 2; ++r) { const double t0 = now_ms(); run_gemm(r, NG); qb[r].wait(); t_g[r] = now_ms() - t0; }
    /* 3. together: on each card, the GEMMs on B and the all-reduces on A at once; A's count sized to B's time */
    const double ar_us = (t_ar[0] + t_ar[1]) / 2.0 / NA * 1000.0, g_us = (t_g[0] + t_g[1]) / 2.0 / NG * 1000.0;
    int NA2 = (int)((double)NG * g_us / ar_us + 0.5); if (NA2 < 2) NA2 = 2; if (NA2 > 4 * NA) NA2 = 4 * NA;
    double tg_tog[2] = { 0, 0 }, ta_tog[2] = { 0, 0 }, t_both = 0;
    {
        const double t0 = now_ms();
        std::thread tg0([&] { run_gemm(0, NG); qb[0].wait(); tg_tog[0] = now_ms() - t0; });
        std::thread tg1([&] { run_gemm(1, NG); qb[1].wait(); tg_tog[1] = now_ms() - t0; });
        std::thread ta0([&] { run_ar(0, NA2); qa[0].wait(); ta_tog[0] = now_ms() - t0; });
        std::thread ta1([&] { run_ar(1, NA2); qa[1].wait(); ta_tog[1] = now_ms() - t0; });
        tg0.join(); tg1.join(); ta0.join(); ta1.join();
        t_both = now_ms() - t0;
    }
    const bool st = stalled(inst[0], (RadStream)&qa[0]) || stalled(inst[1], (RadStream)&qa[1]);
    for (int r = 0; r < 2; ++r) {
        std::printf("RESULT card %d: all-reduce alone %.1f us a launch (%d x 16 MiB) | gemm alone %.1f us a launch (%d x) | "
                    "together: %d gemms in %.1f ms (%.2fx alone) beside %d all-reduces in %.1f ms (%.2fx alone)\n",
                    r, t_ar[r] / NA * 1000.0, NA, t_g[r] / NG * 1000.0, NG, NG, tg_tog[r], tg_tog[r] / t_g[r], NA2, ta_tog[r],
                    ta_tog[r] / (t_ar[r] / NA * NA2));
    }
    const double alone_sum = (t_g[0] + t_g[1]) / 2.0 + (t_ar[0] + t_ar[1]) / 2.0 / NA * NA2;
    std::printf("RESULT overlap: alone A + B %.1f ms, together %.1f ms -> the overlap recovers %.1f ms (%.0f %% of the "
                "all-reduces' time); stalled %s\n",
                alone_sum, t_both, alone_sum - t_both, 100.0 * (alone_sum - t_both) / ((t_ar[0] + t_ar[1]) / 2.0 / NA * NA2),
                st ? "YES" : "no");
    std::printf("VERDICT %s\n", st ? "STALLED (see xe_ar2_stalled)" : (alone_sum - t_both > 0.3 * ((t_ar[0] + t_ar[1]) / 2.0 / NA * NA2)
                                                                           ? "the overlap hides most of the all-reduce" : "little to gain"));
    for (int r = 0; r < 2; ++r) { g[r].free(qb[r]); sycl::free(x[r], qa[r]); }
    fini(inst[0]); fini(inst[1]);
    return st ? 1 : 0;
}
