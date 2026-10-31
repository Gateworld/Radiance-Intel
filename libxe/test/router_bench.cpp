/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * router_bench.cpp -- owner note 129 item 4 (run 110): where router_topk_scatter's ~19 us a decode launch goes. One
 * process, one card, random bf16 logits at Flash-Next's router (512 experts, top 10, one rank's slice of 256), times
 * as host wall over back-to-back launches on an in-order queue:
 *   - the fused selection + sort (r4d_rad_router_topk_scatter, one work-group) at 1, 4 and 8 rows;
 *   - the selection alone (r4d_rad_router_topk), so the sort's share shows by difference;
 *   - an empty one-work-group kernel: the floor a launch costs here.
 * Prints a hash of the fused kernel's outputs for the record.
 *
 *   xe_router_bench LIBXE.so [launches=400] [runs=5]
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "r4d_args.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
RadTensor t2(void* p, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}
RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_router_bench LIBXE.so [launches] [runs]\n"); return 2; }
    const int launches = argc > 2 ? std::atoi(argv[2]) : 400;
    const int runs = argc > 3 ? std::atoi(argv[3]) : 5;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("router_bench: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fused = (LaunchFn)dlsym(h, "r4d_rad_router_topk_scatter");
    LaunchFn sel = (LaunchFn)dlsym(h, "r4d_rad_router_topk");
    if (!fused || !sel) { std::printf("router_bench: missing symbols\n"); return 2; }
    sycl::queue q{ sycl::device{ sycl::gpu_selector_v }, sycl::property_list{ sycl::property::queue::in_order{} } };
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int n_expert = 512, n_local = 256, top_k = 10, Mmax = 8;
    std::vector<uint16_t> hl((size_t)Mmax * n_expert);
    for (size_t i = 0; i < hl.size(); ++i) {   /* bf16 logits in about [-4, 4], ties possible as in the real router */
        const float f = ((float)(h32(7, i) % 2048u) - 1024.0f) / 256.0f;
        uint32_t u; std::memcpy(&u, &f, 4);
        hl[i] = (uint16_t)(u >> 16);
    }
    uint16_t* d_lg = sycl::malloc_device<uint16_t>(hl.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>(Mmax * top_k, q);
    uint16_t* d_w = sycl::malloc_device<uint16_t>(Mmax * top_k, q);
    int32_t* d_sorted = sycl::malloc_device<int32_t>(Mmax * top_k, q);
    int32_t* d_off = sycl::malloc_device<int32_t>(n_local + 1, q);
    int32_t* d_cnt = sycl::malloc_device<int32_t>(n_local, q);
    q.memcpy(d_lg, hl.data(), hl.size() * 2).wait();
    auto wall = [&](auto&& one) {
        std::vector<double> us;
        for (int r = 0; r < runs; ++r) {
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int L = 0; L < launches; ++L) one();
            q.wait_and_throw();
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / launches);
        }
        std::sort(us.begin(), us.end());
        return us[us.size() / 2];
    };
    /* the floor: an empty kernel of one 256-thread work-group */
    auto empty = [&] { q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1>) {}); };
    empty(); q.wait();
    std::printf("RESULT router empty one-work-group kernel: %.2f us a launch\n", wall(empty));
    for (int M : { 1, 4, 8 }) {
        RadParam p[5]{};
        const char* ik[] = { "n_expert", "n_local", "top_k", "norm", "expert_base" };
        const long long iv[] = { n_expert, n_local, top_k, 1, 0 };
        for (int i = 0; i < 5; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
        RadTensor t[6] = { t2(d_lg, RAD_BF16, M, n_expert), t1(d_ids, RAD_I32, M * top_k), t1(d_w, RAD_BF16, M * top_k),
                           t1(d_sorted, RAD_I32, M * top_k), t1(d_off, RAD_I32, n_local + 1), t1(d_cnt, RAD_I32, n_local) };
        RadArgs af{}; af.t = t; af.n_t = 6; af.p = p; af.n_p = 5;
        RadArgs as{}; as.t = t; as.n_t = 3; as.p = p; as.n_p = 5;
        int rc = fused(&af, (RadStream)&q);
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("M %d: fused refused rc %d\n", M, rc); return 1; }
        const double tf = wall([&] { fused(&af, (RadStream)&q); });
        rc = sel(&as, (RadStream)&q);
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("M %d: selection refused rc %d\n", M, rc); return 1; }
        const double ts = wall([&] { sel(&as, (RadStream)&q); });
        fused(&af, (RadStream)&q);
        q.wait();
        std::vector<int32_t> hi(M * top_k), hs(M * top_k), ho(n_local + 1);
        std::vector<uint16_t> hw(M * top_k);
        q.memcpy(hi.data(), d_ids, hi.size() * 4).wait();
        q.memcpy(hw.data(), d_w, hw.size() * 2).wait();
        q.memcpy(hs.data(), d_sorted, hs.size() * 4).wait();
        q.memcpy(ho.data(), d_off, ho.size() * 4).wait();
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](const void* d, size_t n) { const uint8_t* b = (const uint8_t*)d; for (size_t i = 0; i < n; ++i) hash = (hash ^ b[i]) * 1099511628211ull; };
        mix(hi.data(), hi.size() * 4); mix(hw.data(), hw.size() * 2); mix(hs.data(), hs.size() * 4); mix(ho.data(), ho.size() * 4);
        std::printf("RESULT router M %d: fused %.2f us a launch, selection alone %.2f us; outputs hash %016llx\n", M, tf, ts,
                    (unsigned long long)hash);
    }
    std::printf("RESULT router_bench DONE\n");
    return 0;
}
