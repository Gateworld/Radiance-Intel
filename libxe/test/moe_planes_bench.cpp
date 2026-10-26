/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 2).
 *
 * moe_planes_bench.cpp -- device timing of libxe's planes expert GEMV (moe_gemm_q at w4nl64a8h over
 * the joined planes form, xe_moe_planes.cpp) at Flash-Next's decode shapes, as round 2's M6 timed
 * its variants: a pool of experts in device memory (far past L2), one launch per projection with 10
 * routed experts through a pointer table, a profiling queue, GB/s = bytes read / kernel time. The
 * engine's profile (milestone 2 P2, P3) saw ~145 us a gate_up launch where M6's kernel of the same
 * configuration took 43 us; this tells the kernel's own time apart from whatever the engine adds.
 *
 *   moe_planes_bench LIBXE.so [experts=512] [launches=48] [runs=5]
 *   RI_DEVICE=cpu for the CPU SYCL device (a smoke test: the timing means nothing there).
 *   RADIANCE_XE_MOE_V1=1 selects the old kernel inside libxe (the A/B).
 *
 * Prints RESULT lines per projection (gate_up 1280 x 2560 as parts=2 a_order=token, down 2560 x 640
 * as parts=1 a_order=sorted) and compares the two kernels' outputs when both are run (BOTH=1).
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_planes.h"
#include "r4d_args.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
/* an E4M3 byte with exponent field 1..10 (no NaN, no subnormal) */
uint8_t e4(uint32_t r) { return (uint8_t)((r & 0x80u) | ((1u + (r >> 8) % 10u) << 3) | (r & 7u)); }
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

RadTensor tbl(void* p, uint32_t dt, int64_t cnt) {      /* a pointer table: stride[0] == 0 */
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = cnt; t.shape[1] = 1; t.stride[0] = 0; t.stride[1] = 1;
    return t;
}
RadTensor t2(void* p, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}
RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t;
}

/* the joined planes form of one class: codes entry = [hdr | codes [N][K/2] u4 | table 16 f32],
 * scale entry = [hdr | scale [N][K/64] e4m3 | scale.1 f32]; offsets as xe_moe_planes.cpp computes them */
struct Form {
    int64_t p0, t_off, codes_bytes, s1_off, scale_bytes;
};
Form form_of(int64_t N, int64_t K) {
    Form f;
    f.p0 = rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
    f.t_off = rad_planes_round(f.p0 + N * (K / 2));
    f.codes_bytes = rad_planes_round(f.t_off + 16 * 4);
    f.s1_off = rad_planes_round(f.p0 + N * (K / 64));
    f.scale_bytes = rad_planes_round(f.s1_off + 4);
    return f;
}

struct Proj {
    const char* name; int64_t N, K; int parts; bool sorted;
};

double ev_us(const sycl::event& e) {
    const auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (double)(b - a) / 1000.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moe_planes_bench LIBXE.so [experts] [launches] [runs]\n"); return 2; }
    const int n_pool = argc > 2 ? std::atoi(argv[2]) : 512;
    const int launches = argc > 3 ? std::atoi(argv[3]) : 48;
    const int runs = argc > 4 ? std::atoi(argv[4]) : 5;
    const int top_k = 10, n_expert = 512;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("moe_planes_bench: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_w4p");
    if (!fn) { std::printf("moe_planes_bench: no xe_rad_moe_gemm_w4p in %s\n", argv[1]); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const float levels[16] = { -8.f, -5.5f, -4.f, -3.f, -2.25f, -1.5f, -0.875f, -0.25f,
                               0.25f, 0.875f, 1.5f, 2.25f, 3.f, 4.f, 5.5f, 8.f };   /* E4M3-exact */
    /* BENCH_SMALL=1: the tiny model's shapes (K 512 / 128: the UPL=1 instantiation the CPU JIT compiles) */
    const bool small = std::getenv("BENCH_SMALL") && *std::getenv("BENCH_SMALL") == '1';
    const Proj projs[2] = { { "gate_up", small ? 256 : 1280, small ? 512 : 2560, 2, false },
                            { "down", small ? 512 : 2560, small ? 128 : 640, 1, true } };
    bool all_ok = true;
    for (const Proj& P : projs) {
        const Form F = form_of(P.N, P.K);
        /* the pool: n_pool experts split by parity into two classes, each entry its two planes */
        /* the pointer tables index by expert id over n_expert (the kernel's table_of wants shape[0]
         * >= (n_expert + 1 - class) / 2); the pool is smaller, so entry e -> pool expert (e >> 1) % pool */
        const int cnt = (n_expert + 1) / 2;
        std::vector<uint8_t*> ce[2], se[2];
        std::vector<uint8_t> hc(F.codes_bytes), hs(F.scale_bytes);
        for (int e = 0; e < n_pool; ++e) {
            std::fill(hc.begin(), hc.end(), 0);
            std::fill(hs.begin(), hs.end(), 0);
            for (int64_t i = 0; i < P.N * (P.K / 2); ++i) hc[F.p0 + i] = (uint8_t)h32(e * 7919 + 1, i);
            std::memcpy(&hc[F.t_off], levels, sizeof levels);
            for (int64_t i = 0; i < P.N * (P.K / 64); ++i) hs[F.p0 + i] = e4(h32(e * 104729 + 3, i));
            const float s1 = 1.0f / 8192.0f;
            std::memcpy(&hs[F.s1_off], &s1, 4);
            uint8_t* dc = sycl::malloc_device<uint8_t>(hc.size(), q);
            uint8_t* ds = sycl::malloc_device<uint8_t>(hs.size(), q);
            if (!dc || !ds) { std::printf("alloc failed at expert %d\n", e); return 2; }
            q.memcpy(dc, hc.data(), hc.size()).wait();   /* hc and hs are refilled next: wait */
            q.memcpy(ds, hs.data(), hs.size()).wait();
            ce[e & 1].push_back(dc);
            se[e & 1].push_back(ds);
        }
        q.wait();
        const double bytes_launch = (double)top_k * (double)P.N * ((double)P.K / 2 + (double)P.K / 64);
        std::printf("%s: %d experts of %.2f MB (codes + scales) = %.2f GB in device memory; %.2f MB a launch\n",
                    P.name, n_pool, (P.N * (P.K / 2.0 + P.K / 64.0)) / 1e6,
                    n_pool * (P.N * (P.K / 2.0 + P.K / 64.0)) / 1e9, bytes_launch / 1e6);
        /* pointer tables per class (device memory), one per launch */
        std::vector<const void**> tabc[2], tabs[2];
        std::vector<int32_t*> d_sorted, d_off;
        std::vector<int> routing(launches * top_k);
        for (int L = 0; L < launches; ++L) {
            /* 10 distinct experts, ascending (sorted order), a fresh set each launch */
            std::vector<int> ids;
            for (int j = 0; ids.size() < (size_t)top_k; ++j) {
                int e = (int)(h32(L * 31 + 5, j) % (uint32_t)n_expert);
                if (std::find(ids.begin(), ids.end(), e) == ids.end()) ids.push_back(e);
            }
            std::sort(ids.begin(), ids.end());
            for (int j = 0; j < top_k; ++j) routing[L * top_k + j] = ids[j];
            for (int c = 0; c < 2; ++c) {
                std::vector<const void*> hc2(cnt, nullptr), hs2(cnt, nullptr);
                for (int i = 0; i < cnt; ++i) { hc2[i] = ce[c][(size_t)i % ce[c].size()]; hs2[i] = se[c][(size_t)i % se[c].size()]; }
                const void** dc = sycl::malloc_device<const void*>(cnt, q);
                const void** ds = sycl::malloc_device<const void*>(cnt, q);
                q.memcpy(dc, hc2.data(), cnt * sizeof(void*)).wait();
                q.memcpy(ds, hs2.data(), cnt * sizeof(void*)).wait();
                tabc[c].push_back(dc);
                tabs[c].push_back(ds);
            }
            /* sorted[i] = flat index i (token 0, slot i); off[e] = rows with expert < e */
            std::vector<int32_t> so(top_k), off(n_expert + 1);
            for (int i = 0; i < top_k; ++i) so[i] = i;
            for (int e = 0; e <= n_expert; ++e) off[e] = (int32_t)std::count_if(ids.begin(), ids.end(), [e](int x) { return x < e; });
            int32_t* ds2 = sycl::malloc_device<int32_t>(top_k, q);
            int32_t* do2 = sycl::malloc_device<int32_t>(n_expert + 1, q);
            q.memcpy(ds2, so.data(), top_k * 4).wait();
            q.memcpy(do2, off.data(), (n_expert + 1) * 4).wait();
            d_sorted.push_back(ds2);
            d_off.push_back(do2);
        }
        /* the activation: a_order=token -> 1 row (tok = i / top_k = 0); sorted -> T rows */
        const int a_rows = P.sorted ? top_k : 1;
        std::vector<uint8_t> ha((size_t)a_rows * P.K);
        std::vector<float> has((size_t)a_rows * (P.K / 128));
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = e4(h32(11, i));
        for (size_t i = 0; i < has.size(); ++i) has[i] = 0.01f + (float)(h32(13, i) & 255) / 2560.0f;
        uint8_t* d_a = sycl::malloc_device<uint8_t>(ha.size(), q);
        float* d_as = sycl::malloc_device<float>(has.size(), q);
        uint16_t* d_y = sycl::malloc_device<uint16_t>((size_t)top_k * P.N, q);
        q.memcpy(d_a, ha.data(), ha.size());
        q.memcpy(d_as, has.data(), has.size() * 4);
        q.wait();
        char dt[] = "w4nl64a8h";
        RadParam p[11]{};
        const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
        const long long iv[] = { 1, P.N, P.K, n_expert, top_k, 128, P.N, P.K, P.parts };
        for (int i = 0; i < 9; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
        p[9].key = "a_order"; p[9].kind = RAD_P_STR; p[9].sval = P.sorted ? "sorted" : "token";
        p[10].key = "dtype"; p[10].kind = RAD_P_STR; p[10].sval = dt;

        auto launch = [&](int L) -> int {
            RadTensor t[9] = {
                t2(d_a, RAD_F8E4M3, a_rows, P.K), t2(d_as, RAD_F32, a_rows, P.K / 128),
                tbl((void*)tabc[0][L], RAD_U4, cnt), tbl((void*)tabs[0][L], RAD_F8E4M3, cnt),
                t1(d_sorted[L], RAD_I32, top_k), t1(d_off[L], RAD_I32, n_expert + 1),
                tbl((void*)tabc[1][L], RAD_U4, cnt), tbl((void*)tabs[1][L], RAD_F8E4M3, cnt),
                t2(d_y, RAD_BF16, top_k, P.N) };
            RadArgs args{};
            args.t = t; args.n_t = 9; args.p = p; args.n_p = 11;
            return fn(&args, (RadStream)&q);
        };
        /* warm-up, then `runs` passes over the launches; the kernel time from the queue's events:
         * the row's launch returns no event, so the pass is timed as a whole on the host too */
        int rc = launch(0);
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("%s: launch refused rc %d\n", P.name, rc); all_ok = false; continue; }
        std::vector<double> us_launch;
        for (int r = 0; r < runs; ++r) {
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int L = 0; L < launches; ++L) launch(L);
            q.wait_and_throw();
            const auto t1 = std::chrono::steady_clock::now();
            us_launch.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / launches);
        }
        std::sort(us_launch.begin(), us_launch.end());
        const double med = us_launch[us_launch.size() / 2];
        std::printf("RESULT moe_planes %s: %.1f us a launch (min %.1f max %.1f, %d launches x %d runs, host wall "
                    "over back-to-back launches) = %.1f GB/s of %.2f MB\n",
                    P.name, med, us_launch.front(), us_launch.back(), launches, runs, bytes_launch / (med * 1e-6) / 1e9,
                    bytes_launch / 1e6);
        /* a checksum of the output so two kernels (RADIANCE_XE_MOE_V1) can be compared by hand */
        std::vector<uint16_t> hy((size_t)top_k * P.N);
        launch(0); q.wait();
        q.memcpy(hy.data(), d_y, hy.size() * 2).wait();
        double sum = 0, asum = 0; int nz = 0;
        for (uint16_t v : hy) { const float f = bf(v); sum += f; asum += std::fabs(f); nz += f != 0.0f; }
        std::printf("CHECKSUM moe_planes %s launch 0: sum %.6g abs %.6g nonzero %d of %zu  first [%g %g %g %g]\n",
                    P.name, sum, asum, nz, hy.size(), bf(hy[0]), bf(hy[1]), bf(hy[2]), bf(hy[3]));
    }
    std::printf("RESULT moe_planes_bench %s\n", all_ok ? "DONE" : "FAIL");
    return all_ok ? 0 : 1;
}
