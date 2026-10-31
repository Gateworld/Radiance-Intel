/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * gram_accum_test.cpp -- owner note 135 item 3 (run 111): libxe's gram_accum (libr4d's unit, ported) against a host
 * reference in double, h += dequant(a)^T dequant(a) over E4M3 rows with an f32 scale per (row, 128 columns), the
 * accumulator READ and added to (it starts non-zero here). 4 KiB canaries on both sides of h. Small sizes for the CPU
 * SYCL device (amd-host), the expert shapes for a card (owner note 3: the host reference stays small on intel-host).
 *
 *   xe_gram_accum_test LIBXE.so [full]   RI_DEVICE=cpu for the CPU device
 * Prints one line a case and "gram_accum_test: PASS / FAIL".
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <cmath>
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
/* an E4M3 byte with exponent field 1..10 (no NaN), and its value */
uint8_t e4(uint32_t r) { return (uint8_t)((r & 0x80u) | ((1u + (r >> 8) % 10u) << 3) | (r & 7u)); }
double e4m3f(uint8_t b) {
    const int s = b >> 7, ex = (b >> 3) & 15, m = b & 7;
    const double v = ex ? std::ldexp(1.0 + m / 8.0, ex - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}

constexpr int kCanary = 1024;                 /* floats a side: 4 KiB */
constexpr uint32_t kPat = 0x7fa5a5a5u;        /* a NaN pattern no store writes */

bool run_case(LaunchFn fn, sycl::queue& q, int M, int K, uint64_t seed) {
    const int G = 128, ng = (K + G - 1) / G;
    std::vector<uint8_t> a((size_t)M * K);
    std::vector<float> s((size_t)M * ng), h0((size_t)K * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = e4(h32(seed, i));
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.002f + (float)(h32(seed + 1, i) & 1023) / 65536.0f;
    for (size_t i = 0; i < h0.size(); ++i) h0[i] = (float)((int)(h32(seed + 2, i) % 2001) - 1000) / 1000.0f;
    uint8_t* da = sycl::malloc_device<uint8_t>(a.size() + 4, q);
    float* ds = sycl::malloc_device<float>(s.size(), q);
    float* dh = sycl::malloc_device<float>(h0.size() + 2 * kCanary, q);
    std::vector<uint32_t> pat(kCanary, kPat);
    q.memcpy(da, a.data(), a.size()).wait();
    q.memcpy(ds, s.data(), s.size() * 4).wait();
    q.memcpy(dh, pat.data(), kCanary * 4).wait();
    q.memcpy(dh + kCanary, h0.data(), h0.size() * 4).wait();
    q.memcpy(dh + kCanary + h0.size(), pat.data(), kCanary * 4).wait();

    RadTensor t[3] = {};
    t[0].data = da; t[0].dtype = RAD_F8E4M3; t[0].rank = 2; t[0].shape[0] = M; t[0].shape[1] = K; t[0].stride[0] = K; t[0].stride[1] = 1;
    t[1].data = ds; t[1].dtype = RAD_F32; t[1].rank = 2; t[1].shape[0] = M; t[1].shape[1] = ng; t[1].stride[0] = ng; t[1].stride[1] = 1;
    t[2].data = dh + kCanary; t[2].dtype = RAD_F32; t[2].rank = 2; t[2].shape[0] = K; t[2].shape[1] = K; t[2].stride[0] = K; t[2].stride[1] = 1;
    RadParam p[3]{};
    p[0].key = "n"; p[0].kind = RAD_P_INT; p[0].ival = K;
    p[1].key = "group"; p[1].kind = RAD_P_INT; p[1].ival = G;
    char dt[] = "fp8a8";
    p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = dt;
    RadArgs args{};
    args.t = t; args.n_t = 3; args.p = p; args.n_p = 3;
    const int rc = fn(&args, (RadStream)&q);
    q.wait_and_throw();
    std::vector<float> out(h0.size() + 2 * kCanary);
    q.memcpy(out.data(), dh, out.size() * 4).wait();
    sycl::free(da, q); sycl::free(ds, q); sycl::free(dh, q);
    if (rc != RAD_OK) { std::printf("  M %d K %d: launch refused rc %d\n", M, K, rc); return false; }
    bool canary = true;
    for (int i = 0; i < kCanary; ++i) {
        uint32_t lo, hi;
        std::memcpy(&lo, &out[(size_t)i], 4);
        std::memcpy(&hi, &out[(size_t)kCanary + h0.size() + i], 4);
        canary &= lo == kPat && hi == kPat;
    }
    /* the reference in double; the bound: f32 accumulation over M products of O(|x1 x2|) terms */
    std::vector<double> x((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) x[(size_t)m * K + k] = e4m3f(a[(size_t)m * K + k]) * (double)s[(size_t)m * ng + k / G];
    double worst = 0;
    long long bad = 0;
    for (int k1 = 0; k1 < K; ++k1)
        for (int k2 = 0; k2 < K; ++k2) {
            double acc = 0, mag = 0;
            for (int m = 0; m < M; ++m) {
                const double v = x[(size_t)m * K + k1] * x[(size_t)m * K + k2];
                acc += v; mag += std::fabs(v);
            }
            const double ref = (double)h0[(size_t)k1 * K + k2] + acc;
            const double got = out[(size_t)kCanary + (size_t)k1 * K + k2];
            const double tol = (M + 2) * std::ldexp(mag + std::fabs(ref), -23) + 1e-30;
            const double d = std::fabs(got - ref) / tol;
            if (!(d <= 1.0)) ++bad;
            if (d > worst || d != d) worst = d;
        }
    const bool ok = canary && bad == 0;
    std::printf("  M %4d K %4d: %s -- %lld of %d elements past the bound, worst %.3f of it; canaries %s\n", M, K,
                ok ? "PASS" : "FAIL", bad, K * K, worst, canary ? "ok" : "BROKEN");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_gram_accum_test LIBXE.so [full]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("gram_accum_test: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "r4d_rad_gram_accum");
    if (!fn) { std::printf("gram_accum_test: no r4d_rad_gram_accum in %s\n", argv[1]); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q{ cpu ? sycl::device{ sycl::cpu_selector_v } : sycl::device{ sycl::gpu_selector_v },
                   sycl::property_list{ sycl::property::queue::in_order{} } };
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("gram_accum_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    /* M not a multiple of the 32-row stage, K not a multiple of the 64-wide tile (K % 4 == 0 is the contract) */
    const int small[][2] = { { 1, 128 }, { 37, 256 }, { 64, 196 }, { 100, 640 } };
    for (auto& c : small) fails += !run_case(fn, q, c[0], c[1], 0x51 + (uint64_t)c[1]);
    if (argc > 2 && !std::strcmp(argv[2], "full")) {
        /* one rank's expert shapes: gate_up's input (2560) at a short pass, down's (256) at a long one */
        fails += !run_case(fn, q, 64, 2560, 0x61);
        fails += !run_case(fn, q, 1000, 256, 0x62);
    }
    std::printf("gram_accum_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
