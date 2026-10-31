/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * gemv_i8_test.cpp -- libxe's trunk W8A8 GEMV (xe_gemv_i8.cpp, through its RadArgs adapter) against
 * libref's quant_dot (libref/ref_gemm.h: per 128-K group an exact sub-sum, acc += (sub * sa) * sb in
 * ascending group order, one bf16 rounding), re-done here on the host. 4 KiB canaries around y.
 *
 *   gemv_i8_test [full]    RI_DEVICE=cpu for the CPU SYCL device; `full` adds Flash-Next's trunk
 *                          shapes at M=1 and times them (GB/s of weight bytes)
 * Exit 0 only if every case is within tolerance and every canary holds.
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int xe_rad_gemm_i8a8_gemv(const RadArgs* a, RadStream s);

namespace {

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t to_bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
RadTensor tensor(void* p, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xC5;

struct Result { bool ok; double worst; double ms; };

Result run_case(sycl::queue& q, int M, int N, int K, uint64_t seed, int reps) {
    const int G = K / 128;
    std::vector<int8_t> a((size_t)M * K), b((size_t)N * K);
    std::vector<float> as((size_t)M * G);
    std::vector<uint16_t> bs((size_t)N * G);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(h32(seed, i) % 255) - 127);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (int8_t)((int)(h32(seed ^ 7, i) % 255) - 127);
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (h32(seed ^ 9, i) >> 9) * 0x1p-23f, -9);
    for (size_t i = 0; i < bs.size(); ++i) bs[i] = to_bf(std::ldexp(1.0f + (h32(seed ^ 11, i) >> 9) * 0x1p-23f, -12));

    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    int8_t* db = sycl::malloc_device<int8_t>(b.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    uint16_t* dbs = sycl::malloc_device<uint16_t>(bs.size(), q);
    const size_t ybytes = (size_t)M * N * 2;
    uint8_t* dyb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memcpy(da, a.data(), a.size()); q.memcpy(db, b.data(), b.size());
    q.memcpy(das, as.data(), as.size() * 4); q.memcpy(dbs, bs.data(), bs.size() * 2);
    q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
    uint16_t* dy = (uint16_t*)(dyb + kCanary);

    RadTensor t[5] = { tensor(da, RAD_I8, M, K), tensor(das, RAD_F32, M, G), tensor(db, RAD_I8, N, K),
                       tensor(dbs, RAD_BF16, N, G), tensor(dy, RAD_BF16, M, N) };
    char dts[] = "i8a8h";
    RadParam p[5]{};
    const char* keys[4] = { "M", "N", "K", "group" };
    const long long vals[4] = { M, N, K, 128 };
    for (int i = 0; i < 4; ++i) { p[i].key = keys[i]; p[i].kind = RAD_P_INT; p[i].ival = vals[i]; }
    p[4].key = "dtype"; p[4].kind = RAD_P_STR; p[4].sval = dts;
    RadArgs args{};
    args.t = t; args.n_t = 5; args.p = p; args.n_p = 5;
    if (xe_rad_gemm_i8a8_gemv(&args, (RadStream)&q) != RAD_OK) { std::printf("  launch refused\n"); return {false, 0, 0}; }
    q.wait_and_throw();
    double ms = 0;
    if (reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) xe_rad_gemm_i8a8_gemv(&args, (RadStream)&q);
        q.wait_and_throw();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), dyb, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;

    /* libref's quant_dot, element by element; the comparison allows the reordered group sum */
    double worst = 0;
    bool ok = canary;
    const int rows_checked = N < 512 ? N : 512;
    /* run 110 (owner note 129 item 3): the GEMV's own order -- four group lanes, lane j summing groups j, j + 4, ...,
     * then (acc0 + acc1) + (acc2 + acc3) -- emulated; y elements whose bits differ from it are counted, and a hash
     * of all of y is printed so two runs (RADIANCE_XE_GEMV_G4 unset / set) can be compared */
    long long emu_diff = 0;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < ybytes; ++i) hash = (hash ^ hy[kCanary + i]) * 1099511628211ull;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < rows_checked; ++j) {
            const int n = (int)((uint64_t)j * N / rows_checked);
            float acc = 0.0f, lane[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            double mag = 0.0;
            for (int g = 0; g < G; ++g) {
                float sub = 0.0f;
                for (int k = g * 128; k < g * 128 + 128; ++k)
                    sub += (float)a[(size_t)m * K + k] * (float)b[(size_t)n * K + k];
                const float term = (sub * as[(size_t)m * G + g]) * bf(bs[(size_t)n * G + g]);
                acc += term;
                lane[g % 4] += term;
                mag += std::fabs((double)term);
            }
            uint16_t got;
            std::memcpy(&got, &hy[kCanary + ((size_t)m * N + n) * 2], 2);
            emu_diff += got != to_bf((lane[0] + lane[1]) + (lane[2] + lane[3]));
            const double err = std::fabs((double)bf(got) - (double)bf(to_bf(acc)));
            /* one bf16 ulp of the result plus the reordered f32 sum's bound */
            const double tol = std::fabs((double)bf(to_bf(acc))) * 0x1p-7 + mag * G * 0x1p-23 + 1e-30;
            worst = std::max(worst, err / tol);
            ok &= err <= tol;
        }
    std::printf("  M %d N %5d K %5d: %s, worst %.3f of tolerance, canaries %s%s; %lld of %d checked differ from the "
                "GEMV's order; y hash %016llx\n", M, N, K, ok ? "PASS" : "FAIL", worst, canary ? "ok" : "BROKEN",
                reps ? "" : "", emu_diff, M * rows_checked, (unsigned long long)hash);
    if (reps) std::printf("    %.4f ms, %.1f GB/s of weight\n", ms, (double)N * K * (1 + 2.0 / 128) / (ms * 1e6));
    sycl::free(da, q); sycl::free(db, q); sycl::free(das, q); sycl::free(dbs, q); sycl::free(dyb, q);
    return {ok, worst, ms};
}

}  // namespace

int main(int argc, char** argv) {
    const bool full = argc > 1 && std::string(argv[1]) == "full";
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  sycl::property::queue::in_order());
    std::printf("gemv_i8_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    /* small, odd N, M 1..6 (the 4-row pass and its remainder) */
    /* milestone 3 stage B: M >= 64 takes the matrix-unit form on the GPU (full 64-row blocks, the GEMV for the
     * tail rows): 64 exactly, 70 (one block + 6 tail rows), 128 at an N that is not a multiple of 64 */
    const int small[][3] = { {1, 40, 256}, {2, 77, 384}, {4, 64, 1024}, {5, 33, 640}, {6, 130, 2560},
                             {64, 130, 2560}, {70, 96, 1024}, {128, 200, 384} };
    for (auto& c : small) fails += !run_case(q, c[0], c[1], c[2], 0x1234 + c[1], 0).ok;
    if (full) {
        /* Flash-Next's trunk at batch 1 (round 2 M1's shapes) */
        const int big[][2] = { {16384, 2560}, {2560, 6144}, {12288, 2560}, {512, 2560}, {1280, 2560}, {2560, 640} };
        for (auto& c : big) fails += !run_case(q, 1, c[0], c[1], 0x77 + c[0], 20).ok;
        fails += !run_case(q, 4, 12288, 2560, 0x99, 10).ok;
        /* milestone 3 stage B: a prefill-shaped block on the matrix unit (256 rows: the reference check is
         * 256 x 512 columns x K on the host) */
        fails += !run_case(q, 256, 12288, 2560, 0xa1, 5).ok;
        fails += !run_case(q, 256, 2560, 6144, 0xa2, 5).ok;
    }
    std::printf("gemv_i8_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
