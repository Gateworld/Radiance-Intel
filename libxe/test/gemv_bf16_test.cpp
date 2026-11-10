/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, stage B, run 58).
 *
 * gemv_bf16_test.cpp -- gemm_nt over bf16 weights (xe_gemv_bf16.cpp) against a double reference, with canaries on y:
 * the sub-group GEMV at small M and the matrix-unit form at M >= 64 (on a GPU; the CPU device runs the GEMV for
 * every case), bf16 and f32 activations, bf16 and f32 outputs, a K that is not a multiple of 64 and an N that is not
 * a multiple of 64. The products of bf16 pairs are exact in f32, so the only differences are the f32 summation
 * orders (the reference's ascending k, the GEMV's butterfly, the matrix unit's tiles) and the output's rounding:
 * the tolerance is 2^-7 of the row's largest magnitude for a bf16 output and 2^-16 for an f32 one.
 *
 *   gemv_bf16_test [reps]      RI_DEVICE=cpu for the CPU device; RADIANCE_XE_GEMM_BF16_DPAS=0 forces the GEMV
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* a, RadStream s);
/* xe_gemv_bf16.cpp's router row calls the ported scatter (ports/r4d_moe.cpp), which this test never reaches: a stub
 * stands in so the test links without the MoE port and its libr4d configuration symbols. */
extern "C" int r4d_rad_router_topk_scatter(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
/* run 273 (pair 2): the paired launch behind the fused router GEMM op, r4d_moe.cpp's too */
extern "C" int r4d_rad_router_topk_scatter_quant(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

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

/* one case: a [M, K] (bf16 or f32), b [N, K] bf16, y [M, N] (bf16 or f32) */
bool run_case(sycl::queue& q, int M, int N, int K, bool a_f32, bool y_f32, uint64_t seed, int reps) {
    std::vector<uint16_t> ab((size_t)M * K), b((size_t)N * K);
    std::vector<float> af((size_t)M * K);
    for (size_t i = 0; i < ab.size(); ++i) {
        const float v = ((float)(h32(seed, i) % 2001) - 1000.0f) / 500.0f;   /* [-2, 2] */
        ab[i] = to_bf(v);
        af[i] = v;
    }
    for (size_t i = 0; i < b.size(); ++i) b[i] = to_bf(((float)(h32(seed ^ 7, i) % 2001) - 1000.0f) / 1000.0f);
    const size_t ae = a_f32 ? 4 : 2, ye = y_f32 ? 4 : 2, ybytes = (size_t)M * N * ye;
    void* da = sycl::malloc_device<uint8_t>((size_t)M * K * ae, q);
    uint16_t* db = sycl::malloc_device<uint16_t>(b.size(), q);
    uint8_t* dyb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    if (a_f32) q.memcpy(da, af.data(), af.size() * 4); else q.memcpy(da, ab.data(), ab.size() * 2);
    q.memcpy(db, b.data(), b.size() * 2);
    q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
    void* dy = dyb + kCanary;
    RadTensor t[3] = { tensor(da, a_f32 ? RAD_F32 : RAD_BF16, M, K), tensor(db, RAD_BF16, N, K), tensor(dy, y_f32 ? RAD_F32 : RAD_BF16, M, N) };
    RadParam p[2]{};
    p[0].key = "N"; p[0].kind = RAD_P_INT; p[0].ival = N;
    p[1].key = "K"; p[1].kind = RAD_P_INT; p[1].ival = K;
    RadArgs args{};
    args.t = t; args.n_t = 3; args.p = p; args.n_p = 2;
    if (xe_rad_gemm_nt_bf16(&args, (RadStream)&q) != RAD_OK) { std::printf("  launch refused\n"); return false; }
    q.wait_and_throw();
    double ms = 0;
    if (reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) xe_rad_gemm_nt_bf16(&args, (RadStream)&q);
        q.wait_and_throw();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), dyb, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;
    /* the reference in double: the activation as the kernel reads it (bf16 exactly; f32 rounded to bf16 by the
     * matrix form and read as f32 by the GEMV -- both within the tolerance of a bf16 output; an f32 output of an
     * f32 activation is compared against the f32 reading) */
    double worst = 0;
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        std::vector<double> ref((size_t)N, 0.0);
        double amax = 0;
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                const double av = a_f32 ? (double)af[(size_t)m * K + k] : (double)bf(ab[(size_t)m * K + k]);
                s += av * (double)bf(b[(size_t)n * K + k]);
            }
            ref[(size_t)n] = s;
            amax = std::max(amax, std::fabs(s));
        }
        const double tol = (y_f32 && !a_f32 ? 0x1p-16 : (y_f32 ? 0x1p-7 : 0x1p-7)) * (amax > 0 ? amax : 1.0);
        for (int n = 0; n < N; ++n) {
            const uint8_t* yp = hy.data() + kCanary + ((size_t)m * N + n) * ye;
            double got;
            if (y_f32) { float f; std::memcpy(&f, yp, 4); got = f; }
            else { uint16_t u; std::memcpy(&u, yp, 2); got = bf(u); }
            const double d = std::fabs(got - ref[(size_t)n]);
            if (d > worst) worst = d;
            if (!(d <= tol)) { if (bad < 3) std::printf("  bad [%d][%d]: got %.6g ref %.6g (tol %.3g)\n", m, n, got, ref[(size_t)n], tol); ++bad; }
        }
    }
    sycl::free(da, q); sycl::free(db, q); sycl::free(dyb, q);
    const bool ok = canary && bad == 0;
    std::printf("%s M %4d N %4d K %4d a %s y %s: worst |d| %.3g, %d bad, canaries %s%s\n", ok ? "PASS" : "FAIL", M, N, K,
                a_f32 ? "f32 " : "bf16", y_f32 ? "f32 " : "bf16", worst, bad, canary ? "ok" : "BROKEN",
                reps > 0 ? (std::string("; ") + std::to_string(ms) + " ms a launch").c_str() : "");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 0;
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v), { sycl::property::queue::in_order() });
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    bool ok = true;
    /* small M (the GEMV), then the matrix-unit sizes: a full block, a block plus a tail, two blocks; an N and a K
     * off the tile multiples (N 200 = 3 tiles + 8; K 1000 = 15 groups + 40) */
    /* run 209: {1, 33, 2560} and {2, 77, 2560} give the K = 2560 instances (RADIANCE_XE_GEMV_KC=1) odd N at decode M */
    const int cases[][3] = { {1, 96, 256}, {3, 200, 1000}, {64, 128, 2560}, {70, 200, 1000}, {128, 1024, 2048}, {130, 96, 640},
                             {1, 33, 2560}, {2, 77, 2560} };
    for (const auto& c : cases) {
        ok &= run_case(q, c[0], c[1], c[2], false, false, 11, 0);
        ok &= run_case(q, c[0], c[1], c[2], true, false, 13, 0);
        ok &= run_case(q, c[0], c[1], c[2], false, true, 17, 0);
        ok &= run_case(q, c[0], c[1], c[2], true, true, 19, 0);
    }
    if (reps > 0) {   /* a prefill-shaped timing: the GDN a|b projection's 2,048 rows */
        ok &= run_case(q, 2048, 1024, 2048, false, false, 23, reps);
        ok &= run_case(q, 2048, 4096, 2048, true, false, 29, reps);
    }
    std::printf("RESULT gemv_bf16_test %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
