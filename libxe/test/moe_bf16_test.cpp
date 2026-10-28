/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, the bottleneck brief's #11).
 *
 * moe_bf16_test.cpp -- libxe's moe_gemm at bf16 (r4d_rad_moe_gemm_bf16, libxe/ports/r4d_moe.cpp) at decode
 * sizes against a double-precision reference re-done on the host: per sorted row r of expert e (off[e] <= r <
 * off[e+1]), y[r, n] = sum_k a[tok, k] * w_e[n, k] with tok = sorted[r] / top_k (token rows) or r (sorted rows);
 * a dropped slot or a row past the live count is zero; a row below off[0] (another call's, the trailing-segment
 * form a layer's protected experts use) is not touched. A call of at most 256 rows takes the GEMV form under
 * RADIANCE_XE_MOE_BF16_GEMV=1 and the WMMA form by default; both pass the same check. 4 KiB canaries
 * around y; untouched rows keep their fill.
 *
 *   moe_bf16_test LIBXE.so      RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <algorithm>
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
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
/* a bf16 in [-2, 2) with a few mantissa bits, so products and sums are well inside f32 */
uint16_t rbf(uint32_t r) { return to_bf(((float)(r % 4096u) / 1024.0f) - 2.0f); }

RadTensor t2(void* p, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t;
}
RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t;
}
RadTensor t3(void* p, uint32_t dt, int64_t e, int64_t r, int64_t c) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 3;
    t.shape[0] = e; t.shape[1] = r; t.shape[2] = c; t.stride[0] = r * c; t.stride[1] = c; t.stride[2] = 1; return t;
}

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xA7;

/* `skip`: the first `skip` sorted rows belong to another call (off[0] = skip): routed to no expert here */
bool run_case(sycl::queue& q, LaunchFn fn, bool sorted_a, int tokens, int top_k, int n_expert, int N, int K,
              int skip, uint64_t seed) {
    const int T = tokens * top_k;
    /* routing: token t slot s -> expert (t * 3 + s * 5 + seed) % n_expert; slot (0, 1) dropped */
    std::vector<int> ex(T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < top_k; ++s) ex[t * top_k + s] = (t == 0 && s == 1) ? -1 : (int)((t * 3 + s * 5 + seed) % n_expert);
    std::vector<int32_t> sorted, off(n_expert + 1, 0);
    for (int i = 0; i < skip; ++i) sorted.push_back(i % T);   /* another call's rows */
    for (int e = 0; e < n_expert; ++e) {
        off[e] = (int32_t)sorted.size();
        for (int i = 0; i < T; ++i) if (ex[i] == e) sorted.push_back(i);
    }
    off[n_expert] = (int32_t)sorted.size();
    const int TT = (int)sorted.size() + 2;   /* two dropped slots past the live count */
    while ((int)sorted.size() < TT) sorted.push_back(-1);
    /* the launcher checks the activation against y's row count (TT here, with the skipped and dropped rows) */
    const int a_rows = sorted_a ? TT : std::max(tokens, (TT + top_k - 1) / top_k);
    std::vector<uint16_t> a((size_t)a_rows * K), w((size_t)n_expert * N * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = rbf(h32(seed ^ 9, i));
    for (size_t i = 0; i < w.size(); ++i) w[i] = rbf(h32(seed ^ 3, i));

    auto up = [&](const void* h, size_t bytes) { void* d = sycl::malloc_device(bytes ? bytes : 1, q); if (bytes) q.memcpy(d, h, bytes); return d; };
    void* d_a = up(a.data(), a.size() * 2);
    void* d_w = up(w.data(), w.size() * 2);
    void* d_st = up(sorted.data(), sorted.size() * 4);
    void* d_off = up(off.data(), off.size() * 4);
    const size_t ybytes = (size_t)TT * N * 2;
    uint8_t* d_yb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memset(d_yb, kPat, ybytes + 2 * kCanary).wait();

    RadTensor t[5] = { t2(d_a, RAD_BF16, a_rows, K), t3(d_w, RAD_BF16, n_expert, N, K),
                       t1(d_st, RAD_I32, TT), t1(d_off, RAD_I32, n_expert + 1), t2(d_yb + kCanary, RAD_BF16, TT, N) };
    const char* ord = sorted_a ? "sorted" : "token";
    char dt[] = "bf16";
    RadParam p[6]{};
    const char* ik[] = { "N", "K", "n_expert", "top_k" };
    const long long iv[] = { N, K, n_expert, top_k };
    for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
    p[4].key = "a_order"; p[4].kind = RAD_P_STR; p[4].sval = ord;
    p[5].key = "dtype"; p[5].kind = RAD_P_STR; p[5].sval = dt;
    RadArgs args{};
    args.t = t; args.n_t = 5; args.p = p; args.n_p = 6;
    const int rc = fn(&args, (RadStream)&q);
    q.wait_and_throw();
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), d_yb, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;

    bool ok = rc == RAD_OK && canary;
    double worst = 0;
    int untouched_bad = 0;
    const int live = off[n_expert];
    for (int i = 0; i < TT && rc == RAD_OK; ++i) {
        for (int n = 0; n < N; ++n) {
            const uint8_t* yp = &hy[kCanary + ((size_t)i * N + n) * 2];
            if (i < off[0]) { untouched_bad += !(yp[0] == kPat && yp[1] == kPat); continue; }
            uint16_t got;
            std::memcpy(&got, yp, 2);
            double ref = 0.0, mag = 0.0;
            const int flat = i < live ? sorted[i] : -1;
            if (flat >= 0) {
                int lo = 0, hi = n_expert;
                while (lo + 1 < hi) { const int mid = (lo + hi) / 2; if (off[mid] <= i) lo = mid; else hi = mid; }
                const int tok = sorted_a ? i : flat / top_k;
                for (int k = 0; k < K; ++k) {
                    const double v = (double)bf(a[(size_t)tok * K + k]) * bf(w[((size_t)lo * N + n) * K + k]);
                    ref += v;
                    mag += std::fabs(v);
                }
            }
            /* one bf16 rounding of the result plus f32 accumulation in any order */
            const double err = std::fabs((double)bf(got) - ref);
            const double tol = std::fabs(ref) * 0x1p-8 + mag * K * 0x1p-24 + 1e-30;
            worst = std::max(worst, err / tol);
            ok &= err <= tol;
        }
    }
    ok &= untouched_bad == 0;
    std::printf("  %-6s tokens %2d top_k %2d experts %3d N %4d K %4d skip %d: %s (rc %d), worst %.3f of tolerance, "
                "canaries %s, rows below off[0] %s\n", ord, tokens, top_k, n_expert, N, K, skip, ok ? "PASS" : "FAIL",
                rc, worst, canary ? "ok" : "BROKEN", untouched_bad ? "WRITTEN" : "untouched");
    for (void* d : {d_a, d_w, d_st, d_off}) sycl::free(d, q);
    sycl::free(d_yb, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc > 1 ? argv[1] : "/opt/ri/radiance_home/kernels/libxe.so";
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("moe_bf16_test: dlopen %s: %s\n", lib, dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "r4d_rad_moe_gemm_bf16");
    if (!fn) { std::printf("moe_bf16_test: no r4d_rad_moe_gemm_bf16 in %s\n", lib); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v), sycl::property::queue::in_order());
    const char* f = std::getenv("RADIANCE_XE_MOE_BF16_GEMV");
    std::printf("moe_bf16_test: %s, the %s form\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                f && *f == '1' ? "GEMV" : "WMMA");
    int fails = 0;
    fails += !run_case(q, fn, false, 2, 4, 16, 128, 256, 0, 1);
    fails += !run_case(q, fn, true, 3, 2, 16, 192, 128, 0, 2);
    fails += !run_case(q, fn, false, 1, 8, 32, 256, 512, 5, 3);   /* the trailing-segment form */
    fails += !run_case(q, fn, true, 4, 10, 64, 640, 2560, 7, 4);  /* Flash-Next's shapes at decode */
    /* the protected experts' call as the engine makes it (rad_block_moe_fp8.h): a few experts (n_expert = the
     * protected count), more rows than 1.5 n_expert, the other calls' rows below off[0] -- the case the decode
     * band missed (M33) */
    fails += !run_case(q, fn, false, 2, 10, 4, 256, 512, 12, 5);
    fails += !run_case(q, fn, true, 2, 10, 4, 512, 256, 12, 6);
    std::printf("moe_bf16_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
