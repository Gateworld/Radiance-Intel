/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 123, owner note 151 C).
 *
 * logits_i8_test.cpp -- libxe's int8 lm_head (xe_rad_logits_gemm_i8, libxe/xe_gemv_bf16.cpp): the 4-row form against
 * the rows form (RADIANCE_XE_LOGITS_R=2 / 4, R weight rows a sub-group), which must give the same bytes, and a
 * microbench of each (time a call, and an FNV-1a hash of the logits). Rows of K int8 codes then K/128 bf16 scales,
 * padded to 16 bytes (r4d_fp8::lm_row_bytes); x bf16 [M, K]; y f32 [M, V]. 1 row is checked too (it keeps its own
 * form). Form 0 is also checked against a double-precision host reference on the first 256 vocabulary rows.
 *
 *   logits_i8_test LIBXE.so [V=155136] [K=2048] [ITERS=20]     RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <chrono>
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
uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t*)p)[i]; h *= 1099511628211ull; }
    return h;
}

RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}

/* one call; returns its rc */
int call(sycl::queue& q, LaunchFn fn, uint16_t* dx, uint8_t* dw, float* dy, int M, long long V, long long K) {
    RadTensor t[3];
    t[0] = tn(dx, RAD_BF16, { M, K });
    t[1] = tn(dw, RAD_I8, { V, K });
    t[2] = tn(dy, RAD_F32, { M, V });
    RadParam p[2]{};
    p[0].key = "n_vocab"; p[0].kind = RAD_P_INT; p[0].ival = V;
    p[1].key = "n_embd"; p[1].kind = RAD_P_INT; p[1].ival = K;
    RadArgs a{};
    a.t = t; a.n_t = 3; a.p = p; a.n_p = 2;
    return fn(&a, (RadStream)&q);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: logits_i8_test LIBXE.so [V] [K] [ITERS]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_logits_gemm_i8");
    if (!fn) { std::printf("no xe_rad_logits_gemm_i8\n"); return 2; }
    const char* dv = std::getenv("RI_DEVICE");
    sycl::queue q = (dv && !std::strcmp(dv, "cpu")) ? sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order())
                                                    : sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const long long V = argc > 2 ? std::atoll(argv[2]) : 155136, K = argc > 3 ? std::atoll(argv[3]) : 2048;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 20;
    std::printf("logits_i8_test on %s: V %lld K %lld\n", q.get_device().get_info<sycl::info::device::name>().c_str(), V, K);
    const long long w_ld = (K + 2 * (K / 128) + 15) & ~15LL;
    std::vector<uint8_t> w((size_t)(V * w_ld), 0);
    for (long long n = 0; n < V; ++n) {
        uint8_t* row = w.data() + n * w_ld;
        for (long long k = 0; k < K; ++k) row[k] = (uint8_t)(h32(1, (uint64_t)(n * K + k)) & 0xffu);
        uint16_t* sc = (uint16_t*)(row + K);
        for (long long g = 0; g < K / 128; ++g) sc[g] = to_bf(0.0005f + (float)(h32(2, (uint64_t)(n * 64 + g)) % 1000u) * 1e-6f);
    }
    const int MM = 4;
    std::vector<uint16_t> x((size_t)(MM * K));
    for (size_t i = 0; i < x.size(); ++i) x[i] = to_bf(((float)(h32(3, i) % 2001u) - 1000.0f) * 1e-3f);
    uint8_t* dw = sycl::malloc_device<uint8_t>(w.size(), q);
    uint16_t* dx = sycl::malloc_device<uint16_t>(x.size(), q);
    float* dy = sycl::malloc_device<float>((size_t)(MM * V), q);
    q.memcpy(dw, w.data(), w.size());
    q.memcpy(dx, x.data(), x.size() * 2);
    q.wait();
    int fails = 0;
    for (int M : { 1, 2, 4 }) {
        std::vector<float> ref;
        uint64_t ref_hash = 0;
        for (int form : { 0, 2, 4, 12 }) {
            /* form 0 set explicitly: unset is the default R=2 since owner note 155; 12 the split form (run 219) */
            { char v[4]; std::snprintf(v, sizeof v, "%d", form); setenv("RADIANCE_XE_LOGITS_R", v, 1); }
            q.memset(dy, 0xA7, (size_t)(MM * V) * 4).wait();
            const int rc = call(q, fn, dx, dw, dy, M, V, K);
            q.wait_and_throw();
            std::vector<float> y((size_t)(M * V));
            q.memcpy(y.data(), dy, y.size() * 4).wait();
            const uint64_t hh = fnv(y.data(), y.size() * 4);
            /* time: ITERS calls after one warm-up */
            call(q, fn, dx, dw, dy, M, V, K);
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) call(q, fn, dx, dw, dy, M, V, K);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            bool ok = rc == RAD_OK;
            if (form == 0) {
                ref = y; ref_hash = hh;
                /* the host reference on the first 256 rows: the format's definition in double */
                double worst = 0;
                for (int m = 0; m < M; ++m)
                    for (long long n = 0; n < 256 && n < V; ++n) {
                        const uint8_t* row = w.data() + n * w_ld;
                        const uint16_t* sc = (const uint16_t*)(row + K);
                        double s = 0, mag = 0;
                        for (long long g = 0; g < K / 128; ++g) {
                            double gs = 0;
                            for (long long k = g * 128; k < g * 128 + 128; ++k) gs += (double)bf(x[(size_t)(m * K + k)]) * (double)(int8_t)row[k];
                            s += gs * bf(sc[g]);
                            mag += std::fabs(gs * bf(sc[g]));
                        }
                        const double d = std::fabs((double)y[(size_t)(m * V + n)] - s) / (mag + 1e-30);
                        if (d > worst) worst = d;
                    }
                if (worst > 1e-4) ok = false;   /* f32 sums of K products against double */
                std::printf("%s M %d form 0 (4-row): %.1f us a call, hash %016llx, host check worst |d|/sum|t| %.2g\n",
                            ok ? "PASS" : "FAIL", M, us, (unsigned long long)hh, worst);
            } else {
                const bool same = hh == ref_hash && std::memcmp(y.data(), ref.data(), y.size() * 4) == 0;
                ok = ok && same;
                std::printf("%s M %d form %d (rows): %.1f us a call, hash %016llx, %s form 0's bytes\n", ok ? "PASS" : "FAIL",
                            M, form, us, (unsigned long long)hh, same ? "=" : "DIFFERS FROM");
            }
            fails += !ok;
        }
    }
    unsetenv("RADIANCE_XE_LOGITS_R");
    sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q);
    std::printf("%s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
