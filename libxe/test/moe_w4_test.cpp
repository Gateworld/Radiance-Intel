/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * moe_w4_test.cpp -- libxe's w4nl64a8h expert GEMM (moe_gemm_q through libr4d's adapter
 * r4d_rad_moe_gemm_w4a8, libxe/ports/r4d_moe.cpp xe_moe_w4nl64_gemv) against libref's arithmetic
 * (ref_moe.cpp: per 64-K weight group the sub-sum of level * activation, acc += (sub * a_scale) *
 * (e4m3 scale * 2^-13), zero rows for a dropped slot, zero columns past a narrower class), re-done on
 * the host. rad-kbench cannot run this row: its weight is libr4d's plugin-private packed plane.
 *
 * The weight is built in libr4d's stored form (libr4d/r4d_layout.cpp, moe w4): codes [N][K/2] with
 * byte b of each dword holding element b (low nibble) and b+4 (high), dtype R4D_DT_MOEW4; the E4M3
 * scale plane [N][K/64] as stored. Experts by parity: even ones in `w`, odd ones in `w_odd` (the odd
 * class narrower, to check the zero fill). 4 KiB canaries around y.
 *
 *   moe_w4_test LIBXE.so [full]    RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_args.h"

#include <dlfcn.h>

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
/* an E4M3 byte with exponent field 1..10 (no NaN, no subnormal) */
uint8_t e4(uint32_t r) { return (uint8_t)((r & 0x80u) | ((1u + (r >> 8) % 10u) << 3) | (r & 7u)); }

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

struct Class { int n, k, count; std::vector<uint8_t> code; std::vector<uint8_t> scale; };   /* code: [count][n][k] 0..15 */

/* libr4d's stored nibble order: byte b of dword d = element 8d+b (low) | 8d+4+b (high) */
std::vector<uint8_t> pack(const Class& c) {
    std::vector<uint8_t> out((size_t)c.count * c.n * c.k / 2);
    for (int e = 0; e < c.count; ++e)
        for (int r = 0; r < c.n; ++r) {
            const uint8_t* src = &c.code[((size_t)e * c.n + r) * c.k];
            uint8_t* dst = &out[((size_t)e * c.n + r) * c.k / 2];
            for (int d = 0; d < c.k / 8; ++d)
                for (int b = 0; b < 4; ++b) dst[4 * d + b] = (uint8_t)(src[8 * d + b] | (src[8 * d + 4 + b] << 4));
        }
    return out;
}

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xA7;

bool run_case(sycl::queue& q, LaunchFn fn, bool sorted_a, int parts, int tokens, int top_k, int n_expert,
              int N, int N_odd, int K, uint64_t seed) {
    Class ce{N, K, (n_expert + 1) / 2, {}, {}}, co{N_odd, K, n_expert / 2, {}, {}};
    for (Class* c : {&ce, &co}) {
        c->code.resize((size_t)c->count * c->n * c->k);
        c->scale.resize((size_t)c->count * c->n * (c->k / 64));
        for (size_t i = 0; i < c->code.size(); ++i) c->code[i] = (uint8_t)(h32(seed + c->n, i) & 15u);
        for (size_t i = 0; i < c->scale.size(); ++i) c->scale[i] = (uint8_t)(0x30u + h32(seed ^ 5, i) % 0x18u);   /* 2^-1..2^2 */
    }
    /* routing: token t slot s -> expert (t * 3 + s * 5 + seed) % n_expert; slot (0, 1) dropped */
    const int T = tokens * top_k;
    std::vector<int> ex(T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < top_k; ++s) ex[t * top_k + s] = (t == 0 && s == 1) ? -1 : (int)((t * 3 + s * 5 + seed) % n_expert);
    std::vector<int32_t> sorted, off(n_expert + 1, 0);
    for (int e = 0; e < n_expert; ++e) {
        off[e] = (int32_t)sorted.size();
        for (int i = 0; i < T; ++i) if (ex[i] == e) sorted.push_back(i);
    }
    off[n_expert] = (int32_t)sorted.size();
    while ((int)sorted.size() < T) sorted.push_back(-1);
    const int a_rows = sorted_a ? T : tokens;
    std::vector<uint8_t> a((size_t)a_rows * K);
    std::vector<float> as((size_t)a_rows * (K / 128));
    for (size_t i = 0; i < a.size(); ++i) a[i] = e4(h32(seed ^ 9, i));
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (h32(seed ^ 11, i) >> 9) * 0x1p-23f, -6);

    auto up = [&](const void* h, size_t bytes) { void* d = sycl::malloc_device(bytes ? bytes : 1, q); if (bytes) q.memcpy(d, h, bytes); return d; };
    const std::vector<uint8_t> pe = pack(ce), po = pack(co);
    void* d_a = up(a.data(), a.size());
    void* d_as = up(as.data(), as.size() * 4);
    void* d_we = up(pe.data(), pe.size());
    void* d_se = up(ce.scale.data(), ce.scale.size());
    void* d_wo = up(po.data(), po.size());
    void* d_so = up(co.scale.data(), co.scale.size());
    void* d_st = up(sorted.data(), sorted.size() * 4);
    void* d_off = up(off.data(), off.size() * 4);
    const size_t ybytes = (size_t)T * N * 2;
    uint8_t* d_yb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memset(d_yb, kPat, ybytes + 2 * kCanary).wait();

    const uint32_t W4 = (uint32_t)R4D_DT_MOEW4;
    RadTensor t[9] = {
        t2(d_a, RAD_F8E4M3, a_rows, K), t2(d_as, RAD_F32, a_rows, K / 128),
        t3(d_we, W4, ce.count, N, K), t3(d_se, RAD_F8E4M3, ce.count, N, K / 64),
        t1(d_st, RAD_I32, T), t1(d_off, RAD_I32, n_expert + 1),
        t3(d_wo, W4, co.count, N_odd, K), t3(d_so, RAD_F8E4M3, co.count, N_odd, K / 64),
        t2(d_yb + kCanary, RAD_BF16, T, N) };
    const char* ord = sorted_a ? "sorted" : "token";
    char dt[] = "w4nl64a8h";
    RadParam p[11]{};
    const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
    const long long iv[] = { tokens, N, K, n_expert, top_k, 128, N_odd, K, parts };
    for (int i = 0; i < 9; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
    p[9].key = "a_order"; p[9].kind = RAD_P_STR; p[9].sval = ord;
    p[10].key = "dtype"; p[10].kind = RAD_P_STR; p[10].sval = dt;
    RadArgs args{};
    args.t = t; args.n_t = 9; args.p = p; args.n_p = 11;
    const int rc = fn(&args, (RadStream)&q);
    q.wait_and_throw();
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), d_yb, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;

    /* libref's chain */
    bool ok = rc == RAD_OK && canary;
    double worst = 0;
    const int fy = N / parts;
    for (int i = 0; i < T && rc == RAD_OK; ++i) {
        const int flat = sorted[i];
        int e = -1, tok = 0;
        if (flat >= 0) {
            int lo = 0, hi = n_expert;
            while (lo + 1 < hi) { const int mid = (lo + hi) / 2; if (off[mid] <= i) lo = mid; else hi = mid; }
            tok = sorted_a ? i : flat / top_k;
            if (tok < a_rows) e = lo;
        }
        const Class& c = (e >= 0 && (e & 1)) ? co : ce;
        const int fc = e >= 0 ? c.n / parts : 0;
        for (int n = 0; n < N; ++n) {
            const int pp = n / fy, j = n - pp * fy;
            float acc = 0.0f;
            double mag = 0.0;
            if (e >= 0 && j < fc) {
                const int r = pp * fc + j, ent = e / 2;
                for (int g = 0; g < c.k / 64; ++g) {
                    float sub = 0.0f;
                    double smag = 0.0;
                    for (int k = g * 64; k < g * 64 + 64; ++k) {
                        const float v = kW4nlTable[c.code[((size_t)ent * c.n + r) * c.k + k]] *
                                        rad_fp8e4m3_to_f32(a[(size_t)tok * K + k]);
                        sub += v;
                        smag += std::fabs((double)v);
                    }
                    const float sa = as[(size_t)tok * (K / 128) + (g * 64) / 128];
                    const float sb = rad_fp8e4m3_to_f32(c.scale[((size_t)ent * c.n + r) * (c.k / 64) + g]) * kW4G64Scale;
                    acc += (sub * sa) * sb;
                    mag += smag * sa * sb;   /* sum of |terms|: the reordering bound below */
                }
            }
            uint16_t got;
            std::memcpy(&got, &hy[kCanary + ((size_t)i * N + n) * 2], 2);
            const double ref = bf(to_bf(acc));
            const double err = std::fabs((double)bf(got) - ref);
            const double tol = std::fabs(ref) * 0x1p-7 + mag * 64 * 0x1p-23 + 1e-30;
            worst = std::max(worst, err / tol);
            ok &= err <= tol;
        }
    }
    std::printf("  %-6s parts %d tokens %3d top_k %d experts %d N %d N_odd %d K %d: %s (rc %d), worst %.3f of tolerance, canaries %s\n",
                ord, parts, tokens, top_k, n_expert, N, N_odd, K, ok ? "PASS" : "FAIL", rc, worst, canary ? "ok" : "BROKEN");
    for (void* d : {d_a, d_as, d_we, d_se, d_wo, d_so, d_st, d_off}) sycl::free(d, q);
    sycl::free(d_yb, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc > 1 ? argv[1] : "/opt/ri/radiance_home/kernels/libxe.so";
    const bool full = argc > 2 && std::string(argv[2]) == "full";
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("moe_w4_test: dlopen %s: %s\n", lib, dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "r4d_rad_moe_gemm_w4a8");
    if (!fn) { std::printf("moe_w4_test: no r4d_rad_moe_gemm_w4a8 in %s\n", lib); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v), sycl::property::queue::in_order());
    std::printf("moe_w4_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    /* gate_up form (token rows, two parts) and down form (sorted rows, one part); a narrower odd class */
    fails += !run_case(q, fn, false, 2, 5, 2, 4, 256, 128, 256, 1);
    fails += !run_case(q, fn, true, 1, 5, 2, 4, 256, 192, 128, 2);
    fails += !run_case(q, fn, false, 1, 1, 2, 4, 128, 128, 512, 3);
    if (full) {
        /* Flash-Next's expert shapes: gate_up 2 x 640 over 2560 (parts 2), down 2560 over 640 */
        fails += !run_case(q, fn, false, 2, 1, 10, 64, 1280, 1280, 2560, 4);
        fails += !run_case(q, fn, true, 1, 1, 10, 64, 2560, 2560, 640, 5);
        fails += !run_case(q, fn, false, 2, 4, 10, 64, 1280, 1280, 2560, 6);
    }
    std::printf("moe_w4_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
