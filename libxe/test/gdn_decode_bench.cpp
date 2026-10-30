/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, run 101).
 *
 * gdn_decode_bench.cpp -- device timing of libxe's decode GDN update at one rank's decode shape (owner note 108
 * item 1): the fused conv + recurrent update + gated norm + quantiser (gdn_conv_recurrent_update, what a decode
 * step runs), and the recurrent update alone with q / k / v given (gdn_recurrent_update), so the convolution's share
 * shows. rad-kbench's recipe cannot bind this op on the container and its --bench segfaults on this backend (run 101,
 * GK1 / GK2), so this dlopens libxe.so as moe_planes_bench does: synthetic tensors, back-to-back launches, the host
 * wall a launch, and a checksum of the output so two forms of the kernel can be compared.
 *
 *   gdn_decode_bench LIBXE.so [launches=200] [runs=5]
 *   BENCH_T=tokens a sequence (1: MTP off; 4: the 4-row verify), BENCH_H / BENCH_HG = value / key heads a rank
 *   (16 / 8), BENCH_N = sequences (1).
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "r4d_args.h"

#include <dlfcn.h>

#include <algorithm>
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
uint16_t bfb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
float unit(uint64_t a, uint64_t i) { return ((float)(h32(a, i) & 0xffff) / 65535.0f) * 2.0f - 1.0f; }

RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = (int)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t st = 1;
    for (int d = t.rank - 1; d >= 0; --d) { t.stride[d] = st; st *= t.shape[d]; }
    return t;
}
int env_i(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gdn_decode_bench LIBXE.so [launches] [runs]\n"); return 2; }
    const int launches = argc > 2 ? std::atoi(argv[2]) : 200;
    const int runs = argc > 3 ? std::atoi(argv[3]) : 5;
    const int T = env_i("BENCH_T", 1), H = env_i("BENCH_H", 16), Hg = env_i("BENCH_HG", 8), N = env_i("BENCH_N", 1);
    constexpr int K = 128, V = 128, W = 4;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("gdn_decode_bench: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn cr = (LaunchFn)dlsym(h, "r4d_rad_gdn_conv_recurrent_update");
    LaunchFn ru = (LaunchFn)dlsym(h, "r4d_rad_gdn_recurrent_update");
    if (!cr || !ru) { std::printf("gdn_decode_bench: the GDN entry points are missing from %s\n", argv[1]); return 2; }
    sycl::queue q(sycl::device(sycl::gpu_selector_v), { sycl::property::queue::in_order() });
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s; N %d sequences x T %d tokens, H %d value / Hg %d key heads\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), N, T, H, Hg);

    const int TT = N * T;                                  /* rows */
    const int conv_dim = 2 * Hg * K + H * V;
    const int slen = T + (W - 2);                          /* conv state: the window the fold needs */
    const int nstate = N * T;                              /* a slot per candidate (per_candidate) */
    auto upload_bf = [&](size_t n, uint64_t seed, float s) {
        std::vector<uint16_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = bfb(unit(seed, i) * s);
        uint16_t* d = sycl::malloc_device<uint16_t>(n, q);
        q.memcpy(d, v.data(), n * 2).wait();
        return d;
    };
    auto upload_f = [&](size_t n, uint64_t seed, float s, float o) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = unit(seed, i) * s + o;
        float* d = sycl::malloc_device<float>(n, q);
        q.memcpy(d, v.data(), n * 4).wait();
        return d;
    };
    auto upload_i = [&](const std::vector<int32_t>& v) {
        int32_t* d = sycl::malloc_device<int32_t>(v.size(), q);
        q.memcpy(d, v.data(), v.size() * 4).wait();
        return d;
    };
    uint16_t* x   = upload_bf((size_t)TT * conv_dim, 1, 1.0f);
    uint16_t* cw  = upload_bf((size_t)conv_dim * W, 2, 0.5f);
    uint16_t* cb  = upload_bf((size_t)conv_dim, 3, 0.1f);
    uint16_t* cst = upload_bf((size_t)N * conv_dim * slen, 4, 1.0f);
    uint16_t* ta  = upload_bf((size_t)TT * H, 5, 1.0f);
    uint16_t* tb  = upload_bf((size_t)TT * H, 6, 1.0f);
    float* alog   = upload_f((size_t)H, 7, 0.5f, 0.0f);
    float* dtb    = upload_f((size_t)H, 8, 0.5f, 0.0f);
    float* state  = upload_f((size_t)nstate * H * V * K, 9, 0.05f, 0.0f);
    uint16_t* z   = upload_bf((size_t)TT * H * V, 10, 1.0f);
    float* nw     = upload_f((size_t)V, 11, 0.2f, 1.0f);
    uint16_t* qq  = upload_bf((size_t)TT * Hg * K, 12, 1.0f);
    uint16_t* kk  = upload_bf((size_t)TT * Hg * K, 13, 1.0f);
    uint16_t* vv  = upload_bf((size_t)TT * H * V, 14, 1.0f);
    std::vector<int32_t> hcu(N + 1), hsidx((size_t)N * T), hcidx(N);
    for (int n = 0; n <= N; ++n) hcu[n] = n * T;
    for (int i = 0; i < N * T; ++i) hsidx[i] = i;
    for (int n = 0; n < N; ++n) hcidx[n] = n;
    int32_t* cu = upload_i(hcu);
    int32_t* sidx = upload_i(hsidx);
    int32_t* cidx = upload_i(hcidx);
    uint16_t* o  = sycl::malloc_device<uint16_t>((size_t)TT * H * V, q);
    int8_t* oq   = sycl::malloc_device<int8_t>((size_t)TT * H * V, q);
    float* os    = sycl::malloc_device<float>((size_t)TT * H, q);

    RadParam p[9]{};
    const char* ik[] = { "head_k", "head_v", "n_head_v", "conv_width" };
    const long long iv[] = { K, V, H, W };
    for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
    p[4].key = "scale"; p[4].kind = RAD_P_F64; p[4].dval = 1.0 / std::sqrt((double)K);
    p[5].key = "eps"; p[5].kind = RAD_P_F64; p[5].dval = 1e-6;
    p[6].key = "state_form"; p[6].kind = RAD_P_STR; p[6].sval = "per_candidate";
    p[7].key = "softplus_thr"; p[7].kind = RAD_P_F64; p[7].dval = 20.0;
    p[8].key = "l2_eps"; p[8].kind = RAD_P_F64; p[8].dval = (double)1e-6f;

    RadTensor tcst = tn(cst, RAD_BF16, { N, conv_dim, slen });
    RadTensor tst = tn(state, RAD_F32, { nstate, H, V, K });
    RadTensor tsidx = tn(sidx, RAD_I32, { N, T });
    RadTensor tnull{};
    RadTensor tcr[18] = {
        tn(x, RAD_BF16, { TT, conv_dim }), tn(cw, RAD_BF16, { conv_dim, W }), tn(cb, RAD_BF16, { conv_dim }), tcst,
        tn(cidx, RAD_I32, { N }), tn(ta, RAD_BF16, { TT, H }), tn(tb, RAD_BF16, { TT, H }), tn(alog, RAD_F32, { H }),
        tn(dtb, RAD_F32, { H }), tst, tsidx, tnull, tn(cu, RAD_I32, { N + 1 }), tn(z, RAD_BF16, { TT, H * V }),
        tn(nw, RAD_F32, { V }), tn(o, RAD_BF16, { TT, H, V }), tn(oq, RAD_I8, { TT, H * V }), tn(os, RAD_F32, { TT, H }) };
    RadTensor tru[16] = {
        tn(qq, RAD_BF16, { TT, Hg, K }), tn(kk, RAD_BF16, { TT, Hg, K }), tn(vv, RAD_BF16, { TT, H, V }),
        tn(ta, RAD_BF16, { TT, H }), tn(tb, RAD_BF16, { TT, H }), tn(alog, RAD_F32, { H }), tn(dtb, RAD_F32, { H }),
        tst, tsidx, tnull, tn(cu, RAD_I32, { N + 1 }), tn(z, RAD_BF16, { TT, H * V }), tn(nw, RAD_F32, { V }),
        tn(o, RAD_BF16, { TT, H, V }), tn(oq, RAD_I8, { TT, H * V }), tn(os, RAD_F32, { TT, H }) };

    struct Form { const char* name; LaunchFn fn; RadTensor* t; int nt; };
    const Form forms[2] = { { "conv_recurrent", cr, tcr, 18 }, { "recurrent", ru, tru, 16 } };
    bool ok = true;
    for (const Form& F : forms) {
        RadArgs args{};
        args.t = F.t; args.n_t = F.nt; args.p = p; args.n_p = 9;
        int rc = F.fn(&args, (RadStream)&q);
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("%s: launch refused rc %d\n", F.name, rc); ok = false; continue; }
        std::vector<double> us;
        for (int r = 0; r < runs; ++r) {
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int L = 0; L < launches; ++L) F.fn(&args, (RadStream)&q);
            q.wait_and_throw();
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / launches);
        }
        std::sort(us.begin(), us.end());
        std::printf("RESULT gdn_decode %s: %.1f us a launch (min %.1f max %.1f, %d launches x %d runs, host wall over "
                    "back-to-back launches)\n", F.name, us[us.size() / 2], us.front(), us.back(), launches, runs);
        /* a checksum after one launch from a fresh state, so two forms of the kernel can be compared */
        std::vector<uint16_t> ho((size_t)TT * H * V);
        q.memcpy(ho.data(), o, ho.size() * 2).wait();
        double sum = 0, asum = 0;
        for (uint16_t v : ho) { const float f = bf(v); sum += f; asum += std::fabs(f); }
        std::printf("CHECKSUM gdn_decode %s: sum %.6g abs %.6g first [%g %g %g %g]\n", F.name, sum, asum, bf(ho[0]),
                    bf(ho[1]), bf(ho[2]), bf(ho[3]));
    }
    std::printf("RESULT gdn_decode_bench %s\n", ok ? "DONE" : "FAIL");
    return ok ? 0 : 1;
}
