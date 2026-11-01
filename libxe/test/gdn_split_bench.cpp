/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * gdn_split_bench.cpp -- owner note 138 (run 112): the decode GDN fold (gdn_conv_recurrent_update) at one rank's
 * Flash-Next shape (8 key heads, 24 value heads, K = V = 128, conv width 4, per-candidate state, the int8 quantiser
 * folded in), on hashed inputs, at 1 token and the 4-token verify. For each it prints a hash of every output after one
 * call -- o, the int8 codes and scales, the recurrent state and the conv state -- and the time a call over back-to-back
 * launches. Run it once with RADIANCE_XE_GDN_SPLIT unset and once with =2 / =4: the hashes must be equal.
 *
 *   xe_gdn_split_bench LIBXE.so [launches=200]
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
uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float uni(uint64_t s, size_t i, float lo, float hi) { return lo + (hi - lo) * (float)(h32(s, i) & 0xffffff) / 16777216.0f; }

RadTensor ten(void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}
uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_gdn_split_bench LIBXE.so [launches]\n"); return 2; }
    const int launches = argc > 2 ? std::atoi(argv[2]) : 200;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "r4d_rad_gdn_conv_recurrent_update");
    if (!fn) { std::printf("no r4d_rad_gdn_conv_recurrent_update\n"); return 2; }
    /* profiling on, so libxe's launch profiler (RADIANCE_XE_LAUNCH_PROFILE=1) can time each kernel */
    sycl::queue q{ sycl::device{ sycl::gpu_selector_v },
                   sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} } };
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* sp = std::getenv("RADIANCE_XE_GDN_SPLIT");
    std::printf("gdn_split_bench: %s, RADIANCE_XE_GDN_SPLIT=%s\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                sp ? sp : "(unset)");
    const int Hg = 8, H = 24, K = 128, V = 128, W = 4, CD = 2 * Hg * K + H * V, SL = 6;   /* SL = 4 tokens + W - 2 */
    for (int T : { 1, 4 }) {
        const int N = 1, S = T + 1;
        auto dev = [&](size_t bytes) { return sycl::malloc_device<uint8_t>(bytes, q); };
        std::vector<uint16_t> hx((size_t)T * CD), hcw((size_t)CD * W), hcb(CD), hcs((size_t)CD * SL), ha((size_t)T * H),
            hb((size_t)T * H), hz((size_t)T * H * V);
        std::vector<float> halog(H), hdtb(H), hst((size_t)S * H * V * K), hnw(V);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = bf(uni(1, i, -2.f, 2.f));
        for (size_t i = 0; i < hcw.size(); ++i) hcw[i] = bf(uni(2, i, -0.5f, 0.5f));
        for (size_t i = 0; i < hcb.size(); ++i) hcb[i] = bf(uni(3, i, -0.1f, 0.1f));
        for (size_t i = 0; i < hcs.size(); ++i) hcs[i] = bf(uni(4, i, -2.f, 2.f));
        for (size_t i = 0; i < ha.size(); ++i) { ha[i] = bf(uni(5, i, -3.f, 3.f)); hb[i] = bf(uni(6, i, -3.f, 3.f)); }
        for (size_t i = 0; i < hz.size(); ++i) hz[i] = bf(uni(7, i, -3.f, 3.f));
        for (int i = 0; i < H; ++i) { halog[i] = uni(8, i, -1.f, 1.f); hdtb[i] = uni(9, i, -1.f, 1.f); }
        for (size_t i = 0; i < hst.size(); ++i) hst[i] = uni(10, i, -0.2f, 0.2f);
        for (int i = 0; i < V; ++i) hnw[i] = uni(11, i, 0.5f, 1.5f);
        std::vector<int32_t> hcidx = { 0 }, hcu = { 0, T }, hsidx(T);
        for (int t = 0; t < T; ++t) hsidx[t] = 1 + t;    /* read slot sidx[0] (no naccept), write slot t + 1 */
        uint8_t *dx = dev(hx.size() * 2), *dcw = dev(hcw.size() * 2), *dcb = dev(hcb.size() * 2), *dcs = dev(hcs.size() * 2),
                *da = dev(ha.size() * 2), *db = dev(hb.size() * 2), *dz = dev(hz.size() * 2), *dal = dev(H * 4),
                *ddt = dev(H * 4), *dst = dev(hst.size() * 4), *dnw = dev(V * 4), *dci = dev(4), *dcu = dev(8),
                *dsi = dev(T * 4), *dO = dev((size_t)T * H * V * 2), *dq = dev((size_t)T * H * V), *ds = dev((size_t)T * H * 4);
        auto upload = [&] {
            q.memcpy(dx, hx.data(), hx.size() * 2); q.memcpy(dcw, hcw.data(), hcw.size() * 2);
            q.memcpy(dcb, hcb.data(), hcb.size() * 2); q.memcpy(dcs, hcs.data(), hcs.size() * 2);
            q.memcpy(da, ha.data(), ha.size() * 2); q.memcpy(db, hb.data(), hb.size() * 2);
            q.memcpy(dz, hz.data(), hz.size() * 2); q.memcpy(dal, halog.data(), H * 4); q.memcpy(ddt, hdtb.data(), H * 4);
            q.memcpy(dst, hst.data(), hst.size() * 4); q.memcpy(dnw, hnw.data(), V * 4); q.memcpy(dci, hcidx.data(), 4);
            q.memcpy(dcu, hcu.data(), 8); q.memcpy(dsi, hsidx.data(), T * 4);
            q.memset(dO, 0, (size_t)T * H * V * 2); q.memset(dq, 0, (size_t)T * H * V); q.memset(ds, 0, (size_t)T * H * 4);
            q.wait();
        };
        upload();
        RadTensor t[18] = {
            ten(dx, RAD_BF16, { T, CD }), ten(dcw, RAD_BF16, { CD, W }), ten(dcb, RAD_BF16, { CD }),
            ten(dcs, RAD_BF16, { 1, CD, SL }), ten(dci, RAD_I32, { N }), ten(da, RAD_BF16, { T, H }),
            ten(db, RAD_BF16, { T, H }), ten(dal, RAD_F32, { H }), ten(ddt, RAD_F32, { H }),
            ten(dst, RAD_F32, { S, H, V, K }), ten(dsi, RAD_I32, { N, T }), RadTensor{}, ten(dcu, RAD_I32, { N + 1 }),
            ten(dz, RAD_BF16, { T, H * V }), ten(dnw, RAD_F32, { V }), ten(dO, RAD_BF16, { T, H, V }),
            ten(dq, RAD_I8, { T, H * V }), ten(ds, RAD_F32, { T, H }) };
        t[11].data = nullptr;   /* no naccept */
        RadParam p[8]{};
        const char* ik[] = { "head_k", "head_v", "conv_width", "n_head_v" };
        const long long iv[] = { K, V, W, H };
        for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
        p[4].key = "scale"; p[4].kind = RAD_P_F64; p[4].dval = 1.0 / std::sqrt((double)K);
        p[5].key = "eps"; p[5].kind = RAD_P_F64; p[5].dval = 1e-6;
        char sfv[] = "per_candidate";
        p[6].key = "state_form"; p[6].kind = RAD_P_STR; p[6].sval = sfv;
        RadArgs args{};
        args.t = t; args.n_t = 18; args.p = p; args.n_p = 7;
        int rc = fn(&args, (RadStream)&q);
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("T %d: launch refused rc %d\n", T, rc); return 1; }
        std::vector<uint8_t> o((size_t)T * H * V * 2), oq((size_t)T * H * V), os((size_t)T * H * 4), st(hst.size() * 4),
            cs(hcs.size() * 2);
        q.memcpy(o.data(), dO, o.size()); q.memcpy(oq.data(), dq, oq.size()); q.memcpy(os.data(), ds, os.size());
        q.memcpy(st.data(), dst, st.size()); q.memcpy(cs.data(), dcs, cs.size()); q.wait();
        const uint64_t f0 = 1469598103934665603ull;
        std::printf("RESULT gdn_split T %d hashes: o %016llx codes %016llx scales %016llx state %016llx conv %016llx\n", T,
                    (unsigned long long)fnv(o.data(), o.size(), f0), (unsigned long long)fnv(oq.data(), oq.size(), f0),
                    (unsigned long long)fnv(os.data(), os.size(), f0), (unsigned long long)fnv(st.data(), st.size(), f0),
                    (unsigned long long)fnv(cs.data(), cs.size(), f0));
        /* timing: back-to-back calls (the state and conv state move on; the time does not depend on them) */
        for (int i = 0; i < 5; ++i) fn(&args, (RadStream)&q);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < launches; ++i) fn(&args, (RadStream)&q);
        q.wait_and_throw();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / launches;
        std::printf("RESULT gdn_split T %d: %.2f us a call (host wall over %d back-to-back calls)\n", T, us, launches);
        for (uint8_t* x : { dx, dcw, dcb, dcs, da, db, dz, dal, ddt, dst, dnw, dci, dcu, dsi, dO, dq, ds }) sycl::free(x, q);
    }
    std::printf("RESULT gdn_split_bench DONE\n");
    return 0;
}
