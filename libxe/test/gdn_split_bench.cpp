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
 *   xe_gdn_split_bench LIBXE.so [launches=200] [anchor]
 *
 * Run 113 (note 138's rest): `anchor` runs the MTP form instead -- state_form anchor, two accepted tokens replayed
 * from the factor block, the six anchor / factor columns on distinct slots (1-3, 4-6) and the read copy's shift
 * marker at 3 (shift +1), so a slice that read a marker another had already stamped would read the wrong anchor
 * column and change the hashes.
 *
 * Run 282 (owner note 224 item 2, the GDN redesign's gate; logs/run281-gdn-stage1-plan.md): `anchor-seq [reps]` runs
 * the anchor form as a SEQUENCE of calls on ONE state -- the factor block zeroed first (marker 0: the default anchor
 * column), then (T 4, nacc 1), (T 4, nacc 2), (T 2, nacc 4), (T 4, nacc 1), (T 0, fold-only), (T 3, nacc 2), the same
 * sidx row every call (slide 0: the engine's linear pool, one slot a sequence for its life, the anchor read and written
 * at column 2 by the marker's algebra) -- and hashes every output after EVERY call, `reps` times over (default 1; 200
 * shows an intermittent ordering race). Step s + 1 replays step s's factors, so a slice that overwrote a row group's
 * factors while another slice of the group was still replaying them (the run-271 race at VS 8 / 16) changes the state
 * hash. The hashes must equal SPLIT 0's after every call. The sliding cases (+1 / -1) come once the bench copies the
 * engine's column rotation from src/tests/ref_test.cpp; N = 2 with a padding sequence is the next addition.
 *
 *   xe_gdn_split_bench LIBXE.so [launches=200] [anchor | anchor-seq [reps=1]]
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

/* Run 282: the anchor form as a sequence of calls on one state (see the header). The buffers are sized for T <= 4
 * and re-filled per call (the token inputs x / a / b / z from a per-call seed, cu = {0, T}, nacc); the state, the conv
 * state and the factor block carry over. A hash line per call; `reps` repetitions of the whole schedule. */
int run_anchor_seq(LaunchFn fn, sycl::queue& q, int reps) {
    const int Hg = 8, H = 24, K = 128, V = 128, W = 4, CD = 2 * Hg * K + H * V, SL = 6, TM = 4, N = 1, S = 8, SW = 8;
    struct Call { int T, nacc; };
    const Call sched[6] = { { 4, 1 }, { 4, 2 }, { 2, 4 }, { 4, 1 }, { 0, 0 }, { 3, 2 } };
    auto dev = [&](size_t bytes) { return sycl::malloc_device<uint8_t>(bytes, q); };
    std::vector<uint16_t> hx((size_t)TM * CD), hcw((size_t)CD * W), hcb(CD), hcs((size_t)CD * SL), ha((size_t)TM * H),
        hb((size_t)TM * H), hz((size_t)TM * H * V);
    std::vector<float> halog(H), hdtb(H), hst((size_t)S * H * V * K), hnw(V);
    for (size_t i = 0; i < hcw.size(); ++i) hcw[i] = bf(uni(2, i, -0.5f, 0.5f));
    for (size_t i = 0; i < hcb.size(); ++i) hcb[i] = bf(uni(3, i, -0.1f, 0.1f));
    for (size_t i = 0; i < hcs.size(); ++i) hcs[i] = bf(uni(4, i, -2.f, 2.f));
    for (int i = 0; i < H; ++i) { halog[i] = uni(8, i, -1.f, 1.f); hdtb[i] = uni(9, i, -1.f, 1.f); }
    for (size_t i = 0; i < hst.size(); ++i) hst[i] = uni(10, i, -0.2f, 0.2f);
    for (int i = 0; i < V; ++i) hnw[i] = uni(11, i, 0.5f, 1.5f);
    /* the factor slots 4-6 ZEROED (a fresh block: marker 0 reads as absent, so the first call takes the default anchor
     * column 2 = slot 3; its replay of a zero factor leaves the state as it was); the anchor slots 1-3 random */
    const size_t head = (size_t)V * K;
    for (int sl = 4; sl <= 6; ++sl)
        for (size_t i = 0; i < (size_t)H * head; ++i) hst[(size_t)sl * H * head + i] = 0.0f;
    std::vector<int32_t> hcidx = { 0 }, hcu = { 0, 0 }, hsidx(SW), hnacc = { 0 };
    for (int i = 0; i < 6; ++i) hsidx[i] = 1 + i;   /* anchor 1-3, factors 4-6, the same row every call (slide 0) */
    hsidx[6] = 7; hsidx[7] = 7;
    uint8_t *dx = dev(hx.size() * 2), *dcw = dev(hcw.size() * 2), *dcb = dev(hcb.size() * 2), *dcs = dev(hcs.size() * 2),
            *da = dev(ha.size() * 2), *db = dev(hb.size() * 2), *dz = dev(hz.size() * 2), *dal = dev(H * 4),
            *ddt = dev(H * 4), *dst = dev(hst.size() * 4), *dnw = dev(V * 4), *dci = dev(4), *dcu = dev(8),
            *dsi = dev(SW * 4), *dO = dev((size_t)TM * H * V * 2), *dq = dev((size_t)TM * H * V), *ds = dev((size_t)TM * H * 4),
            *dna = dev(4);
    q.memcpy(dcw, hcw.data(), hcw.size() * 2); q.memcpy(dcb, hcb.data(), hcb.size() * 2);
    q.memcpy(dal, halog.data(), H * 4); q.memcpy(ddt, hdtb.data(), H * 4); q.memcpy(dnw, hnw.data(), V * 4);
    q.memcpy(dci, hcidx.data(), 4); q.memcpy(dsi, hsidx.data(), SW * 4);
    q.wait();
    const uint64_t f0 = 1469598103934665603ull;
    int rc_all = 0;
    for (int rep = 0; rep < reps; ++rep) {
        /* the state and the conv state start the same every repetition */
        q.memcpy(dst, hst.data(), hst.size() * 4); q.memcpy(dcs, hcs.data(), hcs.size() * 2); q.wait();
        for (int c = 0; c < 6; ++c) {
            const int T = sched[c].T, nacc = sched[c].nacc;
            const uint64_t seed = 100 + (uint64_t)c;
            for (size_t i = 0; i < (size_t)T * CD; ++i) hx[i] = bf(uni(seed, i, -2.f, 2.f));
            for (size_t i = 0; i < (size_t)T * H; ++i) { ha[i] = bf(uni(seed + 20, i, -3.f, 3.f)); hb[i] = bf(uni(seed + 40, i, -3.f, 3.f)); }
            for (size_t i = 0; i < (size_t)T * H * V; ++i) hz[i] = bf(uni(seed + 60, i, -3.f, 3.f));
            hcu[1] = T; hnacc[0] = nacc;
            if (T > 0) {
                q.memcpy(dx, hx.data(), (size_t)T * CD * 2); q.memcpy(da, ha.data(), (size_t)T * H * 2);
                q.memcpy(db, hb.data(), (size_t)T * H * 2); q.memcpy(dz, hz.data(), (size_t)T * H * V * 2);
            }
            q.memcpy(dcu, hcu.data(), 8); q.memcpy(dna, hnacc.data(), 4);
            q.memset(dO, 0, (size_t)TM * H * V * 2); q.memset(dq, 0, (size_t)TM * H * V); q.memset(ds, 0, (size_t)TM * H * 4);
            q.wait();
            const int Tt = T > 0 ? T : 1;   /* the tensors keep a row at T 0 (cu says 0 tokens) */
            RadTensor t[18] = {
                ten(dx, RAD_BF16, { Tt, CD }), ten(dcw, RAD_BF16, { CD, W }), ten(dcb, RAD_BF16, { CD }),
                ten(dcs, RAD_BF16, { 1, CD, SL }), ten(dci, RAD_I32, { N }), ten(da, RAD_BF16, { Tt, H }),
                ten(db, RAD_BF16, { Tt, H }), ten(dal, RAD_F32, { H }), ten(ddt, RAD_F32, { H }),
                ten(dst, RAD_F32, { S, H, V, K }), ten(dsi, RAD_I32, { N, SW }), ten(dna, RAD_I32, { N }),
                ten(dcu, RAD_I32, { N + 1 }),
                ten(dz, RAD_BF16, { Tt, H * V }), ten(dnw, RAD_F32, { V }), ten(dO, RAD_BF16, { Tt, H, V }),
                ten(dq, RAD_I8, { Tt, H * V }), ten(ds, RAD_F32, { Tt, H }) };
            RadParam p[8]{};
            const char* ik[] = { "head_k", "head_v", "conv_width", "n_head_v" };
            const long long iv[] = { K, V, W, H };
            for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
            p[4].key = "scale"; p[4].kind = RAD_P_F64; p[4].dval = 1.0 / std::sqrt((double)K);
            p[5].key = "eps"; p[5].kind = RAD_P_F64; p[5].dval = 1e-6;
            p[6].key = "state_form"; p[6].kind = RAD_P_STR; p[6].sval = "anchor";
            RadArgs args{};
            args.t = t; args.n_t = 18; args.p = p; args.n_p = 7;
            const int rc = fn(&args, (RadStream)&q);
            q.wait_and_throw();
            if (rc != RAD_OK && T == 0) {   /* a fold-only call the row refuses at N 1: recorded, not a failure (the same for every SPLIT) */
                std::printf("RESULT anchor-seq rep %d call %d (T 0 fold-only): refused rc %d, skipped\n", rep, c, rc);
                continue;
            }
            if (rc != RAD_OK) { std::printf("RESULT anchor-seq rep %d call %d (T %d nacc %d): launch refused rc %d\n", rep, c, T, nacc, rc); rc_all = 1; break; }
            std::vector<uint8_t> o((size_t)Tt * H * V * 2), oq((size_t)Tt * H * V), os((size_t)Tt * H * 4), st(hst.size() * 4),
                cs(hcs.size() * 2);
            q.memcpy(o.data(), dO, o.size()); q.memcpy(oq.data(), dq, oq.size()); q.memcpy(os.data(), ds, os.size());
            q.memcpy(st.data(), dst, st.size()); q.memcpy(cs.data(), dcs, cs.size()); q.wait();
            std::printf("RESULT anchor-seq rep %d call %d (T %d nacc %d) hashes: o %016llx codes %016llx scales %016llx state %016llx conv %016llx\n",
                        rep, c, T, nacc, (unsigned long long)fnv(o.data(), o.size(), f0), (unsigned long long)fnv(oq.data(), oq.size(), f0),
                        (unsigned long long)fnv(os.data(), os.size(), f0), (unsigned long long)fnv(st.data(), st.size(), f0),
                        (unsigned long long)fnv(cs.data(), cs.size(), f0));
        }
        if (rc_all) break;
    }
    for (uint8_t* x : { dx, dcw, dcb, dcs, da, db, dz, dal, ddt, dst, dnw, dci, dcu, dsi, dO, dq, ds, dna }) sycl::free(x, q);
    return rc_all;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_gdn_split_bench LIBXE.so [launches] [anchor | anchor-seq [reps]]\n"); return 2; }
    const int launches = argc > 2 ? std::atoi(argv[2]) : 200;
    const bool anchor = argc > 3 && std::strcmp(argv[3], "anchor") == 0;
    const bool seq = argc > 3 && std::strcmp(argv[3], "anchor-seq") == 0;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "r4d_rad_gdn_conv_recurrent_update");
    if (!fn) { std::printf("no r4d_rad_gdn_conv_recurrent_update\n"); return 2; }
    /* profiling on, so libxe's launch profiler (RADIANCE_XE_LAUNCH_PROFILE=1) can time each kernel */
    sycl::queue q{ sycl::device{ sycl::gpu_selector_v },
                   sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} } };
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* sp = std::getenv("RADIANCE_XE_GDN_SPLIT");
    std::printf("gdn_split_bench: %s, RADIANCE_XE_GDN_SPLIT=%s, %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), sp ? sp : "(unset)",
                seq ? "anchor-seq" : anchor ? "anchor" : "per_candidate");
    if (seq) {
        const int reps = argc > 4 ? std::atoi(argv[4]) : 1;
        const int rc = run_anchor_seq(fn, q, reps);
        std::printf("RESULT gdn_split_bench %s\n", rc ? "FAIL" : "DONE");
        return rc;
    }
    const int Hg = 8, H = 24, K = 128, V = 128, W = 4, CD = 2 * Hg * K + H * V, SL = 6;   /* SL = 4 tokens + W - 2 */
    /* RI_GDN_T (run 283): the token counts to run, e.g. "4" alone, so the launch profiler's exit summary (over every
     * launch of the process) is one T's; default both */
    std::vector<int> Ts = { 1, 4 };
    if (const char* e = std::getenv("RI_GDN_T"); e && *e) {
        Ts.clear();
        for (const char* p = e; *p; ++p) if (*p >= '1' && *p <= '8') Ts.push_back(*p - '0');
        if (Ts.empty()) Ts = { 1, 4 };
    }
    for (int T : Ts) {
        const int N = 1, S = anchor ? 8 : T + 1;
        const int SW = anchor ? 8 : T;   /* the sidx row: anchor's stride is the draft window + 1 */
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
        std::vector<int32_t> hcidx = { 0 }, hcu = { 0, T }, hsidx(SW), hnacc = { 2 };
        if (anchor) {
            for (int i = 0; i < 6; ++i) hsidx[i] = 1 + i;   /* anchor 1-3, factors 4-6 */
            hsidx[6] = 7; hsidx[7] = 7;
            /* the read factor copy's marker (slot 5, every head; RU_MARK_OFF = RU_TMAX 16 x RU_FSTR 648 floats into
             * the head) at 3: m = 2, so the anchor is read at column 1 (slot 2) and written at column 2 (slot 3) */
            const size_t head = (size_t)V * K;
            for (int hv = 0; hv < H; ++hv) hst[(size_t)5 * H * head + (size_t)hv * head + 16 * 648] = 3.0f;
        } else {
            for (int t = 0; t < T; ++t) hsidx[t] = 1 + t;    /* read slot sidx[0] (no naccept), write slot t + 1 */
        }
        uint8_t *dx = dev(hx.size() * 2), *dcw = dev(hcw.size() * 2), *dcb = dev(hcb.size() * 2), *dcs = dev(hcs.size() * 2),
                *da = dev(ha.size() * 2), *db = dev(hb.size() * 2), *dz = dev(hz.size() * 2), *dal = dev(H * 4),
                *ddt = dev(H * 4), *dst = dev(hst.size() * 4), *dnw = dev(V * 4), *dci = dev(4), *dcu = dev(8),
                *dsi = dev(SW * 4), *dO = dev((size_t)T * H * V * 2), *dq = dev((size_t)T * H * V), *ds = dev((size_t)T * H * 4),
                *dna = dev(4);
        auto upload = [&] {
            q.memcpy(dx, hx.data(), hx.size() * 2); q.memcpy(dcw, hcw.data(), hcw.size() * 2);
            q.memcpy(dcb, hcb.data(), hcb.size() * 2); q.memcpy(dcs, hcs.data(), hcs.size() * 2);
            q.memcpy(da, ha.data(), ha.size() * 2); q.memcpy(db, hb.data(), hb.size() * 2);
            q.memcpy(dz, hz.data(), hz.size() * 2); q.memcpy(dal, halog.data(), H * 4); q.memcpy(ddt, hdtb.data(), H * 4);
            q.memcpy(dst, hst.data(), hst.size() * 4); q.memcpy(dnw, hnw.data(), V * 4); q.memcpy(dci, hcidx.data(), 4);
            q.memcpy(dcu, hcu.data(), 8); q.memcpy(dsi, hsidx.data(), SW * 4); q.memcpy(dna, hnacc.data(), 4);
            q.memset(dO, 0, (size_t)T * H * V * 2); q.memset(dq, 0, (size_t)T * H * V); q.memset(ds, 0, (size_t)T * H * 4);
            q.wait();
        };
        upload();
        RadTensor t[18] = {
            ten(dx, RAD_BF16, { T, CD }), ten(dcw, RAD_BF16, { CD, W }), ten(dcb, RAD_BF16, { CD }),
            ten(dcs, RAD_BF16, { 1, CD, SL }), ten(dci, RAD_I32, { N }), ten(da, RAD_BF16, { T, H }),
            ten(db, RAD_BF16, { T, H }), ten(dal, RAD_F32, { H }), ten(ddt, RAD_F32, { H }),
            ten(dst, RAD_F32, { S, H, V, K }), ten(dsi, RAD_I32, { N, SW }), ten(dna, RAD_I32, { N }),
            ten(dcu, RAD_I32, { N + 1 }),
            ten(dz, RAD_BF16, { T, H * V }), ten(dnw, RAD_F32, { V }), ten(dO, RAD_BF16, { T, H, V }),
            ten(dq, RAD_I8, { T, H * V }), ten(ds, RAD_F32, { T, H }) };
        if (!anchor) t[11] = RadTensor{};   /* no naccept */
        RadParam p[8]{};
        const char* ik[] = { "head_k", "head_v", "conv_width", "n_head_v" };
        const long long iv[] = { K, V, W, H };
        for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
        p[4].key = "scale"; p[4].kind = RAD_P_F64; p[4].dval = 1.0 / std::sqrt((double)K);
        p[5].key = "eps"; p[5].kind = RAD_P_F64; p[5].dval = 1e-6;
        char sfv[16];
        std::snprintf(sfv, sizeof sfv, "%s", anchor ? "anchor" : "per_candidate");
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
        for (uint8_t* x : { dx, dcw, dcb, dcs, da, db, dz, dal, ddt, dst, dnw, dci, dcu, dsi, dO, dq, ds, dna }) sycl::free(x, q);
    }
    std::printf("RESULT gdn_split_bench DONE\n");
    return 0;
}
