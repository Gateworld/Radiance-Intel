/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, run 267: owner lever #3).
 *
 * moe_bf16_dpas_test.cpp -- the protected experts' bf16 GEMM on the matrix units (xe_moe_bf16_dpas.cpp) against the
 * WMMA form's arithmetic computed on the host: fp32 products of bf16 pairs (exact) added in ascending k, one
 * nearest-even rounding to bf16 -- what ports/r4d_moe.cpp's moe_gemm_bf16_kernel does through xe::wmma. The matrix
 * unit associates the K sum differently, so the two are not bit-identical by design: this test REPORTS the bf16 ulp
 * distance (max, mean, the share of differing outputs) and fails only above 2 ulps, on a wrong row / column (rows
 * past a run, dropped slots, a tail block, columns past N) or a broken canary. Cases (the coordinator's list): a
 * tail block of dropped slots past the live count, N not a multiple of 64, a dead row inside a run, an expert with
 * fewer than 8 rows, rows below off[0] (another call's) that must stay untouched; plus the launcher's guards (K % 64,
 * a small call, a small scratch) on any device.
 *
 *   moe_bf16_dpas_test            RI_DEVICE=cpu for the CPU device (the guards only: joint_matrix runs on a GPU)
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int xe_moe_bf16_dpas_on();
extern "C" int xe_moe_bf16_dpas_launch(void* stream, const uint16_t* A, const void* const* Wt, size_t w_s0,
                                       const int32_t* sorted, const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N,
                                       uint32_t K, uint32_t n_expert, uint32_t top_k, uint32_t a_sorted, uint32_t a_ld,
                                       uint32_t w_ld, uint32_t y_ld, void* scratch, size_t scratch_bytes);

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
/* bf16 ulp distance of two values of the same sign (an infinite distance across zero, which never happens here) */
int ulp_dist(uint16_t a, uint16_t b) {
    if (a == b) return 0;
    const bool sa = a & 0x8000u, sb = b & 0x8000u;
    if (sa != sb) return (a & 0x7fffu) == 0 && (b & 0x7fffu) == 0 ? 0 : 1 << 15;
    const int d = (int)(a & 0x7fffu) - (int)(b & 0x7fffu);
    return d < 0 ? -d : d;
}
constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xC5;

/* Run 268 (the coordinator's MB4): the OLD kernel too -- ports/r4d_moe.cpp's r4d_rad_moe_gemm_bf16 (the dispatch:
 * T <= 256 the GEMV form when on, else the WMMA emulation's prefill form) loaded from the image's libxe.so (RI_LIBXE,
 * default /opt/ri/radiance_home/kernels/libxe.so), scored against the same references as the matrix form. */
typedef int (*LaunchFn)(const RadArgs*, RadStream);
LaunchFn g_old = nullptr;

RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}

/* the old op on contiguous operands: A [M, K], W [E, N, K] stacked, Y [T, N] (pitch N); 0 = ran */
int run_old(sycl::queue& q, const uint16_t* dA, uint32_t M, const uint16_t* dW, const int32_t* dS, const int32_t* dO,
            uint16_t* dY, uint32_t T, uint32_t N, uint32_t K, uint32_t E, uint32_t top_k, void* scratch, size_t scratch_bytes) {
    RadTensor t[5];
    t[0] = tn((void*)dA, RAD_BF16, {(int64_t)M, (int64_t)K});
    t[1] = tn((void*)dW, RAD_BF16, {(int64_t)E, (int64_t)N, (int64_t)K});
    t[2] = tn((void*)dS, RAD_I32, {(int64_t)T});
    t[3] = tn((void*)dO, RAD_I32, {(int64_t)E + 1});
    t[4] = tn((void*)dY, RAD_BF16, {(int64_t)T, (int64_t)N});
    RadParam p[6]{};
    const char* ik[] = {"N", "K", "n_expert", "top_k"};
    const long long iv[] = {(long long)N, (long long)K, (long long)E, (long long)top_k};
    for (int i = 0; i < 4; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
    p[4].key = "dtype"; p[4].kind = RAD_P_STR; p[4].sval = "bf16";
    p[5].key = "a_order"; p[5].kind = RAD_P_STR; p[5].sval = "token";
    RadArgs args{};
    args.t = t; args.n_t = 5; args.p = p; args.n_p = 6; args.scratch = scratch; args.scratch_bytes = (int64_t)scratch_bytes;
    const int rc = g_old(&args, (RadStream)&q);
    q.wait();
    return rc;
}
const char* old_form_name(uint32_t T) { return T <= 256u ? "T <= 256: the GEMV form (default on)" : "T > 256: the WMMA prefill form"; }

struct Case {
    const char* name;
    uint32_t K, N, n_expert, top_k;
    std::vector<uint32_t> rows_per_expert;   /* live rows an expert (the sort's runs, in expert order) */
    uint32_t dead_in_run;                    /* a sorted row inside a run marked dropped (-1), 0 for none */
    uint32_t dropped_tail;                   /* dropped slots after the live count (a tail block) */
    uint32_t below;                          /* rows below off[0], another call's: untouched */
    uint32_t seed;                           /* the data's seed (0 = the fixed data MB1 saw; run 268: a second seed) */
};

/* The sort: rows below off[0] (another call's), then each expert's run, then the dropped tail. Token rows are
 * assigned round-robin so a token feeds several slots (top_k), as the engine's sort does. */
bool run_case(sycl::queue& q, const Case& cs, bool gpu) {
    const uint32_t K = cs.K, N = cs.N, E = cs.n_expert, top_k = cs.top_k;
    uint32_t live = cs.below;
    std::vector<int32_t> off(E + 1);
    off[0] = (int32_t)cs.below;
    for (uint32_t e = 0; e < E; ++e) { live += cs.rows_per_expert[e]; off[e + 1] = (int32_t)live; }
    const uint32_t T = live + cs.dropped_tail;
    const uint32_t M = (T + top_k - 1) / top_k + 1;   /* token rows */
    std::vector<int32_t> sorted(T);
    for (uint32_t r = 0; r < T; ++r) sorted[r] = r < live ? (int32_t)((r * 7u + 3u) % (M * top_k)) : -1;
    if (cs.dead_in_run) sorted[cs.below + cs.dead_in_run] = -1;
    const uint32_t a_ld = K, w_ld = K, y_ld = N + 8;   /* an output pitch wider than N */
    std::vector<uint16_t> A((size_t)M * a_ld), W((size_t)E * N * w_ld);
    for (size_t i = 0; i < A.size(); ++i) A[i] = to_bf(((float)(h32(1 + 100u * cs.seed, i) % 2001) - 1000.0f) / 500.0f);
    for (size_t i = 0; i < W.size(); ++i) W[i] = to_bf(((float)(h32(2 + 100u * cs.seed, i) % 2001) - 1000.0f) / 1000.0f);
    /* the host reference: the WMMA form's arithmetic; beside it the fp64 sum of the exact products (run 268: a bf16
     * ulp is scale-bound, so an output near zero -- cancellation over K -- can sit several ulps from the sequential
     * fp32 sum through re-association alone; those are judged by the absolute error against the fp64 sum instead) */
    std::vector<uint16_t> ref((size_t)T * y_ld, 0x1234);   /* 0x1234: a row this call must not touch */
    std::vector<double> ref64((size_t)T * y_ld, 0.0), peak64((size_t)T * y_ld, 0.0), row_rms(T, 0.0);
    for (uint32_t e = 0; e < E; ++e)
        for (int32_t r = off[e]; r < off[e + 1]; ++r) {
            const int32_t f = sorted[r];
            double sq = 0.0;
            for (uint32_t n = 0; n < N; ++n) {
                uint16_t v = 0;
                double v64 = 0.0, pk = 0.0;
                if (f >= 0) {
                    const uint32_t tok = (uint32_t)f / top_k;
                    volatile float acc = 0.0f;   /* volatile: no contraction, the sequential order */
                    double acc64 = 0.0;
                    for (uint32_t k = 0; k < K; ++k) {
                        const float p = bf(A[(size_t)tok * a_ld + k]) * bf(W[((size_t)e * N + n) * w_ld + k]);
                        acc = acc + p;
                        acc64 += (double)p;   /* p is exact (8 x 8 significant bits): the sum of the exact products */
                        if (std::fabs(acc64) > pk) pk = std::fabs(acc64);   /* the K sum's largest partial sum */
                    }
                    v = to_bf((float)acc);
                    v64 = acc64;
                }
                ref[(size_t)r * y_ld + n] = v;
                ref64[(size_t)r * y_ld + n] = v64;
                peak64[(size_t)r * y_ld + n] = pk;
                sq += v64 * v64;
            }
            row_rms[r] = std::sqrt(sq / (double)N);
            for (uint32_t n = N; n < y_ld; ++n) ref[(size_t)r * y_ld + n] = 0x1234;   /* past N: untouched */
        }
    for (uint32_t r = live; r < T; ++r)   /* the dropped tail: zeros over the columns < N */
        for (uint32_t n = 0; n < N; ++n) ref[(size_t)r * y_ld + n] = 0;

    /* device buffers; Y carries canaries and the 0x1234 fill */
    uint16_t* dA = sycl::malloc_device<uint16_t>(A.size(), q);
    uint16_t* dW = sycl::malloc_device<uint16_t>(W.size(), q);
    int32_t* dS = sycl::malloc_device<int32_t>(T, q);
    int32_t* dO = sycl::malloc_device<int32_t>(E + 1, q);
    const size_t ybytes = (size_t)T * y_ld * 2;
    uint8_t* dYb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    const uint32_t cap = E + (T + 63) / 64 + 1;
    uint32_t* dMap = sycl::malloc_device<uint32_t>(1 + 2 * cap + 64, q);
    std::vector<uint16_t> yfill((size_t)T * y_ld, 0x1234);
    q.memcpy(dA, A.data(), A.size() * 2); q.memcpy(dW, W.data(), W.size() * 2);
    q.memcpy(dS, sorted.data(), T * 4); q.memcpy(dO, off.data(), (E + 1) * 4);
    q.memset(dYb, kPat, kCanary); q.memset(dYb + kCanary + ybytes, kPat, kCanary);
    q.memcpy(dYb + kCanary, yfill.data(), ybytes).wait();
    uint16_t* dY = (uint16_t*)(dYb + kCanary);
    /* the launcher's guards (any device): K % 64, a small call, a small scratch -> 1 (not taken) */
    bool ok = true;
    const size_t map_bytes = (size_t)(1 + 2 * cap) * 4;
    if (xe_moe_bf16_dpas_launch(&q, dA, (const void* const*)dW, (size_t)N * w_ld * 2, dS, dO, dY, T, N, K + 32, E, top_k, 0,
                                a_ld, w_ld, y_ld, dMap, map_bytes) != 1) { std::printf("FAIL %s: K %% 64 not refused\n", cs.name); ok = false; }
    if (xe_moe_bf16_dpas_launch(&q, dA, (const void* const*)dW, (size_t)N * w_ld * 2, dS, dO, dY, 50, N, K, E, top_k, 0,
                                a_ld, w_ld, y_ld, dMap, map_bytes) != 1) { std::printf("FAIL %s: a small call not refused\n", cs.name); ok = false; }
    if (xe_moe_bf16_dpas_launch(&q, dA, (const void* const*)dW, (size_t)N * w_ld * 2, dS, dO, dY, T, N, K, E, top_k, 0,
                                a_ld, w_ld, y_ld, dMap, 16) != 1) { std::printf("FAIL %s: a small scratch not refused\n", cs.name); ok = false; }
    if (!gpu) {
        std::printf("SKIP %s: the matrix-unit kernel runs on a GPU (guards checked)\n", cs.name);
        sycl::free(dA, q); sycl::free(dW, q); sycl::free(dS, q); sycl::free(dO, q); sycl::free(dYb, q); sycl::free(dMap, q);
        return ok;
    }
    /* the stacked plane: w_s0 = one expert's [N, K] bf16 */
    const int rc = xe_moe_bf16_dpas_launch(&q, dA, (const void* const*)dW, (size_t)N * w_ld * 2, dS, dO, dY, T, N, K, E,
                                           top_k, 0, a_ld, w_ld, y_ld, dMap, map_bytes);
    q.wait();
    if (rc != 0) { std::printf("FAIL %s: the launcher declined (rc %d) at T %u N %u K %u\n", cs.name, rc, T, N, K); ok = false; }
    std::vector<uint8_t> yb(ybytes + 2 * kCanary);
    q.memcpy(yb.data(), dYb, yb.size()).wait();
    for (size_t i = 0; i < kCanary; ++i)
        if (yb[i] != kPat || yb[kCanary + ybytes + i] != kPat) { std::printf("FAIL %s: canary broken at %zu\n", cs.name, i); ok = false; break; }
    const uint16_t* y = (const uint16_t*)(yb.data() + kCanary);
    size_t n_cmp = 0, n_diff = 0, bad = 0; long long sum_ulp = 0; int max_ulp = 0;
    /* run 268 (the coordinator's MB2 criteria): every differing output is printed with both forms' distance from the
     * fp64 sum, |value| / the row's RMS and the cancellation ratio (the K sum's largest partial sum / |value|). A
     * defect: the matrix form farther from fp64 than the wmma order by more than one bf16 ulp of the value, or an
     * outlier above 2 ulps whose cancellation ratio is under 16 (re-association cannot move a value that far). */
    size_t far_new = 0, low_cr = 0, closer_new = 0, closer_old = 0, closer_tie = 0;
    double err_new = 0.0, err_old = 0.0, min_cr_outlier = 0.0;
    for (uint32_t r = 0; r < T; ++r)
        for (uint32_t n = 0; n < y_ld; ++n) {
            const size_t i = (size_t)r * y_ld + n;
            const uint16_t a = y[i], b = ref[i];
            const bool computed = r >= (uint32_t)off[0] && r < live && n < N && sorted[r] >= 0;
            if (!computed) { if (a != b) { if (bad < 5) std::printf("FAIL %s: row %u col %u is %04x, expected %04x\n", cs.name, r, n, a, b); ++bad; } continue; }
            ++n_cmp;
            const int d = ulp_dist(a, b);
            if (!d) continue;
            ++n_diff; sum_ulp += d; if (d > max_ulp) max_ulp = d;
            const double v = ref64[i], av = std::fabs(v);
            const double ea = std::fabs((double)bf(a) - v), eb = std::fabs((double)bf(b) - v);
            const double ulp_v = std::ldexp(1.0, std::ilogb(av > 0.0 ? av : 1e-38) - 7);   /* one bf16 ulp at |v| */
            const double cr = peak64[i] / (av > 0.0 ? av : 1e-300);
            if (ea > err_new) err_new = ea;
            if (eb > err_old) err_old = eb;
            if (ea < eb) ++closer_new; else if (eb < ea) ++closer_old; else ++closer_tie;
            if (ea - eb > ulp_v) ++far_new;
            if (d > 2) { if (cr < 16.0) ++low_cr; if (min_cr_outlier == 0.0 || cr < min_cr_outlier) min_cr_outlier = cr; }
            std::printf("DIFF %s: row %u col %u ulp %d | fp64 %.7g |v|/rms %.3g cancel %.1f | matrix %.7g err %.3g (%.2f ulp) "
                        "| wmma-order %.7g err %.3g (%.2f ulp)%s\n",
                        cs.name, r, n, d, v, row_rms[r] > 0.0 ? av / row_rms[r] : 0.0, cr, (double)bf(a), ea, ea / ulp_v,
                        (double)bf(b), eb, eb / ulp_v, ea - eb > ulp_v ? " FARTHER" : "");
        }
    if (bad) ok = false;
    if (far_new) { std::printf("FAIL %s: the matrix form is farther from fp64 than the wmma order by more than 1 bf16 ulp on %zu outputs\n", cs.name, far_new); ok = false; }
    if (low_cr) { std::printf("FAIL %s: %zu outliers above 2 ulps with a cancellation ratio under 16\n", cs.name, low_cr); ok = false; }
    std::printf("RESULT %s: T %u live %u N %u K %u experts %u | %zu outputs compared, %zu differ (%.3f %%), max ulp %d, "
                "mean ulp over differing %.3f, untouched / zero rows wrong %zu | vs fp64 over the differing: matrix closer %zu, "
                "wmma-order closer %zu, tie %zu, max err matrix %.3g / wmma-order %.3g, farther by > 1 ulp %zu; outliers > 2 ulp: "
                "min cancellation ratio %.1f, under 16: %zu -> %s\n",
                cs.name, T, live, N, K, E, n_cmp, n_diff, n_cmp ? 100.0 * (double)n_diff / (double)n_cmp : 0.0, max_ulp,
                n_diff ? (double)sum_ulp / (double)n_diff : 0.0, bad, closer_new, closer_old, closer_tie, err_new, err_old,
                far_new, min_cr_outlier, low_cr, ok ? "PASS" : "FAIL");
    /* run 268 (MB4): the OLD kernel on the same operands, scored against the same references and against the matrix
     * form's outputs. Its Y is contiguous (pitch N); rows below off[0] must stay 0x1234, the dropped tail zero. */
    if (g_old) {
        const size_t ybo = (size_t)T * N * 2;
        uint8_t* dYo = sycl::malloc_device<uint8_t>(ybo + 2 * kCanary, q);
        const size_t scr_bytes = 4u << 20;
        void* dScr = sycl::malloc_device<uint8_t>(scr_bytes, q);
        std::vector<uint16_t> yfo((size_t)T * N, 0x1234);
        q.memset(dYo, kPat, kCanary); q.memset(dYo + kCanary + ybo, kPat, kCanary);
        q.memcpy(dYo + kCanary, yfo.data(), ybo).wait();
        const int rco = run_old(q, dA, M, dW, dS, dO, (uint16_t*)(dYo + kCanary), T, N, K, E, top_k, dScr, scr_bytes);
        std::vector<uint8_t> yo(ybo + 2 * kCanary);
        q.memcpy(yo.data(), dYo, yo.size()).wait();
        bool oko = rco == 0;
        if (rco != 0) std::printf("FAIL old-%s: the op returned rc %d\n", cs.name, rco);
        for (size_t i = 0; i < kCanary; ++i)
            if (yo[i] != kPat || yo[kCanary + ybo + i] != kPat) { std::printf("FAIL old-%s: canary broken at %zu\n", cs.name, i); oko = false; break; }
        const uint16_t* yold = (const uint16_t*)(yo.data() + kCanary);
        size_t o_cmp = 0, o_diff = 0, o_bad = 0, on_diff = 0, o_closer = 0, m_closer = 0, om_tie = 0; long long o_sum = 0;
        int o_max = 0, on_max = 0; double o_err = 0.0, m_err = 0.0;
        for (uint32_t r = 0; r < T; ++r)
            for (uint32_t n = 0; n < N; ++n) {
                const uint16_t a = yold[(size_t)r * N + n], b = ref[(size_t)r * y_ld + n], m = y[(size_t)r * y_ld + n];
                const bool computed = r >= (uint32_t)off[0] && r < live && sorted[r] >= 0;
                if (!computed) {
                    const uint16_t want = (r >= (uint32_t)off[0] && r < live) ? (uint16_t)0 : b;   /* a dead row in a run: 0 */
                    if (a != want) { if (o_bad < 5) std::printf("FAIL old-%s: row %u col %u is %04x, expected %04x\n", cs.name, r, n, a, want); ++o_bad; }
                    continue;
                }
                ++o_cmp;
                const double v = ref64[(size_t)r * y_ld + n];
                const double ea = std::fabs((double)bf(a) - v), em = std::fabs((double)bf(m) - v);
                if (ea > o_err) o_err = ea;
                if (em > m_err) m_err = em;
                const int d = ulp_dist(a, b);
                if (d) { ++o_diff; o_sum += d; if (d > o_max) o_max = d; }
                const int dm = ulp_dist(a, m);
                if (dm) { ++on_diff; if (dm > on_max) on_max = dm; if (ea < em) ++o_closer; else if (em < ea) ++m_closer; else ++om_tie; }
            }
        if (o_bad) oko = false;
        std::printf("RESULT old-%s: %s | %zu outputs compared: vs the sequential-fp32 reference %zu differ (%.3f %%), max ulp %d, "
                    "mean %.3f; untouched / dead / tail rows wrong %zu | vs the matrix form %zu differ (%.3f %%), max ulp %d; where "
                    "they differ, closer to fp64: old %zu, matrix %zu, tie %zu | max abs err vs fp64: old %.3g, matrix %.3g -> %s\n",
                    cs.name, old_form_name(T), o_cmp, o_diff, o_cmp ? 100.0 * (double)o_diff / (double)o_cmp : 0.0, o_max,
                    o_diff ? (double)o_sum / (double)o_diff : 0.0, o_bad, on_diff, o_cmp ? 100.0 * (double)on_diff / (double)o_cmp : 0.0,
                    on_max, o_closer, m_closer, om_tie, o_err, m_err, oko ? "PASS" : "FAIL");
        ok &= oko;
        sycl::free(dYo, q); sycl::free(dScr, q);
    }
    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dS, q); sycl::free(dO, q); sycl::free(dYb, q); sycl::free(dMap, q);
    return ok;
}

/* Run 268 (MB4): the same live rows at different offsets in the sorted array -- the run starts at `below` = 0, 3, 13,
 * 17, 61 (not multiples of 16) with the same (token, expert) rows each time; each kernel's output per live row is
 * compared bit for bit with its own output at offset 0, and with the sequential reference. T stays above 256 (the
 * dispatch's prefill class) so the old op runs its WMMA prefill form. */
bool offset_sweep(sycl::queue& q) {
    const uint32_t K = 2560, N = 640, E = 2, top_k = 5, R0 = 40, R1 = 19, LIVE = R0 + R1, TAIL = 240;
    const uint32_t belows[] = {0, 3, 13, 17, 61};
    const uint32_t Tmax = 61 + LIVE + TAIL;
    const uint32_t M = (Tmax + top_k - 1) / top_k + 1;
    const uint32_t a_ld = K, w_ld = K;
    std::vector<uint16_t> A((size_t)M * a_ld), W((size_t)E * N * w_ld);
    for (size_t i = 0; i < A.size(); ++i) A[i] = to_bf(((float)(h32(201, i) % 2001) - 1000.0f) / 500.0f);
    for (size_t i = 0; i < W.size(); ++i) W[i] = to_bf(((float)(h32(202, i) % 2001) - 1000.0f) / 1000.0f);
    /* the live rows' tokens, relative to the run (the same at every offset), and their sequential reference */
    std::vector<int32_t> tok(LIVE);
    for (uint32_t j = 0; j < LIVE; ++j) tok[j] = (int32_t)((j * 7u + 3u) % (M * top_k));
    std::vector<uint16_t> ref((size_t)LIVE * N);
    for (uint32_t j = 0; j < LIVE; ++j) {
        const uint32_t e = j < R0 ? 0 : 1, t = (uint32_t)tok[j] / top_k;
        for (uint32_t n = 0; n < N; ++n) {
            volatile float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) acc = acc + bf(A[(size_t)t * a_ld + k]) * bf(W[((size_t)e * N + n) * w_ld + k]);
            ref[(size_t)j * N + n] = to_bf((float)acc);
        }
    }
    uint16_t* dA = sycl::malloc_device<uint16_t>(A.size(), q);
    uint16_t* dW = sycl::malloc_device<uint16_t>(W.size(), q);
    int32_t* dS = sycl::malloc_device<int32_t>(Tmax, q);
    int32_t* dO = sycl::malloc_device<int32_t>(E + 1, q);
    uint16_t* dYn = sycl::malloc_device<uint16_t>((size_t)Tmax * N, q);
    uint16_t* dYo = sycl::malloc_device<uint16_t>((size_t)Tmax * N, q);
    const uint32_t cap = E + (Tmax + 63) / 64 + 1;
    uint32_t* dMap = sycl::malloc_device<uint32_t>(1 + 2 * cap + 64, q);
    const size_t scr_bytes = 4u << 20;
    void* dScr = sycl::malloc_device<uint8_t>(scr_bytes, q);
    q.memcpy(dA, A.data(), A.size() * 2); q.memcpy(dW, W.data(), W.size() * 2); q.wait();
    std::vector<uint16_t> base_n, base_o;
    bool ok = true;
    for (uint32_t below : belows) {
        const uint32_t T = below + LIVE + TAIL;
        std::vector<int32_t> sorted(T), off = {(int32_t)below, (int32_t)(below + R0), (int32_t)(below + LIVE)};
        for (uint32_t r = 0; r < below; ++r) sorted[r] = (int32_t)((r * 5u + 1u) % (M * top_k));   /* another call's rows */
        for (uint32_t j = 0; j < LIVE; ++j) sorted[below + j] = tok[j];
        for (uint32_t r = below + LIVE; r < T; ++r) sorted[r] = -1;
        q.memcpy(dS, sorted.data(), T * 4); q.memcpy(dO, off.data(), 3 * 4);
        q.fill(dYn, (uint16_t)0x1234, (size_t)T * N); q.fill(dYo, (uint16_t)0x1234, (size_t)T * N); q.wait();
        const int rcn = xe_moe_bf16_dpas_launch(&q, dA, (const void* const*)dW, (size_t)N * w_ld * 2, dS, dO, dYn, T, N, K, E,
                                                top_k, 0, a_ld, w_ld, N, dMap, (size_t)(1 + 2 * cap) * 4);
        q.wait();
        const int rco = g_old ? run_old(q, dA, M, dW, dS, dO, dYo, T, N, K, E, top_k, dScr, scr_bytes) : -1;
        std::vector<uint16_t> yn((size_t)T * N), yo((size_t)T * N);
        q.memcpy(yn.data(), dYn, yn.size() * 2); q.memcpy(yo.data(), dYo, yo.size() * 2); q.wait();
        for (int which = 0; which < 2; ++which) {
            if (which == 1 && !g_old) break;
            const std::vector<uint16_t>& yy = which ? yo : yn;
            std::vector<uint16_t>& base = which ? base_o : base_n;
            const int rc = which ? rco : rcn;
            std::vector<uint16_t> live((size_t)LIVE * N);
            for (uint32_t j = 0; j < LIVE; ++j) std::memcpy(&live[(size_t)j * N], &yy[(size_t)(below + j) * N], (size_t)N * 2);
            if (below == 0) base = live;
            uint32_t rows_diff = 0; int max_ulp0 = 0; size_t n_ref = 0; int max_ref = 0;
            for (uint32_t j = 0; j < LIVE; ++j) {
                bool rd = false;
                for (uint32_t n = 0; n < N; ++n) {
                    const int d0 = ulp_dist(live[(size_t)j * N + n], base[(size_t)j * N + n]);
                    if (d0) { rd = true; if (d0 > max_ulp0) max_ulp0 = d0; }
                    const int dr = ulp_dist(live[(size_t)j * N + n], ref[(size_t)j * N + n]);
                    if (dr) { ++n_ref; if (dr > max_ref) max_ref = dr; }
                }
                if (rd) ++rows_diff;
            }
            const bool fine = rc == 0 && (below == 0 || rows_diff == 0);
            if (which == 1 && !fine) ok = false;   /* the old kernel offset-dependent = a defect in today's path */
            if (which == 0 && rc != 0) ok = false;
            std::printf("RESULT offsets %s: run start %2u (T %u): rc %d | rows differing from the start-0 run %u of %u (max ulp %d) | "
                        "vs the sequential reference %zu of %zu outputs differ, max ulp %d%s\n",
                        which ? "old " : "matrix", below, T, rc, rows_diff, LIVE, max_ulp0, n_ref, (size_t)LIVE * N, max_ref,
                        fine ? "" : " <-- OFFSET-DEPENDENT");
        }
    }
    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dS, q); sycl::free(dO, q); sycl::free(dYn, q); sycl::free(dYo, q);
    sycl::free(dMap, q); sycl::free(dScr, q);
    return ok;
}

}  // namespace

int main() {
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q = cpu ? sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order{}}
                        : sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const bool gpu = q.get_device().is_gpu();
    std::printf("moe_bf16_dpas_test: %s (%s)\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                gpu ? "the matrix-unit kernel and the guards" : "the guards only");
    /* run 268 (MB4): the old op from the image's libxe.so, on the GPU only */
    if (gpu) {
        const char* lib = std::getenv("RI_LIBXE");
        if (!lib) lib = "/opt/ri/radiance_home/kernels/libxe.so";
        void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (h) g_old = (LaunchFn)dlsym(h, "r4d_rad_moe_gemm_bf16");
        if (!g_old) std::printf("SKIP the old kernel: %s (%s)\n", lib, h ? "no r4d_rad_moe_gemm_bf16" : dlerror());
        else std::printf("moe_bf16_dpas_test: the old op r4d_rad_moe_gemm_bf16 from %s\n", lib);
    }
    bool ok = true;
    /* 1. the engine's shape class: K 2048, N 192 (not a multiple of 64), 6 experts with a small one, a dead row, a
     *    dropped tail, rows below off[0] */
    ok &= run_case(q, Case{"shape-A", 2048, 192, 6, 2, {131, 3, 64, 70, 200, 65}, 40, 37, 24}, gpu);
    /* 2. one group of K, N exactly 64, an expert of 1 row, a tail of a whole 64-block plus a remainder */
    ok &= run_case(q, Case{"shape-B", 64, 64, 4, 3, {1, 512, 9, 8}, 0, 70, 0}, gpu);
    /* 3. K 2560 (the trunk's), N 320, every expert over a block boundary, a dead row at a block's last slot */
    ok &= run_case(q, Case{"shape-C", 2560, 320, 3, 2, {65, 129, 330}, 63, 5, 7}, gpu);
    /* 4. (run 268, the coordinator) shape-C again on a second seed: the same shapes, other data */
    ok &= run_case(q, Case{"shape-C2", 2560, 320, 3, 2, {65, 129, 330}, 63, 5, 7, 1}, gpu);
    /* 5. (run 268, the guard at one block) the prompts' short last steps at the protected call's real shape: N 640,
     *    K 2560, 2 experts, top_k 5 -- T 300 slots (the dispatch's prefill class is T > 256) with 59 live rows, and
     *    T 180 (a 36-row tail's slots) with 27 live rows; both under the old 512 cut-off */
    ok &= run_case(q, Case{"shape-D", 2560, 640, 2, 5, {40, 19}, 0, 241, 0}, gpu);
    ok &= run_case(q, Case{"shape-E", 2560, 640, 2, 5, {20, 7}, 3, 153, 0}, gpu);
    /* 6. (run 268, MB4) the same live rows at run starts 0 / 3 / 13 / 17 / 61: both kernels, bit for bit across offsets */
    if (gpu) ok &= offset_sweep(q);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
