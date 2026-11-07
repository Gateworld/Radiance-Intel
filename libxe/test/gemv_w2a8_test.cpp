/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * gemv_w2a8_test.cpp -- libxe's w2a8 GEMV (xe_gemv_w2a8.cpp, the MTP draft head) through its layout
 * hooks and RadArgs adapter, against libref's gemm_zp at ZP_I8_ROW (libref/ref_gemm.cpp: per 128-K
 * group sub = isum - zero * a_sum, acc += (sub * a_scale) * scale in ascending group order, one bf16
 * rounding), re-done here on the host. The activation is written in quant_act_i8's step order, the
 * scale and zero joined by xe_w2a8_relayout from an affine encoding; u8 and u4 zeros. 4 KiB canaries
 * around y.
 *
 *   gemv_w2a8_test [full]  RI_DEVICE=cpu for the CPU SYCL device; `full` adds the draft head's shape
 *                          (N 248320, K 2560) at M 1 and 4 and times it (GB/s of weight bytes)
 * Exit 0 only if every case is within tolerance and every canary holds.
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_encoding.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int xe_rad_gemm_w2a8_gemv(const RadArgs* a, RadStream s);
extern "C" int xe_w2a8_layout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                              RadLayout*);
extern "C" int xe_w2a8_relayout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                                void*, int64_t);

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
int64_t step_pos(int64_t c) { return (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1); }

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xC5;

struct Result { bool ok; double worst; double ms; };

Result run_case(sycl::queue& q, int M, int N, int K, bool z8, uint64_t seed, int reps) {
    const int G = K / 128;
    const int mpad = (M + 15) / 16 * 16;
    /* the activation in column order, then stored in the step order */
    std::vector<int8_t> acol((size_t)M * K), a((size_t)M * K);
    for (size_t i = 0; i < acol.size(); ++i) acol[i] = (int8_t)((int)(h32(seed, i) % 255) - 127);
    for (int m = 0; m < M; ++m)
        for (int c = 0; c < K; ++c) a[(size_t)m * K + step_pos(c)] = acol[(size_t)m * K + c];
    std::vector<float> as((size_t)M), asum((size_t)G * mpad, 0.0f);
    for (int m = 0; m < M; ++m) as[(size_t)m] = std::ldexp(1.0f + (h32(seed ^ 9, m) >> 9) * 0x1p-23f, -7);
    for (int g = 0; g < G; ++g)
        for (int m = 0; m < M; ++m) {
            int s = 0;
            for (int k = g * 128; k < g * 128 + 128; ++k) s += acol[(size_t)m * K + k];
            asum[(size_t)g * mpad + m] = (float)s;
        }
    /* codes, f16 scales, zeros (u8 0..3 or u4 nibbles, low first) */
    std::vector<uint8_t> codes((size_t)N * K / 4, 0);
    std::vector<uint8_t> cv((size_t)N * K);
    for (size_t i = 0; i < cv.size(); ++i) {
        cv[i] = (uint8_t)(h32(seed ^ 7, i) & 3);
        codes[i >> 2] |= (uint8_t)(cv[i] << (2 * (i & 3)));
    }
    std::vector<sycl::half> sc((size_t)N * G);
    std::vector<uint8_t> zv((size_t)N * G);
    for (size_t i = 0; i < sc.size(); ++i) {
        sc[i] = sycl::half(std::ldexp(1.0f + (h32(seed ^ 11, i) >> 13) * 0x1p-19f, -10));
        zv[i] = (uint8_t)(h32(seed ^ 13, i) % 4);
    }
    const int zrb = z8 ? G : (G + 1) / 2;
    std::vector<uint8_t> zp((size_t)N * zrb, 0);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < G; ++g) {
            if (z8) zp[(size_t)n * zrb + g] = zv[(size_t)n * G + g];
            else zp[(size_t)n * zrb + g / 2] |= (uint8_t)(zv[(size_t)n * G + g] << (4 * (g & 1)));
        }

    /* b_scale through the row's hooks */
    RadEncoding enc;
    rad_enc_clear(&enc);
    rad_enc_copy_str(enc.scheme, "affine");
    rad_enc_add_plane(&enc, "codes", RAD_U2, 1, 1);
    rad_enc_add_plane(&enc, "scale", RAD_F16, 1, 128);
    rad_enc_add_plane(&enc, "zero", z8 ? RAD_U8 : RAD_U4, 1, 128);
    const int sel[2] = { 1, 2 };
    RadTensor pl[2] = { tensor(sc.data(), RAD_F16, N, G), tensor(zp.data(), z8 ? RAD_U8 : RAD_U4, N, G) };
    char dts[] = "w2a8";
    RadParam p[5]{};
    const char* keys[4] = { "M", "N", "K", "group" };
    const long long vals[4] = { M, N, K, 128 };
    for (int i = 0; i < 4; ++i) { p[i].key = keys[i]; p[i].kind = RAD_P_INT; p[i].ival = vals[i]; }
    p[4].key = "dtype"; p[4].kind = RAD_P_STR; p[4].sval = dts;
    RadLayout lay{};
    if (xe_w2a8_layout(p, 5, 3, &enc, sel, pl, 2, &lay) != RAD_OK) { std::printf("  layout refused\n"); return {false, 0, 0}; }
    std::vector<uint8_t> joined((size_t)lay.bytes);
    if (xe_w2a8_relayout(p, 5, 3, &enc, sel, pl, 2, joined.data(), lay.bytes) != RAD_OK) {
        std::printf("  relayout refused\n");
        return {false, 0, 0};
    }

    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    uint8_t* db = sycl::malloc_device<uint8_t>(codes.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    float* dsum = sycl::malloc_device<float>(asum.size(), q);
    uint8_t* dbs = sycl::malloc_device<uint8_t>(joined.size(), q);
    const size_t ybytes = (size_t)M * N * 2;
    uint8_t* dyb = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memcpy(da, a.data(), a.size()); q.memcpy(db, codes.data(), codes.size());
    q.memcpy(das, as.data(), as.size() * 4); q.memcpy(dsum, asum.data(), asum.size() * 4);
    q.memcpy(dbs, joined.data(), joined.size());
    q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
    uint16_t* dy = (uint16_t*)(dyb + kCanary);

    RadTensor t[6] = { tensor(da, RAD_I8, M, K), tensor(das, RAD_F32, M, 1), tensor(db, RAD_U2, N, K),
                       tensor(dbs, lay.dtype, 1, lay.bytes), tensor(dy, RAD_BF16, M, N),
                       tensor(dsum, RAD_F32, G, mpad) };
    t[3].rank = 1; t[3].shape[0] = lay.bytes; t[3].stride[0] = 1;
    RadArgs args{};
    args.t = t; args.n_t = 6; args.p = p; args.n_p = 5;
    setenv("RADIANCE_XE_W2A8_R", "0", 1);   /* the one-row form, the reference (unset is R=16 since run 218) */
    if (xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q) != RAD_OK) { std::printf("  launch refused\n"); return {false, 0, 0}; }
    q.wait_and_throw();
    double ms = 0;
    if (reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q);
        q.wait_and_throw();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), dyb, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;

    /* libref's gemm_zp (ZP_I8_ROW), element by element; the comparison allows the reordered group sum */
    double worst = 0;
    bool ok = canary;
    const int rows_checked = N < 512 ? N : 512;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < rows_checked; ++j) {
            const int n = (int)((uint64_t)j * N / rows_checked);
            float acc = 0.0f;
            double mag = 0.0;
            for (int g = 0; g < G; ++g) {
                long long is = 0;
                for (int k = g * 128; k < g * 128 + 128; ++k)
                    is += (long long)acol[(size_t)m * K + k] * cv[(size_t)n * K + k];
                const float sub = (float)is - (float)zv[(size_t)n * G + g] * asum[(size_t)g * mpad + m];
                const float term = (sub * as[(size_t)m]) * (float)sc[(size_t)n * G + g];
                acc += term;
                mag += std::fabs((double)term);
            }
            uint16_t got;
            std::memcpy(&got, &hy[kCanary + ((size_t)m * N + n) * 2], 2);
            const double err = std::fabs((double)bf(got) - (double)bf(to_bf(acc)));
            /* one bf16 ulp of the result plus the reordered f32 sum's bound */
            const double tol = std::fabs((double)bf(to_bf(acc))) * 0x1p-7 + mag * G * 0x1p-23 + 1e-30;
            worst = std::max(worst, err / tol);
            ok &= err <= tol;
        }
    std::printf("  M %d N %6d K %5d zero %s: %s, worst %.3f of tolerance, canaries %s\n", M, N, K, z8 ? "u8" : "u4",
                ok ? "PASS" : "FAIL", worst, canary ? "ok" : "BROKEN");
    if (reps) std::printf("    %.4f ms, %.1f GB/s of weight\n", ms, (double)joined.size() / (ms * 1e6) + (double)N * K / 4 / (ms * 1e6));
    /* run 123 (owner note 151 C): the rows form (RADIANCE_XE_W2A8_R=2 / 4) must give the same bytes; run 189 (owner
     * note 183 item 1): and each with its quads by DP4A (RADIANCE_XE_W2A8_DP4=1); run 212: and the wide form (=16) */
    for (int fd : { 2, 4, 12, 14, 16 }) {
        const int form = fd == 16 ? 16 : fd % 10;
        const bool d4 = fd == 12 || fd == 14;
        char v[4];
        std::snprintf(v, sizeof v, "%d", form);
        setenv("RADIANCE_XE_W2A8_R", v, 1);
        if (d4) setenv("RADIANCE_XE_W2A8_DP4", "1", 1); else unsetenv("RADIANCE_XE_W2A8_DP4");
        q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
        const bool rc = xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q) == RAD_OK;
        q.wait_and_throw();
        double fms = 0;
        if (reps > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q);
            q.wait_and_throw();
            fms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        }
        std::vector<uint8_t> hr(ybytes + 2 * kCanary);
        q.memcpy(hr.data(), dyb, hr.size()).wait();
        const bool same = rc && std::memcmp(hr.data(), hy.data(), hr.size()) == 0;
        ok &= same;
        std::printf("    rows form R=%d%s: %s the one-row form's bytes%s", form, d4 ? " DP4A" : "",
                    same ? "=" : "DIFFERS FROM", reps ? "" : "\n");
        if (reps) std::printf(", %.4f ms\n", fms);
    }
    /* run 219 (owner note 207 item A3a): the wide form over a draft vocabulary (RADIANCE_XE_W2A8_VOCAB) -- every
     * third row and the last; listed rows must be the one-row form's bytes, the others -inf (0xFF80) */
    {
        char vp[64];
        std::snprintf(vp, sizeof vp, "/tmp/w2a8-vocab-%d-%d.ids", N, M);
        std::vector<int32_t> ids;
        for (int n = 0; n < N; n += 3) ids.push_back(n);
        if (ids.back() != N - 1) ids.push_back(N - 1);
        FILE* f = std::fopen(vp, "wb");
        const bool wrote = f && std::fwrite(ids.data(), 4, ids.size(), f) == ids.size();
        if (f) std::fclose(f);
        setenv("RADIANCE_XE_W2A8_R", "16", 1);
        unsetenv("RADIANCE_XE_W2A8_DP4");
        setenv("RADIANCE_XE_W2A8_VOCAB", vp, 1);
        q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
        const bool rc = wrote && xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q) == RAD_OK;
        q.wait_and_throw();
        double vms = 0;
        if (reps > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) xe_rad_gemm_w2a8_gemv(&args, (RadStream)&q);
            q.wait_and_throw();
            vms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        }
        std::vector<uint8_t> hv(ybytes + 2 * kCanary);
        q.memcpy(hv.data(), dyb, hv.size()).wait();
        bool vok = rc;
        for (size_t c = 0; c < kCanary; ++c) vok &= hv[c] == kPat && hv[kCanary + ybytes + c] == kPat;
        std::vector<char> listed((size_t)N, 0);
        for (int32_t n : ids) listed[(size_t)n] = 1;
        for (int m = 0; m < M && vok; ++m)
            for (int n = 0; n < N && vok; ++n) {
                const size_t o = kCanary + ((size_t)m * N + n) * 2;
                uint16_t got, want;
                std::memcpy(&got, &hv[o], 2);
                std::memcpy(&want, &hy[o], 2);
                vok &= listed[(size_t)n] ? got == want : got == 0xFF80u;
            }
        ok &= vok;
        std::printf("    wide form, draft vocabulary of %zu / %d rows: %s%s", ids.size(), N,
                    vok ? "listed rows = the one-row form's bytes, the rest -inf" : "WRONG (bytes, -inf or canaries)",
                    reps ? "" : "\n");
        if (reps) std::printf(", %.4f ms\n", vms);
        /* the sharded head (rad_block_mtp_hc.h passes vocab_off): the same list as global ids at an offset of 1000,
         * with ids below the shard and past its end that this rank must ignore -- the same bytes */
        const long long off = 1000;
        std::vector<int32_t> gids = {5, 17};
        for (int32_t n : ids) gids.push_back((int32_t)(n + off));
        gids.push_back((int32_t)(off + N + 3));
        char gp[64];
        std::snprintf(gp, sizeof gp, "/tmp/w2a8-vocab-g-%d-%d.ids", N, M);
        FILE* g = std::fopen(gp, "wb");
        const bool gw = g && std::fwrite(gids.data(), 4, gids.size(), g) == gids.size();
        if (g) std::fclose(g);
        setenv("RADIANCE_XE_W2A8_VOCAB", gp, 1);
        RadParam p6[6];
        for (int k = 0; k < 5; ++k) p6[k] = p[k];
        p6[5] = RadParam{};
        p6[5].key = "vocab_off"; p6[5].kind = RAD_P_INT; p6[5].ival = off;
        RadArgs args6 = args;
        args6.p = p6; args6.n_p = 6;
        q.memset(dyb, kPat, ybytes + 2 * kCanary).wait();
        const bool grc = gw && xe_rad_gemm_w2a8_gemv(&args6, (RadStream)&q) == RAD_OK;
        q.wait_and_throw();
        std::vector<uint8_t> hg(ybytes + 2 * kCanary);
        q.memcpy(hg.data(), dyb, hg.size()).wait();
        const bool gok = grc && std::memcmp(hg.data(), hv.data(), hg.size()) == 0;
        ok &= gok;
        std::printf("    the same vocabulary as global ids, vocab_off %lld: %s\n", off,
                    gok ? "the same bytes" : "DIFFERS");
        unsetenv("RADIANCE_XE_W2A8_VOCAB");
        std::remove(vp);
        std::remove(gp);
    }
    unsetenv("RADIANCE_XE_W2A8_R");
    unsetenv("RADIANCE_XE_W2A8_DP4");
    sycl::free(da, q); sycl::free(db, q); sycl::free(das, q); sycl::free(dsum, q); sycl::free(dbs, q); sycl::free(dyb, q);
    return {ok, worst, ms};
}

}  // namespace

int main(int argc, char** argv) {
    const bool full = argc > 1 && std::string(argv[1]) == "full";
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  sycl::property::queue::in_order());
    std::printf("gemv_w2a8_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    /* small, odd N, odd G (the u4 zero's high nibble), M 1..8 (the 4-row pass and its remainder) */
    const int small[][3] = { {1, 40, 256}, {2, 77, 384}, {4, 64, 1024}, {5, 33, 640}, {8, 130, 2560} };
    for (auto& c : small) {
        fails += !run_case(q, c[0], c[1], c[2], true, 0x1234 + c[1], 0).ok;
        fails += !run_case(q, c[0], c[1], c[2], false, 0x4321 + c[1], 0).ok;
    }
    if (full) {
        fails += !run_case(q, 1, 248320, 2560, true, 0x77, 20).ok;
        fails += !run_case(q, 4, 248320, 2560, true, 0x99, 10).ok;
    }
    std::printf("gemv_w2a8_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
