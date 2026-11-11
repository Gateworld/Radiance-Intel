/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 274: the decode step's pairs 4 and 5, RADIANCE_XE_GEMV_MULTI).
 *
 * gemv_multi_test.cpp -- several projections of one activation as the standalone GEMV rows issue them and as the paired
 * rows do, on the same deterministic inputs in one process (the paired rows are listed only under
 * RADIANCE_XE_GEMV_MULTI=1; RADIANCE_XE_GEMV_NX=1 selects the form the paired kernels copy -- without it the adapter
 * falls back to the separate launches and the comparison proves nothing, so the test refuses to run):
 *   pair 5   three gemm_nt_q (xe_gemm_i8a8_gemv) over one int8 activation, widths 4096 / 256 / 256 (q|gate, k, v a
 *            rank) vs gemm_nt_q_x3;
 *   pair 4   gemm_nt_q (width 5120, the GDN in-projection) + gemm_nt (xe_gemm_nt_bf16_gemv, width 96, the a|b) over the
 *            activation's codes and its bf16 plane vs gemm_nt_q_bf16_pair.
 * K 2048, M 1 and 4; every output byte for byte; RESULT lines; rc 0 only when every byte matches and the outputs are
 * non-trivial. Runs on the CPU SYCL device (RI_DEVICE=cpu) or the card.
 *
 *   gemv_multi_test [LIBXE.so]        RI_LIBXE names the library when no argument is given
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef int (*CountFn)(void);
typedef const RadKernelInfo* (*AtFn)(int);

/* the row by its entry-point name (several rows serve gemm_nt_q) */
const RadKernelInfo* row_named(void* h, const char* name) {
    CountFn cnt = (CountFn)dlsym(h, "rad_kernel_count");
    AtFn at = (AtFn)dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* r = at(i);
        if (r && r->name && !std::strcmp(r->name, name) && r->launch) return r;
    }
    return nullptr;
}

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float unit(uint64_t a, uint64_t b) { return (float)(h32(a, b) % 2000001u) / 1000000.0f - 1.0f; }
RadTensor tn(void* p, uint32_t dt, int64_t s0, int64_t s1) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 2;
    t.shape[0] = s0; t.shape[1] = s1; t.stride[0] = s1; t.stride[1] = 1;
    return t;
}
RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

int64_t count_diff(const void* a, const void* b, size_t bytes) {
    const uint8_t* x = (const uint8_t*)a; const uint8_t* y = (const uint8_t*)b;
    int64_t d = 0;
    for (size_t i = 0; i < bytes; ++i) d += x[i] != y[i];
    return d;
}

struct I8W { int8_t* b; uint16_t* bs; int64_t n; };

/* One row count: 0 on byte-identical pairs, 1 on a difference, 2 on a refused launch or a missing row. */
int run_m(void* hx, sycl::queue& q, int64_t M, int which) {
    const int64_t K = 2048, G = K / 128;
    const int64_t N3[3] = { 4096, 256, 256 }, NIN = 5120, NAB = 96;
    const RadKernelInfo* r_i8 = row_named(hx, "xe_gemm_i8a8_gemv");
    const RadKernelInfo* r_bf = row_named(hx, "xe_gemm_nt_bf16_gemv");
    const RadKernelInfo* r_x3 = row_named(hx, "xe_gemm_nt_q_x3");
    const RadKernelInfo* r_pr = row_named(hx, "xe_gemm_nt_q_bf16_pair");
    if (!r_i8 || !r_bf) { std::printf("gemv_multi_test: a standalone GEMV row is missing\n"); return 2; }
    if (!r_x3 || !r_pr) { std::printf("gemv_multi_test: no paired rows (RADIANCE_XE_GEMV_MULTI=1?)\n"); return 2; }

    auto sh = [&](size_t bytes) { void* p = sycl::malloc_shared<uint8_t>(bytes, q); std::memset(p, 0, bytes); return p; };
    /* the activation: int8 codes, f32 scales a (row, 128), and its bf16 plane */
    int8_t* a = (int8_t*)sh((size_t)(M * K));
    float* as = (float*)sh((size_t)(M * G) * 4);
    uint16_t* x = (uint16_t*)sh((size_t)(M * K) * 2);
    for (int64_t i = 0; i < M * K; ++i) a[i] = (int8_t)(int)(127.0f * unit(81 + which, (uint64_t)i));
    for (int64_t i = 0; i < M * G; ++i) as[i] = 0.01f + 0.04f * (0.5f + 0.5f * unit(82, (uint64_t)i));
    for (int64_t i = 0; i < M * K; ++i) x[i] = bf16_of(2.0f * unit(83 + which, (uint64_t)i));
    /* the weights: int8 with bf16 scales for the four int8 sets, bf16 for the a|b */
    I8W w[4];
    const int64_t NW[4] = { N3[0], N3[1], N3[2], NIN };
    for (int s = 0; s < 4; ++s) {
        w[s].n = NW[s];
        w[s].b = (int8_t*)sh((size_t)(NW[s] * K));
        w[s].bs = (uint16_t*)sh((size_t)(NW[s] * G) * 2);
        for (int64_t i = 0; i < NW[s] * K; ++i) w[s].b[i] = (int8_t)(int)(127.0f * unit(90 + s, (uint64_t)i));
        for (int64_t i = 0; i < NW[s] * G; ++i) w[s].bs[i] = bf16_of(0.002f + 0.01f * (0.5f + 0.5f * unit(95 + s, (uint64_t)i)));
    }
    uint16_t* wab = (uint16_t*)sh((size_t)(NAB * K) * 2);
    for (int64_t i = 0; i < NAB * K; ++i) wab[i] = bf16_of(0.05f * unit(99, (uint64_t)i));
    /* outputs: [0] the standalone rows', [1] the paired rows' */
    uint16_t* y3[2][3]; uint16_t* yin[2]; uint16_t* yab[2];
    for (int f = 0; f < 2; ++f) {
        for (int s = 0; s < 3; ++s) y3[f][s] = (uint16_t*)sh((size_t)(M * N3[s]) * 2);
        yin[f] = (uint16_t*)sh((size_t)(M * NIN) * 2);
        yab[f] = (uint16_t*)sh((size_t)(M * NAB) * 2);
    }
    int rc = RAD_OK;
    /* ---- the standalone rows */
    for (int s = 0; s < 3; ++s) {
        RadTensor t[5] = { tn(a, RAD_I8, M, K), tn(as, RAD_F32, M, G), tn(w[s].b, RAD_I8, N3[s], K),
                           tn(w[s].bs, RAD_BF16, N3[s], G), tn(y3[0][s], RAD_BF16, M, N3[s]) };
        RadParam p[5] = { pint("M", M), pint("N", N3[s]), pint("K", K), pint("group", 128), pstr("dtype", "i8a8") };
        RadArgs ar{}; ar.t = t; ar.n_t = 5; ar.p = p; ar.n_p = 5; ar.world_size = 1;
        rc |= r_i8->launch(&ar, (RadStream)&q);
    }
    {
        RadTensor t[5] = { tn(a, RAD_I8, M, K), tn(as, RAD_F32, M, G), tn(w[3].b, RAD_I8, NIN, K),
                           tn(w[3].bs, RAD_BF16, NIN, G), tn(yin[0], RAD_BF16, M, NIN) };
        RadParam p[5] = { pint("M", M), pint("N", NIN), pint("K", K), pint("group", 128), pstr("dtype", "i8a8") };
        RadArgs ar{}; ar.t = t; ar.n_t = 5; ar.p = p; ar.n_p = 5; ar.world_size = 1;
        rc |= r_i8->launch(&ar, (RadStream)&q);
        RadTensor tb[3] = { tn(x, RAD_BF16, M, K), tn(wab, RAD_BF16, NAB, K), tn(yab[0], RAD_BF16, M, NAB) };
        RadParam pb[4] = { pint("M", M), pint("N", NAB), pint("K", K), pstr("dtype", "bf16") };
        RadArgs ab{}; ab.t = tb; ab.n_t = 3; ab.p = pb; ab.n_p = 4; ab.world_size = 1;
        rc |= r_bf->launch(&ab, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("gemv_multi_test: a standalone row refused (rc %d)\n", rc); return 2; }
    /* ---- the paired rows */
    {
        RadTensor t[11] = { tn(a, RAD_I8, M, K), tn(as, RAD_F32, M, G),
                            tn(w[0].b, RAD_I8, N3[0], K), tn(w[0].bs, RAD_BF16, N3[0], G), tn(y3[1][0], RAD_BF16, M, N3[0]),
                            tn(w[1].b, RAD_I8, N3[1], K), tn(w[1].bs, RAD_BF16, N3[1], G), tn(y3[1][1], RAD_BF16, M, N3[1]),
                            tn(w[2].b, RAD_I8, N3[2], K), tn(w[2].bs, RAD_BF16, N3[2], G), tn(y3[1][2], RAD_BF16, M, N3[2]) };
        RadParam p[7] = { pint("M", M), pint("K", K), pint("N1", N3[0]), pint("N2", N3[1]), pint("N3", N3[2]),
                          pint("group", 128), pstr("dtype", "i8a8") };
        RadArgs ar{}; ar.t = t; ar.n_t = 11; ar.p = p; ar.n_p = 7; ar.world_size = 1;
        rc |= r_x3->launch(&ar, (RadStream)&q);
    }
    {
        RadTensor t[8] = { tn(a, RAD_I8, M, K), tn(as, RAD_F32, M, G),
                           tn(w[3].b, RAD_I8, NIN, K), tn(w[3].bs, RAD_BF16, NIN, G), tn(yin[1], RAD_BF16, M, NIN),
                           tn(x, RAD_BF16, M, K), tn(wab, RAD_BF16, NAB, K), tn(yab[1], RAD_BF16, M, NAB) };
        RadParam p[6] = { pint("M", M), pint("K", K), pint("N1", NIN), pint("N2", NAB), pint("group", 128),
                          pstr("dtype", "i8a8") };
        RadArgs ar{}; ar.t = t; ar.n_t = 8; ar.p = p; ar.n_p = 6; ar.world_size = 1;
        rc |= r_pr->launch(&ar, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("gemv_multi_test: a paired row refused (rc %d)\n", rc); return 2; }

    /* ---- run 285 (group B of owner note 232's audit): the x3 and a bf16 GEMV in one launch (gemm_nt_q_x3_bf16, listed
     * under RADIANCE_XE_GEMV_MULTI2=1 or b): its three int8 outputs against the standalone rows' above, its bf16 output
     * against the bf16 row over a fourth weight (640 wide, the indexer's qk projection's width); skipped when the row is
     * not listed. */
    const RadKernelInfo* r_x4 = row_named(hx, "xe_gemm_nt_q_x3_bf16");
    int64_t d4[4] = { 0, 0, 0, 0 }, nz4 = 0;
    bool have4 = false;
    if (r_x4) {
        const int64_t N4 = 640;
        uint16_t* w4 = (uint16_t*)sh((size_t)(N4 * K) * 2);
        for (int64_t i = 0; i < N4 * K; ++i) w4[i] = bf16_of(0.05f * unit(98, (uint64_t)i));
        uint16_t* y4[2]; uint16_t* y3b[3];
        for (int f = 0; f < 2; ++f) y4[f] = (uint16_t*)sh((size_t)(M * N4) * 2);
        for (int s = 0; s < 3; ++s) y3b[s] = (uint16_t*)sh((size_t)(M * N3[s]) * 2);
        {
            RadTensor tb[3] = { tn(x, RAD_BF16, M, K), tn(w4, RAD_BF16, N4, K), tn(y4[0], RAD_BF16, M, N4) };
            RadParam pb[4] = { pint("M", M), pint("N", N4), pint("K", K), pstr("dtype", "bf16") };
            RadArgs ab{}; ab.t = tb; ab.n_t = 3; ab.p = pb; ab.n_p = 4; ab.world_size = 1;
            rc |= r_bf->launch(&ab, (RadStream)&q);
        }
        q.wait_and_throw();
        {
            RadTensor t[14] = { tn(a, RAD_I8, M, K), tn(as, RAD_F32, M, G),
                                tn(w[0].b, RAD_I8, N3[0], K), tn(w[0].bs, RAD_BF16, N3[0], G), tn(y3b[0], RAD_BF16, M, N3[0]),
                                tn(w[1].b, RAD_I8, N3[1], K), tn(w[1].bs, RAD_BF16, N3[1], G), tn(y3b[1], RAD_BF16, M, N3[1]),
                                tn(w[2].b, RAD_I8, N3[2], K), tn(w[2].bs, RAD_BF16, N3[2], G), tn(y3b[2], RAD_BF16, M, N3[2]),
                                tn(x, RAD_BF16, M, K), tn(w4, RAD_BF16, N4, K), tn(y4[1], RAD_BF16, M, N4) };
            RadParam p[8] = { pint("M", M), pint("K", K), pint("N1", N3[0]), pint("N2", N3[1]), pint("N3", N3[2]),
                              pint("N4", N4), pint("group", 128), pstr("dtype", "i8a8") };
            RadArgs ar{}; ar.t = t; ar.n_t = 14; ar.p = p; ar.n_p = 8; ar.world_size = 1;
            rc |= r_x4->launch(&ar, (RadStream)&q);
        }
        q.wait_and_throw();
        if (rc != RAD_OK) { std::printf("gemv_multi_test: the x3 + bf16 row refused (rc %d)\n", rc); return 2; }
        for (int s = 0; s < 3; ++s) d4[s] = count_diff(y3[0][s], y3b[s], (size_t)(M * N3[s]) * 2);
        d4[3] = count_diff(y4[0], y4[1], (size_t)(M * N4) * 2);
        for (int64_t i = 0; i < M * N4; ++i) nz4 += y4[1][i] != 0;
        std::printf("RESULT M %lld K %lld: x3 + bf16 (4096 / 256 / 256 int8 + 640 bf16) bytes differing %lld / %lld / %lld / "
                    "%lld, %lld nonzero bf16 outputs\n",
                    (long long)M, (long long)K, (long long)d4[0], (long long)d4[1], (long long)d4[2], (long long)d4[3],
                    (long long)nz4);
        sycl::free(w4, q);
        for (int f = 0; f < 2; ++f) sycl::free(y4[f], q);
        for (int s = 0; s < 3; ++s) sycl::free(y3b[s], q);
        have4 = true;
    }

    int64_t d3[3], nz3 = 0;
    for (int s = 0; s < 3; ++s) d3[s] = count_diff(y3[0][s], y3[1][s], (size_t)(M * N3[s]) * 2);
    for (int s = 0; s < 3; ++s) for (int64_t i = 0; i < M * N3[s]; ++i) nz3 += y3[1][s][i] != 0;
    const int64_t din = count_diff(yin[0], yin[1], (size_t)(M * NIN) * 2), dab = count_diff(yab[0], yab[1], (size_t)(M * NAB) * 2);
    int64_t nzin = 0, nzab = 0;
    for (int64_t i = 0; i < M * NIN; ++i) nzin += yin[1][i] != 0;
    for (int64_t i = 0; i < M * NAB; ++i) nzab += yab[1][i] != 0;
    std::printf("RESULT M %lld K %lld: x3 (4096 / 256 / 256) bytes differing %lld / %lld / %lld, %lld nonzero outputs; "
                "pair (5120 int8 + 96 bf16) bytes differing %lld / %lld, %lld / %lld nonzero outputs\n",
                (long long)M, (long long)K, (long long)d3[0], (long long)d3[1], (long long)d3[2], (long long)nz3,
                (long long)din, (long long)dab, (long long)nzin, (long long)nzab);
    sycl::free(a, q); sycl::free(as, q); sycl::free(x, q); sycl::free(wab, q);
    for (int s = 0; s < 4; ++s) { sycl::free(w[s].b, q); sycl::free(w[s].bs, q); }
    for (int f = 0; f < 2; ++f) { for (int s = 0; s < 3; ++s) sycl::free(y3[f][s], q); sycl::free(yin[f], q); sycl::free(yab[f], q); }
    if (nz3 == 0 || nzin == 0 || nzab == 0) { std::printf("gemv_multi_test: a paired row wrote nothing\n"); return 2; }
    if (have4 && nz4 == 0) { std::printf("gemv_multi_test: the x3 + bf16 row wrote nothing\n"); return 2; }
    return (d3[0] || d3[1] || d3[2] || din || dab || d4[0] || d4[1] || d4[2] || d4[3]) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc >= 2 ? argv[1] : std::getenv("RI_LIBXE");
    if (!lib) { std::fprintf(stderr, "usage: gemv_multi_test LIBXE.so   (or RI_LIBXE)\n"); return 2; }
    const char* nx = std::getenv("RADIANCE_XE_GEMV_NX");
    if (!nx || nx[0] != '1') { std::printf("gemv_multi_test: RADIANCE_XE_GEMV_NX=1 is required (the form the paired kernels copy)\n"); return 2; }
    void* hx = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    std::printf("gemv_multi_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t rows[2] = { 1, 4 };   /* a draft pass's row and a verify's */
    int worst = 0;
    for (int i = 0; i < 2; ++i) {
        const int r = run_m(hx, q, rows[i], i);
        if (r == 2) return 2;
        worst = std::max(worst, r);
    }
    std::printf(worst ? "FAIL gemm_nt_q_x3 / gemm_nt_q_bf16_pair: differ from the standalone GEMV rows\n"
                      : "PASS gemm_nt_q_x3 / gemm_nt_q_bf16_pair: bit-identical to the standalone GEMV rows\n");
    return worst;
}
