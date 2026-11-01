/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 114, owner notes 131 / 134 item 3).
 *
 * moe_i4i8_test.cpp -- libxe's moe_gemm_q at w4g64i8h ("format B", xe_rad_moe_gemm_i4i8, libxe/xe_moe_i4i8.cpp)
 * against a double-precision reference re-done on the host from the format's definition: per sorted row r of expert
 * e (off[e] <= r < off[e+1]; class e % 2, entry e / 2), output column pp * fy + j of weight row pp * fc + j,
 *   y[r, n] = sum over 64-K groups g of (isum_g * a_scale[tok, g / 2]) * w_scale[e][row, g],
 *   isum_g = sum_k a[tok, k] * w[e][row, k]   (int8 x int4, exact),
 * with tok = sorted[r] / top_k (token rows) or r (sorted rows). w is two's complement int4, element 2i in the low
 * nibble of byte i; w_scale bf16 per (row, 64 of K); a int8 with an f32 scale per (row, 128 of K). A dropped slot
 * (sorted -1) and columns past a narrower class's width are zero. 4 KiB canaries around y. Cases: 1 and 4 tokens,
 * token and sorted rows, a narrower odd class, stacked planes and pointer tables.
 *
 *   moe_i4i8_test LIBXE.so      RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

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
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xA7;

int w_at(const std::vector<uint8_t>& w, size_t row_bytes, size_t base, int64_t row, int64_t k) {
    const uint8_t b = w[base + (size_t)row * row_bytes + (size_t)(k / 2)];
    const int code = (k & 1) ? (b >> 4) : (b & 15);
    return (code ^ 8) - 8;
}

/* One call. `ptr`: the classes as pointer tables (stride 0), as the engine's weight tables pass them. */
bool run_case(sycl::queue& q, LaunchFn fn, bool sorted_a, int tokens, int top_k, int n_expert, int N, int N1, int K,
              int parts, bool ptr, uint64_t seed) {
    const int T = tokens * top_k, AR = sorted_a ? T : tokens;
    const int ne0 = (n_expert + 1) / 2, ne1 = n_expert / 2;
    const int Ncl[2] = { N, N1 };
    /* routing: token t slot s -> expert h % n_expert (distinct experts a token where possible); slot 1 of token 0 dropped */
    std::vector<int> ex((size_t)T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < top_k; ++s) ex[(size_t)(t * top_k + s)] = (int)((h32(seed, t * 97 + s) + s) % (uint32_t)n_expert);
    if (T > 1) ex[1] = -1;
    std::vector<int32_t> off((size_t)n_expert + 1, 0), sorted((size_t)T, -1);
    for (int i = 0; i < T; ++i) if (ex[(size_t)i] >= 0) ++off[(size_t)ex[(size_t)i] + 1];
    for (int e = 0; e < n_expert; ++e) off[(size_t)e + 1] += off[(size_t)e];
    {
        std::vector<int32_t> cur(off.begin(), off.end() - 1);
        for (int i = 0; i < T; ++i) if (ex[(size_t)i] >= 0) sorted[(size_t)cur[(size_t)ex[(size_t)i]]++] = i;
    }
    /* inputs */
    std::vector<int8_t> a((size_t)AR * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(h32(seed + 1, i) % 255u) - 127);
    std::vector<float> as((size_t)AR * (K / 128));
    for (size_t i = 0; i < as.size(); ++i) as[i] = 0.001f + (float)(h32(seed + 2, i) % 1000u) * 1e-5f;
    const size_t rb = (size_t)K / 2;
    std::vector<uint8_t> w[2];
    std::vector<uint16_t> ws[2];
    const int nec[2] = { ne0, ne1 };
    for (int c = 0; c < 2; ++c) {
        w[c].resize((size_t)nec[c] * Ncl[c] * rb);
        for (size_t i = 0; i < w[c].size(); ++i) w[c][i] = (uint8_t)h32(seed + 3 + c, i);
        ws[c].resize((size_t)nec[c] * Ncl[c] * (K / 64));
        for (size_t i = 0; i < ws[c].size(); ++i) ws[c][i] = to_bf(0.002f + (float)(h32(seed + 5 + c, i) % 997u) * 1e-5f);
    }
    /* device buffers */
    const int Nmax = N > N1 ? N : N1;
    const size_t ybytes = (size_t)T * Nmax * 2;
    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    uint8_t* dw[2];
    uint16_t* dws[2];
    for (int c = 0; c < 2; ++c) {
        dw[c] = sycl::malloc_device<uint8_t>(w[c].size() + 1, q);
        dws[c] = sycl::malloc_device<uint16_t>(ws[c].size() + 1, q);
        q.memcpy(dw[c], w[c].data(), w[c].size());
        q.memcpy(dws[c], ws[c].data(), ws[c].size() * 2);
    }
    int32_t* dso = sycl::malloc_device<int32_t>((size_t)T, q);
    int32_t* doff = sycl::malloc_device<int32_t>(off.size(), q);
    uint8_t* dy = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memcpy(da, a.data(), a.size());
    q.memcpy(das, as.data(), as.size() * 4);
    q.memcpy(dso, sorted.data(), sorted.size() * 4);
    q.memcpy(doff, off.data(), off.size() * 4);
    q.memset(dy, kPat, ybytes + 2 * kCanary);
    /* pointer tables: one entry an expert, each its plane */
    std::vector<void*> tw[2], tws[2];
    void** dtab[4] = { nullptr, nullptr, nullptr, nullptr };
    if (ptr)
        for (int c = 0; c < 2; ++c) {
            for (int e = 0; e < nec[c]; ++e) {
                tw[c].push_back(dw[c] + (size_t)e * Ncl[c] * rb);
                tws[c].push_back(dws[c] + (size_t)e * Ncl[c] * (K / 64));
            }
            dtab[2 * c] = sycl::malloc_device<void*>((size_t)nec[c] + 1, q);
            dtab[2 * c + 1] = sycl::malloc_device<void*>((size_t)nec[c] + 1, q);
            q.memcpy(dtab[2 * c], tw[c].data(), tw[c].size() * sizeof(void*));
            q.memcpy(dtab[2 * c + 1], tws[c].data(), tws[c].size() * sizeof(void*));
        }
    q.wait();
    RadTensor t[9];
    t[0] = tn(da, RAD_I8, { AR, K });
    t[1] = tn(das, RAD_F32, { AR, K / 128 });
    for (int c = 0; c < 2; ++c) {
        RadTensor tw3 = tn(dw[c], RAD_I4, { nec[c], Ncl[c], K });
        RadTensor ts3 = tn(dws[c], RAD_BF16, { nec[c], Ncl[c], K / 64 });
        if (ptr) { tw3.data = dtab[2 * c]; tw3.stride[0] = 0; ts3.data = dtab[2 * c + 1]; ts3.stride[0] = 0; }
        t[c == 0 ? 2 : 6] = tw3;
        t[c == 0 ? 3 : 7] = ts3;
    }
    t[4] = tn(dso, RAD_I32, { T });
    t[5] = tn(doff, RAD_I32, { n_expert + 1 });
    t[8] = tn(dy + kCanary, RAD_BF16, { T, Nmax });
    RadParam p[10]{};
    const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
    const long long iv[] = { tokens, N, K, n_expert, top_k, 128, N1, K, parts };
    for (int i = 0; i < 9; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
    char ord[8], dt[16];
    std::snprintf(ord, sizeof ord, "%s", sorted_a ? "sorted" : "token");
    std::snprintf(dt, sizeof dt, "w4g64i8h");
    RadParam ps[2]{};
    ps[0].key = "a_order"; ps[0].kind = RAD_P_STR; ps[0].sval = ord;
    ps[1].key = "dtype"; ps[1].kind = RAD_P_STR; ps[1].sval = dt;
    RadParam all[11];
    for (int i = 0; i < 9; ++i) all[i] = p[i];
    all[9] = ps[0];
    all[10] = ps[1];
    RadArgs args{};
    args.t = t; args.n_t = 9; args.p = all; args.n_p = 11;
    const int rc = fn(&args, (RadStream)&q);
    q.wait_and_throw();
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    q.memcpy(hy.data(), dy, hy.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;
    bool ok = rc == RAD_OK && canary;
    double worst = 0;
    const int fy = Nmax / parts;
    for (int r = 0; r < T && ok; ++r) {
        int e = -1;
        for (int x = 0; x < n_expert; ++x) if (off[(size_t)x] <= r && r < off[(size_t)x + 1]) e = x;
        const int flat = sorted[(size_t)r];
        const int tok = sorted_a ? r : (flat >= 0 ? flat / top_k : 0);
        for (int n = 0; n < Nmax; ++n) {
            uint16_t got;
            std::memcpy(&got, &hy[kCanary + ((size_t)r * Nmax + n) * 2], 2);
            double ref = 0.0, mag = 0.0;
            if (e >= 0 && flat >= 0) {
                const int c = e & 1, entry = e >> 1, fc = Ncl[c] / parts, pp = n / fy, j = n - pp * fy;
                if (j < fc) {
                    const int row = pp * fc + j;
                    const size_t wbase = (size_t)entry * Ncl[c] * rb;
                    for (int g = 0; g < K / 64; ++g) {
                        long long isum = 0;
                        for (int k = g * 64; k < g * 64 + 64; ++k)
                            isum += (long long)a[(size_t)tok * K + k] * w_at(w[c], rb, wbase, row, k);
                        const double sa = as[(size_t)tok * (K / 128) + g / 2];
                        const double sb = bf(ws[c][((size_t)entry * Ncl[c] + row) * (K / 64) + g]);
                        ref += (double)isum * sa * sb;
                        mag += std::fabs((double)isum * sa * sb);
                    }
                }
            }
            const double err = std::fabs((double)bf(got) - ref);
            const double tol = std::fabs(ref) * 0x1p-8 + mag * 0x1p-20 + 1e-30;
            if (err > tol) {
                if (ok) std::printf("  mismatch at row %d col %d: got %.6g want %.6g (tol %.3g)\n", r, n, bf(got), ref, tol);
                ok = false;
            }
            if (std::fabs(ref) > 0) worst = std::max(worst, err / std::fabs(ref));
        }
    }
    std::printf("%s %s tokens %d top_k %d experts %d N %d/%d K %d parts %d %s: rc %d, worst rel %.3g, canaries %s\n",
                ok ? "PASS" : "FAIL", sorted_a ? "sorted" : "token", tokens, top_k, n_expert, N, N1, K, parts,
                ptr ? "tables" : "stacked", rc, worst, canary ? "ok" : "BROKEN");
    for (void* x : { (void*)da, (void*)das, (void*)dso, (void*)doff, (void*)dy }) sycl::free(x, q);
    for (int c = 0; c < 2; ++c) { sycl::free(dw[c], q); sycl::free(dws[c], q); }
    for (void** x : dtab) if (x) sycl::free(x, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moe_i4i8_test LIBXE.so\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_i4i8");
    if (!fn) { std::printf("no xe_rad_moe_gemm_i4i8\n"); return 2; }
    const char* dv = std::getenv("RI_DEVICE");
    sycl::queue q = (dv && !std::strcmp(dv, "cpu")) ? sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order())
                                                    : sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("moe_i4i8_test on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    /* gate_up's shape (token rows, two parts) and down's (sorted rows), at 1 and 4 tokens; a narrower odd class */
    fails += !run_case(q, fn, false, 1, 10, 16, 256, 256, 512, 2, false, 1);
    fails += !run_case(q, fn, false, 4, 10, 16, 256, 128, 512, 2, false, 2);
    fails += !run_case(q, fn, true, 4, 10, 16, 512, 512, 256, 1, false, 3);
    fails += !run_case(q, fn, false, 4, 10, 16, 256, 256, 512, 2, true, 4);
    fails += !run_case(q, fn, true, 1, 10, 16, 512, 512, 256, 1, true, 5);
    std::printf("%s (%d failing case(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
