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
 * Format C (w4nl64i8h, xe_rad_moe_gemm_w4nl_i8, owner note 147, run 116): the same cases over joined w4nl64a8h
 * entries laid out as xe_moe_planes_relayout writes them (rad_planes.h: codes [N, K/2 bytes] then 16 f32 levels;
 * E4M3 scales [N, K/64] then one f32 scale.1). The reference rounds each entry's levels as libref does,
 * lvl = nearbyint(level * 127 / amax), w = lvl[code], and its scale is e4m3(s) * (scale.1 * (amax / 127)).
 *
 *   moe_i4i8_test LIBXE.so      RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_planes.h"

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

int code_at(const std::vector<uint8_t>& w, size_t row_bytes, size_t base, int64_t row, int64_t k) {
    const uint8_t b = w[base + (size_t)row * row_bytes + (size_t)(k / 2)];
    return (k & 1) ? (b >> 4) : (b & 15);
}
int w_at(const std::vector<uint8_t>& w, size_t row_bytes, size_t base, int64_t row, int64_t k) {
    return (code_at(w, row_bytes, base, row, k) ^ 8) - 8;
}
double e4m3(uint8_t v) {
    const int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
    const double x = ex == 0 ? std::ldexp((double)m, -9) : std::ldexp(1.0 + m / 8.0, ex - 7);
    return s ? -x : x;
}

/* One call. `ptr`: the classes as pointer tables (stride 0), as the engine's weight tables pass them. `fc`: format C. */
bool run_case(sycl::queue& q, LaunchFn fn, bool sorted_a, int tokens, int top_k, int n_expert, int N, int N1, int K,
              int parts, bool ptr, uint64_t seed, bool fc = false) {
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
    std::vector<uint8_t> wsb[2];   /* fc: the scale entries' bytes */
    const int nec[2] = { ne0, ne1 };
    /* bytes an entry (codes, scales) and, fc, the planes' offsets in one (xe_moe_planes.cpp's first_plane) */
    const size_t p0 = fc ? (size_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry)) : 0;
    size_t went[2], sent[2], toff[2], s1off[2];
    std::vector<double> lvl[2], smul[2];   /* fc: each entry's rounded levels (16) and scale.1 x amax / 127 */
    for (int c = 0; c < 2; ++c) {
        toff[c] = (size_t)rad_planes_round((int64_t)(p0 + Ncl[c] * rb));
        s1off[c] = (size_t)rad_planes_round((int64_t)(p0 + (size_t)Ncl[c] * (K / 64)));
        went[c] = fc ? (size_t)rad_planes_round((int64_t)toff[c] + 64) : (size_t)Ncl[c] * rb;
        sent[c] = fc ? (size_t)rad_planes_round((int64_t)s1off[c] + 4) : (size_t)Ncl[c] * (K / 64) * 2;
        w[c].assign((size_t)nec[c] * went[c], 0);
        for (int e = 0; e < nec[c]; ++e)
            for (size_t i = 0; i < (size_t)Ncl[c] * rb; ++i)
                w[c][(size_t)e * went[c] + p0 + i] = (uint8_t)h32(seed + 3 + c, (size_t)e * Ncl[c] * rb + i);
        if (!fc) {
            ws[c].resize((size_t)nec[c] * Ncl[c] * (K / 64));
            for (size_t i = 0; i < ws[c].size(); ++i) ws[c][i] = to_bf(0.002f + (float)(h32(seed + 5 + c, i) % 997u) * 1e-5f);
            wsb[c].resize(ws[c].size() * 2);
            std::memcpy(wsb[c].data(), ws[c].data(), wsb[c].size());
            continue;
        }
        /* the served table (fc's levels, then negated) times a per-entry factor, every third entry off-grid values */
        static const float kTab[8] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f };
        wsb[c].assign((size_t)nec[c] * sent[c], 0);
        lvl[c].resize((size_t)nec[c] * 16);
        smul[c].resize((size_t)nec[c]);
        for (int e = 0; e < nec[c]; ++e) {
            float tab[16];
            const float f = 0.5f + (float)(h32(seed + 9 + c, e) % 100u) * 0.01f;
            for (int t = 0; t < 16; ++t)
                tab[t] = e % 3 == 2 ? (float)((int)(h32(seed + 11 + c, e * 16 + t) % 4001u) - 2000) * 0.01f
                                    : (t < 8 ? kTab[t] : -kTab[t - 8]) * f;
            std::memcpy(&w[c][(size_t)e * went[c] + toff[c]], tab, sizeof tab);
            float amax = 0.0f;
            for (float v : tab) amax = std::max(amax, std::fabs(v));
            for (int t = 0; t < 16; ++t) lvl[c][(size_t)e * 16 + t] = std::nearbyint(tab[t] * 127.0f / amax);
            const float s1 = std::ldexp(1.0f + (float)(e % 5) * 0.125f, -13);
            std::memcpy(&wsb[c][(size_t)e * sent[c] + s1off[c]], &s1, 4);
            smul[c][(size_t)e] = (double)(s1 * (amax / 127.0f));
            for (size_t i = 0; i < (size_t)Ncl[c] * (K / 64); ++i)   /* E4M3 2^-3 .. 2^3, positive and negative */
                wsb[c][(size_t)e * sent[c] + p0 + i] = (uint8_t)(0x20u + h32(seed + 5 + c, (size_t)e * 65536 + i) % 48u) |
                                                       (uint8_t)((h32(seed + 7 + c, i) & 1u) << 7);
        }
    }
    /* device buffers */
    const int Nmax = N > N1 ? N : N1;
    const size_t ybytes = (size_t)T * Nmax * 2;
    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    uint8_t* dw[2];
    uint8_t* dws[2];
    for (int c = 0; c < 2; ++c) {
        dw[c] = sycl::malloc_device<uint8_t>(w[c].size() + 1, q);
        dws[c] = sycl::malloc_device<uint8_t>(wsb[c].size() + 2, q);
        q.memcpy(dw[c], w[c].data(), w[c].size());
        q.memcpy(dws[c], wsb[c].data(), wsb[c].size());
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
                tw[c].push_back(dw[c] + (size_t)e * went[c]);
                tws[c].push_back(dws[c] + (size_t)e * sent[c]);
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
        RadTensor tw3 = tn(dw[c], fc ? RAD_U4 : RAD_I4, { nec[c], Ncl[c], K });
        RadTensor ts3 = tn(dws[c], fc ? RAD_F8E4M3 : RAD_BF16, { nec[c], Ncl[c], K / 64 });
        if (fc) { tw3.stride[0] = (int64_t)went[c] * 2; ts3.stride[0] = (int64_t)sent[c]; }   /* elements an entry */
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
    std::snprintf(dt, sizeof dt, "%s", fc ? "w4nl64i8h" : "w4g64i8h");
    RadParam ps[2]{};
    ps[0].key = "a_order"; ps[0].kind = RAD_P_STR; ps[0].sval = ord;
    ps[1].key = "dtype"; ps[1].kind = RAD_P_STR; ps[1].sval = dt;
    RadParam all[11];
    for (int i = 0; i < 9; ++i) all[i] = p[i];
    all[9] = ps[0];
    all[10] = ps[1];
    RadArgs args{};
    args.t = t; args.n_t = 9; args.p = all; args.n_p = 11;
    if (std::getenv("RI_VERBOSE")) std::printf("  launch: T %d N %d/%d K %d %s\n", T, N, N1, K, fc ? "format C" : "B");
    const int rc = fn(&args, (RadStream)&q);
    q.wait_and_throw();
    if (std::getenv("RI_VERBOSE")) std::printf("  returned rc %d\n", rc);
    std::vector<uint8_t> hy(ybytes + 2 * kCanary);
    if (std::getenv("RI_VERBOSE")) std::printf("  host buffer\n");
    q.memcpy(hy.data(), dy, hy.size()).wait();
    if (std::getenv("RI_VERBOSE")) std::printf("  copied\n");
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i) canary &= hy[i] == kPat && hy[kCanary + ybytes + i] == kPat;
    if (std::getenv("RI_VERBOSE")) std::printf("  canaries %d\n", (int)canary);
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
                const int c = e & 1, entry = e >> 1, fcol = Ncl[c] / parts, pp = n / fy, j = n - pp * fy;
                if (j < fcol) {
                    const int row = pp * fcol + j;
                    const size_t wbase = (size_t)entry * went[c] + p0;
                    for (int g = 0; g < K / 64; ++g) {
                        long long isum = 0;
                        for (int k = g * 64; k < g * 64 + 64; ++k)
                            isum += (long long)a[(size_t)tok * K + k] *
                                    (fc ? (long long)lvl[c][(size_t)entry * 16 + code_at(w[c], rb, wbase, row, k)]
                                        : (long long)w_at(w[c], rb, wbase, row, k));
                        const double sa = as[(size_t)tok * (K / 128) + g / 2];
                        const double sb = fc ? e4m3(wsb[c][(size_t)entry * sent[c] + p0 + (size_t)row * (K / 64) + g]) *
                                                   smul[c][(size_t)entry]
                                             : bf(ws[c][((size_t)entry * Ncl[c] + row) * (K / 64) + g]);
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
    if (std::getenv("RI_VERBOSE")) std::printf("  checked\n");
    std::printf("%s %s%s tokens %d top_k %d experts %d N %d/%d K %d parts %d %s: rc %d, worst rel %.3g, canaries %s\n",
                ok ? "PASS" : "FAIL", fc ? "format C " : "", sorted_a ? "sorted" : "token", tokens, top_k, n_expert, N, N1, K, parts,
                ptr ? "tables" : "stacked", rc, worst, canary ? "ok" : "BROKEN");
    for (void* x : { (void*)da, (void*)das, (void*)dso, (void*)doff, (void*)dy }) sycl::free(x, q);
    for (int c = 0; c < 2; ++c) { sycl::free(dw[c], q); sycl::free(dws[c], q); }
    for (void** x : dtab) if (x) sycl::free(x, q);
    return ok;
}

/* Owner note 147's real-weight check, its device half (run 116): the four real experts formatc_study_test wrote as
 * joined entries (RI_FORMATC_JOINED: <n>.<m>.{c,s}.bin, meta.txt), two classes by parity, through
 * xe_rad_moe_gemm_w4nl_i8 on this queue's device and through libref's ref_moe_gemm_q on the host (f32 out), on the
 * same random int8 activations. libref reads the entries by their headers (rad_planes_find), this kernel by the
 * offsets it assumes: a disagreement in either shows. Bad past 2^-8 |ref| + 2^-12 max |ref|. */
typedef int (*KernelCountFn)(void);
typedef const RadKernelInfo* (*KernelAtFn)(int);

std::vector<uint8_t> read_all(const std::string& p) {
    std::vector<uint8_t> v;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)(n > 0 ? n : 0));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

bool real_case(sycl::queue& q, LaunchFn fn, LaunchFn ref, const std::string& dir, const char* m, bool sorted_a,
               int tokens, int top_k, int parts, uint64_t seed) {
    const int n_expert = 4;
    long long N = 0, K = 0;
    {
        FILE* f = std::fopen((dir + "/meta.txt").c_str(), "r");
        if (!f) { std::printf("FAIL real %s: no meta.txt\n", m); return false; }
        int n, l, e;
        char which[32];
        long long r, c;
        while (std::fscanf(f, "%d %31s %lld %lld %d %d", &n, which, &r, &c, &l, &e) == 6)
            if (!std::strcmp(which, m)) { N = r; K = c; }
        std::fclose(f);
    }
    std::vector<uint8_t> hc[4], hs[4];
    for (int x = 0; x < n_expert; ++x) {
        hc[x] = read_all(dir + "/" + std::to_string(x) + "." + m + ".c.bin");
        hs[x] = read_all(dir + "/" + std::to_string(x) + "." + m + ".s.bin");
        if (hc[x].empty() || hs[x].empty() || N <= 0) { std::printf("FAIL real %s: entry %d unreadable\n", m, x); return false; }
    }
    const int T = tokens * top_k, AR = sorted_a ? T : tokens;
    std::vector<int> ex((size_t)T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < top_k; ++s) ex[(size_t)(t * top_k + s)] = (int)((h32(seed, t * 97 + s) + s) % (uint32_t)n_expert);
    std::vector<int32_t> off((size_t)n_expert + 1, 0), sorted((size_t)T, -1);
    for (int i = 0; i < T; ++i) ++off[(size_t)ex[(size_t)i] + 1];
    for (int e = 0; e < n_expert; ++e) off[(size_t)e + 1] += off[(size_t)e];
    {
        std::vector<int32_t> cur(off.begin(), off.end() - 1);
        for (int i = 0; i < T; ++i) sorted[(size_t)cur[(size_t)ex[(size_t)i]]++] = i;
    }
    std::vector<int8_t> a((size_t)AR * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(h32(seed + 1, i) % 255u) - 127);
    std::vector<float> as((size_t)AR * (K / 128));
    for (size_t i = 0; i < as.size(); ++i) as[i] = 0.001f + (float)(h32(seed + 2, i) % 1000u) * 1e-5f;
    /* device copies and pointer tables; the host's for libref */
    uint8_t* dc[4];
    uint8_t* ds[4];
    for (int x = 0; x < n_expert; ++x) {
        dc[x] = sycl::malloc_device<uint8_t>(hc[x].size(), q);
        ds[x] = sycl::malloc_device<uint8_t>(hs[x].size(), q);
        q.memcpy(dc[x], hc[x].data(), hc[x].size());
        q.memcpy(ds[x], hs[x].data(), hs[x].size());
    }
    void* htab[4][2];      /* [class * 2 + (codes, scales)][entry] */
    void** dtab[4];
    for (int c = 0; c < 2; ++c) {
        void* pc[2] = { dc[c], dc[c + 2] };
        void* ps[2] = { ds[c], ds[c + 2] };
        htab[2 * c][0] = hc[c].data(); htab[2 * c][1] = hc[c + 2].data();
        htab[2 * c + 1][0] = hs[c].data(); htab[2 * c + 1][1] = hs[c + 2].data();
        dtab[2 * c] = sycl::malloc_device<void*>(2, q);
        dtab[2 * c + 1] = sycl::malloc_device<void*>(2, q);
        q.memcpy(dtab[2 * c], pc, sizeof pc);
        q.memcpy(dtab[2 * c + 1], ps, sizeof ps);
    }
    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    int32_t* dso = sycl::malloc_device<int32_t>((size_t)T, q);
    int32_t* doff = sycl::malloc_device<int32_t>(off.size(), q);
    uint16_t* dy = sycl::malloc_device<uint16_t>((size_t)T * N, q);
    q.memcpy(da, a.data(), a.size());
    q.memcpy(das, as.data(), as.size() * 4);
    q.memcpy(dso, sorted.data(), sorted.size() * 4);
    q.memcpy(doff, off.data(), off.size() * 4);
    q.memset(dy, 0xA7, (size_t)T * N * 2);
    q.wait();
    std::vector<float> yref((size_t)T * N, -1.0f);
    RadParam all[11]{};
    const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
    const long long iv[] = { tokens, N, K, n_expert, top_k, 128, N, K, parts };
    for (int i = 0; i < 9; ++i) { all[i].key = ik[i]; all[i].kind = RAD_P_INT; all[i].ival = iv[i]; }
    char ord[8], dt[16];
    std::snprintf(ord, sizeof ord, "%s", sorted_a ? "sorted" : "token");
    std::snprintf(dt, sizeof dt, "w4nl64i8h");
    all[9].key = "a_order"; all[9].kind = RAD_P_STR; all[9].sval = ord;
    all[10].key = "dtype"; all[10].kind = RAD_P_STR; all[10].sval = dt;
    int rc[2];
    for (int side = 0; side < 2; ++side) {   /* 0 libxe on the queue, 1 libref on the host */
        RadTensor t[9];
        t[0] = tn(side ? (void*)a.data() : (void*)da, RAD_I8, { AR, K });
        t[1] = tn(side ? (void*)as.data() : (void*)das, RAD_F32, { AR, K / 128 });
        for (int c = 0; c < 2; ++c) {
            RadTensor tw3 = tn(side ? (void*)htab[2 * c] : (void*)dtab[2 * c], RAD_U4, { 2, N, K });
            RadTensor ts3 = tn(side ? (void*)htab[2 * c + 1] : (void*)dtab[2 * c + 1], RAD_F8E4M3, { 2, N, K / 64 });
            tw3.stride[0] = 0;
            ts3.stride[0] = 0;
            t[c == 0 ? 2 : 6] = tw3;
            t[c == 0 ? 3 : 7] = ts3;
        }
        t[4] = tn(side ? (void*)sorted.data() : (void*)dso, RAD_I32, { T });
        t[5] = tn(side ? (void*)off.data() : (void*)doff, RAD_I32, { n_expert + 1 });
        t[8] = side ? tn(yref.data(), RAD_F32, { T, N }) : tn(dy, RAD_BF16, { T, N });
        RadArgs args{};
        args.t = t; args.n_t = 9; args.p = all; args.n_p = 11;
        rc[side] = side ? ref(&args, nullptr) : fn(&args, (RadStream)&q);
        if (!side) q.wait_and_throw();
    }
    std::vector<uint16_t> hy((size_t)T * N);
    q.memcpy(hy.data(), dy, hy.size() * 2).wait();
    double amax = 0.0;
    for (float v : yref) amax = std::max(amax, (double)std::fabs(v));
    if (std::getenv("RI_VERBOSE")) {
        /* the entries' own headers against the offsets the kernel assumes, and a third opinion on row 0: the format's
         * definition decoded on the host at those offsets */
        const int64_t p0 = rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
        for (int v = 0; v < 2; ++v) {
            const RadPlanesHeader* hh = (const RadPlanesHeader*)(v ? hs[0].data() : hc[0].data());
            const RadPlanesEntry* pe = (const RadPlanesEntry*)((const uint8_t*)hh + sizeof(RadPlanesHeader));
            for (uint32_t x = 0; x < hh->n_planes && x < 2; ++x)
                std::printf("  entry 0 %s plane %s: dtype %u kind %u block %lld,%lld rows %lld cols %lld row_bytes %lld offset %lld\n",
                            v ? "scales" : "codes", pe[x].role, (unsigned)pe[x].dtype, (unsigned)pe[x].kind,
                            (long long)pe[x].block[0], (long long)pe[x].block[1], (long long)pe[x].rows, (long long)pe[x].cols,
                            (long long)pe[x].row_bytes, (long long)pe[x].offset);
        }
        std::printf("  assumed: codes %lld table %lld scale %lld scale.1 %lld\n", (long long)p0,
                    (long long)rad_planes_round(p0 + N * (K / 2)), (long long)p0, (long long)rad_planes_round(p0 + N * (K / 64)));
        int e = -1;
        for (int x = 0; x < n_expert; ++x) if (off[(size_t)x] <= 0 && 0 < off[(size_t)x + 1]) e = x;
        const int tok = sorted_a ? 0 : sorted[0] / top_k;
        const uint8_t* ce = hc[e].data();
        const uint8_t* se = hs[e].data();
        const float* tab = (const float*)(ce + rad_planes_round(p0 + N * (K / 2)));
        float s1, tam = 0.0f;
        std::memcpy(&s1, se + rad_planes_round(p0 + N * (K / 64)), 4);
        for (int t = 0; t < 16; ++t) tam = std::max(tam, std::fabs(tab[t]));
        std::printf("  expert %d table:", e);
        for (int t = 0; t < 16; ++t) std::printf(" %g", tab[t]);
        std::printf(" s1 %g\n", s1);
        const long long fy = N / parts;
        for (int n = 0; n < 4; ++n) {
            const long long row = (n / fy) * (N / parts) + (n % fy);
            double y = 0.0;
            for (long long g = 0; g < K / 64; ++g) {
                long long is = 0;
                for (long long k = g * 64; k < g * 64 + 64; ++k) {
                    const uint8_t b = ce[p0 + row * (K / 2) + k / 2];
                    const int code = (k & 1) ? (b >> 4) : (b & 15);
                    is += (long long)a[(size_t)tok * K + k] * (long long)std::nearbyint(tab[code] * 127.0f / tam);
                }
                y += (double)is * as[(size_t)tok * (K / 128) + g / 2] * e4m3(se[p0 + row * (K / 64) + g]) * (double)(s1 * (tam / 127.0f));
            }
            std::printf("  row 0 col %d: libxe %.6g libref %.6g host-at-offsets %.6g\n", n, bf(hy[(size_t)n]), yref[(size_t)n], y);
        }
    }
    long long bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const double r = yref[i], d = std::fabs((double)bf(hy[i]) - r);
        if (d > std::fabs(r) * 0x1p-8 + amax * 0x1p-12) {
            if (!bad) std::printf("  first bad at row %zu col %zu: got %.6g want %.6g\n", i / (size_t)N, i % (size_t)N, bf(hy[i]), r);
            ++bad;
        }
        if (amax > 0) worst = std::max(worst, d / amax);
    }
    const bool ok = rc[0] == RAD_OK && rc[1] == RAD_OK && bad == 0 && amax > 0;
    std::printf("%s real format C %s (%lld x %lld, 4 experts) %s tokens %d top_k %d parts %d: rc %d / %d, bad %lld of %zu, "
                "worst |d| / max|ref| %.3g, max|ref| %.4g\n", ok ? "PASS" : "FAIL", m, N, K, sorted_a ? "sorted" : "token",
                tokens, top_k, parts, rc[0], rc[1], bad, hy.size(), worst, amax);
    for (void* x : { (void*)da, (void*)das, (void*)dso, (void*)doff, (void*)dy }) sycl::free(x, q);
    for (int x = 0; x < n_expert; ++x) { sycl::free(dc[x], q); sycl::free(ds[x], q); }
    for (void** x : dtab) sycl::free(x, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   /* each case's line survives a crash in a later one */
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
    /* RI_SKIP_B=1: format C's cases only; RI_VERBOSE=1: a line before and after each launch */
    const bool skip_b = std::getenv("RI_SKIP_B") != nullptr;
    if (skip_b) goto format_c;
    /* gate_up's shape (token rows, two parts) and down's (sorted rows), at 1 and 4 tokens; a narrower odd class */
    fails += !run_case(q, fn, false, 1, 10, 16, 256, 256, 512, 2, false, 1);
    fails += !run_case(q, fn, false, 4, 10, 16, 256, 128, 512, 2, false, 2);
    fails += !run_case(q, fn, true, 4, 10, 16, 512, 512, 256, 1, false, 3);
    fails += !run_case(q, fn, false, 4, 10, 16, 256, 256, 512, 2, true, 4);
    fails += !run_case(q, fn, true, 1, 10, 16, 512, 512, 256, 1, true, 5);
    /* the run form's other widths (run 114): K 2560 (16 lanes a column, 5 chunks a lane, gate_up's) and K 384 (8 lanes,
     * the second unit half past the row: a rank's down class) */
    fails += !run_case(q, fn, false, 4, 10, 16, 512, 256, 2560, 2, true, 6);
    fails += !run_case(q, fn, true, 4, 10, 16, 512, 512, 384, 1, true, 7);
format_c:
    /* format C (run 116): the engine's shapes at two cards -- gate_up 768 / 512 columns over K 2560 in two parts, down
     * 2560 columns over K 384 / 256 -- plus a one-token case and stacked entries */
    LaunchFn fc = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_w4nl_i8");
    if (!fc) { std::printf("no xe_rad_moe_gemm_w4nl_i8\n"); return 2; }
    fails += !run_case(q, fc, false, 1, 10, 16, 256, 256, 512, 2, false, 11, true);
    fails += !run_case(q, fc, false, 4, 10, 16, 768, 512, 2560, 2, true, 12, true);
    fails += !run_case(q, fc, true, 4, 10, 16, 512, 512, 384, 1, true, 13, true);
    fails += !run_case(q, fc, true, 1, 10, 16, 512, 512, 256, 1, false, 14, true);
    /* the real-weight check's routing and shapes on synthetic entries (run 116): 4 experts, top_k 2 / 4 */
    fails += !run_case(q, fc, false, 4, 2, 4, 1280, 1280, 2560, 2, true, 15, true);
    fails += !run_case(q, fc, true, 1, 4, 4, 2560, 2560, 640, 1, true, 16, true);
    /* the real-weight check (note 147), when formatc_study_test's entries and a libref are named */
    const char* jd = std::getenv("RI_FORMATC_JOINED");
    const char* lr = std::getenv("RI_LIBREF");
    if (jd && lr) {
        void* hr = dlopen(lr, RTLD_NOW | RTLD_LOCAL);
        KernelCountFn cnt = hr ? (KernelCountFn)dlsym(hr, "rad_kernel_count") : nullptr;
        KernelAtFn at = hr ? (KernelAtFn)dlsym(hr, "rad_kernel_at") : nullptr;
        LaunchFn ref = nullptr;
        for (int i = 0; cnt && at && i < cnt(); ++i)
            if (at(i) && !std::strcmp(at(i)->name, "ref_moe_gemm_q")) ref = at(i)->launch;
        if (!ref) { std::printf("no ref_moe_gemm_q in %s\n", lr); return 2; }
        fails += !real_case(q, fc, ref, jd, "gate_up", false, 4, 2, 2, 21);
        fails += !real_case(q, fc, ref, jd, "gate_up", false, 1, 4, 2, 22);
        fails += !real_case(q, fc, ref, jd, "down", true, 4, 2, 1, 23);
        fails += !real_case(q, fc, ref, jd, "down", true, 1, 4, 1, 24);
    }
    std::printf("%s (%d failing case(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
