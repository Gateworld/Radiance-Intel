/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * xe_moe_i4i8.cpp -- moe_gemm_q at w4g64i8h ("format B", owner notes 131 / 134 item 3): the routed experts as
 * two's-complement int4 codes with a bf16 scale per (row, 64 of K), rotated at convert (fwht128), against int8
 * activations with an f32 scale per (row, 128 of K) -- had_quant_act_i8g's codes, row-major -- on the card's
 * int8 path (DP4A) instead of the E4M3 decode the codebook forms need.
 *
 * The experts are read from their CANONICAL planes, as libref reads them: the codes view one i4 plane [N, K],
 * element 2i in the low nibble of byte i (value (code ^ 8) - 8), the scale view one bf16 plane [N, K/64]. Neither
 * is rearranged at load (no layout hook: a single-plane view is stored as it is).
 *
 * Arithmetic as libref's (ref_moe.cpp, quant_dot): per 64-K group the exact int32 sum of code x activation, then
 * acc += (sum * a_scale(row, k / 128)) * w_scale(row, group). The integer sum is taken on biased codes, u = code ^ 8
 * = value + 8 in 0..15, as dot(u, a) - 8 * sum(a), each a DP4A over four byte lanes (the run-111 probe's form,
 * i4i8_probe.cpp). Rows outside the routing and columns past a narrower class's width are zeros. Experts are two
 * classes by parity (entry e / 2 of class e % 2), as the declaration splits them at one or two ranks; the four-class
 * form (four ranks) is not served here. A sub-group of 32 computes one output, lanes over the weight groups -- the
 * first, plain form; the probe's faster mappings (several rows a team, a group a lane pair) come next.
 */
#include "xe_lanes.h"
#include "r4d_args.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

struct ICls {
    const uint8_t* base;     /* the stacked codes plane, or null */
    const void* const* tab;  /* the pointer table, or null */
    uint64_t stride;         /* bytes an entry of a stacked plane */
    const uint8_t* sbase;
    const void* const* stab;
    uint64_t sstride;
    uint32_t n, k;
    uint32_t w_ld, s_ld;     /* row pitches: bytes of codes, bf16 elements of scales */
};

inline const uint8_t* entry_of(const uint8_t* base, const void* const* tab, uint64_t stride, uint32_t e) {
    return tab ? (const uint8_t*)tab[e] : base + (uint64_t)e * stride;
}
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}

/* sum of the four signed byte products, plus acc: DP4A's arithmetic. sycl/ext/oneapi/dot_product.hpp defines its
 * dot_acc out of line, so a second translation unit including it (xe_gemv_i8.cpp does) fails to link; IGC
 * recognises this form. */
inline int32_t dp4(int32_t a, int32_t b, int32_t acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}

/* Eight activation bytes a0..a7 (lo = a0..a3, hi = a4..a7) as their even and odd lanes, to meet the codes of
 * elements 8d .. 8d + 7 (byte b: element 2b low nibble, 2b + 1 high). */
inline int32_t a_even(uint32_t lo, uint32_t hi) {
    return (int32_t)((lo & 0xFFu) | ((lo >> 8) & 0xFF00u) | ((hi & 0xFFu) << 16) | ((hi << 8) & 0xFF000000u));
}
inline int32_t a_odd(uint32_t lo, uint32_t hi) {
    return (int32_t)(((lo >> 8) & 0xFFu) | ((lo >> 16) & 0xFF00u) | ((hi << 8) & 0xFF0000u) | (hi & 0xFF000000u));
}

void moe_i4i8_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                     const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                     uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                     uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    const int lane = (int)threadIdx.x & 31;
    const uint32_t n = blockIdx.x * 8u + ((uint32_t)threadIdx.x >> 5);
    const uint32_t i = blockIdx.y;
    if (n >= N || i >= T) return;   /* whole sub-groups: no barrier below */
    const int32_t flat = sorted[i];
    int32_t e = -1;
    uint32_t tok = 0;
    if (flat >= 0) {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        tok = a_sorted ? i : (uint32_t)flat / top_k;
        if (tok < a_rows) e = (int32_t)lo;
    }
    const uint32_t fy = N / parts, pp = n / fy, j = n - pp * fy;
    const ICls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    float acc = 0.0f;
    if (e >= 0 && j < fc) {
        const uint32_t entry = (uint32_t)e >> 1, r = pp * fc + j;
        const uint8_t* wrow = entry_of(c.base, c.tab, c.stride, entry) + (size_t)r * c.w_ld;
        const uint16_t* srow = (const uint16_t*)entry_of(c.sbase, c.stab, c.sstride, entry) + (size_t)r * c.s_ld;
        const int8_t* arow = A + (size_t)tok * a_ld;
        for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
            const uint32_t* wd = (const uint32_t*)(wrow + (size_t)g * 32);   /* 8 dwords, 64 codes */
            const uint32_t* ad = (const uint32_t*)(arow + (size_t)g * 64);   /* 16 dwords, 64 activations */
            int32_t s = 0, sa = 0;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const uint32_t u = wd[d] ^ 0x88888888u, lo = ad[2 * d], hi = ad[2 * d + 1];
                s = dp4((int32_t)(u & 0x0F0F0F0Fu), a_even(lo, hi), s);
                s = dp4((int32_t)((u >> 4) & 0x0F0F0F0Fu), a_odd(lo, hi), s);
                sa = dp4((int32_t)lo, (int32_t)0x01010101, sa);
                sa = dp4((int32_t)hi, (int32_t)0x01010101, sa);
            }
            const int32_t isum = s - 8 * sa;
            const float fa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
            const float fb = __uint_as_float((uint32_t)srow[g] << 16);
            acc += ((float)isum * fa) * fb;
        }
        for (int x = 16; x >= 1; x >>= 1) acc += __shfl_xor(acc, x);
    }
    if (lane == 0) Y[(size_t)i * y_ld + n] = to_bf16(acc);
}

/* The run form (moe_planes_v2m's shape, xe_moe_planes.cpp, at format B; the probe's L16 / L8 mappings): the
 * work-group whose row starts a run of up to MT routed rows of one expert stages the run's activations and writes
 * them all, so each weight chunk is loaded once for the run; the run's other work-groups return, and a padding row
 * writes its zeros. A team of LPR lanes owns R output columns; lane l loads 16-byte chunks (32 codes) l + u * LPR.
 * The activations are staged as (even, odd) dword pairs a block of eight, so a canonical code dword (byte b: element
 * 2b low, 2b + 1 high) meets its two activation dwords directly; a group's two chunks sit on lanes l and l ^ 1 and
 * their exact sums join by one xor (as floats: |sum| < 2^24) before the group's -8 sum(a) and libref's scaling. */
constexpr int kMT = 4, kWG = 512;
constexpr uint32_t kMaxK = 4096;
template <int UPL, int LPR, int R>
void moe_i4i8_run_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, ICls c0, ICls c1,
                         const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                         uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                         uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    XE_SHARED(uint32_t, s_a, kMT * (kMaxK / 4));   /* (even, odd) dwords a block of eight */
    XE_SHARED(float, s_g, kMT * (kMaxK / 64));     /* each group's activation sum (exact) */
    XE_SHARED(float, s_as, kMT * (kMaxK / 128));   /* the per-128 scales */
    const int tid = (int)threadIdx.x;
    const uint32_t i = blockIdx.y, nb = blockIdx.x;
    if (i >= T) return;
    const int32_t flat = sorted[i];
    int32_t e = -1;
    if (flat >= 0) {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
        }
        e = (int32_t)lo;
    }
    uint32_t rn = 1;
    if (e >= 0) {
        const uint32_t first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
        if ((i - first) % kMT != 0) return;
        rn = (last - i) < (uint32_t)kMT ? (last - i) : (uint32_t)kMT;
    }
    uint32_t tokr[kMT];
    bool liver[kMT];
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        tokr[q] = 0; liver[q] = false;
        if (e >= 0 && (uint32_t)q < rn) {
            const int32_t fq = sorted[i + q];
            const uint32_t t = a_sorted ? i + q : (uint32_t)fq / top_k;
            tokr[q] = t;
            liver[q] = fq >= 0 && t < a_rows;
        }
    }
    const ICls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
    if (e >= 0) {
#pragma unroll
        for (int q = 0; q < kMT; ++q) {
            if ((uint32_t)q >= rn || !liver[q]) continue;
            const uint32_t* arow = (const uint32_t*)(A + (size_t)tokr[q] * a_ld);
            for (uint32_t b = (uint32_t)tid; b < K / 8u; b += kWG) {
                const uint32_t lo = arow[2 * b], hi = arow[2 * b + 1];
                s_a[q * (kMaxK / 4) + 2 * b] = (uint32_t)a_even(lo, hi);
                s_a[q * (kMaxK / 4) + 2 * b + 1] = (uint32_t)a_odd(lo, hi);
            }
            for (uint32_t g = (uint32_t)tid; g < K / 64u; g += kWG) {
                const uint32_t* ag = arow + g * 16u;
                int32_t sa = 0;
                for (int d = 0; d < 16; ++d) sa = dp4((int32_t)ag[d], (int32_t)0x01010101, sa);
                s_g[q * (kMaxK / 64) + g] = (float)sa;
            }
            for (uint32_t g = (uint32_t)tid; g < K / 128u; g += kWG)
                s_as[q * (kMaxK / 128) + g] = As[(size_t)tokr[q] * as_ld + g];
        }
    }
    __syncthreads();
    const int team = tid / LPR, l = tid % LPR;
    float acc[kMT][R];
    uint32_t rowr[R];
    bool rowok[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[r] = e >= 0 && n < N && j < fc;
        rowr[r] = pp * fc + j;
#pragma unroll
        for (int q = 0; q < kMT; ++q) acc[q][r] = 0.0f;
    }
    if (e >= 0) {
        const uint32_t CPR = K / 32u;
        const uint32_t entry = (uint32_t)e >> 1;
        const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry);
        const uint16_t* se = (const uint16_t*)entry_of(c.sbase, c.stab, c.sstride, entry);
        uint4 w[UPL][R];
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * LPR;
#pragma unroll
            for (int r = 0; r < R; ++r)
                w[u][r] = (jc < CPR && rowok[r]) ? *(const uint4*)(we + (size_t)rowr[r] * c.w_ld + jc * 16u)
                                                 : uint4{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * LPR;
            /* No break: the join below is a sub-group shuffle every lane must reach. A chunk past the row (both of a
             * group's, CPR being even) computes on zero codes and adds nothing; its scales are not read. */
            const bool live = jc < CPR;
            const uint32_t g = live ? jc / 2u : 0u;
            float sbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                sbr[r] = rowok[r] && live ? __uint_as_float((uint32_t)se[(size_t)rowr[r] * c.s_ld + g] << 16) : 0.0f;
#pragma unroll
            for (int q = 0; q < kMT; ++q) {
                if ((uint32_t)q >= rn) break;   /* uniform over the work-group */
                const float asc = s_as[q * (kMaxK / 128) + g / 2u], gs = s_g[q * (kMaxK / 64) + g];
                uint32_t a[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) a[t] = live ? s_a[q * (kMaxK / 4) + jc * 8u + (uint32_t)t] : 0u;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t d[4] = { w[u][r].x, w[u][r].y, w[u][r].z, w[u][r].w };
                    int32_t s = 0;
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        const uint32_t uu = d[x] ^ 0x88888888u;
                        s = dp4((int32_t)(uu & 0x0F0F0F0Fu), (int32_t)a[2 * x], s);
                        s = dp4((int32_t)((uu >> 4) & 0x0F0F0F0Fu), (int32_t)a[2 * x + 1], s);
                    }
                    float fs = (float)s;
                    fs += __shfl_xor(fs, 1);   /* the group's other chunk */
                    if ((l & 1) == 0 && rowok[r] && live) acc[q][r] += ((fs - 8.0f * gs) * asc) * sbr[r];
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        if ((uint32_t)q >= rn) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[q][r];   /* the even lanes' sums (the odd lanes hold 0) */
            if constexpr (LPR == 16) v += __shfl_xor(v, 8);
            if constexpr (LPR >= 8) v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] ? v : 0.0f);
        }
    }
}

/* RADIANCE_XE_I4I8_PLAIN=1: the plain form only (a sub-group an output), for A/B and as the reference form */
bool plain_forced() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_I4I8_PLAIN"); return s && *s == '1'; }();
    return v;
}

/* One class's table: a pointer table (stride 0) or a stacked plane. */
int table_of(const RadTensor* t, long long cnt, const uint8_t** base, const void* const** tab, uint64_t* stride) {
    if (!t || !t->data || t->rank < 2 || t->shape[0] < cnt) return R4D_NO(RAD_E_SHAPE);
    if (t->stride[0] == 0) { *tab = (const void* const*)t->data; *base = nullptr; *stride = 0; return RAD_OK; }
    const int64_t bits = rad_dtype_bits(t->dtype);
    if (bits <= 0 || (bits * t->stride[0]) % 8) return R4D_NO(RAD_E_STRIDE);
    *tab = nullptr;
    *base = (const uint8_t*)t->data;
    *stride = (uint64_t)(bits * t->stride[0] / 8);
    return RAD_OK;
}

}  // namespace

/* moe_gemm_q (two classes): a, a_scale, w, w_scale, sorted, offsets, w_odd, w_scale_odd, y */
extern "C" int xe_rad_moe_gemm_i4i8(const RadArgs* a, RadStream s) {
    const RadTensor* Aq = r4d_opt(a, 0);
    const RadTensor* As = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off = r4d_opt(a, 5);
    const RadTensor* W[2] = { r4d_opt(a, 2), r4d_opt(a, 6) };
    const RadTensor* Ws[2] = { r4d_opt(a, 3), r4d_opt(a, 7) };
    const RadTensor* Y = r4d_opt(a, 8);
    if (!Aq || !As || !sorted || !off || !Y || !W[0] || !W[1] || !Ws[0] || !Ws[1]) return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const char* wdt = r4d_gets(a, "dtype");
    if (!wdt || std::strcmp(wdt, "w4g64i8h")) return R4D_NO(RAD_E_UNSUPPORTED);
    long long Nc[2] = { r4d_geti_or(a, "N", 0), 0 }, Kc[2] = { r4d_geti_or(a, "K", 0), 0 };
    Nc[1] = r4d_geti_or(a, "N_odd", Nc[0]);
    Kc[1] = r4d_geti_or(a, "K_odd", Kc[0]);
    const long long parts = r4d_geti_or(a, "parts", 1), n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0), group = r4d_geti_or(a, "group", 128);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0 || group != 128) return R4D_NO(RAD_E_SHAPE);
    long long N = 0, K = 0;
    for (int q = 0; q < 2; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0 || Kc[q] % 128 || Nc[q] % parts) return R4D_NO(RAD_E_SHAPE);
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    if (Aq->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 || sorted->dtype != RAD_I32 ||
        off->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    const long long as_ld = r4d_rows_pitch(As, K / 128, &Ms);
    if (y_ld < 0 || T <= 0 || a_ld < 0 || M <= 0 || as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (a_ld % 4) return R4D_NO(RAD_E_STRIDE);   /* the activation is read a dword at a time */
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);
    ICls cl[2]{};
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (n_expert + 1 - q) / 2;
        if (W[q]->dtype != RAD_I4 || Ws[q]->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        int rc = table_of(W[q], cnt, &cl[q].base, &cl[q].tab, &cl[q].stride);
        if (rc == RAD_OK) rc = table_of(Ws[q], cnt, &cl[q].sbase, &cl[q].stab, &cl[q].sstride);
        if (rc != RAD_OK) return rc;
        /* the canonical planes: rows of K/2 bytes and K/64 scales (a stacked plane's own row stride if it has one) */
        const long long wr = W[q]->rank >= 3 && W[q]->stride[0] != 0 ? W[q]->stride[1] : Kc[q];
        const long long sr = Ws[q]->rank >= 3 && Ws[q]->stride[0] != 0 ? Ws[q]->stride[1] : Kc[q] / 64;
        if (wr < Kc[q] || wr % 8 || sr < Kc[q] / 64) return R4D_NO(RAD_E_STRIDE);
        cl[q].w_ld = (uint32_t)(wr / 2);
        cl[q].s_ld = (uint32_t)sr;
        cl[q].n = (uint32_t)Nc[q];
        cl[q].k = (uint32_t)Kc[q];
    }
    try {
        sycl::queue* qq = (sycl::queue*)s;
        if (!plain_forced() && K <= (long long)kMaxK) {
            /* the run form: 8 lanes a column at K <= 512 (down, 4-8 groups a row), two columns a team; else 16
             * (gate_up) and one column a team -- the probe's best gate_up mapping (L16R1), and at two columns
             * IGC 2.41.5 spills ~259 bytes of <5, 16, 2> (run 114) */
            const int lpr = K <= 512 ? 8 : 16, rpt = lpr == 8 ? 2 : 1;
            const int upl = (int)((K / 32 + lpr - 1) / lpr);
            const unsigned cpw = (unsigned)((kWG / lpr) * rpt);
            const dim3 grid((unsigned)((N + cpw - 1) / cpw), (unsigned)T), block(kWG);
#define XE_I4I8_RUN(U, L, R)                                                                                 \
    XE_LAUNCH_SG(16, qq, (moe_i4i8_run_kernel<U, L, R>), grid, block, 0, (const int8_t*)Aq->data,            \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts, (uint32_t)M)
            if (lpr == 8) {
                if (upl <= 1) XE_I4I8_RUN(1, 8, 2); else XE_I4I8_RUN(2, 8, 2);
            } else switch (upl) {
                case 1: XE_I4I8_RUN(1, 16, 1); break;
                case 2: XE_I4I8_RUN(2, 16, 1); break;
                case 3: XE_I4I8_RUN(3, 16, 1); break;
                case 4: XE_I4I8_RUN(4, 16, 1); break;
                case 5: XE_I4I8_RUN(5, 16, 1); break;
                case 6: XE_I4I8_RUN(6, 16, 1); break;
                case 7: XE_I4I8_RUN(7, 16, 1); break;
                default: XE_I4I8_RUN(8, 16, 1); break;
            }
#undef XE_I4I8_RUN
            return RAD_OK;
        }
        XE_LAUNCH(qq, moe_i4i8_kernel, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const int8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
