/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_moe_planes.cpp -- moe_gemm_q at w4nl64a8h over the experts' CANONICAL planes, joined as
 * abi/rad_planes.h defines (the form libref reads): the codes view "codes,table" (u4 codes, low
 * nibble first, indexing 16 f32 levels) and the scale view "scale,scale.1" (E4M3 per row per 64 of
 * K, times one f32). The layout hooks are rad_planes.h's, a copy, so the load does no relayout for
 * the experts (owner note 4: intel-host's CPU must stay cool at load); libr4d's packed row
 * (ports/r4d_moe.cpp, r4d_layout.cpp) rearranges every expert on the host at every load.
 *
 * Arithmetic as libref's (ref_moe.cpp, quant_dot): per 64-K group the sub-sum of level * activation,
 * then acc += (sub * a_scale(row, k / 128)) * (e4m3(scale) * scale.1); rows outside the routing and
 * columns past a narrower class's width are zeros. Experts are two classes by parity (entry e / 2 of
 * class e % 2), as the declaration splits them. A sub-group computes one output; lanes take the
 * weight groups. The planes' offsets are computed on the host from the shapes (rad_planes_plan's
 * arithmetic), so a launch reads nothing back from the card.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "rad_planes.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct PCls {
    const uint8_t* base;     /* the stacked plane, or null */
    const void* const* tab;  /* the pointer table, or null */
    uint64_t stride;         /* bytes an entry of a stacked plane */
    const uint8_t* sbase;
    const void* const* stab;
    uint64_t sstride;
    uint32_t n, k;
    uint32_t c_off, t_off;   /* codes and table inside a codes entry */
    uint32_t s_off, s1_off;  /* scale and scale.1 inside a scale entry */
    uint32_t w_ld, s_ld;     /* LOGICAL form only: row pitches in elements */
};

inline float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}
inline const uint8_t* entry_of(const uint8_t* base, const void* const* tab, uint64_t stride, uint32_t e) {
    return tab ? (const uint8_t*)tab[e] : base + (uint64_t)e * stride;
}

/* LOGICAL: the operands hold the values themselves, as rad-kbench and the oracle pass them
 * (libref's enc_selection_logical): an E4M3 level per element and a bf16 scale product per (row,
 * 64 of K). Otherwise the joined planes. */
template <bool LOGICAL>
void moe_planes_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                       const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                       uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                       uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                       uint32_t a_rows) {
    const int lane = (int)threadIdx.x & 31;
    const uint32_t n = blockIdx.x * 8u + ((uint32_t)threadIdx.x >> 5);
    const uint32_t i = blockIdx.y;
    if (n >= N || i >= T) return;
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
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    float acc = 0.0f;
    if (e >= 0 && j < fc) {
        const uint32_t entry = (uint32_t)e >> 1, r = pp * fc + j;
        const uint8_t* we = entry_of(c.base, c.tab, c.stride, entry);
        const uint8_t* se = entry_of(c.sbase, c.stab, c.sstride, entry);
        const uint8_t* arow = A + (size_t)tok * a_ld;
        if constexpr (LOGICAL) {
            const uint8_t* wrow = we + (size_t)r * c.w_ld;
            const uint16_t* srow = (const uint16_t*)se + (size_t)r * c.s_ld;
            for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
                const uint8_t* wb = wrow + (size_t)g * 64;
                const uint8_t* ab = arow + (size_t)g * 64;
                float sub = 0.0f;
                for (int m = 0; m < 64; ++m) sub += e4m3(wb[m]) * e4m3(ab[m]);
                const float sa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
                const float sb = __uint_as_float((uint32_t)srow[g] << 16);
                acc += (sub * sa) * sb;
            }
        } else {
            const uint8_t* wrow = we + c.c_off + (size_t)r * (c.k / 2u);
            const float* lvp = (const float*)(we + c.t_off);
            const uint8_t* srow = se + c.s_off + (size_t)r * (c.k / 64u);
            const float s1 = *(const float*)(se + c.s1_off);
            float lv[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) lv[t] = lvp[t];
            for (uint32_t g = (uint32_t)lane; g < c.k / 64u; g += 32u) {
                const uint8_t* wb = wrow + (size_t)g * 32;   /* 64 codes, two a byte, low nibble first */
                const uint8_t* ab = arow + (size_t)g * 64;
                float sub = 0.0f;
                for (int m = 0; m < 32; ++m) {
                    const uint32_t b = wb[m];
                    sub += lv[b & 15u] * e4m3(ab[2 * m]);
                    sub += lv[b >> 4] * e4m3(ab[2 * m + 1]);
                }
                const float sa = As[(size_t)tok * as_ld + (g * 64u) / 128u];
                const float sb = e4m3(srow[g]) * s1;
                acc += (sub * sa) * sb;
            }
        }
        for (int x = 16; x >= 1; x >>= 1) acc += __shfl_xor(acc, x);
    }
    if (lane == 0) Y[(size_t)i * y_ld + n] = to_bf16(acc);
}

/* The decode kernel (milestone 2, stage 1, kernel 1): round 2 M6's best configuration brought to the
 * planes form -- sub-group 16, 16 lanes a row along K with a 16-byte load each (32 codes), two rows
 * a team sharing the activation, 512 work-items a work-group (32 teams, 64 rows), one routed row a
 * work-group. Per work-group the activation row is staged once in work-group memory as f16 pairs
 * (exact for E4M3), with its per-128 scales, a 256-entry table byte -> the two levels (f32, from the
 * entry's 16-level table) and the 256 E4M3 values. Per code byte: one table read, one activation
 * read, two f32 FMAs (level * activation is exact in f32). Per 16-byte chunk the partial sum takes
 * the chunk's 64-group E4M3 scale and the activation's per-128 scale, as libref's quant_dot does per
 * 64-group ((sub * sa) * sb); the entry's scale.1 multiplies the row's total at the store. Rounding
 * differs from libref in the order of the sums (a half-group at a time, lanes meeting in a
 * butterfly): a tolerance check, as the old kernel was. Rows outside the routing and columns past a
 * narrower class's width write zeros, as before. K up to kV2MaxK; larger K and the LOGICAL form
 * take the kernel above. RADIANCE_XE_MOE_V1=1 forces the old kernel (the A/B on card1). */
constexpr int kV2WG = 512, kV2LPR = 16, kV2R = 2;
constexpr int kV2Teams = kV2WG / kV2LPR;        /* 32 */
constexpr int kV2RPW = kV2Teams * kV2R;         /* 64 rows a work-group */
constexpr uint32_t kV2MaxK = 4096;

inline float h2f(uint32_t bits16) { return (float)sycl::bit_cast<sycl::half>((uint16_t)bits16); }
inline uint32_t f2h(float f) { return (uint32_t)sycl::bit_cast<uint16_t>((sycl::half)f); }

/* UPL: 16-byte chunks a lane (K / 32 / 16, rounded up; the last may be partial): every lane's weight
 * loads of the row pair are issued together before the arithmetic, so the memory system sees
 * 2 x UPL loads a lane in flight (v2 issued one and waited: 144 us, 114 GB/s, 3x round 2 M6's 43 us
 * for this configuration). The level table is one packed f16 pair a byte and the products are f16,
 * exact (levels and activations are E4M3), summed in f32, as M6's v2 (DEC1). */
template <int UPL>
void moe_planes_v2_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As, PCls c0, PCls c1,
                          const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                          uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
                          uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t a_sorted, uint32_t parts,
                          uint32_t a_rows) {
    XE_SHARED(uint32_t, s_a, kV2MaxK / 2);      /* the activation row as f16 pairs */
    XE_SHARED(float, s_as, kV2MaxK / 128);      /* its per-128 scales */
    XE_SHARED(uint32_t, s_lut, 256);            /* byte -> (level of its low nibble, of its high) as f16 */
    XE_SHARED(float, s_e4, 256);                /* byte -> E4M3 value */
    const int tid = (int)threadIdx.x;
    const uint32_t i = blockIdx.y, nb = blockIdx.x;
    if (i >= T) return;
    /* the routed row's expert and token: uniform over the work-group */
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
    const PCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts;
    const uint32_t fc = e >= 0 ? c.n / parts : 0u;
    const uint32_t K = e >= 0 ? c.k : 0u;
    const uint8_t* we = nullptr;
    const uint8_t* se = nullptr;
    float s1 = 0.0f;
    if (e >= 0) {
        const uint32_t entry = (uint32_t)e >> 1;
        we = entry_of(c.base, c.tab, c.stride, entry);
        se = entry_of(c.sbase, c.stab, c.sstride, entry);
        s1 = *(const float*)(se + c.s1_off);
        const float* lvp = (const float*)(we + c.t_off);
        const uint8_t* arow = A + (size_t)tok * a_ld;
        for (uint32_t b = (uint32_t)tid; b < K / 2u; b += kV2WG)
            s_a[b] = f2h(e4m3(arow[2 * b])) | (f2h(e4m3(arow[2 * b + 1])) << 16);
        for (uint32_t g = (uint32_t)tid; g < K / 128u; g += kV2WG) s_as[g] = As[(size_t)tok * as_ld + g];
        for (uint32_t t = (uint32_t)tid; t < 256u; t += kV2WG) {
            s_e4[t] = e4m3(t);
            s_lut[t] = f2h(lvp[t & 15u]) | (f2h(lvp[t >> 4]) << 16);
        }
    }
    __syncthreads();
    const int team = tid / kV2LPR, l = tid % kV2LPR;
    float acc[kV2R];
    uint32_t rowr[kV2R];
    bool rowok[kV2R];
#pragma unroll
    for (int r = 0; r < kV2R; ++r) {
        const uint32_t n = nb * kV2RPW + (uint32_t)(team * kV2R + r);
        const uint32_t pp = fy ? n / fy : 0u, j = n - pp * fy;
        rowok[r] = e >= 0 && n < N && j < fc;
        rowr[r] = pp * fc + j;
        acc[r] = 0.0f;
    }
    if (e >= 0) {
        const uint32_t CPR = K / 32u;                   /* 16-byte chunks a row */
        const uint32_t rb = K / 2u, sb = K / 64u;       /* bytes a codes row, a scale row */
        const uint8_t* wrow0 = we + c.c_off;
        const uint8_t* srow0 = se + c.s_off;
        /* all weight loads of this lane first (chunk u of row r), then the arithmetic */
        uint4 w[UPL][kV2R];
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
#pragma unroll
            for (int r = 0; r < kV2R; ++r)
                w[u][r] = (jc < CPR && rowok[r]) ? *(const uint4*)(wrow0 + (size_t)rowr[r] * rb + jc * 16u)
                                                 : uint4{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = (uint32_t)l + (uint32_t)u * kV2LPR;
            if (jc >= CPR) break;                       /* uniform over the sub-group's tail */
            const uint32_t k0 = jc * 32u;               /* the chunk's first K index (32 codes) */
            const float asc = s_as[k0 / 128u];
            const uint32_t g = k0 / 64u;
            uint32_t a[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) a[t] = s_a[k0 / 2u + (uint32_t)t];
#pragma unroll
            for (int r = 0; r < kV2R; ++r) {
                const uint32_t d[4] = { w[u][r].x, w[u][r].y, w[u][r].z, w[u][r].w };
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const uint32_t byte = (d[x] >> (8 * b)) & 0xffu;
                        const uint32_t ab = a[x * 4 + b];
                        const uint32_t lv = s_lut[byte];
                        const sycl::half l0 = sycl::bit_cast<sycl::half>((uint16_t)(lv & 0xffffu));
                        const sycl::half l1 = sycl::bit_cast<sycl::half>((uint16_t)(lv >> 16));
                        const sycl::half a0 = sycl::bit_cast<sycl::half>((uint16_t)(ab & 0xffffu));
                        const sycl::half a1 = sycl::bit_cast<sycl::half>((uint16_t)(ab >> 16));
                        s += (float)(sycl::half)(l0 * a0);
                        s += (float)(sycl::half)(l1 * a1);
                    }
                if (rowok[r]) acc[r] += (s * asc) * s_e4[srow0[(size_t)rowr[r] * sb + g]];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kV2R; ++r) {
        float v = acc[r];
        v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 4);
        v += __shfl_xor(v, 2);
        v += __shfl_xor(v, 1);
        const uint32_t n = nb * kV2RPW + (uint32_t)(team * kV2R + r);
        if (l == 0 && n < N) Y[(size_t)i * y_ld + n] = to_bf16(rowok[r] ? v * s1 : 0.0f);
    }
}

/* Where the planes of an entry start: rad_planes_plan's arithmetic for two planes. */
uint32_t first_plane() {
    return (uint32_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
}

/* One class's table: a pointer table (stride 0) or a stacked plane; `bits` the operand dtype's. */
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

/* The two joined views this row reads: codes,table (u4 1x1 + 16 f32) and scale,scale.1 (E4M3 1x64
 * + one f32). */
bool view_ok(const RadEncoding* enc, const int* sel, const RadTensor* pl, int n) {
    /* RADIANCE_XE_NO_PLANES=1: decline, so libr4d's packed row serves (the load A/B of owner note 4) */
    static const bool off = [] { const char* v = std::getenv("RADIANCE_XE_NO_PLANES"); return v && *v == '1'; }();
    if (off || !enc || !sel || !pl) return false;
    if (std::getenv("RAD_R4D_TRACE")) {
        std::fprintf(stderr, "[xe] moe planes layout: %d plane(s)", n);
        for (int i = 0; i < n; ++i)
            std::fprintf(stderr, " [%s dtype %u kind %u block %lld,%lld extent %lld,%lld]", enc->plane[sel[i]].role,
                         (unsigned)pl[i].dtype, (unsigned)enc->plane[sel[i]].kind,
                         (long long)enc->plane[sel[i]].block[0], (long long)enc->plane[sel[i]].block[1],
                         (long long)enc->plane[sel[i]].extent[0], (long long)enc->plane[sel[i]].extent[1]);
        std::fprintf(stderr, "\n");
    }
    if (n != 2) return false;
    const RadEncPlane& a = enc->plane[sel[0]];
    const RadEncPlane& b = enc->plane[sel[1]];
    if (!std::strcmp(a.role, "codes") && !std::strcmp(b.role, "table"))
        return pl[0].dtype == RAD_U4 && a.block[0] == 1 && a.block[1] == 1 && pl[1].dtype == RAD_F32 &&
               b.kind == RAD_PLANE_TABLE && b.extent[0] * b.extent[1] == 16;
    if (!std::strcmp(a.role, "scale") && !std::strcmp(b.role, "scale.1"))
        return pl[0].dtype == RAD_F8E4M3 && a.block[0] == 1 && a.block[1] == 64 && pl[1].dtype == RAD_F32;
    return false;
}

bool v1_forced() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_MOE_V1"); return s && *s == '1'; }();
    return v;
}

}  // namespace

extern "C" int xe_moe_planes_layout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                    const RadTensor* pl, int n, RadLayout* out) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, pl, n, out);
}
extern "C" int xe_moe_planes_relayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_relayout(enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int xe_moe_planes_unrelayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                                        const void* src, int64_t src_bytes, const RadTensor* pl, int n) {
    if (!view_ok(enc, sel, pl, n)) return RAD_E_DTYPE;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, pl, n);
}

/* moe_gemm_q (two classes): a, a_scale, w, w_scale, sorted, offsets, w_odd, w_scale_odd, y */
extern "C" int xe_rad_moe_gemm_w4p(const RadArgs* a, RadStream s) {
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
    if (!wdt || std::strcmp(wdt, "w4nl64a8h")) return R4D_NO(RAD_E_UNSUPPORTED);
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
    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 || sorted->dtype != RAD_I32 ||
        off->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    const long long as_ld = r4d_rows_pitch(As, K / 128, &Ms);
    if (y_ld < 0 || T <= 0 || a_ld < 0 || M <= 0 || as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);
    PCls cl[2]{};
    bool logical = false;
    const uint32_t p0 = first_plane();
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (n_expert + 1 - q) / 2;
        int rc = table_of(W[q], cnt, &cl[q].base, &cl[q].tab, &cl[q].stride);
        if (rc == RAD_OK) rc = table_of(Ws[q], cnt, &cl[q].sbase, &cl[q].stab, &cl[q].sstride);
        if (rc != RAD_OK) return rc;
        /* the joined codes view arrives as u4 (its first plane, as libref's cb_codes reads it); the
         * logical form as E4M3 levels with a bf16 scale product (rad-kbench, the oracle) */
        const bool lg = W[q]->dtype == RAD_F8E4M3 && Ws[q]->dtype == RAD_BF16;
        if (q == 0) logical = lg;
        if (lg != logical || (!lg && W[q]->dtype != RAD_U4)) return R4D_NO(RAD_E_DTYPE);
        if (lg) {
            if (W[q]->rank != 3 || Ws[q]->rank != 3 || W[q]->stride[2] != 1 || Ws[q]->stride[2] != 1)
                return R4D_NO(RAD_E_STRIDE);
            cl[q].w_ld = (uint32_t)W[q]->stride[1];
            cl[q].s_ld = (uint32_t)Ws[q]->stride[1];
        }
        cl[q].n = (uint32_t)Nc[q];
        cl[q].k = (uint32_t)Kc[q];
        cl[q].c_off = p0;
        cl[q].t_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 2));
        cl[q].s_off = p0;
        cl[q].s1_off = (uint32_t)rad_planes_round(p0 + Nc[q] * (Kc[q] / 64));
    }
    try {
        sycl::queue* qq = (sycl::queue*)s;
        if (logical)
            XE_LAUNCH(qq, moe_planes_kernel<true>, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const uint8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
        else if (K <= (long long)kV2MaxK && !v1_forced()) {
            /* chunks a lane: K / 32 chunks a row over 16 lanes, rounded up (both classes' K are equal
             * in the shipped model; the larger decides, the smaller leaves lanes idle in the tail) */
            const int upl = (int)((K / 32 + kV2LPR - 1) / kV2LPR);
            const dim3 grid((unsigned)((N + kV2RPW - 1) / kV2RPW), (unsigned)T), block(kV2WG);
#define XE_MOE_V2(U)                                                                                        \
    XE_LAUNCH_SG(16, qq, moe_planes_v2_kernel<U>, grid, block, 0, (const uint8_t*)Aq->data,                 \
                 (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data, (const int32_t*)off->data, \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,          \
                 (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u, (uint32_t)parts,       \
                 (uint32_t)M)
            switch (upl) {
                case 1: XE_MOE_V2(1); break;
                case 2: XE_MOE_V2(2); break;
                case 3: XE_MOE_V2(3); break;
                case 4: XE_MOE_V2(4); break;
                case 5: XE_MOE_V2(5); break;
                case 6: XE_MOE_V2(6); break;
                case 7: XE_MOE_V2(7); break;
                default: XE_MOE_V2(8); break;   /* K = 4096: 128 chunks, 8 a lane */
            }
#undef XE_MOE_V2
        } else
            XE_LAUNCH(qq, moe_planes_kernel<false>, dim3((unsigned)((N + 7) / 8), (unsigned)T), dim3(256), 0,
                  (const uint8_t*)Aq->data, (const float*)As->data, cl[0], cl[1], (const int32_t*)sorted->data,
                  (const int32_t*)off->data, (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                  (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, a_sorted ? 1u : 0u,
                  (uint32_t)parts, (uint32_t)M);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
