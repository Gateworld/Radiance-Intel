/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 190, owner notes 188 / 189: the expert-layout probe, stage 1).
 *
 * fc_tile_probe.cpp -- format C's decode GEMM (moe_i4i8_run_kernel<.., FC>, libxe/xe_moe_i4i8.cpp, reached through
 * libxe's xe_rad_moe_gemm_w4nl_i8 as the engine reaches it) against the same arithmetic over a load-time repack of
 * the same entries (briefs/2026-11-05-expert-layout.md), on one card's decode shapes, in one process:
 *   gate_up  768 / 512 columns (the two parity classes) over K 2560, two parts, token rows;
 *   down     2560 columns over K 384 / 256, sorted rows;
 * at the verify mix (4 tokens x top_k 10 = 40 rows over 27 distinct experts) and at one token (10 rows).
 *
 * The repack (form T1) changes where the bytes are, never a value or the order of a float operation:
 *   - the codes: within each 8-element block of a row, byte b holds element b in its low nibble and element b + 4 in
 *     its high nibble (today: elements 2b and 2b + 1). The 256-entry pair table then yields the levels of elements
 *     0-3 and 4-7 of the block, which meet the activations' dwords in their natural order: no even / odd split, so
 *     the activations are read straight from global memory -- no LDS staging, no per-work-group copy of the rows;
 *   - each entry's pair table (the kernel's s_g words: 0x40000000 | level(low code) | level(high code) << 8, the same
 *     rint(level x 127 / amax) in f32) and its scale factor (scale.1 x amax / 127) precomputed beside the codes: the
 *     work-group copies 1 KB into LDS instead of two 16-level amax loops and 256 rints;
 *   - the scales stay where they are (abl4: 1-4 us).
 * Everything else is today's kernel line for line: the same lanes own the same chunks, each chunk's int32 sum is the
 * same exact sum of the same products, and the per-lane accumulation and the xor tree are the same, so the bf16
 * output must be the same bytes. The probe checks that (and 4 KiB canaries on both outputs) before it times anything.
 *
 *   xe_fc_tile_probe LIBXE.so [LAUNCHES=48] [RUNS=5]   LAUNCHES 0: the byte check only (the CPU device's ctest)
 * RESULT lines: per launch us (host wall over LAUNCHES back-to-back launches, the median of RUNS), today's and T1's.
 */
#include <sycl/sycl.hpp>

#include "xe_lanes.h"
#include "rad_abi.h"
#include "rad_planes.h"

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

/* ---- today's kernel's helpers and constants, verbatim (xe_moe_i4i8.cpp) */
constexpr int kMT = 4, kWG = 512;
inline float e4m3_to_f32(uint32_t v) {
    const uint32_t s = (v & 0x80u) << 24, ex = (v >> 3) & 15u, m = v & 7u;
    if (ex == 15u && m == 7u) return __uint_as_float(s | 0x7fc00000u);
    if (ex == 0u) return __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));   /* m x 2^-9 */
    return __uint_as_float(s | ((ex + 120u) << 23) | (m << 20));
}
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}
inline int32_t dp4(int32_t a, int32_t b, int32_t acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}

/* One class of the repacked entries. An entry: the codes [n, k/2] at 0, the 256 table words at tab_off, the f32 scale
 * factor at smul_off; the scales are today's entries (E4M3 [n, k/64] at s_off). */
struct TCls {
    const uint8_t* base;
    uint64_t stride;
    const uint8_t* sbase;
    uint64_t sstride;
    uint32_t n, k, w_ld, s_ld, s_off, tab_off, smul_off;
};

/* ---- T1: moe_i4i8_run_kernel<UPL, LPR, R, GL, FC = true> with the repacked reads (see the header). The tail skip is
 * left out: the probe's experts are all resident, as the skip's own off-device test would find. */
template <int UPL, int LPR, int R, bool GL>
void t1_run_kernel(const int8_t* __restrict__ A, const float* __restrict__ As, TCls c0, TCls c1,
                   const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
                   uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld,
                   uint32_t y_ld, uint32_t a_sorted, uint32_t parts, uint32_t a_rows) {
    XE_SHARED(float, s_g, 256);   /* the entry's pair table */
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
    const TCls& c = (e >= 0 && (e & 1)) ? c1 : c0;
    const uint32_t fy = N / parts, fc = e >= 0 ? c.n / parts : 0u, K = e >= 0 ? c.k : 0u;
    if (e >= 0) {
        const uint32_t cpw = (uint32_t)((kWG / LPR) * R), n0 = nb * cpw, n1 = n0 + cpw < N ? n0 + cpw : N;
        const uint32_t a = fy ? n0 % fy : 0u;
        const bool live = n0 < N && (n1 - n0 >= fy || a < fc || n0 + (fy - a) < n1);
        if (!live) {
            const int team = tid / LPR, l = tid % LPR;
            if (l == 0)
                for (int q = 0; q < kMT; ++q) {
                    if ((uint32_t)q >= rn) break;
                    for (int r = 0; r < R; ++r) {
                        const uint32_t n = n0 + (uint32_t)(team * R + r);
                        if (n < N) Y[(size_t)(i + q) * y_ld + n] = 0;
                    }
                }
            return;
        }
    }
    const uint8_t* ent = e >= 0 ? c.base + (uint64_t)((uint32_t)e >> 1) * c.stride : nullptr;
    if (e >= 0 && tid < 256) s_g[tid] = __uint_as_float(((const uint32_t*)(ent + c.tab_off))[tid]);
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
        const uint8_t* we = ent;
        const uint8_t* se8 = c.sbase + (uint64_t)((uint32_t)e >> 1) * c.sstride + c.s_off;
        const float smul = *(const float*)(ent + c.smul_off);
        constexpr int CH = GL ? 2 : 1;
        uint4 w[UPL][R][CH];
#pragma unroll
        for (int u = 0; u < UPL; ++u)
#pragma unroll
            for (int ch = 0; ch < CH; ++ch) {
                const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) + (uint32_t)ch : (uint32_t)l + (uint32_t)u * LPR;
#pragma unroll
                for (int r = 0; r < R; ++r)
                    w[u][r][ch] = (jc < CPR && rowok[r]) ? *(const uint4*)(we + (size_t)rowr[r] * c.w_ld + jc * 16u)
                                                         : uint4{0, 0, 0, 0};
            }
#pragma unroll
        for (int u = 0; u < UPL; ++u) {
            const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) : (uint32_t)l + (uint32_t)u * LPR;
            const bool live = jc < CPR;
            const uint32_t g = live ? jc / 2u : 0u;
            float sbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                sbr[r] = rowok[r] && live ? e4m3_to_f32(se8[(size_t)rowr[r] * c.s_ld + g]) * smul : 0.0f;
            uint32_t lv[R][CH][8];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int ch = 0; ch < CH; ++ch) {
                    const uint32_t d[4] = { w[u][r][ch].x, w[u][r][ch].y, w[u][r][ch].z, w[u][r][ch].w };
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                        for (int bb = 0; bb < 4; ++bb) {
                            const uint32_t ev = __float_as_uint(s_g[(d[x] >> (8 * bb)) & 255u]);
                            lo8 |= (ev & 255u) << (8 * bb);
                            hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                        }
                        lv[r][ch][2 * x] = lo8;        /* elements 8x' + 0..3 of the block */
                        lv[r][ch][2 * x + 1] = hi8;    /* elements 8x' + 4..7 */
                    }
                }
#pragma unroll
            for (int q = 0; q < kMT; ++q) {
                if ((uint32_t)q >= rn) break;   /* uniform over the work-group */
                /* the activations in their own order, from global memory (no staging): a chunk's 32 bytes = 2 uint4 */
                const bool rd = liver[q] && live;
                const float asc = rd ? As[(size_t)tokr[q] * as_ld + g / 2u] : 0.0f;
                const uint4* ap = (const uint4*)(A + (size_t)tokr[q] * a_ld);
                uint32_t a[8 * CH];
#pragma unroll
                for (int v = 0; v < 2 * CH; ++v) {
                    const uint4 x = rd ? ap[jc * 2u + (uint32_t)v] : uint4{0, 0, 0, 0};
                    a[4 * v] = x.x; a[4 * v + 1] = x.y; a[4 * v + 2] = x.z; a[4 * v + 3] = x.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int32_t s = 0;
#pragma unroll
                    for (int ch = 0; ch < CH; ++ch)
#pragma unroll
                        for (int x = 0; x < 4; ++x) {
                            s = dp4((int32_t)lv[r][ch][2 * x], (int32_t)a[8 * ch + 2 * x], s);
                            s = dp4((int32_t)lv[r][ch][2 * x + 1], (int32_t)a[8 * ch + 2 * x + 1], s);
                        }
                    float fs = (float)s;
                    if constexpr (!GL) fs += __shfl_xor(fs, 1);   /* the group's other chunk */
                    if ((GL || (l & 1) == 0) && rowok[r] && live) acc[q][r] += (fs * asc) * sbr[r];
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kMT; ++q) {
        if ((uint32_t)q >= rn) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[q][r];
            if constexpr (LPR == 16) v += __shfl_xor(v, 8);
            if constexpr (LPR >= 8) v += __shfl_xor(v, 4);
            if constexpr (LPR >= 4) v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            const uint32_t n = nb * ((kWG / LPR) * R) + (uint32_t)(team * R + r);
            if (l == 0 && n < N) Y[(size_t)(i + q) * y_ld + n] = to_bf16(rowok[r] && liver[q] ? v : 0.0f);
        }
    }
}

/* the launcher's choice of instance, as moe_i4i8()'s run form makes it */
void t1_launch(sycl::queue& q, const int8_t* A, const float* As, const TCls& c0, const TCls& c1, const int32_t* sorted,
               const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert, uint32_t top_k,
               uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, bool a_sorted, uint32_t parts, uint32_t M) {
    const bool gl = K <= 512;
    const int lpr = gl ? 2 : 16, rpt = gl ? 2 : 1;
    const int upl = gl ? (int)((K / 64 + lpr - 1) / lpr) : (int)((K / 32 + lpr - 1) / lpr);
    const unsigned cpw = (unsigned)((kWG / lpr) * rpt);
    const dim3 grid((N + cpw - 1) / cpw, T), block(kWG);
#define T1(U, L, R, G)                                                                                      \
    XE_LAUNCH_SG(16, &q, (t1_run_kernel<U, L, R, G>), grid, block, 0, A, As, c0, c1, sorted, off, Y, T, N,  \
                 n_expert, top_k, a_ld, as_ld, y_ld, a_sorted ? 1u : 0u, parts, M)
    if (gl) switch (upl) {
        case 1: T1(1, 2, 2, true); break;
        case 2: T1(2, 2, 2, true); break;
        case 3: T1(3, 2, 2, true); break;
        default: T1(4, 2, 2, true); break;
    } else switch (upl) {
        case 4: T1(4, 16, 1, false); break;
        case 5: T1(5, 16, 1, false); break;
        case 6: T1(6, 16, 1, false); break;
        default: T1(8, 16, 1, false); break;
    }
#undef T1
}

/* ---- inputs, as moe_i4i8_test.cpp's format C cases build them */
uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
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

struct Shape { const char* name; int N, N1, K, K1, parts; bool sorted_a; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

/* One shape at `tokens` tokens: today's kernel and T1 on the same inputs; the bytes compared; then (launches > 0) the
 * per-launch times. Returns false on a mismatch or a broken canary. */
bool run_shape(sycl::queue& q, LaunchFn fn, const Shape& sh, int tokens, int launches, int runs, uint64_t seed) {
    const int top_k = 10, n_expert = 512, T = tokens * top_k, AR = sh.sorted_a ? T : tokens;
    const int Ncl[2] = { sh.N, sh.N1 }, Kcl[2] = { sh.K, sh.K1 };
    const int Kmax = std::max(sh.K, sh.K1), Nmax = std::max(sh.N, sh.N1);
    const int ne[2] = { (n_expert + 1) / 2, n_expert / 2 };
    /* the routing: the verify mix's 27 distinct experts (or 10 at one token), distinct within a token */
    const int pool_n = tokens == 1 ? 10 : 27;
    std::vector<int> pool;
    for (uint64_t x = 0; (int)pool.size() < pool_n; ++x) {
        const int e = (int)(h32(seed, 1000 + x) % (uint32_t)n_expert);
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    std::vector<int> ex((size_t)T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0, k = 0; s < top_k; ++k) {
            const int e = pool[(size_t)(h32(seed + 50, (uint64_t)t * 1000 + k) % (uint32_t)pool_n)];
            bool dup = false;
            for (int z = 0; z < s; ++z) dup |= ex[(size_t)(t * top_k + z)] == e;
            if (!dup) ex[(size_t)(t * top_k + s++)] = e;
        }
    std::vector<int32_t> off((size_t)n_expert + 1, 0), sorted((size_t)T, -1);
    for (int i = 0; i < T; ++i) ++off[(size_t)ex[(size_t)i] + 1];
    for (int e = 0; e < n_expert; ++e) off[(size_t)e + 1] += off[(size_t)e];
    {
        std::vector<int32_t> cur(off.begin(), off.end() - 1);
        for (int i = 0; i < T; ++i) sorted[(size_t)cur[(size_t)ex[(size_t)i]]++] = i;
    }
    std::vector<int8_t> a((size_t)AR * Kmax);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(h32(seed + 1, i) % 255u) - 127);
    std::vector<float> as((size_t)AR * (Kmax / 128));
    for (size_t i = 0; i < as.size(); ++i) as[i] = 0.001f + (float)(h32(seed + 2, i) % 1000u) * 1e-5f;
    /* today's joined entries (xe_moe_planes_relayout's layout) and T1's repack of them */
    const size_t p0 = (size_t)rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
    std::vector<uint8_t> w[2], wsb[2], tw[2];
    size_t went[2], sent[2], toff[2], s1off[2], tent[2], ttab[2], tsm[2];
    static const float kTab[8] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f };
    for (int c = 0; c < 2; ++c) {
        const size_t rb = (size_t)Kcl[c] / 2;
        toff[c] = (size_t)rad_planes_round((int64_t)(p0 + Ncl[c] * rb));
        s1off[c] = (size_t)rad_planes_round((int64_t)(p0 + (size_t)Ncl[c] * (Kcl[c] / 64)));
        went[c] = (size_t)rad_planes_round((int64_t)toff[c] + 64);
        sent[c] = (size_t)rad_planes_round((int64_t)s1off[c] + 4);
        ttab[c] = (size_t)rad_planes_round((int64_t)(Ncl[c] * rb));
        tsm[c] = ttab[c] + 1024;
        tent[c] = (size_t)rad_planes_round((int64_t)tsm[c] + 4);
        w[c].assign((size_t)ne[c] * went[c], 0);
        wsb[c].assign((size_t)ne[c] * sent[c], 0);
        tw[c].assign((size_t)ne[c] * tent[c], 0);
        for (int e = 0; e < ne[c]; ++e) {
            uint8_t* cw = &w[c][(size_t)e * went[c] + p0];
            for (size_t i = 0; i < (size_t)Ncl[c] * rb; ++i) cw[i] = (uint8_t)h32(seed + 3 + c, (size_t)e * Ncl[c] * rb + i);
            float tab[16];
            const float f = 0.5f + (float)(h32(seed + 9 + c, e) % 100u) * 0.01f;
            for (int t = 0; t < 16; ++t)
                tab[t] = e % 3 == 2 ? (float)((int)(h32(seed + 11 + c, e * 16 + t) % 4001u) - 2000) * 0.01f
                                    : (t < 8 ? kTab[t] : -kTab[t - 8]) * f;
            std::memcpy(&w[c][(size_t)e * went[c] + toff[c]], tab, sizeof tab);
            const float s1 = std::ldexp(1.0f + (float)(e % 5) * 0.125f, -13);
            std::memcpy(&wsb[c][(size_t)e * sent[c] + s1off[c]], &s1, 4);
            for (size_t i = 0; i < (size_t)Ncl[c] * (Kcl[c] / 64); ++i)
                wsb[c][(size_t)e * sent[c] + p0 + i] = (uint8_t)(0x20u + h32(seed + 5 + c, (size_t)e * 65536 + i) % 48u) |
                                                       (uint8_t)((h32(seed + 7 + c, i) & 1u) << 7);
            /* T1: the codes with each 8-element block's nibbles reordered (byte b: elements b, b + 4) */
            uint8_t* t1 = &tw[c][(size_t)e * tent[c]];
            for (int row = 0; row < Ncl[c]; ++row)
                for (int k0 = 0; k0 < Kcl[c]; k0 += 8) {
                    int cd[8];
                    for (int z = 0; z < 8; ++z) {
                        const uint8_t b = cw[(size_t)row * rb + (size_t)((k0 + z) / 2)];
                        cd[z] = ((k0 + z) & 1) ? (b >> 4) : (b & 15);
                    }
                    for (int b = 0; b < 4; ++b) t1[(size_t)row * rb + (size_t)(k0 / 2 + b)] = (uint8_t)(cd[b] | (cd[b + 4] << 4));
                }
            /* T1: the pair table and the scale factor, as the kernel computes them in f32 */
            float amax = 0.0f;
            for (int t = 0; t < 16; ++t) amax = std::fmax(amax, std::fabs(tab[t]));
            uint32_t* tt = (uint32_t*)(t1 + ttab[c]);
            for (int x = 0; x < 256; ++x) {
                const float l0 = tab[x & 15], l1 = tab[x >> 4];
                const int i0 = amax > 0.0f ? (int)std::nearbyint(l0 * 127.0f / amax) : (int)l0;
                const int i1 = amax > 0.0f ? (int)std::nearbyint(l1 * 127.0f / amax) : (int)l1;
                tt[x] = 0x40000000u | ((uint32_t)i0 & 255u) | (((uint32_t)i1 & 255u) << 8);
            }
            const float smul = amax > 0.0f ? s1 * (amax / 127.0f) : s1;
            std::memcpy(t1 + tsm[c], &smul, 4);
        }
    }
    /* device buffers */
    const size_t ybytes = (size_t)T * Nmax * 2;
    int8_t* da = sycl::malloc_device<int8_t>(a.size(), q);
    float* das = sycl::malloc_device<float>(as.size(), q);
    uint8_t *dw[2], *dws[2], *dtw[2];
    for (int c = 0; c < 2; ++c) {
        dw[c] = sycl::malloc_device<uint8_t>(w[c].size() + 64, q);
        dws[c] = sycl::malloc_device<uint8_t>(wsb[c].size() + 64, q);
        dtw[c] = sycl::malloc_device<uint8_t>(tw[c].size() + 64, q);
        q.memcpy(dw[c], w[c].data(), w[c].size());
        q.memcpy(dws[c], wsb[c].data(), wsb[c].size());
        q.memcpy(dtw[c], tw[c].data(), tw[c].size());
    }
    int32_t* dso = sycl::malloc_device<int32_t>((size_t)T, q);
    int32_t* doff = sycl::malloc_device<int32_t>(off.size(), q);
    uint8_t* dy0 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    uint8_t* dy1 = sycl::malloc_device<uint8_t>(ybytes + 2 * kCanary, q);
    q.memcpy(da, a.data(), a.size());
    q.memcpy(das, as.data(), as.size() * 4);
    q.memcpy(dso, sorted.data(), sorted.size() * 4);
    q.memcpy(doff, off.data(), off.size() * 4);
    q.memset(dy0, kPat, ybytes + 2 * kCanary);
    q.memset(dy1, kPat, ybytes + 2 * kCanary);
    q.wait();
    /* today's call, as moe_i4i8_test.cpp makes it (stacked joined entries) */
    RadTensor t[9];
    t[0] = tn(da, RAD_I8, { AR, Kmax });
    t[1] = tn(das, RAD_F32, { AR, Kmax / 128 });
    for (int c = 0; c < 2; ++c) {
        RadTensor tw3 = tn(dw[c], RAD_U4, { ne[c], Ncl[c], Kcl[c] });
        RadTensor ts3 = tn(dws[c], RAD_F8E4M3, { ne[c], Ncl[c], Kcl[c] / 64 });
        tw3.stride[0] = (int64_t)went[c] * 2;
        ts3.stride[0] = (int64_t)sent[c];
        t[c == 0 ? 2 : 6] = tw3;
        t[c == 0 ? 3 : 7] = ts3;
    }
    t[4] = tn(dso, RAD_I32, { T });
    t[5] = tn(doff, RAD_I32, { n_expert + 1 });
    t[8] = tn(dy0 + kCanary, RAD_BF16, { T, Nmax });
    RadParam all[11]{};
    const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
    const long long iv[] = { AR, sh.N, sh.K, n_expert, top_k, 128, sh.N1, sh.K1, sh.parts };
    for (int i = 0; i < 9; ++i) { all[i].key = ik[i]; all[i].kind = RAD_P_INT; all[i].ival = iv[i]; }
    char ord[8], dt[16];
    std::snprintf(ord, sizeof ord, "%s", sh.sorted_a ? "sorted" : "token");
    std::snprintf(dt, sizeof dt, "w4nl64i8h");
    all[9].key = "a_order"; all[9].kind = RAD_P_STR; all[9].sval = ord;
    all[10].key = "dtype"; all[10].kind = RAD_P_STR; all[10].sval = dt;
    RadArgs args{};
    args.t = t; args.n_t = 9; args.p = all; args.n_p = 11;
    TCls tc[2];
    for (int c = 0; c < 2; ++c)
        tc[c] = TCls{ dtw[c], tent[c], dws[c], sent[c], (uint32_t)Ncl[c], (uint32_t)Kcl[c], (uint32_t)(Kcl[c] / 2),
                      (uint32_t)(Kcl[c] / 64), (uint32_t)p0, (uint32_t)ttab[c], (uint32_t)tsm[c] };
    auto today = [&]() { return fn(&args, (RadStream)&q); };
    auto t1 = [&]() {
        t1_launch(q, da, das, tc[0], tc[1], dso, doff, (uint16_t*)(dy1 + kCanary), (uint32_t)T, (uint32_t)Nmax,
                  (uint32_t)Kmax, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)Kmax, (uint32_t)(Kmax / 128),
                  (uint32_t)Nmax, sh.sorted_a, (uint32_t)sh.parts, (uint32_t)AR);
        return 0;
    };
    const int rc = today();
    t1();
    q.wait_and_throw();
    std::vector<uint8_t> h0(ybytes + 2 * kCanary), h1(ybytes + 2 * kCanary);
    q.memcpy(h0.data(), dy0, h0.size());
    q.memcpy(h1.data(), dy1, h1.size()).wait();
    bool canary = true;
    for (size_t i = 0; i < kCanary; ++i)
        canary &= h0[i] == kPat && h0[kCanary + ybytes + i] == kPat && h1[i] == kPat && h1[kCanary + ybytes + i] == kPat;
    size_t diff = 0, first = (size_t)-1, nonzero = 0;
    for (size_t i = 0; i < ybytes; i += 2) {
        const bool d = h0[kCanary + i] != h1[kCanary + i] || h0[kCanary + i + 1] != h1[kCanary + i + 1];
        if (d && first == (size_t)-1) first = i / 2;
        diff += d;
        nonzero += h0[kCanary + i] != 0 || h0[kCanary + i + 1] != 0;
    }
    const bool ok = rc == RAD_OK && canary && diff == 0 && nonzero > 0;
    std::printf("%s %s tokens %d: T %d over %d experts, N %d/%d K %d/%d -- today rc %d; %zu of %zu outputs differ%s; "
                "%zu nonzero; canaries %s\n", ok ? "PASS" : "FAIL", sh.name, tokens, T, pool_n, sh.N, sh.N1, sh.K, sh.K1,
                rc, diff, ybytes / 2, diff ? "" : " (bit-identical)", nonzero, canary ? "intact" : "BROKEN");
    if (diff) std::printf("  first difference at output %zu (row %zu, column %zu)\n", first, first / Nmax, first % Nmax);
    if (ok && launches > 0) {
        auto time = [&](auto&& f) {
            for (int z = 0; z < 3; ++z) f();
            q.wait();
            std::vector<double> per;
            for (int r = 0; r < runs; ++r) {
                const auto a0 = std::chrono::steady_clock::now();
                for (int z = 0; z < launches; ++z) f();
                q.wait();
                per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count() /
                              launches);
            }
            return median(per);
        };
        const double u0 = time(today), u1 = time(t1), u0b = time(today);
        std::printf("RESULT %s tokens %d: today %.1f us (re-timed %.1f), T1 %.1f us a launch -> %.0f %% of today\n",
                    sh.name, tokens, u0, u0b, u1, 100.0 * u1 / std::min(u0, u0b));
    }
    for (int c = 0; c < 2; ++c) { sycl::free(dw[c], q); sycl::free(dws[c], q); sycl::free(dtw[c], q); }
    sycl::free(da, q); sycl::free(das, q); sycl::free(dso, q); sycl::free(doff, q); sycl::free(dy0, q); sycl::free(dy1, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: xe_fc_tile_probe LIBXE.so [LAUNCHES=48] [RUNS=5]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_w4nl_i8");
    if (!fn) { std::printf("no xe_rad_moe_gemm_w4nl_i8\n"); return 2; }
    const int launches = argc > 2 ? std::atoi(argv[2]) : 48, runs = argc > 3 ? std::atoi(argv[3]) : 5;
    const char* dv = std::getenv("RI_DEVICE");
    sycl::queue q = (dv && !std::strcmp(dv, "cpu")) ? sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order())
                                                    : sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("fc_tile_probe on %s, %d launches x %d runs\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                launches, runs);
    const Shape gate_up{ "gate_up", 768, 512, 2560, 2560, 2, false }, down{ "down", 2560, 2560, 384, 256, 1, true };
    int fails = 0;
    for (int tokens : { 4, 1 }) {
        fails += !run_shape(q, fn, gate_up, tokens, launches, runs, 101 + tokens);
        fails += !run_shape(q, fn, down, tokens, launches, runs, 201 + tokens);
    }
    std::printf("fc_tile_probe: %d of 4 cases failed\n", fails);
    return fails ? 1 : 0;
}
