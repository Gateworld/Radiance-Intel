/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * i4i8_probe.cpp -- owner note 131's format probe ("option B"), kernel level only: the routed experts as plain
 * int4 (two's complement -8..7, a bf16 scale every 64 weights) on the card's int8 path (DP4A), against
 * activations as int8 with one f32 scale a row, timed against today's libxe expert GEMV (w4nl64a8h through
 * xe_rad_moe_gemm_w4p, whatever the launcher picks under the caller's environment) at one rank's shapes
 * (gate_up 512 x 2560, down 2560 x 256), in the same process:
 *   - 1 row: one token, 10 distinct experts (MTP off);
 *   - the verify mix: 4 tokens, 40 routed rows over 27 distinct experts (13 of them twice: 1.48 rows an expert).
 * Random weights; nothing here is in the serve path.
 *
 * The prototype's format: an expert's codes [N][K/2] bytes, a 16-byte chunk = 32 weights with byte j's low
 * nibble weight j and its high nibble weight j + 16, stored biased (code = w + 8); a group = 2 chunks = 64
 * weights; scales [N][K/64] bf16. The integer dot over a group is sum(code * a) - 8 * sum(a) (the activation
 * group sums are made when the run's rows are staged), exact in int32; then acc += (float)isum * scale, once a
 * group; y = acc * the row's scale, as bf16.
 *
 *   xe_i4i8_probe LIBXE.so [launches=48] [runs=5]   timing (RESULT lines); DUMP=dir writes launch 0's outputs
 *                                                   and exact group sums of every shape and variant to dir
 *   xe_i4i8_probe REF dir                           the host reference (no device): recomputes the inputs and
 *                                                   compares; for amd-host's CPU, not intel-host's (owner note 3)
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

#include "rad_abi.h"
#include "rad_planes.h"
#include "r4d_args.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

constexpr int kTopK = 10, kNExpert = 512, kWG = 512, kMT = 4;
constexpr uint32_t kMaxK = 2560;

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
uint8_t e4(uint32_t r) { return (uint8_t)((r & 0x80u) | ((1u + (r >> 8) % 10u) << 3) | (r & 7u)); }
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

/* ---- the shapes and the routing (shared by the timing and the reference: everything from h32) ---- */
/* N1 / K1: the odd experts' class (0: as the even's). The engine's shapes (run 116, AGT1's trace): the two-card parity
 * split gives gate_up classes of 768 / 512 columns and down K 384 / 256 -- timed with today's form and format B's row
 * only (the prototype's kernels take one shape) */
struct Proj { const char* name; uint32_t N, K; bool sorted; uint32_t N1 = 0, K1 = 0; int parts = 1; };
const Proj kProj[4] = { { "gate_up", 512, 2560, false, 0, 0, 2 }, { "down", 2560, 256, true, 0, 0, 1 },
                        { "gate_up-engine", 768, 2560, false, 512, 2560, 2 }, { "down-engine", 2560, 384, true, 2560, 256, 1 } };
uint32_t n_of(const Proj& P, int e) { return (e & 1) && P.N1 ? P.N1 : P.N; }
uint32_t k_of(const Proj& P, int e) { return (e & 1) && P.K1 ? P.K1 : P.K; }

struct Route {
    int T, tokens;
    std::vector<int32_t> sorted, off;   /* sorted[i] = flat index (token * top_k + slot); off[e] = rows below e */
    std::vector<int> row_e;
};
/* mode 0: one token, 10 distinct experts; mode 1: the verify mix (4 tokens, 27 experts, 13 of them twice) */
Route route_of(int mode, int L) {
    Route R;
    std::vector<int> ids;
    const int distinct = mode ? 27 : kTopK;
    for (int j = 0; (int)ids.size() < distinct; ++j) {
        const int e = (int)(h32(L * 31 + 5 + mode * 7777, j) % (uint32_t)kNExpert);
        if (std::find(ids.begin(), ids.end(), e) == ids.end()) ids.push_back(e);
    }
    std::vector<int> twice(distinct, 0);
    if (mode) for (int j = 0, n = 0; n < 13; ++j) { const int x = (int)(h32(L * 37 + 9, j) % 27u); if (!twice[x]) { twice[x] = 1; ++n; } }
    std::vector<std::pair<int, int>> rows;   /* (expert, copy) */
    for (int x = 0; x < distinct; ++x) for (int c = 0; c <= twice[x]; ++c) rows.push_back({ ids[x], c });
    std::sort(rows.begin(), rows.end());
    R.T = (int)rows.size();
    R.tokens = mode ? 4 : 1;
    std::vector<int> slot(R.tokens, 0);
    for (int i = 0; i < R.T; ++i) {
        const int tok = i % R.tokens;        /* an expert's two rows are adjacent: two tokens */
        R.sorted.push_back(tok * kTopK + slot[tok]++);
        R.row_e.push_back(rows[i].first);
    }
    R.off.assign(kNExpert + 1, 0);
    for (int e = 0; e <= kNExpert; ++e)
        R.off[e] = (int32_t)std::count_if(rows.begin(), rows.end(), [e](const std::pair<int, int>& r) { return r.first < e; });
    return R;
}

/* the i4 form of expert e: codes and scales */
void i4_codes(int e, const Proj& P, std::vector<uint8_t>& c) {
    c.resize((size_t)n_of(P, e) * k_of(P, e) / 2);
    for (size_t i = 0; i < c.size(); ++i) c[i] = (uint8_t)h32(e * 7919 + 101, i);
}
void i4_scales(int e, const Proj& P, std::vector<uint16_t>& s) {
    s.resize((size_t)n_of(P, e) * (k_of(P, e) / 64));
    for (size_t i = 0; i < s.size(); ++i) {   /* 2^-15 .. 2^-10, 7 mantissa bits: exact bf16 */
        const uint32_t h = h32(e * 104729 + 103, i);
        const float f = std::ldexp(1.0f + (float)(h & 127u) / 128.0f, -15 + (int)((h >> 7) % 6u));
        uint32_t u; std::memcpy(&u, &f, 4);
        s[i] = (uint16_t)(u >> 16);
    }
}
/* Format C (run 116, owner note 146 item 4): the same u4 codes as indices into w4nl's 16 levels rounded to int8 --
 * code c < 8 is +kLvl8[c], c >= 8 is -kLvl8[c - 8] (the table's order: 1.125 3.25 5.5 8 11 14 18 22, then negated;
 * x 127 / 22). The magnitudes packed in one u64 for the register lookup. */
constexpr int kLvl8[8] = { 6, 19, 32, 46, 64, 81, 104, 127 };
constexpr uint64_t kLvl8Packed = 6ull | 19ull << 8 | 32ull << 16 | 46ull << 24 | 64ull << 32 | 81ull << 40 | 104ull << 48 |
                                 127ull << 56;
inline int lvl8_of(int c) { return c < 8 ? kLvl8[c] : -kLvl8[c - 8]; }

int8_t i8_act(int proj, size_t i) { return (int8_t)((int)(h32(211 + proj, i) % 255u) - 127); }
float i8_scale(int proj, int row) { return 0.001f + (float)(h32(213 + proj, row) & 255) / 25600.0f; }

/* ---- the prototype kernel ----
 * The grid and the run handling are v2m's: work-group (row i, column block nb) of kWG threads; the work-group whose
 * row starts a run of up to kMT rows of one expert stages the run and writes all of them, the others return. A team
 * of LPR lanes owns R output columns; lane l loads 16-byte chunks l + u * LPR (u < UPL) of each column, all loads
 * first. A group's two chunks sit on lanes l and l ^ 1; their int32 sums are joined by one xor before the scale.
 * GL (a group a lane): lane l owns groups l + u * LPR whole (two chunks, 32 bytes), no xor. WGS threads a work-group.
 * The activation group sums: a DP4A against ones a word, then a sub-group sum (16 words = one group).
 * DUMP writes each group's exact sum to D[row][n][g].
 * LUT (run 116, format C): 0 the int4 codes themselves (format B, sum(code * a) - 8 * sum(a)); 1 each code mapped to
 * its int8 level in registers (the eight magnitudes in one u64, a shift a code, the sign from bit 3); 2 through a
 * 256-entry LDS table, a code byte (two codes) to its two int8 levels. Then sum(level * a), with no bias term. */
template <int LPR, int UPL, int R, bool GL, int WGS, bool DUMP, int LUT = 0>
sycl::event i4a8_launch(sycl::queue& q, const uint8_t* const* wtab, const uint16_t* const* stab, const int8_t* A,
                        const float* As, const int32_t* sorted, const int32_t* off, uint16_t* Y, int32_t* D, uint32_t T,
                        uint32_t N, uint32_t K, uint32_t a_sorted) {
    constexpr int CH = GL ? 2 : 1;   /* chunks a unit */
    const uint32_t cpw = (uint32_t)(WGS / LPR) * R;
    const uint32_t gx = (N + cpw - 1) / cpw;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> s_a(sycl::range<1>(kMT * kMaxK / 4), h);
        sycl::local_accessor<int32_t, 1> s_g(sycl::range<1>(kMT * kMaxK / 64), h);
        sycl::local_accessor<uint32_t, 1> s_t(sycl::range<1>(LUT == 2 ? 256 : 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(T, gx * WGS), sycl::range<2>(1, WGS)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t i = (uint32_t)it.get_group(0), nb = (uint32_t)it.get_group(1);
            const int tid = (int)it.get_local_id(1);
            const int32_t flat = sorted[i];
            if (flat < 0) return;
            uint32_t lo = 0, hi = kNExpert;
            while (lo + 1 < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
            }
            const uint32_t e = lo, first = (uint32_t)off[e], last = (uint32_t)off[e + 1];
            if ((i - first) % kMT != 0) return;
            const uint32_t rn = (last - i) < (uint32_t)kMT ? (last - i) : (uint32_t)kMT;
            const uint32_t G = K / 64u, CPR = K / 32u, rb = K / 2u;
            auto sg = it.get_sub_group();
            uint32_t tok[kMT];
#pragma unroll
            for (int qq = 0; qq < kMT; ++qq) {
                tok[qq] = 0;
                if ((uint32_t)qq < rn) tok[qq] = a_sorted ? i + qq : (uint32_t)sorted[i + qq] / kTopK;
                if ((uint32_t)qq >= rn) continue;
                const uint32_t* arow = (const uint32_t*)(A + (size_t)tok[qq] * K);
                /* WGS and K / 4 are multiples of 16: a sub-group's 16 words are one group, all lanes live */
                for (uint32_t b0 = 0; b0 < K / 4u; b0 += WGS) {
                    const uint32_t b = b0 + (uint32_t)tid;
                    const uint32_t x = b < K / 4u ? arow[b] : 0u;
                    if (b < K / 4u) s_a[qq * (kMaxK / 4) + b] = x;
                    const int s = sycl::reduce_over_group(sg, sycl::ext::oneapi::dot_acc((int32_t)x, (int32_t)0x01010101, 0),
                                                          sycl::plus<int>());
                    if (b < K / 4u && (b & 15u) == 0) s_g[qq * (kMaxK / 64) + b / 16u] = s;
                }
            }
            if constexpr (LUT == 2) {   /* byte v (codes v & 15, v >> 4) -> its two int8 levels in the low half-word */
                for (int v = tid; v < 256; v += WGS) {
                    const int c0 = v & 15, c1 = v >> 4;
                    const int l0 = c0 < 8 ? (int)((kLvl8Packed >> (8 * c0)) & 255u) : -(int)((kLvl8Packed >> (8 * (c0 - 8))) & 255u);
                    const int l1 = c1 < 8 ? (int)((kLvl8Packed >> (8 * c1)) & 255u) : -(int)((kLvl8Packed >> (8 * (c1 - 8))) & 255u);
                    s_t[v] = ((uint32_t)l0 & 255u) | (((uint32_t)l1 & 255u) << 8);
                }
            }
            sycl::group_barrier(it.get_group());
            const int team = tid / LPR, l = tid % LPR;
            const uint8_t* wb = wtab[e];
            const uint16_t* sb = stab[e];
            uint32_t col[R];
            bool ok[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t n = nb * cpw + (uint32_t)(team * R + r);
                ok[r] = n < N;
                col[r] = ok[r] ? n : 0u;
            }
            sycl::uint4 w[UPL][R][CH];
#pragma unroll
            for (int u = 0; u < UPL; ++u)
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int c = 0; c < CH; ++c) {
                        const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) + (uint32_t)c : (uint32_t)l + (uint32_t)u * LPR;
                        w[u][r][c] = (jc < CPR && ok[r]) ? *(const sycl::uint4*)(wb + (size_t)col[r] * rb + jc * 16u)
                                                         : sycl::uint4{ 0, 0, 0, 0 };
                    }
            float acc[kMT][R];
#pragma unroll
            for (int qq = 0; qq < kMT; ++qq)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[qq][r] = 0.0f;
#pragma unroll
            for (int u = 0; u < UPL; ++u) {
                /* the unit's first chunk; CPR (GL: G) % LPR == 0, so uniform */
                const uint32_t jc = GL ? 2u * ((uint32_t)l + (uint32_t)u * LPR) : (uint32_t)l + (uint32_t)u * LPR;
                const uint32_t g = jc / 2u;
                float ws[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t bits = (uint32_t)sb[(size_t)col[r] * G + g] << 16;
                    ws[r] = sycl::bit_cast<float>(bits);
                }
#pragma unroll
                for (int qq = 0; qq < kMT; ++qq) {
                    if ((uint32_t)qq >= rn) break;                   /* uniform over the work-group */
                    uint32_t a[8 * CH];
#pragma unroll
                    for (int t = 0; t < 8 * CH; ++t) a[t] = s_a[qq * (kMaxK / 4) + jc * 8u + (uint32_t)t];
                    const int gs = s_g[qq * (kMaxK / 64) + g];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int32_t s = 0;
#pragma unroll
                        for (int c = 0; c < CH; ++c) {
                            const uint32_t d[4] = { w[u][r][c].x(), w[u][r][c].y(), w[u][r][c].z(), w[u][r][c].w() };
#pragma unroll
                            for (int x = 0; x < 4; ++x) {
                                uint32_t lo4 = d[x] & 0x0f0f0f0fu, hi4 = (d[x] >> 4) & 0x0f0f0f0fu;
                                if constexpr (LUT == 1) {   /* each code byte to its signed int8 level */
                                    uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                                    for (int bb = 0; bb < 4; ++bb) {
                                        const uint32_t c0 = (lo4 >> (8 * bb)) & 15u, c1 = (hi4 >> (8 * bb)) & 15u;
                                        const uint32_t m0 = (uint32_t)(kLvl8Packed >> (8 * (c0 & 7u))) & 255u;
                                        const uint32_t m1 = (uint32_t)(kLvl8Packed >> (8 * (c1 & 7u))) & 255u;
                                        lo8 |= ((c0 & 8u) ? (0u - m0) & 255u : m0) << (8 * bb);
                                        hi8 |= ((c1 & 8u) ? (0u - m1) & 255u : m1) << (8 * bb);
                                    }
                                    lo4 = lo8; hi4 = hi8;
                                } else if constexpr (LUT == 2) {   /* a code byte (low code, high code) to two levels */
                                    uint32_t lo8 = 0, hi8 = 0;
#pragma unroll
                                    for (int bb = 0; bb < 4; ++bb) {
                                        const uint32_t ev = s_t[(d[x] >> (8 * bb)) & 255u];
                                        lo8 |= (ev & 255u) << (8 * bb);
                                        hi8 |= ((ev >> 8) & 255u) << (8 * bb);
                                    }
                                    lo4 = lo8; hi4 = hi8;
                                }
                                s = sycl::ext::oneapi::dot_acc((int32_t)lo4, (int32_t)a[8 * c + x], s);
                                s = sycl::ext::oneapi::dot_acc((int32_t)hi4, (int32_t)a[8 * c + 4 + x], s);
                            }
                        }
                        if constexpr (!GL) s += sycl::permute_group_by_xor(sg, s, 1);   /* the group's other chunk */
                        if constexpr (LUT == 0) s -= 8 * gs;
                        if (GL || (l & 1) == 0) {
                            acc[qq][r] += (float)s * ws[r];
                            if constexpr (DUMP)
                                if (ok[r]) D[((size_t)(i + qq) * N + col[r]) * G + g] = s;
                        }
                    }
                }
            }
#pragma unroll
            for (int qq = 0; qq < kMT; ++qq) {
                if ((uint32_t)qq >= rn) break;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float v = acc[qq][r];   /* not GL: the even lanes' sum (the odd lanes hold 0) */
                    if constexpr (GL && LPR >= 2) v += sycl::permute_group_by_xor(sg, v, 1);
                    if constexpr (LPR >= 4) v += sycl::permute_group_by_xor(sg, v, 2);
                    if constexpr (LPR >= 8) v += sycl::permute_group_by_xor(sg, v, 4);
                    if constexpr (LPR == 16) v += sycl::permute_group_by_xor(sg, v, 8);
                    if (l == 0 && ok[r]) {
                        const uint32_t u = sycl::bit_cast<uint32_t>(v * As[tok[qq]]);
                        Y[(size_t)(i + qq) * N + col[r]] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
                    }
                }
            }
        });
    });
}

/* name: L lanes a column, R columns a team, G a group a lane, W256 a 256-thread work-group (else 512) */
/* lut: 0 format B's int4 codes; 1 / 2 format C (run 116), the u4 codes to int8 levels in registers / through LDS, at
 * format B's best mappings (gate_up L16R1W256, down GL2R2) */
struct Variant { const char* name; int id, proj, lut = 0; };
const Variant kVar[] = { { "L16R1", 0, 0 }, { "L16R2", 1, 0 }, { "L16R1W256", 2, 0 }, { "GL8R1", 3, 0 }, { "GL8R2", 4, 0 },
                         { "L8R2", 10, 1 }, { "L8R8", 11, 1 }, { "L8R2W256", 12, 1 }, { "GL4R2", 13, 1 },
                         { "GL4R4", 14, 1 }, { "GL2R2", 15, 1 }, { "GL4R2W256", 16, 1 }, { "GL2R4W256", 17, 1 },
                         { "C-reg L16R1W256", 20, 0, 1 }, { "C-lds L16R1W256", 21, 0, 2 },
                         { "C-reg GL2R2", 22, 1, 1 }, { "C-lds GL2R2", 23, 1, 2 } };

template <bool DUMP>
sycl::event run_var(const Variant& v, sycl::queue& q, const uint8_t* const* wt, const uint16_t* const* st, const int8_t* A,
                    const float* As, const int32_t* so, const int32_t* of, uint16_t* Y, int32_t* D, uint32_t T,
                    const Proj& P) {
    const uint32_t as = P.sorted ? 1u : 0u;
#define I4A8(LPR, UPL, R, GL, WGS) i4a8_launch<LPR, UPL, R, GL, WGS, DUMP>(q, wt, st, A, As, so, of, Y, D, T, P.N, P.K, as)
    switch (v.id) {   /* gate_up K 2560: 80 chunks, 40 groups a column; down K 256: 8 chunks, 4 groups */
    case 0: return I4A8(16, 5, 1, false, 512);
    case 1: return I4A8(16, 5, 2, false, 512);
    case 2: return I4A8(16, 5, 1, false, 256);
    case 3: return I4A8(8, 5, 1, true, 512);
    case 4: return I4A8(8, 5, 2, true, 512);
    case 10: return I4A8(8, 1, 2, false, 512);
    case 11: return I4A8(8, 1, 8, false, 512);
    case 12: return I4A8(8, 1, 2, false, 256);
    case 13: return I4A8(4, 1, 2, true, 512);
    case 14: return I4A8(4, 1, 4, true, 512);
    case 15: return I4A8(2, 2, 2, true, 512);
    case 16: return I4A8(4, 1, 2, true, 256);
    case 17: return I4A8(2, 2, 4, true, 256);
    }
#undef I4A8
#define I4C8(LPR, UPL, R, GL, WGS, LUT) \
    i4a8_launch<LPR, UPL, R, GL, WGS, DUMP, LUT>(q, wt, st, A, As, so, of, Y, D, T, P.N, P.K, as)
    switch (v.id) {
    case 20: return I4C8(16, 5, 1, false, 256, 1);
    case 21: return I4C8(16, 5, 1, false, 256, 2);
    case 22: return I4C8(2, 2, 2, true, 512, 1);
    default: return I4C8(2, 2, 2, true, 512, 2);
    }
#undef I4C8
}

/* ---- today's form (w4nl64a8h planes, as moe_planes_bench builds it) ---- */
struct Form { int64_t p0, t_off, codes_bytes, s1_off, scale_bytes; };
Form form_of(int64_t N, int64_t K) {
    Form f;
    f.p0 = rad_planes_round((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
    f.t_off = rad_planes_round(f.p0 + N * (K / 2));
    f.codes_bytes = rad_planes_round(f.t_off + 16 * 4);
    f.s1_off = rad_planes_round(f.p0 + N * (K / 64));
    f.scale_bytes = rad_planes_round(f.s1_off + 4);
    return f;
}
RadTensor tbl(void* p, uint32_t dt, int64_t cnt) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = cnt; t.shape[1] = 1; t.stride[0] = 0; t.stride[1] = 1;
    return t;
}
RadTensor t2(void* p, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}
RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t;
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

const char* kModeName[2] = { "1row", "verify" };

int time_main(const char* lib, int launches, int runs) {
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("i4i8_probe: dlopen %s: %s\n", lib, dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_w4p");
    if (!fn) { std::printf("i4i8_probe: no xe_rad_moe_gemm_w4p\n"); return 2; }
    /* run 114: format B's engine row (xe_moe_i4i8.cpp), timed beside today's and the probe's forms when present */
    LaunchFn fnb = (LaunchFn)dlsym(h, "xe_rad_moe_gemm_i4i8");
    sycl::queue q(sycl::device(sycl::gpu_selector_v), { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const char* dump = std::getenv("DUMP");
    const float levels[16] = { -8.f, -5.5f, -4.f, -3.f, -2.25f, -1.5f, -0.875f, -0.25f,
                               0.25f, 0.875f, 1.5f, 2.25f, 3.f, 4.f, 5.5f, 8.f };
    const int cnt = kNExpert / 2;
    for (int pi = 0; pi < 4; ++pi) {
        const Proj& P = kProj[pi];
        const uint32_t G = P.K / 64;
        /* both forms of all 512 experts in device memory (far past L2); each expert at its class's shape */
        std::vector<const void*> pc[2], ps[2];
        std::vector<const uint8_t*> wt(kNExpert);
        std::vector<const uint16_t*> st(kNExpert);
        std::vector<uint8_t> hc, hs, ic;
        std::vector<uint16_t> is;
        for (int e = 0; e < kNExpert; ++e) {
            const int64_t Ne = n_of(P, e), Ke = k_of(P, e);
            const Form F = form_of(Ne, Ke);
            hc.assign((size_t)F.codes_bytes, 0);
            hs.assign((size_t)F.scale_bytes, 0);
            for (int64_t i = 0; i < Ne * (Ke / 2); ++i) hc[F.p0 + i] = (uint8_t)h32(e * 7919 + 1, i);
            std::memcpy(&hc[F.t_off], levels, sizeof levels);
            for (int64_t i = 0; i < Ne * (Ke / 64); ++i) hs[F.p0 + i] = e4(h32(e * 104729 + 3, i));
            const float s1 = 1.0f / 8192.0f;
            std::memcpy(&hs[F.s1_off], &s1, 4);
            uint8_t* dc = sycl::malloc_device<uint8_t>(hc.size(), q);
            uint8_t* ds = sycl::malloc_device<uint8_t>(hs.size(), q);
            i4_codes(e, P, ic);
            i4_scales(e, P, is);
            uint8_t* dic = sycl::malloc_device<uint8_t>(ic.size(), q);
            uint16_t* dis = sycl::malloc_device<uint16_t>(is.size(), q);
            if (!dc || !ds || !dic || !dis) { std::printf("alloc failed at expert %d\n", e); return 2; }
            q.memcpy(dc, hc.data(), hc.size()).wait();
            q.memcpy(ds, hs.data(), hs.size()).wait();
            q.memcpy(dic, ic.data(), ic.size()).wait();
            q.memcpy(dis, is.data(), is.size() * 2).wait();
            pc[e & 1].push_back(dc); ps[e & 1].push_back(ds);
            wt[e] = dic; st[e] = dis;
        }
        const void** tabc[2]; const void** tabs[2];
        for (int c = 0; c < 2; ++c) {
            tabc[c] = sycl::malloc_device<const void*>(cnt, q);
            tabs[c] = sycl::malloc_device<const void*>(cnt, q);
            q.memcpy(tabc[c], pc[c].data(), cnt * sizeof(void*)).wait();
            q.memcpy(tabs[c], ps[c].data(), cnt * sizeof(void*)).wait();
        }
        /* format B's row reads its classes as pointer tables over the same i4 planes and bf16 scales */
        const void** tabbi[2]; const void** tabbs[2];
        for (int c = 0; c < 2; ++c) {
            std::vector<const void*> vi, vs;
            for (int e = c; e < kNExpert; e += 2) { vi.push_back(wt[e]); vs.push_back(st[e]); }
            tabbi[c] = sycl::malloc_device<const void*>(cnt, q);
            tabbs[c] = sycl::malloc_device<const void*>(cnt, q);
            q.memcpy(tabbi[c], vi.data(), cnt * sizeof(void*)).wait();
            q.memcpy(tabbs[c], vs.data(), cnt * sizeof(void*)).wait();
        }
        const uint8_t** d_wt = sycl::malloc_device<const uint8_t*>(kNExpert, q);
        const uint16_t** d_st = sycl::malloc_device<const uint16_t*>(kNExpert, q);
        q.memcpy(d_wt, wt.data(), kNExpert * sizeof(void*)).wait();
        q.memcpy(d_st, st.data(), kNExpert * sizeof(void*)).wait();
        const double eb4 = (double)P.N * (P.K / 2.0 + P.K / 64.0), ebi = (double)P.N * (P.K / 2.0 + 2.0 * P.K / 64.0);
        std::printf("%s: %u x %u; an expert %.3f MB (w4nl) / %.3f MB (i4 + bf16 scales)\n", P.name, P.N, P.K, eb4 / 1e6, ebi / 1e6);
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<Route> rt;
            for (int L = 0; L < launches; ++L) rt.push_back(route_of(mode, L));
            const int T = rt[0].T, tokens = rt[0].tokens;
            const int a_rows = P.sorted ? T : tokens;
            std::vector<int32_t*> d_so, d_of;
            for (int L = 0; L < launches; ++L) {
                int32_t* a = sycl::malloc_device<int32_t>(T, q);
                int32_t* b = sycl::malloc_device<int32_t>(kNExpert + 1, q);
                q.memcpy(a, rt[L].sorted.data(), T * 4).wait();
                q.memcpy(b, rt[L].off.data(), (kNExpert + 1) * 4).wait();
                d_so.push_back(a); d_of.push_back(b);
            }
            /* today's activations (E4M3, f32 per 128) and the probe's (int8, f32 a row) */
            std::vector<uint8_t> ha((size_t)a_rows * P.K);
            std::vector<float> has((size_t)a_rows * (P.K / 128));
            for (size_t i = 0; i < ha.size(); ++i) ha[i] = e4(h32(11, i));
            for (size_t i = 0; i < has.size(); ++i) has[i] = 0.01f + (float)(h32(13, i) & 255) / 2560.0f;
            std::vector<int8_t> hi((size_t)a_rows * P.K);
            std::vector<float> his((size_t)a_rows);
            for (size_t i = 0; i < hi.size(); ++i) hi[i] = i8_act(pi, i);
            for (int r = 0; r < a_rows; ++r) his[r] = i8_scale(pi, r);
            uint8_t* d_a = sycl::malloc_device<uint8_t>(ha.size(), q);
            float* d_as = sycl::malloc_device<float>(has.size(), q);
            int8_t* d_i = sycl::malloc_device<int8_t>(hi.size(), q);
            float* d_is = sycl::malloc_device<float>(his.size(), q);
            uint16_t* d_y = sycl::malloc_device<uint16_t>((size_t)T * P.N, q);
            int32_t* d_d = sycl::malloc_device<int32_t>((size_t)T * P.N * G, q);
            q.memcpy(d_a, ha.data(), ha.size()).wait();
            q.memcpy(d_as, has.data(), has.size() * 4).wait();
            q.memcpy(d_i, hi.data(), hi.size()).wait();
            q.memcpy(d_is, his.data(), his.size() * 4).wait();
            char dt[] = "w4nl64a8h";
            RadParam p[11]{};
            const char* ik[] = { "M", "N", "K", "n_expert", "top_k", "group", "N_odd", "K_odd", "parts" };
            const long long iv[] = { a_rows, P.N, P.K, kNExpert, kTopK, 128, P.N1 ? P.N1 : P.N, P.K1 ? P.K1 : P.K, P.parts };
            for (int i = 0; i < 9; ++i) { p[i].key = ik[i]; p[i].kind = RAD_P_INT; p[i].ival = iv[i]; }
            p[9].key = "a_order"; p[9].kind = RAD_P_STR; p[9].sval = P.sorted ? "sorted" : "token";
            p[10].key = "dtype"; p[10].kind = RAD_P_STR; p[10].sval = dt;
            auto today = [&](int L) -> int {
                RadTensor t[9] = {
                    t2(d_a, RAD_F8E4M3, a_rows, P.K), t2(d_as, RAD_F32, a_rows, P.K / 128),
                    tbl((void*)tabc[0], RAD_U4, cnt), tbl((void*)tabs[0], RAD_F8E4M3, cnt),
                    t1(d_so[L], RAD_I32, T), t1(d_of[L], RAD_I32, kNExpert + 1),
                    tbl((void*)tabc[1], RAD_U4, cnt), tbl((void*)tabs[1], RAD_F8E4M3, cnt),
                    t2(d_y, RAD_BF16, T, P.N) };
                RadArgs args{};
                args.t = t; args.n_t = 9; args.p = p; args.n_p = 11;
                return fn(&args, (RadStream)&q);
            };
            /* distinct experts a launch: the bytes both forms read */
            int distinct = 0;
            for (int e = 0; e < kNExpert; ++e) distinct += rt[0].off[e + 1] > rt[0].off[e];
            /* host wall over back-to-back launches, `runs` passes; today's first, then each variant, then today again */
            auto wall = [&](auto&& one) {
                std::vector<double> us;
                for (int r = 0; r < runs; ++r) {
                    q.wait();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int L = 0; L < launches; ++L) one(L);
                    q.wait_and_throw();
                    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / launches);
                }
                return us;
            };
            if (today(0) != RAD_OK) { std::printf("%s %s: today's launch refused\n", P.name, kModeName[mode]); return 1; }
            q.wait_and_throw();
            const double t_a = median(wall(today));
            std::printf("RESULT i4i8 %s %s today: %.1f us a launch (T %d, %d distinct experts) = %.1f GB/s\n", P.name,
                        kModeName[mode], t_a, T, distinct, distinct * eb4 / (t_a * 1e-6) / 1e9);
            for (const Variant& v : kVar) {
                if (v.proj != pi) continue;
                run_var<false>(v, q, d_wt, d_st, d_i, d_is, d_so[0], d_of[0], d_y, d_d, T, P).wait_and_throw();
                const double t_v = median(wall([&](int L) { run_var<false>(v, q, d_wt, d_st, d_i, d_is, d_so[L], d_of[L], d_y, d_d, T, P); }));
                /* the kernel's own time from the events, for the record */
                std::vector<double> ev;
                for (int L = 0; L < launches; ++L) {
                    sycl::event x = run_var<false>(v, q, d_wt, d_st, d_i, d_is, d_so[L], d_of[L], d_y, d_d, T, P);
                    x.wait();
                    ev.push_back((x.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                  x.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0);
                }
                std::printf("RESULT i4i8 %s %s i4a8 %s: %.1f us a launch (kernel %.1f) = %.1f GB/s; %.1f %% of today's time\n",
                            P.name, kModeName[mode], v.name, t_v, median(ev), distinct * ebi / (t_v * 1e-6) / 1e9, 100.0 * t_v / t_a);
                if (dump) {
                    q.memset(d_y, 0xff, (size_t)T * P.N * 2).wait();
                    q.memset(d_d, 0x7f, (size_t)T * P.N * G * 4).wait();
                    run_var<true>(v, q, d_wt, d_st, d_i, d_is, d_so[0], d_of[0], d_y, d_d, T, P).wait_and_throw();
                    std::vector<uint16_t> hy((size_t)T * P.N);
                    std::vector<int32_t> hd((size_t)T * P.N * G);
                    q.memcpy(hy.data(), d_y, hy.size() * 2).wait();
                    q.memcpy(hd.data(), d_d, hd.size() * 4).wait();
                    const std::string fnm = std::string(dump) + "/i4i8-" + P.name + "-" + kModeName[mode] + "-" + v.name + ".bin";
                    FILE* f = std::fopen(fnm.c_str(), "wb");
                    if (!f) { std::printf("cannot write %s\n", fnm.c_str()); return 1; }
                    std::fwrite(hy.data(), 2, hy.size(), f);
                    std::fwrite(hd.data(), 4, hd.size(), f);
                    std::fclose(f);
                }
            }
            if (fnb) {
                char dtb[] = "w4g64i8h";
                RadParam pb[11];
                for (int i = 0; i < 11; ++i) pb[i] = p[i];
                pb[10].sval = dtb;
                auto rowb = [&](int L) -> int {
                    RadTensor t[9] = {
                        t2(d_i, RAD_I8, a_rows, P.K), t2(d_as, RAD_F32, a_rows, P.K / 128),
                        tbl((void*)tabbi[0], RAD_I4, cnt), tbl((void*)tabbs[0], RAD_BF16, cnt),
                        t1(d_so[L], RAD_I32, T), t1(d_of[L], RAD_I32, kNExpert + 1),
                        tbl((void*)tabbi[1], RAD_I4, cnt), tbl((void*)tabbs[1], RAD_BF16, cnt),
                        t2(d_y, RAD_BF16, T, P.N) };
                    RadArgs args{};
                    args.t = t; args.n_t = 9; args.p = pb; args.n_p = 11;
                    return fnb(&args, (RadStream)&q);
                };
                const int rb = rowb(0);
                q.wait_and_throw();
                if (rb != RAD_OK) {
                    std::printf("RESULT i4i8 %s %s row w4g64i8h: refused (rc %d)\n", P.name, kModeName[mode], rb);
                } else {
                    const double t_r = median(wall(rowb));
                    std::printf("RESULT i4i8 %s %s row w4g64i8h: %.1f us a launch = %.1f GB/s; %.1f %% of today's time\n",
                                P.name, kModeName[mode], t_r, distinct * ebi / (t_r * 1e-6) / 1e9, 100.0 * t_r / t_a);
                }
            }
            const double t_b = median(wall(today));
            std::printf("RESULT i4i8 %s %s today again: %.1f us a launch (drift %.1f %%)\n", P.name, kModeName[mode], t_b,
                        100.0 * (t_b - t_a) / t_a);
            for (auto* x : d_so) sycl::free(x, q);
            for (auto* x : d_of) sycl::free(x, q);
            sycl::free(d_a, q); sycl::free(d_as, q); sycl::free(d_i, q); sycl::free(d_is, q); sycl::free(d_y, q); sycl::free(d_d, q);
        }
        for (int c = 0; c < 2; ++c) {
            for (auto* x : pc[c]) sycl::free((void*)x, q);
            for (auto* x : ps[c]) sycl::free((void*)x, q);
            sycl::free(tabc[c], q); sycl::free(tabs[c], q);
            sycl::free(tabbi[c], q); sycl::free(tabbs[c], q);
        }
        for (int e = 0; e < kNExpert; ++e) { sycl::free((void*)wt[e], q); sycl::free((void*)st[e], q); }
        sycl::free(d_wt, q); sycl::free(d_st, q);
    }
    std::printf("RESULT i4i8_probe DONE\n");
    return 0;
}

/* ---- the host reference: launch 0 of every shape, mode and variant, from the same generators ----
 * The group sums must match exactly; y within the bf16 rounding (2^-8 relative) plus the f32 sum's (G * 2^-23 of
 * sum |terms|). */
int ref_main(const char* dir) {
    bool all_ok = true;
    for (int pi = 0; pi < 2; ++pi) {
        const Proj& P = kProj[pi];
        const uint32_t G = P.K / 64;
        for (int mode = 0; mode < 2; ++mode) {
            const Route rt = route_of(mode, 0);
            const int T = rt.T;
            /* the exact sums and y in double */
            for (int lut = 0; lut < 2; ++lut) {   /* format B's int4 codes, then format C's int8 levels (run 116) */
            std::vector<int32_t> rd((size_t)T * P.N * G);
            std::vector<double> ry((size_t)T * P.N), rb((size_t)T * P.N);
            std::vector<uint8_t> c;
            std::vector<uint16_t> s;
            int last_e = -1;
            for (int i = 0; i < T; ++i) {
                const int e = rt.row_e[i];
                if (e != last_e) { i4_codes(e, P, c); i4_scales(e, P, s); last_e = e; }
                const int tok = P.sorted ? i : rt.sorted[i] / kTopK;
                const double as = (double)i8_scale(pi, tok);
                for (uint32_t n = 0; n < P.N; ++n) {
                    double y = 0, ab = 0;
                    for (uint32_t g = 0; g < G; ++g) {
                        int32_t sum = 0;
                        for (uint32_t k = g * 64; k < g * 64 + 64; ++k) {
                            const uint32_t ch = k / 32, j = k % 32;   /* chunk, weight in chunk */
                            const uint8_t byte = c[(size_t)n * (P.K / 2) + ch * 16 + (j % 16)];
                            const int code = (int)(j < 16 ? (byte & 15) : (byte >> 4));
                            const int w = lut ? lvl8_of(code) : code - 8;
                            sum += w * (int)i8_act(pi, (size_t)tok * P.K + k);
                        }
                        rd[((size_t)i * P.N + n) * G + g] = sum;
                        const double t = (double)sum * (double)bf(s[(size_t)n * G + g]);
                        y += t; ab += std::fabs(t);
                    }
                    ry[(size_t)i * P.N + n] = y * as;
                    rb[(size_t)i * P.N + n] = std::ldexp(std::fabs(y * as), -8) + G * std::ldexp(ab * as, -23);
                }
            }
            for (const Variant& v : kVar) {
                if (v.proj != pi || (v.lut != 0) != (lut != 0)) continue;
                const std::string fnm = std::string(dir) + "/i4i8-" + P.name + "-" + kModeName[mode] + "-" + v.name + ".bin";
                FILE* f = std::fopen(fnm.c_str(), "rb");
                if (!f) { std::printf("REF %s: missing\n", fnm.c_str()); all_ok = false; continue; }
                std::vector<uint16_t> hy((size_t)T * P.N);
                std::vector<int32_t> hd((size_t)T * P.N * G);
                const bool rd_ok = std::fread(hy.data(), 2, hy.size(), f) == hy.size() && std::fread(hd.data(), 4, hd.size(), f) == hd.size();
                std::fclose(f);
                if (!rd_ok) { std::printf("REF %s: short file\n", fnm.c_str()); all_ok = false; continue; }
                int64_t bad_d = 0, bad_y = 0;
                double worst = 0;
                for (size_t x = 0; x < hd.size(); ++x) bad_d += hd[x] != rd[x];
                for (size_t x = 0; x < hy.size(); ++x) {
                    const double d = std::fabs((double)bf(hy[x]) - ry[x]) / rb[x];
                    if (!(d <= 1.0)) ++bad_y;
                    if (d > worst || d != d) worst = d;
                }
                const bool ok = bad_d == 0 && bad_y == 0;
                all_ok = all_ok && ok;
                std::printf("RESULT i4i8_ref %s %s %s: %s -- group sums %lld of %zu differ; y %lld of %zu past the bound, "
                            "worst d/bound %.3g\n", P.name, kModeName[mode], v.name, ok ? "PASS" : "FAIL", (long long)bad_d,
                            hd.size(), (long long)bad_y, hy.size(), worst);
            }
            }   /* lut */
        }
    }
    std::printf("RESULT i4i8_ref %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_i4i8_probe LIBXE.so [launches] [runs] | REF dir\n"); return 2; }
    if (!std::strcmp(argv[1], "REF")) return argc > 2 ? ref_main(argv[2]) : 2;
    return time_main(argv[1], argc > 2 ? std::atoi(argv[2]) : 48, argc > 3 ? std::atoi(argv[3]) : 5);
}
