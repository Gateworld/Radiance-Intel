/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 279, owner note 218 stage 0).
 *
 * esimd_fc_probe.cpp -- format C's decode expert GEMM at gate_up's shape (K 2560 in 40 groups of 64, two parts, token
 * rows; today's moe_i4i8_run_kernel<5,16,1,0,FC> in libxe/xe_moe_i4i8.cpp) as a WHOLE-KERNEL ESIMD form. Stage 0's
 * question (owner note 218): does a pure ESIMD kernel with the format C block load, the register-table decode and the
 * DP4A loop compile ahead of time for the B70 under the image's IGC 2.42.6 / ocloc 26.40 (run 265's invoke_simd mix of
 * SYCL and ESIMD crashed that ocloc)? Stage 2 runs it on card1 against today's kernel for the bytes and the time.
 *
 * The form (stage 1's candidate; the arithmetic is today's, in today's order, so the bf16 bytes are the same by
 * construction):
 *   - a thread owns 16 columns as its SIMD lanes, the up to kMT = 4 rows of one expert's run, and ONE of today's eight
 *     even lanes' partial sums: gi = 0..7 takes the groups g = gi + 8 u, u = 0..4, in u order (today: lane l even,
 *     gi = l / 2, acc += (float(s) * asc) * sbr a group, the odd lanes 0). A work-group is 8 x CB threads = CB blocks of
 *     16 columns; the eight partials meet in shared memory and are added in today's xor-tree order
 *     ((0 + 4) + (2 + 6)) + ((1 + 5) + (3 + 7)), then + 0 (the last xor adds the odd lane's +0.0, which turns a -0 to +0);
 *   - a group's codes for the 16 columns are ONE transposed 2D block load (16 rows x 8 dwords, run 206's E1b);
 *   - the 256-entry pair table (a code byte -> its two int8 levels) sits in registers as two dword tables (run 206's
 *     DEC2), built from the entry's 16 levels with today's rounding (rint(level * 127 / amax), IEEE division);
 *   - the group's int32 dot by dp4a over the packed levels and the row's activation dwords in natural order (no LDS
 *     staging, no even / odd split). Today sums the group's two 32-K chunks separately and adds them as floats:
 *     |a product| <= 16,384, 64 of them < 2^24, so float(s_lo) + float(s_hi) == float(s_lo + s_hi) exactly;
 *   - each product goes through an integer xor with a runtime zero before the add (P100: VC fuses mul + add into a mad
 *     otherwise, -ffp-contract=off notwithstanding);
 *   - 256 GRF a thread (grf_size<256>): the two tables alone are 32 GRF, the activations 20, the partials 4.
 *
 *   xe_esimd_fc_probe [LAUNCHES=48] [RUNS=5] [TOKENS=4]
 * On a GPU: the bytes against a host emulation of today's order (the kernel's own exactness; stage 2 adds today's kernel
 * through libxe as fc_tile_probe does), then the per-launch time (host wall over LAUNCHES back-to-back launches, the
 * median of RUNS). On a CPU device: ESIMD is GPU-only -> "skip", exit 0.
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace es = sycl::ext::intel::esimd;
namespace sxe = sycl::ext::oneapi::experimental;

constexpr int kMT = 4;            /* rows a run (today's kMT) */
constexpr int kGI = 8;            /* today's even lanes a column: the partial sums */
constexpr int kUPL = 5;           /* groups a partial: today's units a lane */
constexpr int kK = 2560;          /* gate_up's K */
constexpr int kG = kK / 64;       /* 40 groups */
static_assert(kG == kGI * kUPL, "gate_up's 40 groups are 8 partials x 5 units");
constexpr uint32_t kWLd = kK / 2;         /* a weight row's code bytes */
constexpr uint32_t kSLd = kK / 64;        /* a weight row's E4M3 scale bytes */

/* One class of experts (the parity split): the codes [n][kWLd] an entry, the 16 f32 levels, scale.1, the scales
 * [n][kSLd]; entry eh at base + eh * stride. */
struct ECls {
    const uint8_t* codes;
    uint64_t cstride;
    const float* tab;     /* 16 an entry */
    const float* s1;      /* 1 an entry */
    const uint8_t* scales;
    uint64_t sstride;
    uint32_t n;
};

/* ---- the ESIMD kernel */
template <int CB>
void esimd_fc_launch(sycl::queue& q, const int8_t* A, const float* As, const ECls& c0, const ECls& c1,
                     const int32_t* sorted, const int32_t* off, uint16_t* Y, uint32_t T, uint32_t N, uint32_t n_expert,
                     uint32_t top_k, uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t parts, uint32_t a_rows,
                     uint32_t zero) {
    constexpr uint32_t kLW = kGI * CB;
    constexpr uint32_t kSlm = CB * kGI * kMT * 16 * 4;
    const sycl::nd_range<2> r({ (size_t)T, (size_t)(N / 16 / CB) * kLW }, { 1, (size_t)kLW });
    q.parallel_for(r, sxe::properties{ sycl::ext::intel::experimental::grf_size<256> },
                   [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        es::slm_init<kSlm>();
        const uint32_t i = (uint32_t)it.get_global_id(0);
        const uint32_t lid = (uint32_t)it.get_local_id(1), gi = lid & 7u, cb = lid >> 3;
        const uint32_t n0 = ((uint32_t)it.get_group(1) * CB + cb) * 16u;
        /* the run: today's routing, uniform over the work-group (one row i a work-group) */
        const int32_t flat = sorted[i];
        int32_t e = -1;
        if (flat >= 0 && i < (uint32_t)off[n_expert]) {
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
        for (int q2 = 0; q2 < kMT; ++q2) {
            tokr[q2] = 0; liver[q2] = false;
            if (e >= 0 && (uint32_t)q2 < rn) {
                const int32_t fq = sorted[i + q2];
                const uint32_t t = (uint32_t)fq / top_k;   /* token rows (a_sorted 0) */
                tokr[q2] = t;
                liver[q2] = fq >= 0 && t < a_rows;
            }
        }
        if (e < 0) {   /* a padding row: zeros, as today's rowok-false columns */
            if (gi == 0 && n0 < N) es::block_store<uint16_t, 16>(Y + (size_t)i * y_ld + n0, es::simd<uint16_t, 16>(0));
            return;
        }
        const ECls& c = (e & 1) ? c1 : c0;
        const uint32_t fy = N / parts, fc = c.n / parts;
        const uint32_t pp = n0 / fy, j0 = n0 - pp * fy;   /* the 16 columns share their part (fy % 16 == 0) */
        const es::simd<uint32_t, 16> lane(0u, 1u);
        const es::simd<uint32_t, 16> jj = lane + j0;
        const es::simd_mask<16> dead = (jj >= fc) | ((lane + n0) >= N);   /* today's !rowok */
        const uint32_t row0 = pp * fc + j0;   /* the block's first weight row */
        const uint32_t eh = (uint32_t)e >> 1;
        const uint8_t* we = c.codes + (uint64_t)eh * c.cstride;
        const uint8_t* se = c.scales + (uint64_t)eh * c.sstride;
        /* the entry's 16 levels -> the int8 levels (today's rounding) -> the two pair tables */
        const es::simd<float, 16> tv = es::block_load<float, 16>(c.tab + (uint64_t)eh * 16);
        const float amax = es::hmax<float>(es::abs(tv));
        es::simd<int32_t, 16> lev;
        float smul;
        const float s1 = c.s1[eh];
        if (amax > 0.0f) {
            const es::simd<float, 16> l127 = tv * 127.0f;
            const es::simd<float, 16> ld = es::div_ieee(l127, es::simd<float, 16>(amax));
            lev = es::simd<int32_t, 16>(es::rnde<float, 16>(ld));
            const es::simd<float, 1> d = es::div_ieee(es::simd<float, 1>(amax), es::simd<float, 1>(127.0f));
            smul = s1 * d[0];
        } else {
            lev = es::simd<int32_t, 16>(tv);
            smul = s1;
        }
        /* ESIMD's iselect and bit_cast_view are non-const members over uint16 indices: the tables stay mutable */
        const es::simd<uint16_t, 256> xs(0, 1);
        const es::simd<uint16_t, 256> x_lo = es::simd<uint16_t, 256>(xs & 15u), x_hi = es::simd<uint16_t, 256>(xs >> 4u);
        const es::simd<uint32_t, 256> l_lo = es::simd<uint32_t, 256>(lev.iselect(x_lo)) & 255u;
        const es::simd<uint32_t, 256> l_hi = es::simd<uint32_t, 256>(lev.iselect(x_hi)) & 255u;
        es::simd<uint32_t, 256> P0 = l_lo | (l_hi << 8);   /* a code byte -> (level lo, level hi) in bytes 0, 1 */
        es::simd<uint32_t, 256> P1 = P0 << 16;              /* the same in bytes 2, 3 */
        /* the rows' activations for this partial's five groups, up front (20 GRF) */
        es::simd<int32_t, 16> av[kMT][kUPL];
#pragma unroll
        for (int q2 = 0; q2 < kMT; ++q2)
#pragma unroll
            for (int u = 0; u < kUPL; ++u) {
                if ((uint32_t)q2 < rn && liver[q2])
                    av[q2][u] = es::block_load<int32_t, 16>((const int32_t*)(A + (size_t)tokr[q2] * a_ld) + 16 * (gi + 8 * u));
                else
                    av[q2][u] = 0;
            }
        /* the 16 columns' scales for the five groups: a dword gather a group (the row pitch is 40 B: no 2D load),
         * e4m3 -> f32 lane-wise as today's e4m3_to_f32, times smul */
        es::simd<uint32_t, 16> rclamp = es::simd<uint32_t, 16>(row0) + lane;
        rclamp.merge(0u, dead);
        es::simd<float, 16> sbr[kUPL];
#pragma unroll
        for (int u = 0; u < kUPL; ++u) {
            const uint32_t b = gi + 8u * (uint32_t)u;
            const es::simd<uint32_t, 16> sw = es::gather<uint32_t, 16>((const uint32_t*)se, rclamp * kSLd + (b & ~3u));
            const es::simd<uint32_t, 16> v = (sw >> (8u * (b & 3u))) & 255u;
            const es::simd<uint32_t, 16> s = (v & 0x80u) << 24, ex = (v >> 3) & 15u, m = v & 7u;
            es::simd<float, 16> mf = es::simd<float, 16>(m) * 0.001953125f;
            es::simd<uint32_t, 16> rr = s | ((ex + 120u) << 23) | (m << 20);
            rr.merge(s | mf.template bit_cast_view<uint32_t>(), ex == 0u);
            rr.merge(s | 0x7fc00000u, (v & 0x7fu) == 0x7fu);
            sbr[u] = rr.template bit_cast_view<float>() * smul;
        }
        es::simd<float, 16> acc[kMT];
#pragma unroll
        for (int q2 = 0; q2 < kMT; ++q2) acc[q2] = 0.0f;
#pragma unroll
        for (int u = 0; u < kUPL; ++u) {
            const uint32_t g = gi + 8u * (uint32_t)u;
            /* the group's codes for the 16 columns: 16 rows x 8 dwords, transposed -> [dword w][column] */
            es::simd<uint32_t, 128> cw = es::load_2d<uint32_t, 8, 16, 1, true>(
                (const uint32_t*)we, kWLd - 1u, c.n - 1u, kWLd - 1u, (int)(g * 8u), (int)row0);
            es::simd<int32_t, 16> lv[16];   /* the levels of elements 4 x .. 4 x + 3, packed, a column a lane */
            auto cb8 = cw.template bit_cast_view<uint8_t>();   /* byte b of dword (w, column l) at 64 w + 4 l + b */
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const es::simd<uint16_t, 16> i0 = es::simd<uint8_t, 16>(cb8.template select<16, 4>(64 * w));
                const es::simd<uint16_t, 16> i1 = es::simd<uint8_t, 16>(cb8.template select<16, 4>(64 * w + 1));
                const es::simd<uint16_t, 16> i2 = es::simd<uint8_t, 16>(cb8.template select<16, 4>(64 * w + 2));
                const es::simd<uint16_t, 16> i3 = es::simd<uint8_t, 16>(cb8.template select<16, 4>(64 * w + 3));
                lv[2 * w] = es::simd<int32_t, 16>(P0.iselect(i0) | P1.iselect(i1));
                lv[2 * w + 1] = es::simd<int32_t, 16>(P0.iselect(i2) | P1.iselect(i3));
            }
#pragma unroll
            for (int q2 = 0; q2 < kMT; ++q2) {
                if ((uint32_t)q2 >= rn || !liver[q2]) continue;   /* uniform */
                const float asc = As[(size_t)tokr[q2] * as_ld + g / 2u];
                es::simd<int32_t, 16> s = 0;
#pragma unroll
                for (int x = 0; x < 16; ++x) s = es::dp4a<int32_t>(s, lv[x], es::simd<int32_t, 16>((int32_t)av[q2][u][x]));
                const es::simd<float, 16> fs = s;
                const es::simd<float, 16> p = fs * asc;
                es::simd<float, 16> m = p * sbr[u];
                es::simd<uint32_t, 16> mb = m.template bit_cast_view<uint32_t>();
                mb ^= zero;   /* no mad: the add takes the rounded product */
                acc[q2] += mb.template bit_cast_view<float>();
            }
        }
        /* the eight partials meet: thread gi's rows to shared memory, then thread q < rn adds row q in today's order */
#pragma unroll
        for (int q2 = 0; q2 < kMT; ++q2)
            es::slm_block_store<float, 16>(((cb * kGI + gi) * kMT + (uint32_t)q2) * 64u, acc[q2]);
        es::barrier();
        if (gi < (uint32_t)kMT && gi < rn) {
            const uint32_t q2 = gi;
            es::simd<float, 16> pa[kGI];
#pragma unroll
            for (int g2 = 0; g2 < kGI; ++g2) pa[g2] = es::slm_block_load<float, 16>(((cb * kGI + (uint32_t)g2) * kMT + q2) * 64u);
            es::simd<float, 16> v = ((pa[0] + pa[4]) + (pa[2] + pa[6])) + ((pa[1] + pa[5]) + (pa[3] + pa[7]));
            es::simd<uint32_t, 16> zb(zero);
            v = v + zb.template bit_cast_view<float>();   /* + the odd lane's +0.0 */
            es::simd<uint32_t, 16> ub = v.template bit_cast_view<uint32_t>();
            es::simd<uint32_t, 16> h = (ub + 0x7fffu + ((ub >> 16) & 1u)) >> 16;
            h.merge((ub >> 16) | 0x40u, (ub & 0x7fffffffu) > 0x7f800000u);
            es::simd<uint16_t, 16> o = h;
            o.merge(0, dead);
            if (!liver[q2]) o = 0;
            if (n0 < N) es::block_store<uint16_t, 16>(Y + (size_t)(i + q2) * y_ld + n0, o);
        }
    });
}

/* ---- the host emulation of today's kernel's arithmetic and order (moe_i4i8_run_kernel<5,16,1,0,FC>) */
inline float e4m3_to_f32_h(uint32_t v) {
    uint32_t u;
    const uint32_t s = (v & 0x80u) << 24, ex = (v >> 3) & 15u, m = v & 7u;
    if (ex == 15u && m == 7u) u = s | 0x7fc00000u;
    else if (ex == 0u) { const float f = (float)m * 0.001953125f; uint32_t fb; std::memcpy(&fb, &f, 4); u = s | fb; }
    else u = s | ((ex + 120u) << 23) | (m << 20);
    float f; std::memcpy(&f, &u, 4); return f;
}
inline uint16_t to_bf16_h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}

struct HostIn {
    int tokens, T, N, top_k, n_expert, parts;
    std::vector<int32_t> sorted, off;
    std::vector<int8_t> a;        /* [tokens][kK] */
    std::vector<float> as;        /* [tokens][kK / 128] */
    int ncl[2];
    std::vector<uint8_t> codes[2], scales[2];
    std::vector<float> tab[2], s1[2];
    size_t cstride[2], sstride[2];
};

void today_host(const HostIn& in, std::vector<uint16_t>& Y) {
#pragma clang fp contract(off)
    const int N = in.N, fy = N / in.parts;
    Y.assign((size_t)in.T * N, 0xffff);
    for (int i = 0; i < in.T; ++i) {
        const int32_t flat = in.sorted[(size_t)i];
        int e = -1;
        if (flat >= 0 && i < in.off[(size_t)in.n_expert]) {
            int lo = 0, hi = in.n_expert;
            while (lo + 1 < hi) { const int mid = (lo + hi) / 2; if (in.off[(size_t)mid] <= i) lo = mid; else hi = mid; }
            e = lo;
        }
        if (e < 0) { for (int n = 0; n < N; ++n) Y[(size_t)i * N + n] = 0; continue; }
        const int first = in.off[(size_t)e], last = in.off[(size_t)e + 1];
        if ((i - first) % kMT != 0) continue;
        const int rn = std::min(last - i, kMT);
        const int cl = e & 1, eh = e >> 1, fc = in.ncl[cl] / in.parts;
        const float* tab = &in.tab[cl][(size_t)eh * 16];
        float amax = 0.0f;
        for (int t = 0; t < 16; ++t) amax = std::fmax(amax, std::fabs(tab[t]));
        int lev[16];
        for (int t = 0; t < 16; ++t) {
            const float l127 = tab[t] * 127.0f;
            lev[t] = amax > 0.0f ? (int)std::nearbyint(l127 / amax) : (int)tab[t];
        }
        const float s1 = in.s1[cl][(size_t)eh];
        const float smul = amax > 0.0f ? s1 * (amax / 127.0f) : s1;
        const uint8_t* we = &in.codes[cl][(size_t)eh * in.cstride[cl]];
        const uint8_t* se = &in.scales[cl][(size_t)eh * in.sstride[cl]];
        for (int q = 0; q < rn; ++q) {
            const int32_t fq = in.sorted[(size_t)(i + q)];
            const int t = fq / in.top_k;
            const bool live = fq >= 0 && t < in.tokens;
            for (int n = 0; n < N; ++n) {
                const int pp = n / fy, j = n - pp * fy;
                if (!live || j >= fc) { Y[(size_t)(i + q) * N + n] = 0; continue; }
                const int row = pp * fc + j;
                const uint8_t* wr = we + (size_t)row * kWLd;
                const int8_t* ar = &in.a[(size_t)t * kK];
                float part[kGI];
                for (int gi = 0; gi < kGI; ++gi) {
                    float acc = 0.0f;
                    for (int u = 0; u < kUPL; ++u) {
                        const int g = gi + 8 * u;
                        int32_t s_lo = 0, s_hi = 0;
                        for (int k = 0; k < 64; ++k) {
                            const uint8_t by = wr[(size_t)(g * 64 + k) / 2];
                            const int code = (k & 1) ? (by >> 4) : (by & 15);
                            const int l = (int)(int8_t)((uint32_t)lev[code] & 255u);
                            const int prod = l * (int)ar[g * 64 + k];
                            if (k < 32) s_lo += prod; else s_hi += prod;
                        }
                        float fs = (float)s_lo;
                        fs += (float)s_hi;
                        const float asc = in.as[(size_t)t * (kK / 128) + g / 2];
                        const float sbr = e4m3_to_f32_h(se[(size_t)row * kSLd + g]) * smul;
                        const float p = fs * asc;
                        const float m = p * sbr;
                        acc += m;
                    }
                    part[gi] = acc;
                }
                float v = ((part[0] + part[4]) + (part[2] + part[6])) + ((part[1] + part[5]) + (part[3] + part[7]));
                v += 0.0f;
                Y[(size_t)(i + q) * N + n] = to_bf16_h(v);
            }
        }
    }
}

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}

/* the inputs as fc_tile_probe builds them: gate_up's classes 768 / 512 x K 2560, parts 2, token rows; the routing =
 * the verify mix (4 tokens x top_k 10 over 27 distinct experts) or one token over 10 */
void build(HostIn& in, int tokens, uint64_t seed) {
    in.tokens = tokens; in.top_k = 10; in.n_expert = 512; in.parts = 2; in.N = 768;
    in.T = tokens * in.top_k;
    in.ncl[0] = 768; in.ncl[1] = 512;
    const int ne[2] = { (in.n_expert + 1) / 2, in.n_expert / 2 };
    const int pool_n = tokens == 1 ? 10 : 27;
    std::vector<int> pool;
    for (uint64_t x = 0; (int)pool.size() < pool_n; ++x) {
        const int e = (int)(h32(seed, 1000 + x) % (uint32_t)in.n_expert);
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    std::vector<int> ex((size_t)in.T);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0, k = 0; s < in.top_k; ++k) {
            const int e = pool[(size_t)(h32(seed + 50, (uint64_t)t * 1000 + k) % (uint32_t)pool_n)];
            bool dup = false;
            for (int z = 0; z < s; ++z) dup |= ex[(size_t)(t * in.top_k + z)] == e;
            if (!dup) ex[(size_t)(t * in.top_k + s++)] = e;
        }
    in.off.assign((size_t)in.n_expert + 1, 0); in.sorted.assign((size_t)in.T, -1);
    for (int i = 0; i < in.T; ++i) ++in.off[(size_t)ex[(size_t)i] + 1];
    for (int e = 0; e < in.n_expert; ++e) in.off[(size_t)e + 1] += in.off[(size_t)e];
    {
        std::vector<int32_t> cur(in.off.begin(), in.off.end() - 1);
        for (int i = 0; i < in.T; ++i) in.sorted[(size_t)cur[(size_t)ex[(size_t)i]]++] = i;
    }
    in.a.resize((size_t)tokens * kK);
    for (size_t i = 0; i < in.a.size(); ++i) in.a[i] = (int8_t)((int)(h32(seed + 1, i) % 255u) - 127);
    in.as.resize((size_t)tokens * (kK / 128));
    for (size_t i = 0; i < in.as.size(); ++i) in.as[i] = 0.001f + (float)(h32(seed + 2, i) % 1000u) * 1e-5f;
    static const float kTab[8] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f };
    for (int c = 0; c < 2; ++c) {
        in.cstride[c] = (size_t)in.ncl[c] * kWLd;
        in.sstride[c] = (size_t)in.ncl[c] * kSLd;
        in.codes[c].resize((size_t)ne[c] * in.cstride[c]);
        in.scales[c].resize((size_t)ne[c] * in.sstride[c]);
        in.tab[c].resize((size_t)ne[c] * 16);
        in.s1[c].resize((size_t)ne[c]);
        for (size_t i = 0; i < in.codes[c].size(); ++i) in.codes[c][i] = (uint8_t)h32(seed + 3 + c, i);
        for (size_t i = 0; i < in.scales[c].size(); ++i)
            in.scales[c][i] = (uint8_t)(0x20u + h32(seed + 5 + c, i) % 48u) | (uint8_t)((h32(seed + 7 + c, i) & 1u) << 7);
        for (int e = 0; e < ne[c]; ++e) {
            const float f = 0.5f + (float)(h32(seed + 9 + c, e) % 100u) * 0.01f;
            for (int t = 0; t < 16; ++t)
                in.tab[c][(size_t)e * 16 + t] = e % 3 == 2 ? (float)((int)(h32(seed + 11 + c, e * 16 + t) % 4001u) - 2000) * 0.01f
                                                           : (t < 8 ? kTab[t] : -kTab[t - 8]) * f;
            in.s1[c][(size_t)e] = std::ldexp(1.0f + (float)(e % 5) * 0.125f, -13);
        }
    }
}

template <class T>
T* dev_copy(sycl::queue& q, const std::vector<T>& v) {
    T* p = sycl::malloc_device<T>(v.size() ? v.size() : 1, q);
    if (v.size()) q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
    return p;
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

}  // namespace

int main(int argc, char** argv) {
    const int launches = argc > 1 ? std::atoi(argv[1]) : 48;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    const int tokens = argc > 3 ? std::atoi(argv[3]) : 4;
    sycl::queue q;
    const std::string dev = q.get_device().get_info<sycl::info::device::name>();
    std::printf("esimd_fc_probe: device %s\n", dev.c_str());
    if (!q.get_device().is_gpu()) { std::printf("esimd_fc_probe: skip (ESIMD runs on a GPU only)\n"); return 0; }
    bool ok = true;
    for (int tk : { tokens, 1 }) {
        HostIn in;
        build(in, tk, 0x5eed0000ull + (uint64_t)tk);
        std::vector<uint16_t> ref;
        today_host(in, ref);
        int8_t* dA = dev_copy(q, in.a);
        float* dAs = dev_copy(q, in.as);
        int32_t* dsorted = dev_copy(q, in.sorted);
        int32_t* doff = dev_copy(q, in.off);
        uint16_t* dY = sycl::malloc_device<uint16_t>((size_t)in.T * in.N, q);
        ECls c[2];
        for (int k = 0; k < 2; ++k) {
            c[k].codes = dev_copy(q, in.codes[k]); c[k].cstride = in.cstride[k];
            c[k].tab = dev_copy(q, in.tab[k]); c[k].s1 = dev_copy(q, in.s1[k]);
            c[k].scales = dev_copy(q, in.scales[k]); c[k].sstride = in.sstride[k];
            c[k].n = (uint32_t)in.ncl[k];
        }
        q.memset(dY, 0xff, (size_t)in.T * in.N * 2).wait();
        esimd_fc_launch<2>(q, dA, dAs, c[0], c[1], dsorted, doff, dY, (uint32_t)in.T, (uint32_t)in.N, (uint32_t)in.n_expert,
                           (uint32_t)in.top_k, (uint32_t)kK, (uint32_t)(kK / 128), (uint32_t)in.N, (uint32_t)in.parts,
                           (uint32_t)in.tokens, 0u);
        q.wait();
        std::vector<uint16_t> out((size_t)in.T * in.N);
        q.memcpy(out.data(), dY, out.size() * 2).wait();
        size_t diff = 0, first = out.size();
        for (size_t x = 0; x < out.size(); ++x) if (out[x] != ref[x]) { if (first == out.size()) first = x; ++diff; }
        std::printf("ESIMD gate_up tokens %d (T %d, N %d): %zu of %zu outputs differ from today's order on the host%s\n",
                    tk, in.T, in.N, diff, out.size(), diff ? "" : " -> BIT-IDENTICAL");
        if (diff) {
            std::printf("  first difference at output %zu (row %zu, column %zu): %04x vs host %04x\n", first,
                        first / (size_t)in.N, first % (size_t)in.N, out[first], ref[first]);
            ok = false;
        }
        if (launches > 0) {
            std::vector<double> us;
            for (int r = 0; r < runs; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int l = 0; l < launches; ++l)
                    esimd_fc_launch<2>(q, dA, dAs, c[0], c[1], dsorted, doff, dY, (uint32_t)in.T, (uint32_t)in.N,
                                       (uint32_t)in.n_expert, (uint32_t)in.top_k, (uint32_t)kK, (uint32_t)(kK / 128),
                                       (uint32_t)in.N, (uint32_t)in.parts, (uint32_t)in.tokens, 0u);
                q.wait();
                const auto t1 = std::chrono::steady_clock::now();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / launches);
            }
            std::printf("RESULT-ESIMD gate_up tokens %d: %.1f us a launch (median of %d x %d)\n", tk, median(us), runs, launches);
        }
        sycl::free(dA, q); sycl::free(dAs, q); sycl::free(dsorted, q); sycl::free(doff, q); sycl::free(dY, q);
        for (int k = 0; k < 2; ++k) {
            sycl::free((void*)c[k].codes, q); sycl::free((void*)c[k].tab, q); sycl::free((void*)c[k].s1, q);
            sycl::free((void*)c[k].scales, q);
        }
    }
    std::printf("%s esimd_fc_probe\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
