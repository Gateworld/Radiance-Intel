/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 109).
 *
 * attn_paged on the matrix units for a prefill pass (included by xe_attn.cpp). Opt-in:
 * RADIANCE_XE_ATTN_JM=1, a single sequence of at least 64 query rows, GPU only; otherwise the
 * split-KV kernel runs.
 *
 * Work-group: 8 sub-groups of 16, a 64-row query block of one query head; sub-group w owns rows
 * 8w .. 8w+7. Keys in blocks of 32: their cache rows (E4M3 or bf16; E4M3 values are exact in bf16)
 * are staged as K^T [d][key] and V [key][d] in LDS, Q [row][d] once. Two passes over the keys:
 * the first takes each row's maximum score, the second forms p = e^(s - max) (bf16) and sums
 * P @ V into accumulators that are never rescaled. The descales fold in: s = (q . k) * kds * scale,
 * out = vds * (P @ V) / l, l summed from the rounded p. A row with no visible key is zero, as in
 * the split kernel; the arithmetic differs (bf16 p, the sums' order), so this is a tolerance check. */
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace attn_jm {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kSG = 16, kNW = 8, kNT = kSG * kNW;
constexpr int BQ = 64, BK = 32;

using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;
using MA  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major>;
using MB  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::row_major>;

template <int HD> struct Smem {
    uint16_t q[BQ][HD];
    uint16_t kT[HD][BK];
    uint16_t v[BK][HD];
    uint16_t p[BQ][BK];
    float c[kNW][8 * BK];          /* a sub-group's scores (8 x BK), or one 8 x 16 output tile */
    float m[BQ], l[BQ];
    long slot[BK];
};

template <class T> inline auto lp(T* x) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(x);
}

inline uint16_t bf16_rn(float f) {             /* round to nearest even (finite inputs) */
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* eight cache elements at element offset o as bf16 bits; zeros for a missing slot */
template <bool KV8>
inline void ld8(const uint8_t* kv, long o, uint16_t* x) {
    if (o < 0) {
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = 0;
        return;
    }
    if constexpr (KV8) {
        /* one 8-byte load when aligned (a cache row is 2 x head_dim bytes and c a multiple of 8), else bytes */
        if ((o & 7) == 0) {
            const uint64_t b = *(const uint64_t*)(kv + o);
#pragma unroll
            for (int e = 0; e < 8; ++e)
                x[e] = (uint16_t)(__float_as_uint(e4m3((uint32_t)(b >> (8 * e)) & 0xffu)) >> 16);   /* exact */
        } else {
            const uint8_t* b = kv + o;
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = (uint16_t)(__float_as_uint(e4m3(b[e])) >> 16);
        }
    } else {
        const uint16_t* b = (const uint16_t*)kv + o;
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = b[e];
    }
}

template <int E, bool KV8, bool OUT_F32>
void kernel(AttnP p) {
    constexpr int HD = 32 * E;
    constexpr int NO = HD / 16;                 /* output column tiles a sub-group */
    XE_SHARED_VAR(Smem<HD>, s);
    const int tid = (int)threadIdx.x, w = tid / kSG, lane = tid % kSG;
    const int qb = (int)blockIdx.x, h = (int)blockIdx.y, kh = h / p.gqa;
    const int base = p.cu ? p.cu[0] : 0;
    const int qlen = p.cu ? p.cu[1] - p.cu[0] : p.q_len;
    const int i0 = qb * BQ;
    if (i0 >= qlen) return;
    const int nrow = qlen - i0 < BQ ? qlen - i0 : BQ;
    const int ctx = p.seqused[0];
    const int cap = p.max_blocks * p.block_size;
    const int kmax = ctx < cap ? ctx : cap;
    const int pos0 = ctx - qlen + i0, posl = pos0 + nrow - 1;
    int jstart = 0, jstop = kmax;
    if (p.window > 0 && pos0 - p.window + 1 > 0) jstart = pos0 - p.window + 1;
    if (p.causal && jstop > posl + 1) jstop = posl + 1;
    if (p.window > 0 && !p.causal && jstop > posl + p.window) jstop = posl + p.window;
    const float kds = p.kds ? p.kds[kh] : 1.0f, vds = p.vds ? p.vds[kh] : 1.0f;
    const float sc = p.scale * kds;

    for (int idx = tid; idx < BQ * HD / 8; idx += kNT) {
        const int r = idx / (HD / 8), c = (idx % (HD / 8)) * 8;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (r < nrow) x = *(const uint4*)(p.q + (size_t)(base + i0 + r) * p.q_rs + (size_t)h * HD + c);
        *(uint4*)&s.q[r][c] = x;
    }
    if (tid < BQ) { s.m[tid] = -INFINITY; s.l[tid] = 0.0f; }

    sycl::sub_group sg = xe::sg();
    const int r0 = 8 * w;
    float* const cw = s.c[w];
    /* this lane's row and half of a key block, for the softmax steps */
    const int rr = lane & 7, hh = lane >> 3, row = r0 + rr, pos = pos0 + row;
    int jlo = 0, jhi = kmax;
    if (p.window > 0) jlo = pos - p.window + 1;
    if (p.causal && jhi > pos + 1) jhi = pos + 1;
    if (p.window > 0 && !p.causal && jhi > pos + p.window) jhi = pos + p.window;
    const bool live = row < nrow;

    Acc O[NO];
#pragma unroll
    for (int t = 0; t < NO; ++t) jm::joint_matrix_fill(sg, O[t], 0.0f);

    for (int pass = 0; pass < 2; ++pass) {
        for (int j0 = jstart; j0 < jstop; j0 += BK) {
            xe::syncthreads();                 /* the previous block's stages are consumed */
            if (tid < BK) {
                const int j = j0 + tid;
                long sl = -1;
                if (j < jstop) {
                    const int blk = p.bt[j / p.block_size];
                    if (blk >= 0)
                        sl = (long)blk * p.kv_bs + (long)kh * p.kv_hs + (long)(j % p.block_size) * p.kv_row;
                }
                s.slot[tid] = sl;
            }
            xe::syncthreads();
            for (int idx = tid; idx < BK * HD / 8; idx += kNT) {
                const int kk = idx / (HD / 8), c = (idx % (HD / 8)) * 8;
                const long sl = s.slot[kk];
                uint16_t x[8];
                ld8<KV8>(p.kv, sl < 0 ? -1 : sl + c, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) s.kT[c + e][kk] = x[e];
                if (pass == 1) {
                    ld8<KV8>(p.kv, sl < 0 ? -1 : sl + HD + c, x);
                    uint4 y;
                    __builtin_memcpy(&y, x, sizeof y);
                    *(uint4*)&s.v[kk][c] = y;
                }
            }
            xe::syncthreads();

            /* S = Q K^T for this sub-group's 8 rows */
            {
                Acc S[BK / 16];
#pragma unroll
                for (int t = 0; t < BK / 16; ++t) jm::joint_matrix_fill(sg, S[t], 0.0f);
                MA a;
                MB b;
#pragma unroll
                for (int ks = 0; ks < HD / 16; ++ks) {
                    jm::joint_matrix_load(sg, a, lp((bf16*)&s.q[r0][ks * 16]), HD);
#pragma unroll
                    for (int t = 0; t < BK / 16; ++t) {
                        jm::joint_matrix_load(sg, b, lp((bf16*)&s.kT[ks * 16][t * 16]), BK);
                        jm::joint_matrix_mad(sg, S[t], a, b, S[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < BK / 16; ++t)
                    jm::joint_matrix_store(sg, S[t], lp(cw + t * 16), BK, jm::layout::row_major);
            }
            xe::wave_barrier();

            if (pass == 0) {
                float mx = -INFINITY;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int kk = hh * 16 + e, j = j0 + kk;
                    if (live && j >= jlo && j < jhi && s.slot[kk] >= 0) {
                        const float x = cw[rr * BK + kk] * sc;
                        mx = x > mx ? x : mx;
                    }
                }
                const float o = sycl::permute_group_by_xor(sg, mx, 8);
                mx = o > mx ? o : mx;
                if (hh == 0 && mx > s.m[row]) s.m[row] = mx;
                xe::wave_barrier();
            } else {
                const float mrow = s.m[row];
                float sum = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int kk = hh * 16 + e, j = j0 + kk;
                    uint16_t pb = 0;
                    if (live && j >= jlo && j < jhi && s.slot[kk] >= 0) {
                        pb = bf16_rn(sycl::exp(cw[rr * BK + kk] * sc - mrow));
                        sum += __uint_as_float((uint32_t)pb << 16);
                    }
                    s.p[row][kk] = pb;
                }
                sum += sycl::permute_group_by_xor(sg, sum, 8);
                if (hh == 0) s.l[row] += sum;
                xe::wave_barrier();
                MA a;
                MB b;
#pragma unroll
                for (int ks = 0; ks < BK / 16; ++ks) {
                    jm::joint_matrix_load(sg, a, lp((bf16*)&s.p[r0][ks * 16]), BK);
#pragma unroll
                    for (int t = 0; t < NO; ++t) {
                        jm::joint_matrix_load(sg, b, lp((bf16*)&s.v[ks * 16][t * 16]), HD);
                        jm::joint_matrix_mad(sg, O[t], a, b, O[t]);
                    }
                }
            }
        }
    }

    /* out = vds * O / l per row (the barrier publishes l's initial zeros when a block saw no key) */
    xe::syncthreads();
#pragma unroll
    for (int t = 0; t < NO; ++t) {
        jm::joint_matrix_store(sg, O[t], lp(cw), 16, jm::layout::row_major);
        xe::wave_barrier();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int rw = r0 + e;
            if (rw < nrow) {
                const float lv = s.l[rw];
                const float val = lv > 0.0f ? cw[e * 16 + lane] * vds / lv : 0.0f;
                const size_t o = ((size_t)(base + i0 + rw) * p.q_heads + h) * HD + t * 16 + lane;
                if constexpr (OUT_F32) ((float*)p.out)[o] = val;
                else ((uint16_t*)p.out)[o] = to_bf16(val);
            }
        }
        xe::wave_barrier();
    }
}

inline bool on(sycl::queue* q) {
    static const bool want = [] { const char* e = std::getenv("RADIANCE_XE_ATTN_JM"); return e && *e == '1'; }();
    static const bool gpu = q->get_device().is_gpu();
    return want && gpu;
}

template <int E, bool KV8, bool OF>
void launch(sycl::queue* q, const AttnP& p, int total_q) {
    constexpr auto K = kernel<E, KV8, OF>;
    xe::launch_as_grf256<K, kSG>(K, *q, dim3((unsigned)((total_q + BQ - 1) / BQ), (unsigned)p.q_heads), dim3(kNT), p);
}
}  // namespace attn_jm

/* ---- RADIANCE_XE_ATTN_MX=1 (run 109, opt-in): the split kernel's work on the matrix units, per query row.
 *
 * The rows this model's prefill issues are sparse (rad_block_attn_gated_fp8.h: one row a query, its own
 * block selection), so no two rows share keys; a kv head's gqa query heads do. A work-group is one
 * (split, kv head, query row), as the split kernel's grid, with one sub-group of 16 per 8 heads (gqa
 * <= 16; this model's 12 take two, the second padded with zeros), the heads the rows of 8 x 16 tiles;
 * the sub-groups share the key stages. Q stays in registers as HD / 16 A tiles. Keys in
 * blocks of 16: K^T and V staged in LDS (E4M3 exact in bf16), S = Q K^T, an online softmax whose
 * reference max moves only when a block's max passes it by more than 8 (then the HD / 16 output tiles
 * are rescaled through the scratch), p = e^(s - ref) in bf16, O += P V. The partials (acc, ref, l) are
 * the split kernel's layout, so the merge is shared: a reference below the true max cancels in it. */
namespace attn_mx {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;
using MA  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major>;
using MB  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::row_major>;
constexpr int BK = 16;
constexpr float kLazy = 8.0f;     /* e^8: the most a p may exceed 1 before the reference moves */

/* A sub-group a row tile of 8 heads (gqa <= 16: one or two), sharing the key stages. */
constexpr int kMaxRT = 2;
template <int HD> struct Smem {
    uint16_t q[8 * kMaxRT][HD];    /* the heads' rows, zeros past gqa; loaded once into A tiles */
    uint16_t kT[HD][BK];
    uint16_t v[BK][HD];
    uint16_t p[kMaxRT][8][BK];
    float c[kMaxRT][8 * BK];       /* a sub-group's S, or one 8 x 16 output tile */
};

/* P @ V in fp16 when the cache is E4M3 (its values are exact in fp16; p <= e^8 fits): p rounded to bf16 moved
 * the 7K prompt's top logprobs by up to 0.4 nat (B66 against B65). A bf16 cache keeps bf16 (its range). */
template <bool KV8> struct PV { using T = bf16; };
template <> struct PV<true> { using T = sycl::half; };
template <bool KV8> inline uint16_t pv_bits(float f) {
    if constexpr (KV8) return sycl::bit_cast<uint16_t>(sycl::half(f));
    else return attn_jm::bf16_rn(f);
}
template <bool KV8> inline float pv_float(uint16_t b) {
    if constexpr (KV8) return (float)sycl::bit_cast<sycl::half>(b);
    else return __uint_as_float((uint32_t)b << 16);
}
/* eight V elements at element offset o in the P @ V type's bits; zeros for a missing slot */
template <bool KV8>
inline void ld8v(const uint8_t* kv, long o, uint16_t* x) {
    if constexpr (KV8) {
        if (o < 0) {
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = 0;
            return;
        }
        if ((o & 7) == 0) {
            const uint64_t b = *(const uint64_t*)(kv + o);
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = pv_bits<true>(e4m3((uint32_t)(b >> (8 * e)) & 0xffu));
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = pv_bits<true>(e4m3(kv[o + e]));
        }
    } else {
        attn_jm::ld8<false>(kv, o, x);
    }
}

template <int E, bool KV8, bool DIRECT, bool OUT_F32>
void kernel(AttnP p) {
    constexpr int HD = 32 * E, NT = HD / 16;
    using PT = typename PV<KV8>::T;
    using PA = jm::joint_matrix<sycl::sub_group, PT, jm::use::a, 8, 16, jm::layout::row_major>;
    using PB = jm::joint_matrix<sycl::sub_group, PT, jm::use::b, 16, 16, jm::layout::row_major>;
    XE_SHARED_VAR(Smem<HD>, s);
    const int tid = (int)threadIdx.x, w = tid / 16, lane = tid % 16;
    const int nrt = (p.gqa + 7) / 8;           /* sub-groups in the work-group, one a row tile */
    const int split = (int)blockIdx.x, kh = (int)blockIdx.y, tok = (int)blockIdx.z;
    int sq, i, qlen;
    seq_of(p, tok, &sq, &i, &qlen);
    if (qlen <= 0 || i >= qlen) return;
    const int ctx = p.seqused[sq];
    const int pos = ctx - qlen + i;
    int jlo = 0, jend = ctx < p.max_blocks * p.block_size ? ctx : p.max_blocks * p.block_size;
    if (p.causal && jend > pos + 1) jend = pos + 1;
    if (p.window > 0) {
        if (jlo < pos - p.window + 1) jlo = pos - p.window + 1;
        if (!p.causal && jend > pos + p.window) jend = pos + p.window;
    }
    const int n = jend > jlo ? jend - jlo : 0;
    const int chunk = (((n + p.splits - 1) / p.splits) + 3) & ~3;
    const int j0s = jlo + split * chunk;
    const int j1 = j0s + chunk < jend ? j0s + chunk : jend;
    const float kds = p.kds ? p.kds[sq * p.kv_heads + kh] : 1.0f;
    const float vds = p.vds ? p.vds[sq * p.kv_heads + kh] : 1.0f;
    const float sc = p.scale * kds;
    const int gq = p.gqa;
    sycl::sub_group sg = xe::sg();

    /* Q: the gqa heads' rows (contiguous heads, HD apart), zeros past gqa */
    for (int idx = tid; idx < 8 * nrt * HD / 8; idx += 16 * nrt) {
        const int r = idx / (HD / 8), c = (idx % (HD / 8)) * 8;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (r < gq) x = *(const uint4*)(p.q + (size_t)tok * p.q_rs + (size_t)(kh * gq + r) * HD + c);
        *(uint4*)&s.q[r][c] = x;
    }
    xe::syncthreads();
    MA qa[NT];
#pragma unroll
    for (int ks = 0; ks < NT; ++ks) jm::joint_matrix_load(sg, qa[ks], attn_jm::lp((bf16*)&s.q[8 * w][ks * 16]), HD);
    float* const cw = s.c[w];
    Acc O[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) jm::joint_matrix_fill(sg, O[t], 0.0f);
    float ref[8], l[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { ref[r] = -INFINITY; l[r] = 0.0f; }
    const int* btr = p.bt + (size_t)sq * p.bt_rs;

    for (int j0 = j0s; j0 < j1; j0 += BK) {
        /* this lane's key: its cache row, or none */
        const int j = j0 + lane;
        long sl = -1;
        if (j < j1) {
            const int blk = btr[j / p.block_size];
            if (blk >= 0) sl = (long)blk * p.kv_bs + (long)kh * p.kv_hs + (long)(j % p.block_size) * p.kv_row;
        }
        /* stage the block: key kk's row read by one sub-group's 16 lanes (HD / 16 elements each) */
        xe::syncthreads();                     /* every sub-group's reads of the previous kT / v are done */
#pragma unroll 4
        for (int kk = w; kk < BK; kk += nrt) {
            const long o = sycl::select_from_group(sg, sl, kk);
#pragma unroll
            for (int h8 = 0; h8 < HD / 16; h8 += 8) {
                const int c = lane * (HD / 16) + h8;
                uint16_t x[8];
                attn_jm::ld8<KV8>(p.kv, o < 0 ? -1 : o + c, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) s.kT[c + e][kk] = x[e];
                ld8v<KV8>(p.kv, o < 0 ? -1 : o + HD + c, x);
                uint4 y;
                __builtin_memcpy(&y, x, sizeof y);
                *(uint4*)&s.v[kk][c] = y;
            }
        }
        xe::syncthreads();
        {
            Acc S;
            jm::joint_matrix_fill(sg, S, 0.0f);
            MB b;
#pragma unroll
            for (int ks = 0; ks < NT; ++ks) {
                jm::joint_matrix_load(sg, b, attn_jm::lp((bf16*)&s.kT[ks * 16][0]), BK);
                jm::joint_matrix_mad(sg, S, qa[ks], b, S);
            }
            jm::joint_matrix_store(sg, S, attn_jm::lp(cw), BK, jm::layout::row_major);
        }
        xe::wave_barrier();
        /* lane = key; per head row: the block's max, the reference, p and its sum */
        const bool kv_ok = sl >= 0;
        float sv[8];
        bool move = false;
        float fac[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            sv[r] = kv_ok ? cw[r * BK + lane] * sc : -INFINITY;
            const float bm = sycl::reduce_over_group(sg, sv[r], sycl::maximum<float>());
            fac[r] = 1.0f;
            if (bm != -INFINITY && (ref[r] == -INFINITY || bm > ref[r] + kLazy)) {
                fac[r] = ref[r] == -INFINITY ? 0.0f : sycl::exp(ref[r] - bm);
                ref[r] = bm;
                move = true;
            }
        }
        if (move) {   /* uniform: every lane took the same decisions from the reduced maxima */
#pragma unroll
            for (int r = 0; r < 8; ++r) l[r] *= fac[r];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                jm::joint_matrix_store(sg, O[t], attn_jm::lp(cw), 16, jm::layout::row_major);
                xe::wave_barrier();
#pragma unroll
                for (int r = 0; r < 8; ++r) cw[r * 16 + lane] *= fac[r];
                xe::wave_barrier();
                jm::joint_matrix_load(sg, O[t], attn_jm::lp(cw), 16, jm::layout::row_major);
                xe::wave_barrier();
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            uint16_t pb = 0;
            if (sv[r] != -INFINITY) pb = pv_bits<KV8>(sycl::exp(sv[r] - ref[r]));
            s.p[w][r][lane] = pb;
            l[r] += sycl::reduce_over_group(sg, pv_float<KV8>(pb), sycl::plus<float>());
        }
        xe::wave_barrier();
        {
            PA a;
            jm::joint_matrix_load(sg, a, attn_jm::lp((PT*)&s.p[w][0][0]), BK);
            PB b;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                jm::joint_matrix_load(sg, b, attn_jm::lp((PT*)&s.v[0][t * 16]), HD);
                jm::joint_matrix_mad(sg, O[t], a, b, O[t]);
            }
        }
    }

    /* out (DIRECT) or the split's partials: acc = vds * O, the reference as m, l */
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        xe::wave_barrier();
        jm::joint_matrix_store(sg, O[t], attn_jm::lp(cw), 16, jm::layout::row_major);
        xe::wave_barrier();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (8 * w + r >= gq) continue;
            const int h = kh * gq + 8 * w + r;
            const float acc = cw[r * 16 + lane] * vds;
            if constexpr (DIRECT) {
                const float val = l[r] > 0.0f ? acc / l[r] : 0.0f;
                const size_t o = ((size_t)tok * p.q_heads + h) * HD + t * 16 + lane;
                if constexpr (OUT_F32) ((float*)p.out)[o] = val;
                else ((uint16_t*)p.out)[o] = to_bf16(val);
            } else {
                float* w = p.part + (((size_t)tok * p.q_heads + h) * p.splits + split) * (HD + 2);
                w[t * 16 + lane] = acc;
                if (t == 0 && lane == 0) { w[HD] = ref[r]; w[HD + 1] = l[r]; }
            }
        }
    }
}

inline bool on(sycl::queue* q) {
    static const bool want = [] { const char* e = std::getenv("RADIANCE_XE_ATTN_MX"); return e && *e == '1'; }();
    static const bool gpu = q->get_device().is_gpu();
    return want && gpu;
}

template <int E, bool KV8, bool OF>
void launch(sycl::queue* q, const AttnP& p, int total_q) {
    const dim3 grid((unsigned)p.splits, (unsigned)p.kv_heads, (unsigned)total_q), block((unsigned)(16 * ((p.gqa + 7) / 8)));
    if (p.splits == 1) {
        constexpr auto K = kernel<E, KV8, true, OF>;
        xe::launch_as_grf256<K, 16>(K, *q, grid, block, p);
    } else {
        constexpr auto K = kernel<E, KV8, false, OF>;
        xe::launch_as_grf256<K, 16>(K, *q, grid, block, p);
        XE_LAUNCH(q, (attn_merge_kernel<E, OF>), dim3((unsigned)((long long)total_q * p.q_heads)), dim3(32), 0, p);
    }
}
}  // namespace attn_mx
