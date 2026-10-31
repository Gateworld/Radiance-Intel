/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_attn.cpp -- attn_paged for decode and speculative verify (libr4d's kernels are WMMA): split-KV
 * over the paged cache. A work-group is one (split, kv head, query row) and a sub-group one query
 * head of the kv head's gqa group; lane l holds head_dim / 32 consecutive elements of q, k and v.
 * Per group of four keys: the dot products meet in a butterfly, then an online softmax (f32 max,
 * sum and accumulator, rescaled when the max moves). Each split writes (acc, max, sum) to scratch
 * and a merge launch combines the splits; a single split writes out directly. The arithmetic is
 * libref's (dot = sum q * (k * k_descale), score = dot * scale, v * v_descale, out = acc * 1/sum, a
 * row with no visible key is zero) in a different summation order, so this is a tolerance check.
 */
#include "xe_lanes.h"
#include "r4d_args.h"

#include <cmath>
#include <cstdint>

namespace {

constexpr int kTargetWG = 512;   /* splits are added until this many work-groups are in flight */
constexpr int kMaxSplits = 32;

enum { AP_Q = 0, AP_KV, AP_BT, AP_SEQ, AP_KDS, AP_VDS, AP_CU, AP_OUT };

struct AttnP {
    const uint16_t* q; long q_rs;
    const uint8_t* kv; long kv_bs, kv_hs, kv_row;   /* strides in elements of the cache dtype */
    const int* bt; long bt_rs; int max_blocks;
    const int* seqused; const int* cu; int n_seq; int q_len;
    const float* kds; const float* vds;
    void* out; float* part;
    int q_heads, kv_heads, gqa, block_size, causal, window, splits;
    float scale;
};

inline float ld_bf16(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
inline uint16_t to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
}
/* OCP E4M3 (fn): bias 7, no infinities, S.1111.111 NaN */
inline float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;                     /* m * 2^-9 */
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

/* E elements of a cache row at element offset o (bytes for E4M3, bf16 elements otherwise). VX (RADIANCE_XE_ATTN_VX=1,
 * run 110, owner note 129 item 5): an E4M3 lane's 8 bytes as one 8-byte load (the launcher checks the alignment), the
 * same bytes decoded alike. */
template <int E, bool KV8, bool VX = false>
inline void ld_kv(const uint8_t* kv, long o, float* x) {
    if constexpr (KV8 && VX && E == 8) {
        const uint2 v = *(const uint2*)(kv + o);
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[e] = e4m3((v.x >> (8 * e)) & 255u); x[4 + e] = e4m3((v.y >> (8 * e)) & 255u); }
    } else if constexpr (KV8) {
        const uint8_t* p = kv + o;
#pragma unroll
        for (int e = 0; e < E; ++e) x[e] = e4m3(p[e]);
    } else {
        const uint16_t* p = (const uint16_t*)kv + o;
#pragma unroll
        for (int e = 0; e < E; ++e) x[e] = ld_bf16(p[e]);
    }
}

/* The query row's sequence, its index in the sequence's query and the sequence's query length. */
inline void seq_of(const AttnP& p, int tok, int* s, int* i, int* qlen) {
    if (p.cu) {
        int lo = 0, hi = p.n_seq - 1;                     /* last s with cu[s] <= tok */
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (p.cu[mid] <= tok) lo = mid; else hi = mid - 1;
        }
        *s = lo; *i = tok - p.cu[lo]; *qlen = p.cu[lo + 1] - p.cu[lo];
    } else {
        *s = tok / p.q_len; *i = tok - *s * p.q_len; *qlen = p.q_len;
    }
}

/* VX: the K / V loads as ld_kv's VX, and the dot product's five butterfly steps as permute_group_by_xor -- the same
 * partner lanes in the same order (run 102's native xor), so the same values. */
template <int E, bool KV8, bool DIRECT, bool OUT_F32, bool VX = false>
void attn_split_kernel(AttnP p) {
    constexpr int HD = 32 * E;
    const int lane = (int)threadIdx.x & 31, g = (int)threadIdx.x >> 5;
    const int split = (int)blockIdx.x, kh = (int)blockIdx.y, tok = (int)blockIdx.z;
    const int h = kh * p.gqa + g;
    int s, i, qlen;
    seq_of(p, tok, &s, &i, &qlen);
    if (qlen <= 0 || i >= qlen) return;
    const int ctx = p.seqused[s];
    const int pos = ctx - qlen + i;
    int jlo = 0, jend = ctx < p.max_blocks * p.block_size ? ctx : p.max_blocks * p.block_size;
    if (p.causal && jend > pos + 1) jend = pos + 1;
    if (p.window > 0) {
        if (jlo < pos - p.window + 1) jlo = pos - p.window + 1;
        if (!p.causal && jend > pos + p.window) jend = pos + p.window;
    }
    /* this split's keys: whole groups of four */
    const int n = jend > jlo ? jend - jlo : 0;
    const int chunk = (((n + p.splits - 1) / p.splits) + 3) & ~3;
    const int j0 = jlo + split * chunk;
    const int j1 = j0 + chunk < jend ? j0 + chunk : jend;

    const float kds = p.kds ? p.kds[s * p.kv_heads + kh] : 1.0f;
    const float vds = p.vds ? p.vds[s * p.kv_heads + kh] : 1.0f;
    float qf[E], acc[E];
    const uint16_t* qr = p.q + (size_t)tok * p.q_rs + (size_t)h * HD + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) { qf[e] = ld_bf16(qr[e]); acc[e] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    const int* btr = p.bt + (size_t)s * p.bt_rs;
    for (int j = j0; j < j1; j += 4) {
        long slot[4];
        float sc[4];
        float mt = -INFINITY;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int jj = j + u;
            const int blk = jj < j1 ? btr[jj / p.block_size] : -1;
            slot[u] = blk < 0 ? -1 : (long)blk * p.kv_bs + (long)kh * p.kv_hs + (long)(jj % p.block_size) * p.kv_row;
            float d = 0.0f;
            if (slot[u] >= 0) {
                float kf[E];
                ld_kv<E, KV8, VX>(p.kv, slot[u] + lane * E, kf);
#pragma unroll
                for (int e = 0; e < E; ++e) d += qf[e] * (kf[e] * kds);
            }
            if constexpr (VX) {
                const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                for (int x = 16; x >= 1; x >>= 1) d += sycl::permute_group_by_xor(sg, d, x);
            } else
            for (int x = 16; x >= 1; x >>= 1) d += __shfl_xor(d, x);
            sc[u] = slot[u] >= 0 ? d * p.scale : -INFINITY;
            if (sc[u] > mt) mt = sc[u];
        }
        if (mt == -INFINITY) continue;
        const float mn = m > mt ? m : mt;
        const float corr = sycl::exp(m - mn);           /* 0 while m is -inf */
        l *= corr;
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] *= corr;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (slot[u] < 0) continue;
            const float w = sycl::exp(sc[u] - mn);
            l += w;
            float vf[E];
            ld_kv<E, KV8, VX>(p.kv, slot[u] + HD + lane * E, vf);
#pragma unroll
            for (int e = 0; e < E; ++e) acc[e] += w * (vf[e] * vds);
        }
        m = mn;
    }
    if constexpr (DIRECT) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        const size_t o = ((size_t)tok * p.q_heads + h) * HD + lane * E;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            if constexpr (OUT_F32) ((float*)p.out)[o + e] = acc[e] * inv;
            else ((uint16_t*)p.out)[o + e] = to_bf16(acc[e] * inv);
        }
    } else {
        float* w = p.part + (((size_t)tok * p.q_heads + h) * p.splits + split) * (HD + 2);
#pragma unroll
        for (int e = 0; e < E; ++e) w[lane * E + e] = acc[e];
        if (lane == 0) { w[HD] = m; w[HD + 1] = l; }
    }
}

/* one sub-group per (query row, head): out = sum_p acc_p e^(m_p - M) / sum_p l_p e^(m_p - M) */
template <int E, bool OUT_F32>
void attn_merge_kernel(AttnP p) {
    constexpr int HD = 32 * E;
    const int lane = (int)threadIdx.x & 31;
    const size_t r = (size_t)blockIdx.x;
    const int tok = (int)(r / p.q_heads);
    int s, i, qlen;
    seq_of(p, tok, &s, &i, &qlen);
    if (qlen <= 0 || i >= qlen) return;
    const float* w = p.part + r * p.splits * (HD + 2);
    float M = -INFINITY;
    for (int k = 0; k < p.splits; ++k) {
        const float mk = w[(size_t)k * (HD + 2) + HD];
        if (mk > M) M = mk;
    }
    float acc[E], L = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.0f;
    if (M != -INFINITY)
        for (int k = 0; k < p.splits; ++k) {
            const float* wk = w + (size_t)k * (HD + 2);
            if (wk[HD] == -INFINITY) continue;
            const float c = sycl::exp(wk[HD] - M);
            L += wk[HD + 1] * c;
#pragma unroll
            for (int e = 0; e < E; ++e) acc[e] += wk[lane * E + e] * c;
        }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    const size_t o = r * HD + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        if constexpr (OUT_F32) ((float*)p.out)[o + e] = acc[e] * inv;
        else ((uint16_t*)p.out)[o + e] = to_bf16(acc[e] * inv);
    }
}

/* The split count for `rows` query rows over `kv_heads` heads: enough work-groups to fill the card,
 * one split (no scratch) once the rows alone do. */
int splits_for(long long rows, long long kv_heads) {
    const long long wg = rows * kv_heads;
    if (wg >= kTargetWG) return 1;
    const long long sp = (kTargetWG + wg - 1) / wg;
    return (int)(sp < kMaxSplits ? sp : kMaxSplits);
}
/* bytes for every split's partials: bounded by (kTargetWG + rows * kv_heads) * gqa rows of
 * head_dim + 2 floats while more than one split is used, i.e. by 2 * kTargetWG * gqa */
long long part_bytes(long long head_dim, long long gqa) {
    return 2LL * kTargetWG * gqa * (head_dim + 2) * (long long)sizeof(float);
}

template <int E, bool KV8, bool OF>
void launch(sycl::queue* q, const AttnP& p, int total_q) {
    const dim3 grid((unsigned)p.splits, (unsigned)p.kv_heads, (unsigned)total_q), block(32 * p.gqa);
    /* RADIANCE_XE_ATTN_VX=1 (run 110): an E4M3 cache at E 8 whose every row offset is a multiple of 8 bytes */
    static const bool vx_on = [] { const char* e = std::getenv("RADIANCE_XE_ATTN_VX"); return e && e[0] == '1'; }();
    const bool vx = vx_on && KV8 && E == 8 && ((uintptr_t)p.kv | (uintptr_t)p.kv_bs | (uintptr_t)p.kv_hs | (uintptr_t)p.kv_row) % 8 == 0;
    if (vx) {
        if (p.splits == 1) XE_LAUNCH(q, (attn_split_kernel<E, KV8, true, OF, true>), grid, block, 0, p);
        else XE_LAUNCH(q, (attn_split_kernel<E, KV8, false, OF, true>), grid, block, 0, p);
        if (p.splits > 1)
            XE_LAUNCH(q, (attn_merge_kernel<E, OF>), dim3((unsigned)((long long)total_q * p.q_heads)), dim3(32), 0, p);
    } else if (p.splits == 1) {
        XE_LAUNCH(q, (attn_split_kernel<E, KV8, true, OF>), grid, block, 0, p);
    } else {
        XE_LAUNCH(q, (attn_split_kernel<E, KV8, false, OF>), grid, block, 0, p);
        XE_LAUNCH(q, (attn_merge_kernel<E, OF>), dim3((unsigned)((long long)total_q * p.q_heads)), dim3(32), 0, p);
    }
}

}  // namespace

#include "xe_attn_jm.inc"   /* the matrix-unit form for a single-sequence prefill (RADIANCE_XE_ATTN_JM=1, opt-in) */

extern "C" int64_t xe_rad_attn_paged_scratch(const RadArgs* a) {
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 0);
    if (hd <= 0 || gqa <= 0) return R4D_NO(RAD_E_INVAL);
    return part_bytes(hd, gqa);
}

/* attn_paged: q, kv_cache, block_table, seqused, k_descale?, v_descale?, cu_seqlens?, out */
extern "C" int xe_rad_attn_paged(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 8) return R4D_NO(RAD_E_INVAL);
    const RadTensor* q  = r4d_opt(a, AP_Q);
    const RadTensor* kv = r4d_opt(a, AP_KV);
    const RadTensor* bt = r4d_opt(a, AP_BT);
    const RadTensor* su = r4d_opt(a, AP_SEQ);
    const RadTensor* kd = r4d_opt(a, AP_KDS);
    const RadTensor* vd = r4d_opt(a, AP_VDS);
    const RadTensor* cu = r4d_opt(a, AP_CU);
    const RadTensor* o  = r4d_opt(a, AP_OUT);
    if (!q || !kv || !bt || !su || !o || !q->data || !kv->data || !bt->data || !su->data || !o->data)
        return R4D_NO(RAD_E_INVAL);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if ((hd != 128 && hd != 256) || gqa < 1 || gqa > 32) return R4D_NO(RAD_E_SHAPE);
    if (kv->rank != 4 || bt->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const bool kv8 = kv->dtype == RAD_F8E4M3;
    if ((!kv8 && kv->dtype != RAD_BF16) || q->dtype != RAD_BF16 || (o->dtype != RAD_BF16 && o->dtype != RAD_F32) ||
        bt->dtype != RAD_I32 || su->dtype != RAD_I32 || (cu && cu->data && cu->dtype != RAD_I32) ||
        (kd && kd->data && kd->dtype != RAD_F32) || (vd && vd->data && vd->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const long long kvh = kv->shape[1], bs = kv->shape[2], qh = gqa * kvh;
    if (kv->shape[3] < 2 * hd || kv->stride[3] != 1) return R4D_NO(RAD_E_SHAPE);
    long long total_q = 0, orows = 0;
    const long long q_rs = r4d_rows_pitch(q, qh * hd, &total_q);
    if (q_rs < 0 || r4d_rows_pitch(o, qh * hd, &orows) != qh * hd) return R4D_NO(RAD_E_STRIDE);
    if (orows != total_q || total_q <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long n_seq = r4d_numel(su);
    const bool ragged = cu && cu->data;
    if (n_seq <= 0 || bt->shape[0] != n_seq || !r4d_contig(su) || bt->stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (ragged ? (r4d_numel(cu) != n_seq + 1 || !r4d_contig(cu)) : (total_q % n_seq != 0)) return R4D_NO(RAD_E_SHAPE);
    if (total_q > 65535) return R4D_NO(RAD_E_SHAPE);

    AttnP p{};
    p.q = (const uint16_t*)q->data; p.q_rs = (long)q_rs;
    p.kv = (const uint8_t*)kv->data; p.kv_bs = (long)kv->stride[0]; p.kv_hs = (long)kv->stride[1];
    p.kv_row = (long)kv->stride[2];
    p.bt = (const int*)bt->data; p.bt_rs = (long)bt->stride[0]; p.max_blocks = (int)bt->shape[1];
    p.seqused = (const int*)su->data; p.cu = ragged ? (const int*)cu->data : nullptr;
    p.n_seq = (int)n_seq; p.q_len = ragged ? 0 : (int)(total_q / n_seq);
    p.kds = kd && kd->data ? (const float*)kd->data : nullptr;
    p.vds = vd && vd->data ? (const float*)vd->data : nullptr;
    p.out = o->data;
    p.q_heads = (int)qh; p.kv_heads = (int)kvh; p.gqa = (int)gqa; p.block_size = (int)bs;
    p.causal = (int)r4d_geti_or(a, "causal", 1); p.window = (int)r4d_geti_or(a, "window", 0);
    p.scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    /* run 109: one sequence of 64 rows or more takes the matrix-unit form under RADIANCE_XE_ATTN_JM=1 */
    const bool jm = n_seq == 1 && total_q >= attn_jm::BQ && attn_jm::on((sycl::queue*)s);
    p.splits = jm ? 1 : splits_for(total_q, kvh);
    if (p.splits > 1) {
        const long long need = total_q * qh * p.splits * (hd + 2) * (long long)sizeof(float);
        if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
        p.part = (float*)a->scratch;
    }
    try {
        sycl::queue* qq = (sycl::queue*)s;
        const bool of = o->dtype == RAD_F32;
        const int tq = (int)total_q;
        /* run 109: the per-row matrix form (RADIANCE_XE_ATTN_MX=1) for calls of 64 rows or more -- a prefill's;
         * decode and verify keep the split kernel (on few rows the form's stages cost more, and it moved M97''s
         * text at a near-tie, M155) */
        if (!jm && total_q >= 64 && gqa <= 8 * attn_mx::kMaxRT && attn_mx::on(qq)) {
            if (hd == 256) {
                if (kv8) { if (of) attn_mx::launch<8, true, true>(qq, p, tq); else attn_mx::launch<8, true, false>(qq, p, tq); }
                else     { if (of) attn_mx::launch<8, false, true>(qq, p, tq); else attn_mx::launch<8, false, false>(qq, p, tq); }
            } else {
                if (kv8) { if (of) attn_mx::launch<4, true, true>(qq, p, tq); else attn_mx::launch<4, true, false>(qq, p, tq); }
                else     { if (of) attn_mx::launch<4, false, true>(qq, p, tq); else attn_mx::launch<4, false, false>(qq, p, tq); }
            }
        } else if (jm) {
            if (hd == 256) {
                if (kv8) { if (of) attn_jm::launch<8, true, true>(qq, p, tq); else attn_jm::launch<8, true, false>(qq, p, tq); }
                else     { if (of) attn_jm::launch<8, false, true>(qq, p, tq); else attn_jm::launch<8, false, false>(qq, p, tq); }
            } else {
                if (kv8) { if (of) attn_jm::launch<4, true, true>(qq, p, tq); else attn_jm::launch<4, true, false>(qq, p, tq); }
                else     { if (of) attn_jm::launch<4, false, true>(qq, p, tq); else attn_jm::launch<4, false, false>(qq, p, tq); }
            }
        } else if (hd == 256) {
            if (kv8) { if (of) launch<8, true, true>(qq, p, tq); else launch<8, true, false>(qq, p, tq); }
            else     { if (of) launch<8, false, true>(qq, p, tq); else launch<8, false, false>(qq, p, tq); }
        } else {
            if (kv8) { if (of) launch<4, true, true>(qq, p, tq); else launch<4, true, false>(qq, p, tq); }
            else     { if (of) launch<4, false, true>(qq, p, tq); else launch<4, false, false>(qq, p, tq); }
        }
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
