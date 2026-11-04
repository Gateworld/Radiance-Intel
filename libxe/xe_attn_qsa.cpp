/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 167, owner note 168 item 1; run 169, owner note 173).
 *
 * xe_attn_qsa.cpp -- attn_qsa_fmha: a sparse (QSA) prefill pass's attention on the matrix units. This model's
 * prefill is per-query sparse from 2,051 tokens of context: every query row attends its own list of 4-token pages
 * (the indexer's top-512 plus its partial page, page ids in radix order), and attn_paged serves it one row at a
 * time, re-reading the cache per row (B90: vector-ALU-bound, the matrix pipe idle, ~100 ms a 4,096-row chunk a
 * layer). Here one launch takes a whole one-sequence chunk: (1) the sequence's block table is inverted once (page
 * id -> block index); (2) each row's list becomes a bitmap over block indices (the partial page included; the
 * count from `seqused` as attn_paged reads it); (3) the context's K and V are gathered from the cache into dense
 * bf16 [kv_heads][ctx][256] (E4M3 exact in bf16); (4) sycl-tla's flash-attention forward (fmha/xe_fmha256.cpp)
 * runs causal over the dense context with the bitmap as a mask -- the same key set as attn_paged's per-row walk
 * (every selected page lies before the row, the partial page is cut at the row's position), summed in position
 * order instead of table order, P in bf16: a tolerance form, judged by the KL replay (owner note 149's bar).
 * Rows' positions are ctx - T + i (a chunk is consecutive).
 *
 * Run 169 (after B97 / B98 / B100 faulted the card): nothing of the op touches the host between launches and the
 * op owns no memory. The context length (the `ctx` operand, a device word of the batch) is read ON THE DEVICE by
 * a one-item kernel into a scratch word, clamped to [T, max_ctx]; every prep kernel is launched at the geometry's
 * maximum and bounded by that word; the FMHA reads it too (the kernel header's copy under fmha/); all memory is
 * the op's declared scratch (the engine's arena, resident on the card); every launch -- the FMHA's included, with
 * the recorder hooks handed to its module -- goes through the engine's launch path, so it is ordered behind the
 * engine's native command lists like every other libxe kernel. GPU only: the mainloop's 2D block loads do not run
 * on the CPU device; RADIANCE_XE_ATTN_QSA_PREP_ONLY=1 runs the checks and the prep kernels alone (the CPU test). */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "fmha/xe_fmha256.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

enum { AQ_Q = 0, AQ_KV, AQ_SEL, AQ_SEQ, AQ_BT, AQ_CTX, AQ_OUT };
constexpr int kHD = 256;
constexpr long long kAlign = 256;

/* The FMHA kernel lives in xe/libxe_fmha.so under this library's directory (its own SYCL module, the B70 target
 * alone, libxe/CMakeLists.txt; a subdirectory because the engine's plugin scan takes every .so beside libxe.so
 * for a plugin); loaded once, at the first call. Null when the module is absent. */
xe_fmha256_run_fn fmha_fn() {
    static xe_fmha256_run_fn fn = [] () -> xe_fmha256_run_fn {
        Dl_info di;
        std::string path = "xe/libxe_fmha.so";
        if (dladdr((void*)&fmha_fn, &di) && di.dli_fname) {
            std::string self = di.dli_fname;
            const size_t cut = self.rfind('/');
            if (cut != std::string::npos) path = self.substr(0, cut + 1) + "xe/libxe_fmha.so";
        }
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "[xe] attn_qsa_fmha: no %s (%s)\n", path.c_str(), dlerror()); return nullptr; }
        void* s = dlsym(h, "xe_fmha256_run");
        if (!s) fprintf(stderr, "[xe] attn_qsa_fmha: %s has no xe_fmha256_run\n", path.c_str());
        return (xe_fmha256_run_fn)s;
    }();
    return fn;
}

/* the engine's recorder hooks as this library resolved them (weak: null outside the engine) */
const XeFmhaHooks g_hooks{rad_l0k_rec_device, rad_l0k_rec_launch};

bool prep_only() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_XE_ATTN_QSA_PREP_ONLY"); return e && *e == '1'; }();
    return v;
}

inline long long align_up(long long x) { return (x + kAlign - 1) / kAlign * kAlign; }

/* The scratch layout from the declared geometry -- the same at declare (the scratch hook) and at issue (the
 * params are the band's frozen geometry): [words][inv][bits][kd][vd]. */
struct Plan {
    long long kvh = 0, ctx_max = 0, nblk_max = 0, nb_max = 0, nw_max = 0, T_pad = 0, bs = 0;
    long long off_inv = 0, off_bits = 0, off_kd = 0, off_vd = 0, bytes = 0;
};
const char* plan_of(const RadArgs* a, Plan* p) {
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    const long long nh = r4d_geti_or(a, "n_head", 0), bs = r4d_geti_or(a, "block_size", 0);
    const long long max_ctx = r4d_geti_or(a, "max_ctx", 0), q_len = r4d_geti_or(a, "q_len", 0);
    if (hd != kHD || gqa < 1 || gqa > 32) return "head_dim / gqa";
    if (nh <= 0 || nh % gqa) return "n_head / gqa";
    if (bs <= 0 || bs > 64 || max_ctx <= 0 || q_len <= 0) return "block_size / max_ctx / q_len";
    p->kvh      = nh / gqa;
    p->bs       = bs;
    p->ctx_max  = (max_ctx + bs - 1) / bs * bs;
    p->nblk_max = p->ctx_max / bs;
    p->nb_max   = 2 * p->nblk_max;                 /* page ids are the pool's block ids: twice the sequence's */
    p->nw_max   = (p->nblk_max + 31) / 32;
    p->T_pad    = (q_len + 255) / 256 * 256;
    const long long kv_bytes = p->kvh * p->ctx_max * kHD * 2;
    p->off_inv  = kAlign;                          /* the words: [0] ctx, [1] nblk */
    p->off_bits = align_up(p->off_inv + p->nb_max * 4);
    p->off_kd   = align_up(p->off_bits + p->T_pad * p->nw_max * 4);
    p->off_vd   = align_up(p->off_kd + kv_bytes);
    p->bytes    = align_up(p->off_vd + kv_bytes);
    return nullptr;
}

/* the context length into the scratch words, clamped: [0] the length, [1] its blocks */
void ctx_kernel(const int32_t* cx, int32_t* w, int T, int ctx_max, int bs) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int v = cx[0];
    if (v < T) v = T;
    if (v > ctx_max) v = ctx_max;
    w[0] = v;
    w[1] = (v + bs - 1) / bs;
}

void fill_i32_kernel(int32_t* p, uint32_t n, int32_t v) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

void zero_u32_kernel(uint32_t* p, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0u;
}

/* page id -> block index, for the sequence's first nblk blocks (the device word) */
void inv_kernel(const int32_t* bt, int32_t* inv, const int32_t* w, uint32_t max_blocks, uint32_t nb_total) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks || b >= (uint32_t)w[1]) return;
    const int32_t id = bt[b];
    if (id >= 0 && (uint32_t)id < nb_total) inv[id] = (int32_t)b;
}

/* a row's list -> its bitmap over block indices; a work-group a row, its threads over the list */
void bits_kernel(const int32_t* sel, long sel_rs, const int32_t* seq, const int32_t* inv, uint32_t* bits,
                 uint32_t nw, uint32_t T, uint32_t entries, uint32_t bs, uint32_t nb_total) {
    const uint32_t m = blockIdx.x;
    if (m >= T) return;
    const int32_t su = seq[m];
    uint32_t n = su > 0 ? ((uint32_t)su + bs - 1u) / bs : 0u;
    if (n > entries) n = entries;
    for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) {
        const int32_t id = sel[(size_t)m * sel_rs + k];
        if (id < 0 || (uint32_t)id >= nb_total) continue;
        const int32_t b = inv[id];
        if (b < 0 || ((uint32_t)b >> 5) >= nw) continue;
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> wd(bits[(size_t)m * nw + ((uint32_t)b >> 5)]);
        wd.fetch_or(1u << ((uint32_t)b & 31u));
    }
}

/* the context's K and V rows from the cache (E4M3 bytes or bf16) into dense bf16 [kv_heads][ctx_max][256], the
 * first ctx (the device word) rows of each head */
template <bool KV8>
void gather_kernel(const uint8_t* kv, long kv_bs, long kv_hs, long kv_row, const int32_t* bt, const int32_t* w,
                   uint32_t ctx_max, uint32_t bs, uint32_t kvh, uint16_t* kd, uint16_t* vd) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t per_row = kHD / 8;
    const size_t total = (size_t)kvh * ctx_max * per_row;
    if (idx >= total) return;
    const uint32_t j = (uint32_t)((idx / per_row) % ctx_max);
    if (j >= (uint32_t)w[0]) return;
    const uint32_t c = (uint32_t)(idx % per_row) * 8u;
    const uint32_t kh = (uint32_t)(idx / per_row / ctx_max);
    const int32_t blk = bt[j / bs];
    uint16_t kx[8], vx[8];
    if (blk < 0) {
#pragma unroll
        for (int e = 0; e < 8; ++e) { kx[e] = 0; vx[e] = 0; }
    } else {
        const long base = (long)blk * kv_bs + (long)kh * kv_hs + (long)(j % bs) * kv_row;   /* element offset */
        if constexpr (KV8) {
            const uint8_t* kp = kv + base + c;
            const uint8_t* vp = kv + base + kHD + c;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                kx[e] = (uint16_t)(__float_as_uint(xe::e4m3_to_f32(kp[e])) >> 16);   /* exact */
                vx[e] = (uint16_t)(__float_as_uint(xe::e4m3_to_f32(vp[e])) >> 16);
            }
        } else {
            const uint16_t* kp = (const uint16_t*)kv + base + c;
            const uint16_t* vp = (const uint16_t*)kv + base + kHD + c;
#pragma unroll
            for (int e = 0; e < 8; ++e) { kx[e] = kp[e]; vx[e] = vp[e]; }
        }
    }
    uint4 kw, vw;
    __builtin_memcpy(&kw, kx, sizeof kw);
    __builtin_memcpy(&vw, vx, sizeof vw);
    const size_t o = ((size_t)kh * ctx_max + j) * kHD + c;
    *(uint4*)(kd + o) = kw;
    *(uint4*)(vd + o) = vw;
}

}  // namespace

/* a refusal names its check and the operands' shapes (run 168: B100 refused "shape" without a trace) */
static int qsa_refuse(const RadArgs* a, int code, const char* why) {
    fprintf(stderr, "[xe] attn_qsa_fmha refused: %s (code %d)\n", why, code);
    for (int i = 0; i < a->n_t && i < 7; ++i) {
        const RadTensor* t = &a->t[i];
        fprintf(stderr, "[xe]   t[%d] data %p dtype %u rank %u shape", i, t->data, (unsigned)t->dtype, (unsigned)t->rank);
        for (unsigned k = 0; k < t->rank; ++k) fprintf(stderr, " %lld", (long long)t->shape[k]);
        fprintf(stderr, " stride");
        for (unsigned k = 0; k < t->rank; ++k) fprintf(stderr, " %lld", (long long)t->stride[k]);
        fprintf(stderr, "\n");
    }
    return R4D_NO(code);
}
#define QSA_REFUSE(code, why) qsa_refuse(a, (code), (why))

/* The scratch hook (declare time, from the geometry alone): the words, the inverted table, the bitmap at the band's
 * rows, dense K and V at max_ctx. ~72 MiB at this model's prefill geometry (max_ctx 65536, one kv head a rank). */
extern "C" int64_t xe_rad_attn_qsa_fmha_scratch(const RadArgs* a) {
    Plan p;
    if (const char* why = plan_of(a, &p)) { (void)why; return R4D_NO(RAD_E_INVAL); }
    return p.bytes;
}

/* attn_qsa_fmha: q, kv_cache, sel, seqused, block_table (the sequence's), ctx, out */
extern "C" int xe_rad_attn_qsa_fmha(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 7) return R4D_NO(RAD_E_INVAL);
    const xe_fmha256_run_fn run = fmha_fn();
    if (!run && !prep_only()) return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor* q   = r4d_opt(a, AQ_Q);
    const RadTensor* kv  = r4d_opt(a, AQ_KV);
    const RadTensor* sel = r4d_opt(a, AQ_SEL);
    const RadTensor* su  = r4d_opt(a, AQ_SEQ);
    const RadTensor* bt  = r4d_opt(a, AQ_BT);
    const RadTensor* cx  = r4d_opt(a, AQ_CTX);
    const RadTensor* o   = r4d_opt(a, AQ_OUT);
    if (!q || !kv || !sel || !su || !bt || !cx || !o || !q->data || !kv->data || !sel->data || !su->data ||
        !bt->data || !cx->data || !o->data)
        return R4D_NO(RAD_E_INVAL);
    Plan P;
    if (const char* why = plan_of(a, &P)) return QSA_REFUSE(RAD_E_SHAPE, why);
    const long long hd = kHD, gqa = r4d_geti_or(a, "gqa", 1);
    if (r4d_geti_or(a, "causal", 1) != 1 || r4d_geti_or(a, "window", 0) != 0)
        return QSA_REFUSE(RAD_E_UNSUPPORTED, "causal / window");
    if (kv->rank != 4 || sel->rank != 2) return QSA_REFUSE(RAD_E_SHAPE, "kv rank 4, sel rank 2");
    const bool kv8 = kv->dtype == RAD_F8E4M3;
    if ((!kv8 && kv->dtype != RAD_BF16) || q->dtype != RAD_BF16 || o->dtype != RAD_BF16 || sel->dtype != RAD_I32 ||
        su->dtype != RAD_I32 || bt->dtype != RAD_I32 || cx->dtype != RAD_I32)
        return QSA_REFUSE(RAD_E_DTYPE, "dtypes");
    const long long kvh = kv->shape[1], bs = kv->shape[2], nb_total = kv->shape[0], qh = gqa * kvh;
    if (kv->shape[3] < 2 * hd || kv->stride[3] != 1 || bs <= 0 || nb_total <= 0)
        return QSA_REFUSE(RAD_E_SHAPE, "kv cache shape");
    if (kvh != P.kvh || bs != P.bs) return QSA_REFUSE(RAD_E_SHAPE, "kv heads / block size against the geometry");
    long long T = 0, orows = 0;
    const long long q_rs = r4d_rows_pitch(q, qh * hd, &T);
    const long long o_rs = r4d_rows_pitch(o, qh * hd, &orows);
    if (q_rs < 0 || o_rs < 0 || orows != T || T <= 0) return QSA_REFUSE(RAD_E_STRIDE, "q / out rows");
    if (T > P.T_pad) return QSA_REFUSE(RAD_E_SHAPE, "rows past the declared q_len");
    if (sel->shape[0] < T || sel->stride[1] != 1) return QSA_REFUSE(RAD_E_SHAPE, "sel rows / inner stride");
    if (r4d_numel(su) < T || !r4d_contig(su)) return QSA_REFUSE(RAD_E_SHAPE, "seqused rows / contiguity");
    const long long entries = sel->shape[1], sel_rs = sel->stride[0];
    const long long max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : r4d_numel(bt);
    if (entries <= 0 || max_blocks <= 0) return QSA_REFUSE(RAD_E_SHAPE, "sel entries / table width");
    if (nb_total > P.nb_max) return QSA_REFUSE(RAD_E_SCRATCH, "the cache's blocks past the inverted table's size");
    if (!a->scratch || a->scratch_bytes < P.bytes) return QSA_REFUSE(RAD_E_SCRATCH, "scratch");
    sycl::queue* qq = (sycl::queue*)s;
    const bool gpu = qq->get_device().is_gpu();
    if (!gpu && !prep_only()) return R4D_NO(RAD_E_DEVICE);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));

    char* base = (char*)a->scratch;
    int32_t*  words = (int32_t*)base;
    int32_t*  inv   = (int32_t*)(base + P.off_inv);
    uint32_t* bits  = (uint32_t*)(base + P.off_bits);
    uint16_t* kd    = (uint16_t*)(base + P.off_kd);
    uint16_t* vd    = (uint16_t*)(base + P.off_vd);
    const uint32_t nw = (uint32_t)P.nw_max;
    const uint32_t ctx_max = (uint32_t)P.ctx_max;
    try {
        XE_LAUNCH(qq, ctx_kernel, dim3(1), dim3(1), 0, (const int32_t*)cx->data, words, (int)T, (int)ctx_max, (int)bs);
        XE_LAUNCH(qq, fill_i32_kernel, dim3((unsigned)((nb_total + 255) / 256)), dim3(256), 0, inv, (uint32_t)nb_total, (int32_t)-1);
        XE_LAUNCH(qq, inv_kernel, dim3((unsigned)((max_blocks + 255) / 256)), dim3(256), 0,
                  (const int32_t*)bt->data, inv, (const int32_t*)words, (uint32_t)max_blocks, (uint32_t)nb_total);
        const uint32_t nbits = (uint32_t)(P.T_pad * nw);
        XE_LAUNCH(qq, zero_u32_kernel, dim3((nbits + 255u) / 256u), dim3(256), 0, bits, nbits);
        XE_LAUNCH(qq, bits_kernel, dim3((unsigned)T), dim3(128), 0,
                  (const int32_t*)sel->data, (long)sel_rs, (const int32_t*)su->data, (const int32_t*)inv, bits,
                  nw, (uint32_t)T, (uint32_t)entries, (uint32_t)bs, (uint32_t)nb_total);
        const size_t total = (size_t)kvh * ctx_max * (kHD / 8);
        const dim3 gg((unsigned)((total + 255) / 256)), gb(256);
        if (kv8)
            XE_LAUNCH(qq, gather_kernel<true>, gg, gb, 0, (const uint8_t*)kv->data, (long)kv->stride[0],
                      (long)kv->stride[1], (long)kv->stride[2], (const int32_t*)bt->data, (const int32_t*)words,
                      ctx_max, (uint32_t)bs, (uint32_t)kvh, kd, vd);
        else
            XE_LAUNCH(qq, gather_kernel<false>, gg, gb, 0, (const uint8_t*)kv->data, (long)kv->stride[0],
                      (long)kv->stride[1], (long)kv->stride[2], (const int32_t*)bt->data, (const int32_t*)words,
                      ctx_max, (uint32_t)bs, (uint32_t)kvh, kd, vd);
    } catch (const sycl::exception&) {
        return R4D_NO(RAD_E_DEVICE);
    }
    if (prep_only()) return RAD_OK;
    XeFmhaArgs fa{};
    fa.q = (const uint16_t*)q->data; fa.q_rs = (long)q_rs; fa.q_hs = kHD;
    fa.k = kd; fa.k_hs = (long)ctx_max * kHD;
    fa.v = vd; fa.v_hs = (long)ctx_max * kHD;
    fa.out = (uint16_t*)o->data; fa.o_rs = (long)o_rs; fa.o_hs = kHD;
    fa.sel_bits = bits; fa.sel_words = (int)nw;
    fa.T = (int)T; fa.ctx = (int)ctx_max; fa.heads = (int)qh; fa.kv_heads = (int)kvh;
    fa.scale = scale;
    fa.ctx_dev = words;
    const int rc = run(qq, &fa, &g_hooks);
    if (rc == -1) return QSA_REFUSE(RAD_E_INVAL, "the FMHA's arguments");
    if (rc == -2) return QSA_REFUSE(RAD_E_SHAPE, "the FMHA's can_implement");
    if (rc != 0) return QSA_REFUSE(RAD_E_DEVICE, "the FMHA's launch threw");
    return RAD_OK;
}
