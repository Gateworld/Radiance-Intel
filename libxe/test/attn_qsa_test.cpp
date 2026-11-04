/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 169, owner note 173 item 1).
 *
 * attn_qsa_test.cpp -- libxe's attn_qsa_fmha (xe_attn_qsa.cpp) on the CPU SYCL device with the operands in the forms
 * the engine builds for a sparse prefill chunk (arch/common/rad_block_attn_gated_fp8.h's issue, core/runtime/
 * issue.cpp's tensors): q / out = brows(w.q, T) -> [T][n_head * 256] bf16; kv_cache -> [blocks][kv_heads][block][512]
 * E4M3; sel = brow_slice(qsa_sel, 0, T, topk + 1) -> [T][topk + 1] i32; seqused -> [T] i32; the block table =
 * praw2(table, I32, 1, pitch) -> [1][pitch] i32; ctx = praw(used, I32, 1) -> [1] i32. Under
 * RADIANCE_XE_ATTN_QSA_PREP_ONLY=1 the op runs its checks and its prep kernels (the context word, the inverted table,
 * the bitmap, the dense K / V) and skips the flash-attention kernel (card-only), so every check's verdict and the
 * prep kernels' output are seen here, against a host re-computation. The scratch comes from the op's own hook.
 *
 *   attn_qsa_test LIBXE.so [T=4096] [ctx=T]      RI_DEVICE=cpu ONEAPI_DEVICE_SELECTOR=opencl:cpu
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);

constexpr int kHD = 256, kHeads = 12, kGqa = 12, kBs = 4, kMaxCtx = 65536, kQLen = 4096, kTopk = 512;
constexpr int kNbTotal = 16365, kPitch = 16384;   /* the pool's blocks (65,460 tokens / 4) and the table's width */

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
float e4m3(uint8_t v) {
    const int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
    const float x = ex == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, ex - 7);
    return s ? -x : x;
}
uint16_t bf16_trunc(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }   /* exact for E4M3 */

RadTensor tn(void* p, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int r = t.rank - 1; r >= 0; --r) { t.stride[r] = st; st *= t.shape[r]; }
    return t;
}
RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }

/* the op's scratch layout (xe_attn_qsa.cpp plan_of), re-done here to find the regions */
struct Plan { long long kvh, ctx_max, nblk_max, nb_max, nw_max, T_pad, off_inv, off_bits, off_kd, off_vd, bytes; };
long long align_up(long long x) { return (x + 255) / 256 * 256; }
Plan plan() {
    Plan p{};
    p.kvh = kHeads / kGqa; p.ctx_max = (kMaxCtx + kBs - 1) / kBs * kBs; p.nblk_max = p.ctx_max / kBs;
    p.nb_max = 2 * p.nblk_max; p.nw_max = (p.nblk_max + 31) / 32; p.T_pad = (kQLen + 255) / 256 * 256;
    const long long kv_bytes = p.kvh * p.ctx_max * kHD * 2;
    p.off_inv = 256; p.off_bits = align_up(p.off_inv + p.nb_max * 4);
    p.off_kd = align_up(p.off_bits + p.T_pad * p.nw_max * 4); p.off_vd = align_up(p.off_kd + kv_bytes);
    p.bytes = align_up(p.off_vd + kv_bytes);
    return p;
}

bool run_case(sycl::queue& q, LaunchFn fn, ScratchFn sfn, int T, int ctx_given) {
    const Plan P = plan();
    const int ctx = std::max(ctx_given, T) > kMaxCtx ? kMaxCtx : std::max(ctx_given, T);   /* the clamp the op applies */
    const int nblk = (ctx + kBs - 1) / kBs;
    std::printf("== case T %d, ctx operand %d (expected after the clamp: %d, %d blocks)\n", T, ctx_given, ctx, nblk);

    /* the sequence's block table: a permutation of the pool's ids, -1 past the sequence */
    std::vector<int32_t> bt((size_t)kPitch, -1);
    for (int b = 0; b < nblk && b < kPitch; ++b) bt[(size_t)b] = (int32_t)(((long long)b * 7919 + 13) % kNbTotal);
    /* the cache: E4M3 codes, NaN codes avoided */
    const size_t kv_n = (size_t)kNbTotal * P.kvh * kBs * 2 * kHD;
    std::vector<uint8_t> kv(kv_n);
    for (size_t i = 0; i < kv_n; ++i) { uint8_t c = (uint8_t)h32(i, 7); if ((c & 0x7f) == 0x7f) c &= 0x7e; kv[i] = c; }
    /* each row's list: the last min(topk, complete blocks) complete blocks' page ids in radix order, then the partial
     * page, then -1; seqused by ref_qsa_select's rule */
    std::vector<int32_t> sel((size_t)T * (kTopk + 1), -1), su((size_t)T, 0);
    std::vector<std::vector<int32_t>> want_blocks((size_t)T);
    for (int i = 0; i < T; ++i) {
        const int p = ctx - T + i;
        const int nc = (p + 1) / kBs;
        const int k = std::min(nc, kTopk);
        std::vector<int32_t> ids;
        for (int b = nc - k; b < nc; ++b) { ids.push_back(bt[(size_t)b]); want_blocks[(size_t)i].push_back(b); }
        std::sort(ids.begin(), ids.end());
        int n = 0;
        for (int32_t id : ids) sel[(size_t)i * (kTopk + 1) + (size_t)n++] = id;
        if ((p + 1) % kBs != 0 && nc < kPitch) { sel[(size_t)i * (kTopk + 1) + (size_t)n++] = bt[(size_t)nc]; want_blocks[(size_t)i].push_back(nc); }
        su[(size_t)i] = n > 0 ? (int32_t)((n - 1) * kBs + p % kBs + 1) : 0;
    }
    const int32_t cx_host = ctx_given;

    /* device copies in the engine's forms */
    auto dev = [&](size_t bytes) { void* p = sycl::malloc_device(bytes, q); if (!p) { std::printf("no device memory\n"); std::exit(2); } return p; };
    uint16_t* d_q   = (uint16_t*)dev((size_t)T * kHeads * kHD * 2);
    uint16_t* d_out = (uint16_t*)dev((size_t)T * kHeads * kHD * 2);
    uint8_t*  d_kv  = (uint8_t*)dev(kv_n);
    int32_t*  d_sel = (int32_t*)dev(sel.size() * 4);
    int32_t*  d_su  = (int32_t*)dev(su.size() * 4);
    int32_t*  d_bt  = (int32_t*)dev(bt.size() * 4);
    int32_t*  d_cx  = (int32_t*)dev(64);
    char*     d_scr = (char*)dev((size_t)P.bytes);
    q.memset(d_q, 0, (size_t)T * kHeads * kHD * 2);
    q.memcpy(d_kv, kv.data(), kv_n);
    q.memcpy(d_sel, sel.data(), sel.size() * 4);
    q.memcpy(d_su, su.data(), su.size() * 4);
    q.memcpy(d_bt, bt.data(), bt.size() * 4);
    q.memcpy(d_cx, &cx_host, 4);
    q.memset(d_scr, 0x5a, (size_t)P.bytes);   /* stale bytes everywhere: the op must overwrite what it reads */
    q.wait();

    RadTensor t[7];
    t[0] = tn(d_q, RAD_BF16, {T, (int64_t)kHeads * kHD});
    t[1] = tn(d_kv, RAD_F8E4M3, {kNbTotal, P.kvh, kBs, 2 * kHD});
    t[2] = tn(d_sel, RAD_I32, {T, kTopk + 1});
    t[3] = tn(d_su, RAD_I32, {T});
    t[4] = tn(d_bt, RAD_I32, {1, kPitch});
    t[5] = tn(d_cx, RAD_I32, {1});
    t[6] = tn(d_out, RAD_BF16, {T, (int64_t)kHeads * kHD});
    RadParam prm[] = { pi("q_len", kQLen), pi("head_dim", kHD), pi("gqa", kGqa), pi("causal", 1), pi("window", 0),
                       pi("n_head", kHeads), pi("block_size", kBs), pi("max_seqs", 1), pi("max_ctx", kMaxCtx) };
    RadArgs a{};
    a.t = t; a.n_t = 7; a.p = prm; a.n_p = (int)(sizeof prm / sizeof prm[0]);
    a.rank = 0; a.world_size = 1;
    const int64_t need = sfn(&a);
    std::printf("   scratch hook: %lld bytes (%.1f MiB); the test's plan %lld\n", (long long)need, need / 1048576.0, (long long)P.bytes);
    bool ok = need == P.bytes;
    a.scratch = d_scr; a.scratch_bytes = need > 0 ? need : 0;
    const int rc = fn(&a, (RadStream)&q);
    q.wait();
    std::printf("   the op: rc %d (%s)\n", rc, rc == RAD_OK ? "ok" : "REFUSED");
    ok = ok && rc == RAD_OK;
    if (rc != RAD_OK) return false;

    /* the prep kernels' output against the host */
    std::vector<int32_t> words(64), inv((size_t)P.nb_max);
    std::vector<uint32_t> bits((size_t)T * P.nw_max);
    std::vector<uint16_t> kd((size_t)ctx * kHD), vd((size_t)ctx * kHD);
    q.memcpy(words.data(), d_scr, 64 * 4);
    q.memcpy(inv.data(), d_scr + P.off_inv, inv.size() * 4);
    q.memcpy(bits.data(), d_scr + P.off_bits, bits.size() * 4);
    q.memcpy(kd.data(), d_scr + P.off_kd, kd.size() * 2);   /* kv head 0 */
    q.memcpy(vd.data(), d_scr + P.off_vd, vd.size() * 2);
    q.wait();
    std::printf("   words: ctx %d (want %d), blocks %d (want %d)\n", words[0], ctx, words[1], nblk);
    ok = ok && words[0] == ctx && words[1] == nblk;
    long long bad = 0;
    for (int id = 0; id < kNbTotal; ++id) {
        int32_t want = -1;
        for (int b = 0; b < nblk; ++b) if (bt[(size_t)b] == id) { want = b; break; }
        if (inv[(size_t)id] != want) ++bad;
    }
    std::printf("   inverted table: %lld of %d ids wrong\n", bad, kNbTotal);
    ok = ok && bad == 0;
    bad = 0;
    for (int i = 0; i < T; ++i) {
        std::vector<uint32_t> want((size_t)P.nw_max, 0u);
        for (int32_t b : want_blocks[(size_t)i]) want[(size_t)b >> 5] |= 1u << (b & 31);
        for (long long w = 0; w < P.nw_max; ++w) if (bits[(size_t)i * P.nw_max + (size_t)w] != want[(size_t)w]) { ++bad; break; }
    }
    std::printf("   bitmap: %lld of %d rows wrong\n", bad, T);
    ok = ok && bad == 0;
    bad = 0;
    for (int j = 0; j < ctx; ++j) {
        const int32_t blk = bt[(size_t)(j / kBs)];
        for (int c = 0; c < kHD; ++c) {
            const size_t base = ((size_t)blk * P.kvh * kBs + (size_t)(j % kBs)) * 2 * kHD;
            const uint16_t wk = bf16_trunc(e4m3(kv[base + (size_t)c])), wv = bf16_trunc(e4m3(kv[base + kHD + (size_t)c]));
            if (kd[(size_t)j * kHD + (size_t)c] != wk || vd[(size_t)j * kHD + (size_t)c] != wv) { ++bad; break; }
        }
    }
    std::printf("   dense K / V: %lld of %d rows wrong\n", bad, ctx);
    ok = ok && bad == 0;
    std::printf("   %s\n", ok ? "PASS" : "FAIL");
    for (void* p : {(void*)d_q, (void*)d_out, (void*)d_kv, (void*)d_sel, (void*)d_su, (void*)d_bt, (void*)d_cx, (void*)d_scr}) sycl::free(p, q);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: attn_qsa_test LIBXE.so [T] [ctx]\n"); return 2; }
    setenv("RADIANCE_XE_ATTN_QSA_PREP_ONLY", "1", 1);
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    LaunchFn fn = (LaunchFn)dlsym(h, "xe_rad_attn_qsa_fmha");
    ScratchFn sfn = (ScratchFn)dlsym(h, "xe_rad_attn_qsa_fmha_scratch");
    if (!fn || !sfn) { std::fprintf(stderr, "no xe_rad_attn_qsa_fmha / _scratch in %s\n", argv[1]); return 2; }
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int T = argc > 2 ? std::atoi(argv[2]) : kQLen;
    const int ctx = argc > 3 ? std::atoi(argv[3]) : T;
    bool ok = run_case(q, fn, sfn, T, ctx);                 /* the 7K's first chunk: T 4096, ctx 4096 */
    if (argc <= 2) {
        ok = run_case(q, fn, sfn, 1508, 5604) && ok;       /* its second chunk */
        ok = run_case(q, fn, sfn, 300, 0) && ok;           /* a context word below T: clamped, never refused */
    }
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? 0 : 1;
}
