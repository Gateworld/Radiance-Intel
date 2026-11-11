/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 285: group A of owner note 232's audit, RADIANCE_XE_GEMV_MULTI2).
 *
 * router_sh_test.cpp -- the FFN block's router launch with the shared expert's gate_up folded in, against the rows the
 * decode step issues today, on the same deterministic inputs in one process: router_gemm_topk_scatter_quant (the router
 * GEMV, the shared gate column, the selection + sort and the block input's rotated int8 quantiser) and the standalone
 * int8 GEMV row (xe_gemm_i8a8_gemv, the shared expert's gate_up over the same input's codes) against
 * router_gemm_topk_scatter_quant_sh (listed only under RADIANCE_XE_GEMV_MULTI2=1; RADIANCE_XE_GEMV_NX=1 selects the
 * GEMV form the paired kernel copies -- without it the adapter runs the two launches, which proves nothing).
 * Compares the logits, the gate, the gate_up output, the expert ids, weights, sorted slots, offsets, counts, codes and
 * scales byte for byte at M = 1, 4, 8 and 12 rows with 512 experts, a 256-expert slice, top_k 10, K 2048 and a 640-wide
 * gate_up (the served geometry's shape at a smaller K). RESULT lines; rc 0 only when every byte matches and the outputs
 * are non-trivial. Runs on the CPU SYCL device (RI_DEVICE=cpu) or the card.
 *
 *   router_sh_test [LIBXE.so]        RI_LIBXE names the library when no argument is given
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

typedef int (*CountFn)(void);
typedef const RadKernelInfo* (*AtFn)(int);

/* the highest-priority device row of an op */
const RadKernelInfo* row_of(void* h, const char* op) {
    CountFn cnt = (CountFn)dlsym(h, "rad_kernel_count");
    AtFn at = (AtFn)dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    const RadKernelInfo* best = nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* r = at(i);
        if (r && r->op && !std::strcmp(r->op, op) && r->domain == RAD_DOMAIN_DEVICE && r->launch &&
            (!best || r->priority > best->priority))
            best = r;
    }
    return best;
}
/* the row by its entry-point name (several rows serve gemm_nt_q) */
const RadKernelInfo* row_named(void* h, const char* name) {
    CountFn cnt = (CountFn)dlsym(h, "rad_kernel_count");
    AtFn at = (AtFn)dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* r = at(i);
        if (r && r->name && !std::strcmp(r->name, name) && r->launch) return r;
    }
    return nullptr;
}

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}
uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float unit(uint64_t a, uint64_t b) { return (float)(h32(a, b) % 2000001u) / 1000000.0f - 1.0f; }
RadTensor tn(void* p, uint32_t dt, int64_t s0, int64_t s1) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 2;
    t.shape[0] = s0; t.shape[1] = s1; t.stride[0] = s1; t.stride[1] = 1;
    return t;
}
RadTensor tn1(void* p, uint32_t dt, int64_t s0) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 1;
    t.shape[0] = s0; t.stride[0] = 1;
    return t;
}
RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

int64_t count_diff(const void* a, const void* b, size_t bytes) {
    const uint8_t* x = (const uint8_t*)a; const uint8_t* y = (const uint8_t*)b;
    int64_t d = 0;
    for (size_t i = 0; i < bytes; ++i) d += x[i] != y[i];
    return d;
}

/* every output of the fused router op, plus the gate_up's */
struct Outs {
    uint16_t* lg; int32_t* ids; uint16_t* w; int32_t* sorted; int32_t* off; int32_t* cnt; uint16_t* gate;
    uint8_t* q; float* s; uint16_t* sgu;
};

/* One row count: 0 on a byte-identical pair, 1 on a difference, 2 on a refused launch or a missing row. */
int run_m(void* hx, sycl::queue& q, int64_t M, int which) {
    const int64_t NE = 512, NL = 256, TK = 10, K = 2048, G = K / 128, T = M * TK, NSH = 640;
    const RadKernelInfo* r_fz = row_of(hx, "router_gemm_topk_scatter_quant");
    const RadKernelInfo* r_i8 = row_named(hx, "xe_gemm_i8a8_gemv");
    const RadKernelInfo* r_sh = row_of(hx, "router_gemm_topk_scatter_quant_sh");
    if (!r_fz) { std::printf("router_sh_test: no router_gemm_topk_scatter_quant row (RADIANCE_XE_ROUTER_QUANT_PAIR=0?)\n"); return 2; }
    if (!r_i8) { std::printf("router_sh_test: the standalone int8 GEMV row is missing\n"); return 2; }
    if (!r_sh) { std::printf("router_sh_test: no router_gemm_topk_scatter_quant_sh row (RADIANCE_XE_GEMV_MULTI2=1?)\n"); return 2; }

    auto sh = [&](size_t bytes) { void* p = sycl::malloc_shared<uint8_t>(bytes, q); std::memset(p, 0, bytes); return p; };
    /* the block input: its bf16 plane (the router and the gate read it) and its int8 codes + f32 scales (the gate_up) */
    uint16_t* x = (uint16_t*)sh((size_t)(M * K) * 2);
    int8_t* aq = (int8_t*)sh((size_t)(M * K));
    float* as = (float*)sh((size_t)(M * G) * 4);
    for (int64_t i = 0; i < M * K; ++i) x[i] = bf16_of(1.5f * unit(71 + which, (uint64_t)i));
    for (int64_t i = 0; i < M * K; ++i) aq[i] = (int8_t)(int)(127.0f * unit(81 + which, (uint64_t)i));
    for (int64_t i = 0; i < M * G; ++i) as[i] = 0.01f + 0.04f * (0.5f + 0.5f * unit(82, (uint64_t)i));
    /* the weights: the router (bf16), the shared gate (bf16, one row), the gate_up (int8 codes, bf16 scales a 128) */
    uint16_t* wb = (uint16_t*)sh((size_t)(NE * K) * 2);
    uint16_t* wg = (uint16_t*)sh((size_t)K * 2);
    int8_t* wsh = (int8_t*)sh((size_t)(NSH * K));
    uint16_t* wshs = (uint16_t*)sh((size_t)(NSH * G) * 2);
    for (int64_t i = 0; i < NE * K; ++i) wb[i] = bf16_of(0.05f * unit(72, (uint64_t)i));
    for (int64_t i = 0; i < K; ++i) wg[i] = bf16_of(0.05f * unit(73, (uint64_t)i));
    for (int64_t i = 0; i < NSH * K; ++i) wsh[i] = (int8_t)(int)(127.0f * unit(90, (uint64_t)i));
    for (int64_t i = 0; i < NSH * G; ++i) wshs[i] = bf16_of(0.002f + 0.01f * (0.5f + 0.5f * unit(95, (uint64_t)i)));
    /* outputs: [0] today's two launches, [1] the one launch */
    Outs o[2];
    for (int f = 0; f < 2; ++f) {
        o[f].lg = (uint16_t*)sh((size_t)(M * NE) * 2);
        o[f].ids = (int32_t*)sh((size_t)T * 4); o[f].w = (uint16_t*)sh((size_t)T * 2);
        o[f].sorted = (int32_t*)sh((size_t)T * 4); o[f].off = (int32_t*)sh((size_t)(NL + 1) * 4);
        o[f].cnt = (int32_t*)sh((size_t)NL * 4); o[f].gate = (uint16_t*)sh((size_t)M * 2);
        o[f].q = (uint8_t*)sh((size_t)(M * K)); o[f].s = (float*)sh((size_t)(M * G) * 4);
        o[f].sgu = (uint16_t*)sh((size_t)(M * NSH) * 2);
    }
    int rc = RAD_OK;
    /* ---- today's two launches -> o[0]: the fused router with the gate, then the standalone int8 GEMV */
    {
        RadTensor t[13] = { tn(x, RAD_BF16, M, K), tn(wb, RAD_BF16, NE, K), tn(wg, RAD_BF16, 1, K),
                            tn(o[0].lg, RAD_BF16, M, NE), tn(o[0].ids, RAD_I32, M, TK), tn(o[0].w, RAD_BF16, M, TK),
                            tn1(o[0].sorted, RAD_I32, T), tn1(o[0].off, RAD_I32, NL + 1), tn1(o[0].cnt, RAD_I32, NL),
                            tn(o[0].gate, RAD_BF16, M, 1),
                            tn(x, RAD_BF16, M, K), tn(o[0].q, RAD_I8, M, K), tn(o[0].s, RAD_F32, M, G) };
        RadParam p[10] = { pint("M", M), pint("K", K), pint("n_expert", NE), pint("n_local", NL), pint("top_k", TK),
                           pint("norm", 1), pstr("dtype", "bf16"), pint("expert_base", 0), pint("n", K), pint("group", 128) };
        RadArgs a{}; a.t = t; a.n_t = 13; a.p = p; a.n_p = 10; a.world_size = 1;
        rc |= r_fz->launch(&a, (RadStream)&q);
        RadTensor ti[5] = { tn(aq, RAD_I8, M, K), tn(as, RAD_F32, M, G), tn(wsh, RAD_I8, NSH, K),
                            tn(wshs, RAD_BF16, NSH, G), tn(o[0].sgu, RAD_BF16, M, NSH) };
        RadParam pi[5] = { pint("M", M), pint("N", NSH), pint("K", K), pint("group", 128), pstr("dtype", "i8a8") };
        RadArgs ai{}; ai.t = ti; ai.n_t = 5; ai.p = pi; ai.n_p = 5; ai.world_size = 1;
        rc |= r_i8->launch(&ai, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_sh_test: today's launches refused (rc %d)\n", rc); return 2; }
    /* ---- the one launch -> o[1] */
    {
        RadTensor t[18] = { tn(x, RAD_BF16, M, K), tn(wb, RAD_BF16, NE, K), tn(wg, RAD_BF16, 1, K),
                            tn(o[1].lg, RAD_BF16, M, NE), tn(o[1].ids, RAD_I32, M, TK), tn(o[1].w, RAD_BF16, M, TK),
                            tn1(o[1].sorted, RAD_I32, T), tn1(o[1].off, RAD_I32, NL + 1), tn1(o[1].cnt, RAD_I32, NL),
                            tn(o[1].gate, RAD_BF16, M, 1),
                            tn(x, RAD_BF16, M, K), tn(o[1].q, RAD_I8, M, K), tn(o[1].s, RAD_F32, M, G),
                            tn(aq, RAD_I8, M, K), tn(as, RAD_F32, M, G), tn(wsh, RAD_I8, NSH, K),
                            tn(wshs, RAD_BF16, NSH, G), tn(o[1].sgu, RAD_BF16, M, NSH) };
        RadParam p[11] = { pint("M", M), pint("K", K), pint("n_expert", NE), pint("n_local", NL), pint("top_k", TK),
                           pint("norm", 1), pstr("dtype", "bf16"), pint("expert_base", 0), pint("n", K), pint("group", 128),
                           pint("N_sh", NSH) };
        RadArgs a{}; a.t = t; a.n_t = 18; a.p = p; a.n_p = 11; a.world_size = 1;
        rc |= r_sh->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_sh_test: the one launch refused (rc %d)\n", rc); return 2; }

    const int64_t d_lg = count_diff(o[0].lg, o[1].lg, (size_t)(M * NE) * 2),
                  d_gate = count_diff(o[0].gate, o[1].gate, (size_t)M * 2),
                  d_sgu = count_diff(o[0].sgu, o[1].sgu, (size_t)(M * NSH) * 2),
                  d_ids = count_diff(o[0].ids, o[1].ids, (size_t)T * 4), d_w = count_diff(o[0].w, o[1].w, (size_t)T * 2),
                  d_sorted = count_diff(o[0].sorted, o[1].sorted, (size_t)T * 4),
                  d_off = count_diff(o[0].off, o[1].off, (size_t)(NL + 1) * 4),
                  d_cnt = count_diff(o[0].cnt, o[1].cnt, (size_t)NL * 4),
                  d_q = count_diff(o[0].q, o[1].q, (size_t)(M * K)), d_s = count_diff(o[0].s, o[1].s, (size_t)(M * G) * 4);
    int64_t nzl = 0, nzg = 0, nzs = 0, nzq = 0, nzc = 0;
    for (int64_t i = 0; i < M * NE; ++i) nzl += o[1].lg[i] != 0;
    for (int64_t i = 0; i < M; ++i) nzg += o[1].gate[i] != 0;
    for (int64_t i = 0; i < M * NSH; ++i) nzs += o[1].sgu[i] != 0;
    for (int64_t i = 0; i < M * K; ++i) nzq += o[1].q[i] != 0;
    for (int64_t i = 0; i < NL; ++i) nzc += o[1].cnt[i];
    std::printf("RESULT M %lld (top_k %lld, %lld experts / %lld local, K %lld, gate_up %lld): bytes differing -- logits %lld, "
                "gate %lld, gate_up %lld, ids %lld, weights %lld, sorted %lld, offsets %lld, counts %lld, codes %lld, scales "
                "%lld; %lld nonzero logits, %lld nonzero gates, %lld nonzero gate_up outputs, %lld nonzero codes, %lld slots "
                "counted\n",
                (long long)M, (long long)TK, (long long)NE, (long long)NL, (long long)K, (long long)NSH, (long long)d_lg,
                (long long)d_gate, (long long)d_sgu, (long long)d_ids, (long long)d_w, (long long)d_sorted, (long long)d_off,
                (long long)d_cnt, (long long)d_q, (long long)d_s, (long long)nzl, (long long)nzg, (long long)nzs,
                (long long)nzq, (long long)nzc);
    sycl::free(x, q); sycl::free(aq, q); sycl::free(as, q); sycl::free(wb, q); sycl::free(wg, q); sycl::free(wsh, q);
    sycl::free(wshs, q);
    for (int f = 0; f < 2; ++f) {
        sycl::free(o[f].lg, q); sycl::free(o[f].ids, q); sycl::free(o[f].w, q); sycl::free(o[f].sorted, q);
        sycl::free(o[f].off, q); sycl::free(o[f].cnt, q); sycl::free(o[f].gate, q); sycl::free(o[f].q, q);
        sycl::free(o[f].s, q); sycl::free(o[f].sgu, q);
    }
    if (nzl == 0 || nzg == 0 || nzs == 0 || nzq == 0 || nzc == 0) {
        std::printf("router_sh_test: the one launch wrote nothing somewhere\n");
        return 2;
    }
    return (d_lg || d_gate || d_sgu || d_ids || d_w || d_sorted || d_off || d_cnt || d_q || d_s) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc >= 2 ? argv[1] : std::getenv("RI_LIBXE");
    if (!lib) { std::fprintf(stderr, "usage: router_sh_test LIBXE.so   (or RI_LIBXE)\n"); return 2; }
    const char* nx = std::getenv("RADIANCE_XE_GEMV_NX");
    if (!nx || nx[0] != '1') { std::printf("router_sh_test: RADIANCE_XE_GEMV_NX=1 is required (the form the paired kernel copies)\n"); return 2; }
    void* hx = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    std::printf("router_sh_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t rows[4] = { 1, 4, 8, 12 };   /* a draft pass, a verify, the 8-wave bound, the 16-wave form */
    int worst = 0;
    for (int i = 0; i < 4; ++i) {
        const int r = run_m(hx, q, rows[i], i);
        if (r == 2) return 2;
        worst = std::max(worst, r);
    }
    std::printf(worst ? "FAIL router_gemm_topk_scatter_quant_sh: differs from router_gemm_topk_scatter_quant + the int8 GEMV\n"
                      : "PASS router_gemm_topk_scatter_quant_sh: bit-identical to router_gemm_topk_scatter_quant + the int8 GEMV\n");
    return worst;
}
