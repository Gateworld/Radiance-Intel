/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 273: the decode step's pair 2, RADIANCE_XE_ROUTER_QUANT_PAIR).
 *
 * router_quant_test.cpp -- the router's selection + sort and the block input's rotated int8 quantiser, as the two rows
 * libxe serves today (router_topk_scatter, had_quant_act_i8g) and as the paired row (router_topk_scatter_quant, listed
 * only under RADIANCE_XE_ROUTER_QUANT_PAIR=1), on the same deterministic inputs in one process. Compares expert ids,
 * weights, sorted slots, offsets, counts, codes and scales byte for byte at M = 1, 4, 8 and 12 rows (the router's 8-
 * and 16-wave forms) with 512 experts, a 256-expert slice, top_k 10 and n 2048 (the served geometry). RESULT lines;
 * rc 0 only when every byte matches and the outputs are non-trivial. Runs on the CPU SYCL device (RI_DEVICE=cpu) or
 * the card.
 *
 *   router_quant_test [LIBXE.so]        RI_LIBXE names the library when no argument is given
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef int (*CountFn)(void);
typedef const RadKernelInfo* (*AtFn)(int);

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

struct Outs {
    int32_t* ids; uint16_t* w; int32_t* sorted; int32_t* off; int32_t* cnt; uint8_t* q; float* s;
};

int64_t count_diff(const void* a, const void* b, size_t bytes) {
    const uint8_t* x = (const uint8_t*)a; const uint8_t* y = (const uint8_t*)b;
    int64_t d = 0;
    for (size_t i = 0; i < bytes; ++i) d += x[i] != y[i];
    return d;
}

/* One row count: 0 on a byte-identical pair, 1 on a difference, 2 on a refused launch or a missing row. */
int run_m(void* hx, sycl::queue& q, int64_t M, int which) {
    const int64_t NE = 512, NL = 256, K = 10, N = 2048, G = N / 128, T = M * K;
    const RadKernelInfo* r_rt = row_of(hx, "router_topk_scatter");
    const RadKernelInfo* r_hq = row_of(hx, "had_quant_act_i8g");
    const RadKernelInfo* r_fz = row_of(hx, "router_topk_scatter_quant");
    if (!r_rt || !r_hq) { std::printf("router_quant_test: a chain row is missing\n"); return 2; }
    if (!r_fz) { std::printf("router_quant_test: no router_topk_scatter_quant row (RADIANCE_XE_ROUTER_QUANT_PAIR=1?)\n"); return 2; }

    auto sh = [&](size_t bytes) { void* p = sycl::malloc_shared<uint8_t>(bytes, q); std::memset(p, 0, bytes); return p; };
    uint16_t* lg = (uint16_t*)sh((size_t)(M * NE) * 2);
    uint16_t* x  = (uint16_t*)sh((size_t)(M * N) * 2);
    for (int64_t i = 0; i < M * NE; ++i) lg[i] = bf16_of(4.0f * unit(61 + which, (uint64_t)i));
    for (int64_t i = 0; i < M * N; ++i) x[i] = bf16_of(3.0f * unit(62 + which, (uint64_t)i));
    Outs o[2];
    for (int f = 0; f < 2; ++f) {
        o[f].ids = (int32_t*)sh((size_t)T * 4); o[f].w = (uint16_t*)sh((size_t)T * 2);
        o[f].sorted = (int32_t*)sh((size_t)T * 4); o[f].off = (int32_t*)sh((size_t)(NL + 1) * 4);
        o[f].cnt = (int32_t*)sh((size_t)NL * 4); o[f].q = (uint8_t*)sh((size_t)(M * N));
        o[f].s = (float*)sh((size_t)(M * G) * 4);
    }
    int rc = RAD_OK;
    /* ---- the two rows -> o[0] */
    {
        RadTensor t[6] = { tn(lg, RAD_BF16, M, NE), tn(o[0].ids, RAD_I32, M, K), tn(o[0].w, RAD_BF16, M, K),
                           tn1(o[0].sorted, RAD_I32, T), tn1(o[0].off, RAD_I32, NL + 1), tn1(o[0].cnt, RAD_I32, NL) };
        RadParam p[7] = { pint("M", M), pint("n_expert", NE), pint("n_local", NL), pint("top_k", K), pint("norm", 1),
                          pstr("dtype", "bf16"), pint("expert_base", 0) };
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 7; a.world_size = 1;
        rc |= r_rt->launch(&a, (RadStream)&q);
    }
    {
        RadTensor t[3] = { tn(x, RAD_BF16, M, N), tn(o[0].q, RAD_I8, M, N), tn(o[0].s, RAD_F32, M, G) };
        RadParam p[4] = { pint("M", M), pint("n", N), pint("group", 128), pstr("dtype", "bf16") };
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 4; a.world_size = 1;
        rc |= r_hq->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_quant_test: the chain refused (rc %d)\n", rc); return 2; }
    /* ---- the paired row -> o[1] */
    {
        RadTensor t[9] = { tn(lg, RAD_BF16, M, NE), tn(o[1].ids, RAD_I32, M, K), tn(o[1].w, RAD_BF16, M, K),
                           tn1(o[1].sorted, RAD_I32, T), tn1(o[1].off, RAD_I32, NL + 1), tn1(o[1].cnt, RAD_I32, NL),
                           tn(x, RAD_BF16, M, N), tn(o[1].q, RAD_I8, M, N), tn(o[1].s, RAD_F32, M, G) };
        RadParam p[9] = { pint("M", M), pint("n_expert", NE), pint("n_local", NL), pint("top_k", K), pint("norm", 1),
                          pstr("dtype", "bf16"), pint("expert_base", 0), pint("n", N), pint("group", 128) };
        RadArgs a{}; a.t = t; a.n_t = 9; a.p = p; a.n_p = 9; a.world_size = 1;
        rc |= r_fz->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_quant_test: the paired row refused (rc %d)\n", rc); return 2; }

    const int64_t d_ids = count_diff(o[0].ids, o[1].ids, (size_t)T * 4), d_w = count_diff(o[0].w, o[1].w, (size_t)T * 2),
                  d_sorted = count_diff(o[0].sorted, o[1].sorted, (size_t)T * 4),
                  d_off = count_diff(o[0].off, o[1].off, (size_t)(NL + 1) * 4),
                  d_cnt = count_diff(o[0].cnt, o[1].cnt, (size_t)NL * 4),
                  d_q = count_diff(o[0].q, o[1].q, (size_t)(M * N)), d_s = count_diff(o[0].s, o[1].s, (size_t)(M * G) * 4);
    int64_t nzq = 0, nzc = 0, nzw = 0;
    for (int64_t i = 0; i < M * N; ++i) nzq += o[1].q[i] != 0;
    for (int64_t i = 0; i < NL; ++i) nzc += o[1].cnt[i];
    for (int64_t i = 0; i < T; ++i) nzw += o[1].w[i] != 0;
    std::printf("RESULT M %lld (top_k %lld, %lld experts / %lld local, n %lld): bytes differing -- ids %lld, weights %lld, "
                "sorted %lld, offsets %lld, counts %lld, codes %lld, scales %lld; %lld nonzero codes, %lld slots counted, "
                "%lld nonzero weights\n",
                (long long)M, (long long)K, (long long)NE, (long long)NL, (long long)N, (long long)d_ids, (long long)d_w,
                (long long)d_sorted, (long long)d_off, (long long)d_cnt, (long long)d_q, (long long)d_s, (long long)nzq,
                (long long)nzc, (long long)nzw);
    sycl::free(lg, q); sycl::free(x, q);
    for (int f = 0; f < 2; ++f) {
        sycl::free(o[f].ids, q); sycl::free(o[f].w, q); sycl::free(o[f].sorted, q); sycl::free(o[f].off, q);
        sycl::free(o[f].cnt, q); sycl::free(o[f].q, q); sycl::free(o[f].s, q);
    }
    if (nzq == 0 || nzc == 0 || nzw == 0) { std::printf("router_quant_test: the paired row wrote nothing\n"); return 2; }
    return (d_ids || d_w || d_sorted || d_off || d_cnt || d_q || d_s) ? 1 : 0;
}

/* The same behind the fused router GEMM op the decode step takes: router_gemm_topk_scatter (logits = a @ b^T, then the
 * selection + sort; no gate column here) + had_quant_act_i8g, against router_gemm_topk_scatter_quant, every output
 * byte for byte (the logits too). */
int run_gemm_m(void* hx, sycl::queue& q, int64_t M, int which) {
    const int64_t NE = 512, NL = 256, K = 10, N = 2048, G = N / 128, T = M * K;
    const RadKernelInfo* r_rg = row_of(hx, "router_gemm_topk_scatter");
    const RadKernelInfo* r_hq = row_of(hx, "had_quant_act_i8g");
    const RadKernelInfo* r_fz = row_of(hx, "router_gemm_topk_scatter_quant");
    if (!r_rg || !r_hq) { std::printf("router_quant_test: a fused-router chain row is missing\n"); return 2; }
    if (!r_fz) { std::printf("router_quant_test: no router_gemm_topk_scatter_quant row (RADIANCE_XE_ROUTER_QUANT_PAIR=1?)\n"); return 2; }
    auto sh = [&](size_t bytes) { void* p = sycl::malloc_shared<uint8_t>(bytes, q); std::memset(p, 0, bytes); return p; };
    uint16_t* x = (uint16_t*)sh((size_t)(M * N) * 2);
    uint16_t* wb = (uint16_t*)sh((size_t)(NE * N) * 2);
    for (int64_t i = 0; i < M * N; ++i) x[i] = bf16_of(1.5f * unit(71 + which, (uint64_t)i));
    for (int64_t i = 0; i < NE * N; ++i) wb[i] = bf16_of(0.05f * unit(72, (uint64_t)i));
    uint16_t* lg[2]; Outs o[2];
    for (int f = 0; f < 2; ++f) {
        lg[f] = (uint16_t*)sh((size_t)(M * NE) * 2);
        o[f].ids = (int32_t*)sh((size_t)T * 4); o[f].w = (uint16_t*)sh((size_t)T * 2);
        o[f].sorted = (int32_t*)sh((size_t)T * 4); o[f].off = (int32_t*)sh((size_t)(NL + 1) * 4);
        o[f].cnt = (int32_t*)sh((size_t)NL * 4); o[f].q = (uint8_t*)sh((size_t)(M * N));
        o[f].s = (float*)sh((size_t)(M * G) * 4);
    }
    int rc = RAD_OK;
    {
        RadTensor t[10] = { tn(x, RAD_BF16, M, N), tn(wb, RAD_BF16, NE, N), RadTensor{}, tn(lg[0], RAD_BF16, M, NE),
                            tn(o[0].ids, RAD_I32, M, K), tn(o[0].w, RAD_BF16, M, K), tn1(o[0].sorted, RAD_I32, T),
                            tn1(o[0].off, RAD_I32, NL + 1), tn1(o[0].cnt, RAD_I32, NL), RadTensor{} };
        RadParam p[8] = { pint("M", M), pint("K", N), pint("n_expert", NE), pint("n_local", NL), pint("top_k", K),
                          pint("norm", 1), pstr("dtype", "bf16"), pint("expert_base", 0) };
        RadArgs a{}; a.t = t; a.n_t = 10; a.p = p; a.n_p = 8; a.world_size = 1;
        rc |= r_rg->launch(&a, (RadStream)&q);
        RadTensor tq[3] = { tn(x, RAD_BF16, M, N), tn(o[0].q, RAD_I8, M, N), tn(o[0].s, RAD_F32, M, G) };
        RadParam pq[4] = { pint("M", M), pint("n", N), pint("group", 128), pstr("dtype", "bf16") };
        RadArgs aq{}; aq.t = tq; aq.n_t = 3; aq.p = pq; aq.n_p = 4; aq.world_size = 1;
        rc |= r_hq->launch(&aq, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_quant_test: the fused-router chain refused (rc %d)\n", rc); return 2; }
    {
        RadTensor t[13] = { tn(x, RAD_BF16, M, N), tn(wb, RAD_BF16, NE, N), RadTensor{}, tn(lg[1], RAD_BF16, M, NE),
                            tn(o[1].ids, RAD_I32, M, K), tn(o[1].w, RAD_BF16, M, K), tn1(o[1].sorted, RAD_I32, T),
                            tn1(o[1].off, RAD_I32, NL + 1), tn1(o[1].cnt, RAD_I32, NL), RadTensor{},
                            tn(x, RAD_BF16, M, N), tn(o[1].q, RAD_I8, M, N), tn(o[1].s, RAD_F32, M, G) };
        RadParam p[10] = { pint("M", M), pint("K", N), pint("n_expert", NE), pint("n_local", NL), pint("top_k", K),
                           pint("norm", 1), pstr("dtype", "bf16"), pint("expert_base", 0), pint("n", N), pint("group", 128) };
        RadArgs a{}; a.t = t; a.n_t = 13; a.p = p; a.n_p = 10; a.world_size = 1;
        rc |= r_fz->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("router_quant_test: the fused-router paired row refused (rc %d)\n", rc); return 2; }
    const int64_t d_lg = count_diff(lg[0], lg[1], (size_t)(M * NE) * 2),
                  d_ids = count_diff(o[0].ids, o[1].ids, (size_t)T * 4), d_w = count_diff(o[0].w, o[1].w, (size_t)T * 2),
                  d_sorted = count_diff(o[0].sorted, o[1].sorted, (size_t)T * 4),
                  d_off = count_diff(o[0].off, o[1].off, (size_t)(NL + 1) * 4),
                  d_cnt = count_diff(o[0].cnt, o[1].cnt, (size_t)NL * 4),
                  d_q = count_diff(o[0].q, o[1].q, (size_t)(M * N)), d_s = count_diff(o[0].s, o[1].s, (size_t)(M * G) * 4);
    int64_t nzl = 0, nzq = 0, nzc = 0;
    for (int64_t i = 0; i < M * NE; ++i) nzl += lg[1][i] != 0;
    for (int64_t i = 0; i < M * N; ++i) nzq += o[1].q[i] != 0;
    for (int64_t i = 0; i < NL; ++i) nzc += o[1].cnt[i];
    std::printf("RESULT gemm M %lld: bytes differing -- logits %lld, ids %lld, weights %lld, sorted %lld, offsets %lld, "
                "counts %lld, codes %lld, scales %lld; %lld nonzero logits, %lld nonzero codes, %lld slots counted\n",
                (long long)M, (long long)d_lg, (long long)d_ids, (long long)d_w, (long long)d_sorted, (long long)d_off,
                (long long)d_cnt, (long long)d_q, (long long)d_s, (long long)nzl, (long long)nzq, (long long)nzc);
    sycl::free(x, q); sycl::free(wb, q);
    for (int f = 0; f < 2; ++f) {
        sycl::free(lg[f], q); sycl::free(o[f].ids, q); sycl::free(o[f].w, q); sycl::free(o[f].sorted, q);
        sycl::free(o[f].off, q); sycl::free(o[f].cnt, q); sycl::free(o[f].q, q); sycl::free(o[f].s, q);
    }
    if (nzl == 0 || nzq == 0 || nzc == 0) { std::printf("router_quant_test: the fused-router paired row wrote nothing\n"); return 2; }
    return (d_lg || d_ids || d_w || d_sorted || d_off || d_cnt || d_q || d_s) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc >= 2 ? argv[1] : std::getenv("RI_LIBXE");
    if (!lib) { std::fprintf(stderr, "usage: router_quant_test LIBXE.so   (or RI_LIBXE)\n"); return 2; }
    void* hx = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    std::printf("router_quant_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t rows[4] = { 1, 4, 8, 12 };   /* a draft pass, a verify, the 8-wave bound, the 16-wave form */
    int worst = 0;
    for (int i = 0; i < 4; ++i) {
        const int r = run_m(hx, q, rows[i], i);
        if (r == 2) return 2;
        worst = std::max(worst, r);
    }
    std::printf(worst ? "FAIL router_topk_scatter_quant: differs from router_topk_scatter + had_quant_act_i8g\n"
                      : "PASS router_topk_scatter_quant: bit-identical to router_topk_scatter + had_quant_act_i8g\n");
    int worst_g = 0;
    for (int i = 0; i < 4; ++i) {
        const int r = run_gemm_m(hx, q, rows[i], i);
        if (r == 2) return 2;
        worst_g = std::max(worst_g, r);
    }
    std::printf(worst_g ? "FAIL router_gemm_topk_scatter_quant: differs from router_gemm_topk_scatter + had_quant_act_i8g\n"
                        : "PASS router_gemm_topk_scatter_quant: bit-identical to router_gemm_topk_scatter + had_quant_act_i8g\n");
    return std::max(worst, worst_g);
}
