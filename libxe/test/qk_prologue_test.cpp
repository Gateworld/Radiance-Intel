/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 273: the decode step's pair 3, RADIANCE_XE_QK_PROLOGUE).
 *
 * qk_prologue_test.cpp -- the gated attention's unfused prologue against libxe's qk_norm_rope_gate row, on the same
 * deterministic inputs, in one process (the row is listed only under RADIANCE_XE_QK_PROLOGUE=1; the chain's rows
 * always are):
 *   the chain   rmsnorm over the q half of the [T][n_head][2 head_dim] projection (a column view, as the architecture
 *               issues it: rad_block_attn_gated_fp8.h's step), rmsnorm over k in place, rope over q, rope over k --
 *               the four launches a gated-attention pass issues at decode today;
 *   the row     rope_table (positions -> the cos/sin plane at each position's own row) then qk_norm_rope_gate: the
 *               same q and k written.
 * Compares q and k byte for byte over two shapes (T 7 and T 1 at the served geometry: head_dim 256, 8 q heads, 2 kv
 * heads, rotary 64, GemmaRMSNorm's `1 + w`), prints RESULT lines, and returns 0 only when every byte matches and the
 * outputs are non-trivial. Runs on the CPU SYCL device (RI_DEVICE=cpu) or the card.
 *
 *   qk_prologue_test [LIBXE.so]        RI_LIBXE names the library when no argument is given
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"

#include <dlfcn.h>

#include <cmath>
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
/* [T][H][W] with the row pitch `p0` and the head pitch `p1` (a column view when W < p1) */
RadTensor tn3(void* p, uint32_t dt, int64_t s0, int64_t s1, int64_t s2, int64_t p0, int64_t p1) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 3;
    t.shape[0] = s0; t.shape[1] = s1; t.shape[2] = s2;
    t.stride[0] = p0; t.stride[1] = p1; t.stride[2] = 1;
    return t;
}
RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadParam pf64(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

/* The bf16 distance of two words, as an integer step count on the same sign (for the report only). */
int ulp_of(uint16_t a, uint16_t b) {
    const int ia = (a & 0x8000) ? -(int)(a & 0x7fff) : (int)a;
    const int ib = (b & 0x8000) ? -(int)(b & 0x7fff) : (int)b;
    return std::abs(ia - ib);
}

struct Shape { int64_t T, n_head, n_kv, hd, rot; };

/* One shape: 0 on a byte-identical pair, 1 on a difference, 2 on a refused launch or a missing row. */
int run_shape(void* hx, sycl::queue& q, const Shape& s, int which) {
    const int64_t T = s.T, NH = s.n_head, NKV = s.n_kv, HD = s.hd, ROT = s.rot;
    const int64_t q_dim = NH * HD, kv_dim = NKV * HD, cs_rows = 1100;
    const double eps = 1e-6, wadd = 1.0, theta = 10000.0, scale = 1.0;
    const RadKernelInfo* r_norm = row_of(hx, "rmsnorm");
    const RadKernelInfo* r_rope = row_of(hx, "rope");
    const RadKernelInfo* r_tab  = row_of(hx, "rope_table");
    const RadKernelInfo* r_qk   = row_of(hx, "qk_norm_rope_gate");
    if (!r_norm || !r_rope || !r_tab) { std::printf("qk_prologue_test: a chain row is missing\n"); return 2; }
    if (!r_qk) { std::printf("qk_prologue_test: no qk_norm_rope_gate row (RADIANCE_XE_QK_PROLOGUE=1?)\n"); return 2; }

    auto sh = [&](size_t bytes) { void* p = sycl::malloc_shared<uint8_t>(bytes, q); std::memset(p, 0, bytes); return p; };
    uint16_t* qg = (uint16_t*)sh((size_t)(T * NH * 2 * HD) * 2);
    uint16_t* k1 = (uint16_t*)sh((size_t)(T * kv_dim) * 2);
    uint16_t* k2 = (uint16_t*)sh((size_t)(T * kv_dim) * 2);
    uint16_t* q1 = (uint16_t*)sh((size_t)(T * q_dim) * 2);
    uint16_t* q2 = (uint16_t*)sh((size_t)(T * q_dim) * 2);
    float* wq = (float*)sh((size_t)HD * 4);
    float* wk = (float*)sh((size_t)HD * 4);
    int32_t* pos = (int32_t*)sh((size_t)T * 4);
    float* cs = (float*)sh((size_t)(cs_rows * ROT) * 4);
    for (int64_t i = 0; i < T * NH * 2 * HD; ++i) qg[i] = bf16_of(2.5f * unit(51 + which, (uint64_t)i));
    for (int64_t i = 0; i < T * kv_dim; ++i) k1[i] = k2[i] = bf16_of(3.0f * unit(52 + which, (uint64_t)i));
    for (int64_t i = 0; i < HD; ++i) { wq[i] = 0.5f * unit(53, (uint64_t)i); wk[i] = 0.5f * unit(54, (uint64_t)i); }
    const int32_t some[8] = { 0, 1, 5, 130, 1023, 77, 2, 640 };
    for (int64_t i = 0; i < T; ++i) pos[i] = some[(which * 3 + i) % 8];

    int rc = RAD_OK;
    /* ---- the chain: rmsnorm q (the q half of each head of [q|gate], a column view) -> q1 */
    {
        RadTensor t[3] = { tn3(qg, RAD_BF16, T, NH, HD, NH * 2 * HD, 2 * HD), tn1(wq, RAD_F32, HD), tn(q1, RAD_BF16, T, q_dim) };
        RadParam p[5] = { pint("M", T * NH), pint("n", HD), pf64("eps", eps), pf64("wadd", wadd), pstr("dtype", "bf16") };
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 5; a.world_size = 1;
        rc |= r_norm->launch(&a, (RadStream)&q);
    }
    /* rmsnorm k in place -> k1 */
    {
        RadTensor t[3] = { tn(k1, RAD_BF16, T, kv_dim), tn1(wk, RAD_F32, HD), tn(k1, RAD_BF16, T, kv_dim) };
        RadParam p[5] = { pint("M", T * NKV), pint("n", HD), pf64("eps", eps), pf64("wadd", wadd), pstr("dtype", "bf16") };
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 5; a.world_size = 1;
        rc |= r_norm->launch(&a, (RadStream)&q);
    }
    /* rope q, rope k (decl_rope's parameters: mode mrope with one-component positions reads as NeoX) */
    for (int which_rope = 0; which_rope < 2; ++which_rope) {
        uint16_t* x = which_rope == 0 ? q1 : k1;
        const int64_t heads = which_rope == 0 ? NH : NKV;
        RadTensor t[2] = { tn(x, RAD_BF16, T, heads * HD), tn1(pos, RAD_I32, T) };
        RadParam p[8] = { pint("M", T), pint("head_dim", HD), pint("n_head", heads), pint("n_head_kv", 0),
                          pf64("theta", theta), pf64("scale", scale), pstr("mode", "mrope"), pint("rotary_dim", ROT) };
        RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 8; a.world_size = 1;
        rc |= r_rope->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("qk_prologue_test: the chain refused (rc %d)\n", rc); return 2; }

    /* ---- the row: the table, then the prologue over qg and k2 -> q2, k2 */
    {
        RadTensor t[2] = { tn1(pos, RAD_I32, T), tn(cs, RAD_F32, cs_rows, ROT) };
        RadParam p[5] = { pint("M", T), pint("rot", ROT), pf64("theta", theta), pf64("scale", scale), pstr("dtype", "f32") };
        RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; a.world_size = 1;
        rc |= r_tab->launch(&a, (RadStream)&q);
    }
    {
        RadTensor t[9] = { tn3(qg, RAD_BF16, T, NH, 2 * HD, NH * 2 * HD, 2 * HD), tn(k2, RAD_BF16, T, kv_dim),
                           tn(cs, RAD_F32, cs_rows, ROT), tn1(pos, RAD_I32, T), tn1(wq, RAD_F32, HD), tn1(wk, RAD_F32, HD),
                           tn(q2, RAD_BF16, T, q_dim), tn(k2, RAD_BF16, T, kv_dim), RadTensor{} };
        RadParam p[9] = { pint("M", T), pint("head_dim", HD), pint("n_head", NH), pint("n_head_kv", NKV), pint("rot", ROT),
                          pstr("q_dtype", "bf16"), pf64("eps", eps), pf64("wadd", wadd), pint("cs_f32", 1) };
        RadArgs a{}; a.t = t; a.n_t = 9; a.p = p; a.n_p = 9; a.world_size = 1;
        rc |= r_qk->launch(&a, (RadStream)&q);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) { std::printf("qk_prologue_test: the row refused (rc %d)\n", rc); return 2; }

    int64_t dq = 0, dk = 0, nzq = 0, nzk = 0;
    int mq = 0, mk = 0;
    for (int64_t i = 0; i < T * q_dim; ++i) {
        nzq += q2[i] != 0;
        if (q1[i] != q2[i]) { ++dq; mq = std::max(mq, ulp_of(q1[i], q2[i])); }
    }
    for (int64_t i = 0; i < T * kv_dim; ++i) {
        nzk += k2[i] != 0;
        if (k1[i] != k2[i]) { ++dk; mk = std::max(mk, ulp_of(k1[i], k2[i])); }
    }
    std::printf("RESULT T %lld heads %lld/%lld head_dim %lld rot %lld: q %lld elements, %lld differ (max %d bf16 steps), "
                "%lld nonzero; k %lld elements, %lld differ (max %d), %lld nonzero\n",
                (long long)T, (long long)NH, (long long)NKV, (long long)HD, (long long)ROT, (long long)(T * q_dim),
                (long long)dq, mq, (long long)nzq, (long long)(T * kv_dim), (long long)dk, mk, (long long)nzk);
    sycl::free(qg, q); sycl::free(k1, q); sycl::free(k2, q); sycl::free(q1, q); sycl::free(q2, q);
    sycl::free(wq, q); sycl::free(wk, q); sycl::free(pos, q); sycl::free(cs, q);
    if (nzq == 0 || nzk == 0) { std::printf("qk_prologue_test: the row wrote nothing\n"); return 2; }
    return (dq || dk) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* lib = argc >= 2 ? argv[1] : std::getenv("RI_LIBXE");
    if (!lib) { std::fprintf(stderr, "usage: qk_prologue_test LIBXE.so   (or RI_LIBXE)\n"); return 2; }
    void* hx = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    std::printf("qk_prologue_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    /* the served geometry (Qwen3-Next: head_dim 256, 16 / 2 heads over two ranks = 8 / 1 a rank; rotary 64), at a
     * verify's rows (4, and 7) and a draft's one, the two-kv-head form, and the architecture's bound (64 rows: the
     * fused row is taken for any T <= 64; coordinator, run 273) */
    const Shape shapes[5] = { { 7, 8, 1, 256, 64 }, { 1, 8, 1, 256, 64 }, { 5, 8, 2, 256, 64 },
                              { 4, 8, 1, 256, 64 }, { 64, 8, 1, 256, 64 } };
    int worst = 0;
    for (int i = 0; i < 5; ++i) {
        const int r = run_shape(hx, q, shapes[i], i);
        if (r == 2) return 2;
        worst = std::max(worst, r);
    }
    std::printf(worst ? "FAIL qk_norm_rope_gate: differs from rmsnorm q + rmsnorm k + rope q + rope k\n"
                      : "PASS qk_norm_rope_gate: bit-identical to rmsnorm q + rmsnorm k + rope q + rope k\n");
    return worst;
}
