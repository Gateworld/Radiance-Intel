/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 206: the fusion map's pair 4, RADIANCE_XE_GATE_QUANT).
 *
 * gate_quant_test.cpp -- the gated attention's chain against its fused row, on the same deterministic inputs:
 *   RI_GQ=0  libxe's sigmoid row (the gate into a bf16 buffer), mul (x *= gate, in place), quant_act_i8g (int8
 *            codes and f32 scales a 128-column group) -- the three ops the architecture emits today;
 *   RI_GQ=1  libxe's gate_quant_fp8 row (listed under RADIANCE_XE_GATE_QUANT=1), int8 codes, x written in place.
 * Writes the codes, the scales and the bf16 product to OUT; ctest compares the two files byte for byte.
 *
 *   gate_quant_test LIBXE.so OUT.bin        RI_DEVICE=cpu for the CPU SYCL device
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
RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: gate_quant_test LIBXE.so OUT.bin\n"); return 2; }
    void* hx = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const char* mv = std::getenv("RI_GQ");
    const int mode = mv ? std::atoi(mv) : 0;
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    /* (token, head) rows of a head_dim of 256: two 128-column scale groups a row */
    const int64_t R = 12, n = 256, G = n / 128;
    std::vector<uint16_t> hg((size_t)(R * n)), hxv((size_t)(R * n));
    for (int64_t i = 0; i < R * n; ++i) {
        hg[(size_t)i] = bf16_of(6.0f * unit(41, (uint64_t)i));
        hxv[(size_t)i] = bf16_of(3.0f * unit(42, (uint64_t)i));
    }
    auto sh = [&](size_t bytes) { return sycl::malloc_shared<uint8_t>(bytes, q); };
    uint8_t* d_g = sh(hg.size() * 2); uint8_t* d_x = sh(hxv.size() * 2); uint8_t* d_t = sh(hg.size() * 2);
    uint8_t* d_q = sh((size_t)(R * n)); float* d_s = (float*)sh((size_t)(R * G * 4));
    std::memcpy(d_g, hg.data(), hg.size() * 2); std::memcpy(d_x, hxv.data(), hxv.size() * 2);
    std::memset(d_q, 0, (size_t)(R * n)); std::memset(d_s, 0, (size_t)(R * G * 4));
    int rc = RAD_OK;
    if (mode == 1) {
        const RadKernelInfo* row = row_of(hx, "gate_quant_fp8");
        if (!row) { std::printf("gate_quant_test: no gate_quant_fp8 row (RADIANCE_XE_GATE_QUANT=1?)\n"); return 2; }
        RadTensor t[5] = { tn(d_g, RAD_BF16, R, n), tn(d_x, RAD_BF16, R, n), tn(d_q, RAD_I8, R, n),
                           tn(d_s, RAD_F32, R, G), tn(d_x, RAD_BF16, R, n) };
        RadParam p[5] = { pint("M", R), pint("n", n), pint("group", 128), pstr("dtype", "bf16"), pstr("act", "sigmoid") };
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 5; a.world_size = 1;
        rc = row->launch(&a, (RadStream)&q);
        std::printf("gate_quant_test: %s rc %d\n", row->name, rc);
    } else {
        const RadKernelInfo* rs = row_of(hx, "sigmoid");
        const RadKernelInfo* rm = row_of(hx, "mul");
        const RadKernelInfo* rq = row_of(hx, "quant_act_i8g");
        if (!rs || !rm || !rq) { std::printf("gate_quant_test: a chain row is missing\n"); return 2; }
        RadParam p[4] = { pint("M", R), pint("n", n), pstr("dtype", "bf16"), pint("group", 128) };
        RadTensor ts[2] = { tn(d_g, RAD_BF16, R, n), tn(d_t, RAD_BF16, R, n) };
        RadArgs as{}; as.t = ts; as.n_t = 2; as.p = p; as.n_p = 3; as.world_size = 1;
        rc |= rs->launch(&as, (RadStream)&q);
        RadTensor tm[3] = { tn(d_x, RAD_BF16, R, n), tn(d_t, RAD_BF16, R, n), tn(d_x, RAD_BF16, R, n) };
        RadArgs am{}; am.t = tm; am.n_t = 3; am.p = p; am.n_p = 3; am.world_size = 1;
        rc |= rm->launch(&am, (RadStream)&q);
        RadTensor tq[3] = { tn(d_x, RAD_BF16, R, n), tn(d_q, RAD_I8, R, n), tn(d_s, RAD_F32, R, G) };
        RadParam pq[4] = { pint("M", R), pint("n", n), pint("group", 128), pstr("dtype", "bf16") };
        RadArgs aq{}; aq.t = tq; aq.n_t = 3; aq.p = pq; aq.n_p = 4; aq.world_size = 1;
        rc |= rq->launch(&aq, (RadStream)&q);
        std::printf("gate_quant_test: %s + %s + %s rc %d\n", rs->name, rm->name, rq->name, rc);
    }
    q.wait_and_throw();
    if (rc != RAD_OK) return 1;
    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) return 2;
    std::fwrite(d_q, 1, (size_t)(R * n), f);
    std::fwrite(d_s, 4, (size_t)(R * G), f);
    std::fwrite(d_x, 2, (size_t)(R * n), f);
    std::fclose(f);
    int nz = 0;
    for (int64_t i = 0; i < R * n; ++i) nz += d_q[i] != 0;
    std::printf("gate_quant_test: %d nonzero codes -> %s\n", nz, argv[2]);
    return nz ? 0 : 1;
}
