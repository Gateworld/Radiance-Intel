/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, owner note 186 item 3b: the megakernel's stage 1a).
 *
 * hc_persist_test.cpp -- libxe's hc_read at a verify's rows (2, 3 and 4), on deterministic inputs with valid E4M3
 * rows (r4d_hc_fp8.h: K codes, K / G f32 scales, padded to 16 bytes), its outputs written to a file: `x`, the inject
 * gains and the down product in the scratch (the persistent form's dbuf). ctest runs it twice -- with
 * RADIANCE_XE_HC_PERSIST unset (three launches) and =1 (one persistent launch), both at RADIANCE_XE_HC_V2=8 -- and
 * compares the two files byte for byte: the persistent form must be bit-identical.
 *
 *   hc_persist_test LIBXE.so OUT.bin        RI_DEVICE=cpu for the CPU SYCL device
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

uint16_t bf16_of(float f) {   /* round to nearest even */
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* a value in [-1, 1) from a hash */
float unit(uint64_t a, uint64_t b) { return (float)(h32(a, b) % 2000001u) / 1000000.0f - 1.0f; }

RadTensor tn(void* p, uint32_t dt, int rank, int64_t s0, int64_t s1 = 1) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = rank;
    t.shape[0] = s0; t.shape[1] = s1;
    if (rank == 1) t.stride[0] = 1;
    if (rank == 2) { t.stride[0] = s1; t.stride[1] = 1; }
    return t;
}

/* E4M3 stored rows of K codes, group G: codes from the hash with the two NaN patterns (0x7f, 0xff) avoided, the scales
 * small and positive */
std::vector<uint8_t> e4m3_rows(int64_t nrows, int64_t K, int64_t G, uint64_t salt) {
    const int64_t rb = ((K + 4 * (K / G)) + 15) & ~(int64_t)15;
    std::vector<uint8_t> v((size_t)(nrows * rb), 0);
    for (int64_t r = 0; r < nrows; ++r) {
        uint8_t* row = v.data() + r * rb;
        for (int64_t k = 0; k < K; ++k) {
            uint8_t c = (uint8_t)(h32(salt, (uint64_t)(r * K + k)) & 0xffu);
            if ((c & 0x7fu) == 0x7fu) c ^= 0x01u;
            row[k] = c;
        }
        for (int64_t g = 0; g < K / G; ++g) {
            const float s = 0.002f + 0.001f * (float)(h32(salt + 1, (uint64_t)(r * 64 + g)) % 1000u) / 1000.0f;
            std::memcpy(row + K + 4 * g, &s, 4);
        }
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: hc_persist_test LIBXE.so OUT.bin\n"); return 2; }
    void* hx = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!hx) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const RadKernelInfo* row = row_of(hx, "hc_read");
    if (!row) { std::printf("libxe: no device row for hc_read\n"); return 2; }
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    const char* pers = std::getenv("RADIANCE_XE_HC_PERSIST");
    std::printf("hc_persist_test: %s on %s, RADIANCE_XE_HC_PERSIST=%s\n", row->name,
                q.get_device().get_info<sycl::info::device::name>().c_str(), pers ? pers : "(unset)");

    /* hc 4 streams of n 512 (HN 2048: a whole number of the S8 form's 2048-element steps), lowrank 128 */
    const int64_t hc = 4, n = 512, HN = hc * n, lr = 128, M = 4;
    auto sh = [&](size_t bytes) { return sycl::malloc_shared<uint8_t>(bytes, q); };
    std::vector<uint16_t> hh((size_t)(M * HN)), hi((size_t)(hc * HN));
    std::vector<float> hw((size_t)HN);
    for (int64_t i = 0; i < M * HN; ++i) hh[(size_t)i] = bf16_of(2.0f * unit(11, (uint64_t)i));
    for (int64_t i = 0; i < HN; ++i) hw[(size_t)i] = 0.5f + 0.5f * unit(12, (uint64_t)i);
    for (int64_t i = 0; i < hc * HN; ++i) hi[(size_t)i] = bf16_of(0.05f * unit(13, (uint64_t)i));
    const std::vector<uint8_t> hd = e4m3_rows(lr, HN, 128, 21), hu = e4m3_rows(HN, lr, lr / 4, 31);

    uint8_t* d_h = sh(hh.size() * 2); uint8_t* d_w = sh(hw.size() * 4); uint8_t* d_inj_w = sh(hi.size() * 2);
    uint8_t* d_down = sh(hd.size()); uint8_t* d_up = sh(hu.size());
    uint8_t* d_x = sh((size_t)(M * n * 2)); uint8_t* d_inj = sh((size_t)(M * hc * 2));
    std::memcpy(d_h, hh.data(), hh.size() * 2); std::memcpy(d_w, hw.data(), hw.size() * 4);
    std::memcpy(d_inj_w, hi.data(), hi.size() * 2);
    std::memcpy(d_down, hd.data(), hd.size()); std::memcpy(d_up, hu.data(), hu.size());

    RadParam pp[5]{};
    pp[0].key = "M"; pp[0].kind = RAD_P_INT; pp[0].ival = M;
    pp[1].key = "n"; pp[1].kind = RAD_P_INT; pp[1].ival = n;
    pp[2].key = "hc"; pp[2].kind = RAD_P_INT; pp[2].ival = hc;
    pp[3].key = "lowrank"; pp[3].kind = RAD_P_INT; pp[3].ival = lr;
    pp[4].key = "eps"; pp[4].kind = RAD_P_F64; pp[4].dval = 1e-6;

    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) { std::printf("cannot write %s\n", argv[2]); return 2; }
    int fails = 0;
    for (int64_t rows = 2; rows <= 4; ++rows) {
        RadTensor t[11]{};
        t[0] = tn(d_h, RAD_BF16, 2, rows, HN);
        t[1] = tn(d_w, RAD_F32, 1, HN);
        t[2] = tn(d_down, RAD_F8E4M3, 2, lr, HN);
        t[3] = tn(d_up, RAD_F8E4M3, 2, HN, lr);
        t[4] = tn(d_inj_w, RAD_BF16, 2, hc, HN);
        t[5] = tn(d_x, RAD_BF16, 2, rows, n);
        t[6] = tn(d_inj, RAD_BF16, 2, rows, hc);
        RadArgs a{}; a.t = t; a.n_t = 11; a.p = pp; a.n_p = 5; a.world_size = 1;
        const int64_t sb = row->scratch ? row->scratch(&a) : 0;
        uint8_t* scratch = sb > 0 ? sh((size_t)sb) : nullptr;
        if (scratch) std::memset(scratch, 0, (size_t)sb);
        a.scratch = scratch; a.scratch_bytes = sb;
        std::memset(d_x, 0, (size_t)(M * n * 2));
        std::memset(d_inj, 0, (size_t)(M * hc * 2));
        const int rc = row->launch(&a, (RadStream)&q);
        q.wait_and_throw();
        std::printf("hc_persist_test: rows %lld: rc %d, scratch %lld bytes\n", (long long)rows, rc, (long long)sb);
        if (rc != RAD_OK) { ++fails; continue; }
        std::fwrite(d_x, 2, (size_t)(rows * n), f);
        std::fwrite(d_inj, 2, (size_t)(rows * hc), f);
        if (scratch) std::fwrite(scratch, 2, (size_t)(rows * lr), f);   /* dbuf: the down product, at the scratch's start */
        /* the values are live: some x and some gain non-zero */
        int nz = 0;
        for (int64_t i = 0; i < rows * n; ++i) nz += ((uint16_t*)d_x)[i] != 0;
        for (int64_t i = 0; i < rows * hc; ++i) nz += ((uint16_t*)d_inj)[i] != 0;
        if (nz == 0) { std::printf("hc_persist_test: rows %lld wrote nothing\n", (long long)rows); ++fails; }
        if (scratch) sycl::free(scratch, q);
    }
    std::fclose(f);
    std::printf("hc_persist_test: %s\n", fails ? "FAIL" : "written");
    return fails ? 1 : 0;
}
