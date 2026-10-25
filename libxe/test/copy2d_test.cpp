/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * copy2d_test.cpp -- the shim's r4d_p2p_copy2d (xe_support.cpp; libr4d's cast_bf16_bf16 copies through
 * it) launched plainly and recorded into a SYCL graph (sycl_ext_oneapi_graph does not record
 * ext_oneapi_memcpy2d, so the recorded form is a kernel): both must give the expected rows, byte for
 * byte, with 4 KiB canaries around the destination and the pitch gaps untouched.
 *
 *   copy2d_test     RI_DEVICE=cpu for the CPU SYCL device
 * Exit 0 only if every case matches.
 */
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height, long stream);

namespace sx = sycl::ext::oneapi::experimental;

namespace {

constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xC5;

bool run_case(sycl::queue& q, long dp, long sp, long w, long h, long off, bool record) {
    const size_t sbytes = (size_t)(sp * h + off), dbytes = (size_t)(dp * h + off);
    std::vector<uint8_t> hs(sbytes);
    for (size_t i = 0; i < sbytes; ++i) hs[i] = (uint8_t)(i * 131 + 7);
    uint8_t* ds = sycl::malloc_device<uint8_t>(sbytes, q);
    uint8_t* dd = sycl::malloc_device<uint8_t>(dbytes + 2 * kCanary, q);
    q.memcpy(ds, hs.data(), sbytes);
    q.memset(dd, kPat, dbytes + 2 * kCanary).wait();
    const long dst = (long)(uintptr_t)(dd + kCanary + off), src = (long)(uintptr_t)(ds + off);
    int rc;
    if (record) {
        sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
        g.begin_recording(q);
        rc = r4d_p2p_copy2d(dst, src, dp, sp, w, h, (long)(uintptr_t)&q);
        g.end_recording(q);
        if (rc == 0) {
            auto e = g.finalize();
            q.ext_oneapi_graph(e);
            q.ext_oneapi_graph(e);   /* replayed twice: the same bytes */
        }
    } else {
        rc = r4d_p2p_copy2d(dst, src, dp, sp, w, h, (long)(uintptr_t)&q);
    }
    q.wait_and_throw();
    std::vector<uint8_t> hd(dbytes + 2 * kCanary);
    q.memcpy(hd.data(), dd, hd.size()).wait();
    bool ok = rc == 0;
    for (size_t i = 0; i < kCanary; ++i) ok &= hd[i] == kPat && hd[kCanary + dbytes + i] == kPat;
    for (long r = 0; r < h; ++r)
        for (long c = 0; c < dp; ++c) {
            const uint8_t got = hd[kCanary + off + r * dp + c];
            const uint8_t want = c < w ? hs[(size_t)(off + r * sp + c)] : kPat;
            ok &= got == want;
        }
    for (long i = 0; i < off; ++i) ok &= hd[kCanary + i] == kPat;
    std::printf("  %s dp %ld sp %ld w %ld h %ld off %ld: rc %d %s\n", record ? "recorded" : "plain   ", dp, sp, w, h,
                off, rc, ok ? "PASS" : "FAIL");
    sycl::free(ds, q); sycl::free(dd, q);
    return ok;
}

}  // namespace

int main() {
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  sycl::property::queue::in_order());
    std::printf("copy2d_test: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    /* the MTP tap's shape (bf16 rows of 10240 into a 2x wider destination), a dense one, a byte path
     * (odd width), an unaligned base */
    const long cases[][5] = { {40960, 20480, 20480, 8, 0}, {20480, 20480, 20480, 3, 0}, {64, 48, 37, 5, 0},
                              {64, 64, 64, 4, 1}, {4096, 2048, 2048, 1, 0} };
    int fails = 0;
    for (auto& c : cases)
        for (int rec = 0; rec < 2; ++rec) fails += !run_case(q, c[0], c[1], c[2], c[3], c[4], rec != 0);
    std::printf("copy2d_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
