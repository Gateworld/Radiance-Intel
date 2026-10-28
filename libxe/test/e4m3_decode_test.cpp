// SPDX-License-Identifier: Apache-2.0
// Part of the Intel port of radiance (milestone 3, owner note 52 item 2). xe::e4m3_to_f32, the branch-free E4M3
// decode behind the shim's v_cvt_pk_f32_fp8 / v_cvt_f32_fp8, against the ABI's rad_fp8e4m3_to_f32 on all 256
// codes, bit for bit (the NaN by its bits too): on the host, then in a kernel on the SYCL device the test runs on
// (the CPU device in ctest; the card as is), through both builtins' forms.
#include "xe_lanes.h"

#include <cstdio>
#include <cstring>
#include <vector>

static unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }

int main() {
    int bad = 0;
    for (unsigned b = 0; b < 256; ++b) {
        const unsigned got = bits(xe::e4m3_to_f32(b)), want = bits(rad_fp8e4m3_to_f32((uint8_t)b));
        if (got != want) {
            if (bad < 8) std::printf("host: code 0x%02x got 0x%08x want 0x%08x\n", b, got, want);
            ++bad;
        }
    }
    std::printf("host: %d of 256 codes differ\n", bad);

    sycl::queue q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    float* out = sycl::malloc_shared<float>(3 * 256, q);
    q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const unsigned w = (unsigned)id[0] * 4u;   // four codes a work-item, packed as the kernels load them
        const int v = (int)(w | ((w + 1) << 8) | ((w + 2) << 16) | ((w + 3) << 24));
        const xe::v2f lo = xe::cvt_pk_f32_fp8(v, false), hi = xe::cvt_pk_f32_fp8(v, true);
        out[w] = lo.x; out[w + 1] = lo.y; out[w + 2] = hi.x; out[w + 3] = hi.y;
        for (int s = 0; s < 4; ++s) out[256 + w + s] = xe::cvt_f32_fp8(v, s);
        for (int s = 0; s < 4; ++s) out[512 + w + s] = xe::e4m3_to_f32(w + (unsigned)s);
    }).wait();
    int dbad = 0;
    for (int form = 0; form < 3; ++form)
        for (unsigned b = 0; b < 256; ++b) {
            const unsigned got = bits(out[form * 256 + b]), want = bits(rad_fp8e4m3_to_f32((uint8_t)b));
            if (got != want) {
                if (dbad < 8) std::printf("device form %d: code 0x%02x got 0x%08x want 0x%08x\n", form, b, got, want);
                ++dbad;
            }
        }
    std::printf("device: %d of 768 decodes differ\n", dbad);
    sycl::free(out, q);
    const bool ok = bad == 0 && dbad == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
