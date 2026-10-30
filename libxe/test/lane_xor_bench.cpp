/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, run 102).
 *
 * lane_xor_bench.cpp -- owner note 111's first question about the lane layer: what the ported kernels' cross-lane
 * butterfly costs on Xe. libr4d's r4d_xor_lane<N> is AMD's DPP row_xmask (N < 16) / v_permlanex16 (N = 16), which
 * xe_lanes.h emulates with sycl::select_from_group over a lane index computed a lane (update_dpp / permlanex16).
 * SYCL's own permute_group_by_xor names the xor pattern. Both give every lane the same partner, so the same values;
 * this times a 32-lane sum butterfly (16, 8, 4, 2, 1) repeated ITERS times a work-item, in both forms, and prints
 * each form's device time and an output checksum (the two must be equal). The first run of each is a warm-up.
 *
 *   xe_lane_xor_bench [iters=4096] [groups=4096] [runs=5]
 */
#include "xe_lanes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kWG = 256;

/* r4d_xor_lane<N> as libxe compiles it: the emulated DPP row_xmask / permlanex16 */
template <int N>
inline float emu_xor(float v) {
    const int b = sycl::bit_cast<int>(v);
    int r;
    if constexpr (N < 16) r = xe::update_dpp(0, b, 0x160 | N, 0xf, 0xf, true);
    else r = xe::permlanex16(b, b, 0x76543210u, 0xFEDCBA98u, false, false);
    return sycl::bit_cast<float>(r);
}

template <int N>
inline float sycl_xor(float v) {
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, N);
}

/* HIP's __shfl_xor as xe_lanes.h emulates it (select_from_group over (l & ~(w-1)) | ((l ^ m) & (w-1))) */
template <int N>
inline float shfl_xor_emu(float v) { return xe::shfl_xor(v, N); }

/* kForm: 0 the emulated DPP / permlanex16, 1 permute_group_by_xor, 2 xe::shfl_xor */
template <int kForm>
struct Butterfly;

template <int kForm>
double run(sycl::queue& q, float* out, int iters, int groups, double* checksum) {
    sycl::event e = q.parallel_for<Butterfly<kForm>>(
        sycl::nd_range<1>((size_t)groups * kWG, kWG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            float v = (float)(it.get_global_id(0) & 1023) * (1.0f / 1024.0f);
            for (int i = 0; i < iters; ++i) {
                float s = v;
                if constexpr (kForm == 0) {
                    s += emu_xor<16>(s); s += emu_xor<8>(s); s += emu_xor<4>(s); s += emu_xor<2>(s); s += emu_xor<1>(s);
                } else if constexpr (kForm == 1) {
                    s += sycl_xor<16>(s); s += sycl_xor<8>(s); s += sycl_xor<4>(s); s += sycl_xor<2>(s); s += sycl_xor<1>(s);
                } else {
                    s += shfl_xor_emu<16>(s); s += shfl_xor_emu<8>(s); s += shfl_xor_emu<4>(s); s += shfl_xor_emu<2>(s);
                    s += shfl_xor_emu<1>(s);
                }
                v = s * (1.0f / 32.0f);   /* the lane's mean: stays bounded, keeps the chain dependent */
            }
            out[it.get_global_id(0)] = v;
        });
    e.wait();
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::vector<float> h((size_t)groups * kWG);
    q.memcpy(h.data(), out, h.size() * sizeof(float)).wait();
    double c = 0.0;
    for (float x : h) c += x;
    *checksum = c;
    return (double)(t1 - t0) * 1e-6;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters  = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int groups = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int runs   = argc > 3 ? std::atoi(argv[3]) : 5;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    std::printf("device: %s; %d work-groups of %d, %d butterflies a work-item\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), groups, kWG, iters);
    float* out = sycl::malloc_device<float>((size_t)groups * kWG, q);
    if (!out) { std::fprintf(stderr, "no device memory\n"); return 1; }
    double ce = 0, cs = 0, ch = 0;
    for (int r = 0; r < runs; ++r) {
        const double te = run<0>(q, out, iters, groups, &ce);
        const double ts = run<1>(q, out, iters, groups, &cs);
        const double th = run<2>(q, out, iters, groups, &ch);
        std::printf("run %d: emulated (update_dpp / permlanex16) %.3f ms, permute_group_by_xor %.3f ms, xe::shfl_xor "
                    "%.3f ms; checksums %.6f / %.6f / %.6f%s\n", r, te, ts, th, ce, cs, ch,
                    ce == cs && cs == ch ? " (equal)" : " (DIFFER)");
    }
    sycl::free(out, q);
    return ce == cs && cs == ch ? 0 : 1;
}
