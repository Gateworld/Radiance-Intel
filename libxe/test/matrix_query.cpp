/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, stage B).
 *
 * matrix_query.cpp -- the device's joint_matrix (DPAS) combinations, from the runtime's query: the (M, N, K)
 * tiles and the A, B, C, D element types the matrix unit takes. The static query header lists dg2 and pvc
 * only, so for Xe2 (the B70) the runtime's answer is the authority; it decides the tile shapes of the prefill
 * GEMM kernels (the trunk's int8 x int8 -> int32 and the experts' bf16 x bf16 -> f32).
 *
 *   matrix_query [device-index=0]      on the Level Zero device (GPU); RI_DEVICE=cpu asks the CPU device
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
const char* type_name(sycl::ext::oneapi::experimental::matrix::matrix_type t) {
    using sycl::ext::oneapi::experimental::matrix::matrix_type;
    switch (t) {
    case matrix_type::bf16: return "bf16";
    case matrix_type::fp16: return "fp16";
    case matrix_type::tf32: return "tf32";
    case matrix_type::fp32: return "fp32";
    case matrix_type::fp64: return "fp64";
    case matrix_type::sint8: return "sint8";
    case matrix_type::sint16: return "sint16";
    case matrix_type::sint32: return "sint32";
    case matrix_type::sint64: return "sint64";
    case matrix_type::uint8: return "uint8";
    case matrix_type::uint16: return "uint16";
    case matrix_type::uint32: return "uint32";
    case matrix_type::uint64: return "uint64";
    }
    return "?";
}
}  // namespace

int main(int argc, char** argv) {
    const int want = argc > 1 ? std::atoi(argv[1]) : 0;
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* cpu = std::getenv("RI_DEVICE");
    std::vector<sycl::device> devs = sycl::device::get_devices(cpu && !std::strcmp(cpu, "cpu") ? sycl::info::device_type::cpu
                                                                                             : sycl::info::device_type::gpu);
    if (devs.empty() || want >= (int)devs.size()) { std::printf("matrix_query: no device %d\n", want); return 2; }
    sycl::device dev = devs[(size_t)want];
    std::printf("matrix_query: %s (%s)\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "not level_zero");
    /* milestone 3, run 53 (owner note 47 item 2): the MTP coupling's kernels are strict FP64, so whether the
     * device has it, and whether as a native or an emulated capability, decides their port */
    std::printf("matrix_query: fp64 aspect %s, double_fp_config %zu flag(s)%s\n", dev.has(sycl::aspect::fp64) ? "yes" : "no",
                dev.get_info<sycl::info::device::double_fp_config>().size(),
                dev.get_info<sycl::info::device::double_fp_config>().empty() ? " (none: no double arithmetic)" : "");
    if (dev.has(sycl::aspect::fp64)) {
        /* a 1,000-term double sum on the device against the host's: emulated FP64 is slow but exact */
        try {
            sycl::queue q(dev);
            double* d = sycl::malloc_shared<double>(1, q);
            *d = 0.0;
            q.single_task([=] {
                double s = 0.0;
                for (int i = 1; i <= 1000; ++i) s += 1.0 / (double)i;
                *d = s;
            }).wait();
            double h = 0.0;
            for (int i = 1; i <= 1000; ++i) h += 1.0 / (double)i;
            std::printf("matrix_query: fp64 harmonic(1000) device %.17g host %.17g %s\n", *d, h, *d == h ? "equal" : "DIFFER");
            sycl::free(d, q);
        } catch (const sycl::exception& e) {
            std::printf("matrix_query: the fp64 kernel failed: %s\n", e.what());
        }
    }
    std::vector<sycl::ext::oneapi::experimental::matrix::combination> combs;
    try {
        combs = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    } catch (const sycl::exception& e) {
        std::printf("matrix_query: the query failed: %s\n", e.what());
        return 1;
    }
    std::printf("matrix_query: %zu combination(s)\n", combs.size());
    std::printf("%5s %5s %5s %8s %8s %8s  %-7s %-7s %-7s %-7s\n", "M", "N", "K", "maxM", "maxN", "maxK", "A", "B", "C", "D");
    for (const auto& c : combs)
        std::printf("%5zu %5zu %5zu %8zu %8zu %8zu  %-7s %-7s %-7s %-7s\n", c.msize, c.nsize, c.ksize, c.max_msize, c.max_nsize,
                    c.max_ksize, type_name(c.atype), type_name(c.btype), type_name(c.ctype), type_name(c.dtype));
    return 0;
}
