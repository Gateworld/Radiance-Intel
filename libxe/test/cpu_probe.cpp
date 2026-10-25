/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * cpu_probe.cpp -- which kernel does the OpenCL CPU compiler fail on? Loads a library built with one
 * device image per kernel (xe_moe_cpuprobe: the ported MoE unit, spir64 only), then builds each
 * kernel's image for the device alone, printing its name first (flushed), so the last name printed
 * before a crash is the kernel the compiler cannot take.
 *
 *   xe_cpu_probe LIB.so [first=0]
 */
#include <sycl/sycl.hpp>

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_cpu_probe LIB.so [first]\n"); return 2; }
    if (!dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL) && !dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL)) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    const int first = argc > 2 ? std::atoi(argv[2]) : 0;
    sycl::queue q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const std::vector<sycl::kernel_id> ids = sycl::get_kernel_ids();
    std::printf("%zu kernel ids\n", ids.size());
    std::fflush(stdout);
    int ok = 0, bad = 0;
    for (size_t i = (size_t)first; i < ids.size(); ++i) {
        std::printf("[%zu] %s ... ", i, ids[i].get_name());
        std::fflush(stdout);
        try {
            auto b = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), { q.get_device() },
                                                                             { ids[i] });
            (void)b;
            std::printf("ok\n");
            ++ok;
        } catch (const sycl::exception& e) {
            std::printf("FAILED: %s\n", e.what());
            ++bad;
        }
        std::fflush(stdout);
    }
    std::printf("built %d, failed %d\n", ok, bad);
    return bad ? 1 : 0;
}
