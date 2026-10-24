/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_registry.cpp -- libxe's kernel rows and plugin exports.
 *
 * libxe declares no schemas of its own: every op it implements is in libref's conventional
 * vocabulary, so libref's schema (which loads either way) is the one its rows bind to.
 */
#include "rad_abi.h"

static const RadKernelInfo kKernels[] = {
    {}  /* placeholder so the array is not empty; rad_kernel_count reports 0 */
};
static const int kNumKernels = 0;

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libxe",
    "0.1.0",
    "SYCL kernels for Intel Xe2 (Arc Pro B70) on the Level Zero backend: the ops qwen4exp decode "
    "issues on one card",
    "xe2-bmg-g31 sycl-cpu"
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return kNumKernels; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < kNumKernels) ? &kKernels[i] : nullptr;
}
