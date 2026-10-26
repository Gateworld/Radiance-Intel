/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 2).
 *
 * kernel_props.cpp -- every kernel the process knows (libxe's, and librad_l0k's core kernels), with the
 * Level Zero driver's per-kernel properties: spill memory (register spills -> the scratch surface the
 * driver allocates per queue), private memory, SIMD width, required/suggested group sizes. Owner note 24:
 * "name the kernels that take scratch and how much". The driver's scratch allocation for a queue is the
 * per-thread spill size times the hardware's thread count, so a kernel's spillMemSize is what matters.
 *
 *   kernel_props LIBXE.so [device-index=0]   on the Level Zero device (GPU); on the CPU device (RI_DEVICE=cpu)
 *   it lists the kernels only (no Level Zero handle to ask).
 */
#include <level_zero/ze_api.h>   /* the toolchain's /usr/include/level_zero; the SYCL interop header no longer pulls it in */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
typedef ze_result_t (*GetPropsFn)(ze_kernel_handle_t, ze_kernel_properties_t*);
typedef ze_result_t (*GetNameFn)(ze_kernel_handle_t, size_t*, char*);
struct Row { std::string name; uint32_t spill, priv, simd, nargs; };
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: kernel_props LIBXE.so [device-index]\n"); return 2; }
    const int want = argc > 2 ? std::atoi(argv[2]) : 0;
    setvbuf(stdout, nullptr, _IONBF, 0);
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("kernel_props: dlopen %s: %s\n", argv[1], dlerror()); return 2; }
    const char* cpu = std::getenv("RI_DEVICE");
    std::vector<sycl::device> devs = sycl::device::get_devices(cpu && !std::strcmp(cpu, "cpu") ? sycl::info::device_type::cpu
                                                                                             : sycl::info::device_type::gpu);
    if (devs.empty() || want >= (int)devs.size()) { std::printf("kernel_props: no device %d\n", want); return 2; }
    sycl::device dev = devs[(size_t)want];
    sycl::context ctx(dev);
    std::printf("kernel_props: %s (%s)\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "not level_zero");
    const bool l0 = dev.get_backend() == sycl::backend::ext_oneapi_level_zero;
    /* The SYCL runtime loads the Level Zero loader RTLD_LOCAL (run 10, T3: dlsym(RTLD_DEFAULT) found nothing),
     * so open it by name: the same library the adapter holds, one more reference. */
    void* ze = l0 ? dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL) : nullptr;
    GetPropsFn get_props = ze ? (GetPropsFn)dlsym(ze, "zeKernelGetProperties") : nullptr;
    GetNameFn get_name = ze ? (GetNameFn)dlsym(ze, "zeKernelGetName") : nullptr;
    if (l0 && (!get_props || !get_name))
        std::printf("kernel_props: zeKernelGetProperties not found (%s); names only\n", ze ? "no symbol" : dlerror());

    std::vector<sycl::kernel_id> ids = sycl::get_kernel_ids();
    std::printf("kernel_props: %zu kernel id(s) registered\n", ids.size());
    std::vector<Row> rows;
    size_t built = 0, failed = 0;
    for (const sycl::kernel_id& id : ids) {
        Row r{id.get_name(), 0, 0, 0, 0};
        try {
            if (!sycl::is_compatible({id}, dev)) { r.name += "  [not compatible with this device]"; rows.push_back(r); continue; }
            auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev}, {id});
            sycl::kernel k = kb.get_kernel(id);
            ++built;
            if (l0 && get_props) {
                ze_kernel_handle_t zk = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(k);
                ze_kernel_properties_t p{};
                p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
                if (get_props(zk, &p) == ZE_RESULT_SUCCESS) {
                    r.spill = p.spillMemSize; r.priv = p.privateMemSize; r.simd = p.maxSubgroupSize; r.nargs = p.numKernelArgs;
                }
                size_t n = 0;
                if (get_name(zk, &n, nullptr) == ZE_RESULT_SUCCESS && n > 1) {
                    std::string s(n, '\0');
                    if (get_name(zk, &n, s.data()) == ZE_RESULT_SUCCESS) { s.resize(std::strlen(s.c_str())); r.name = s; }
                }
            }
        } catch (const sycl::exception& e) {
            ++failed;
            r.name += std::string("  [build failed: ") + e.what() + "]";
        }
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.spill != b.spill ? a.spill > b.spill : a.name < b.name; });
    std::printf("kernel_props: %zu built, %zu failed\n", built, failed);
    std::printf("%10s %10s %5s %5s  %s\n", "spill_B", "private_B", "simd", "args", "kernel");
    uint64_t spill_total = 0; size_t n_spill = 0;
    for (const Row& r : rows) {
        std::printf("%10u %10u %5u %5u  %s\n", r.spill, r.priv, r.simd, r.nargs, r.name.c_str());
        if (r.spill) { spill_total += r.spill; ++n_spill; }
    }
    std::printf("kernel_props: %zu kernel(s) spill, %llu bytes a thread in total; the driver's scratch surface for a "
                "queue is sized by the largest spill it has seen times the hardware thread count\n",
                n_spill, (unsigned long long)spill_total);
    return 0;
}
