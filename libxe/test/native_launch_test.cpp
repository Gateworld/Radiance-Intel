/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 2).
 *
 * native_launch_test.cpp -- the feasibility probe for owner note 28 (tapes on direct Level Zero command
 * lists). Three questions, on the card:
 *  1. Can a libxe-style kernel (xe::launch_as<K>: one by-value std::tuple<P...> argument) be launched
 *     natively -- its ze_kernel_handle_t from the SYCL kernel bundle, zeKernelSetArgumentValue(0, the
 *     tuple), zeKernelSetGroupSize, zeCommandListAppendLaunchKernel on a regular command list, executed
 *     on a native command queue of the same context -- with results identical to the SYCL launch?
 *  2. What does a submission cost on the device: N one-kernel SYCL graphs submitted one by one (the tape's
 *     direct play, G3: ~13 us a graph) against N one-kernel regular command lists executed in ONE
 *     zeCommandQueueExecuteCommandLists call, against one list holding N kernels?
 *  3. Does the SYCL queue's native handle come out (get_native<level_zero>(queue)), or does the backend
 *     need its own command queue (and events to order it against the SYCL queue)?
 *
 *   native_launch_test [N=1000]      RI_DEVICE=cpu: prints "no Level Zero device" and passes (a ctest).
 */
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "xe_lanes.h"

#include <dlfcn.h>

#include <chrono>
#include <optional>
#include <variant>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
/* the kernel under test, in libxe's HIP-dialect form */
__global__ void probe_kernel(int n, const float* a, float* b, float scale, int bias) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * scale + (float)bias + (float)threadIdx.x * 0.001f;
}

struct Ze {
    void* lib = nullptr;
#define ZE_FN(name) decltype(&name) name = nullptr;
    ZE_FN(zeKernelSetArgumentValue) ZE_FN(zeKernelSetGroupSize) ZE_FN(zeCommandListCreate)
    ZE_FN(zeCommandListAppendLaunchKernel) ZE_FN(zeCommandListClose) ZE_FN(zeCommandListDestroy)
    ZE_FN(zeCommandQueueCreate) ZE_FN(zeCommandQueueExecuteCommandLists) ZE_FN(zeCommandQueueSynchronize)
    ZE_FN(zeCommandQueueDestroy) ZE_FN(zeKernelGetName)
#undef ZE_FN
    bool open() {
        lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) return false;
#define ZE_LD(name) name = (decltype(&::name))dlsym(lib, #name); if (!name) return false;
        ZE_LD(zeKernelSetArgumentValue) ZE_LD(zeKernelSetGroupSize) ZE_LD(zeCommandListCreate)
        ZE_LD(zeCommandListAppendLaunchKernel) ZE_LD(zeCommandListClose) ZE_LD(zeCommandListDestroy)
        ZE_LD(zeCommandQueueCreate) ZE_LD(zeCommandQueueExecuteCommandLists) ZE_LD(zeCommandQueueSynchronize)
        ZE_LD(zeCommandQueueDestroy) ZE_LD(zeKernelGetName)
#undef ZE_LD
        return true;
    }
};
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1000;
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* cpu = std::getenv("RI_DEVICE");
    if (cpu && !std::strcmp(cpu, "cpu")) { std::printf("native_launch_test: no Level Zero device (RI_DEVICE=cpu); nothing to probe\n"); return 0; }
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty() || gpus[0].get_backend() != sycl::backend::ext_oneapi_level_zero) { std::printf("native_launch_test: no Level Zero GPU\n"); return 0; }
    sycl::device dev = gpus[0];
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());
    std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    Ze ze;
    if (!ze.open()) { std::printf("FAIL: libze_loader symbols\n"); return 1; }

    const int n = 4096;
    float* a = sycl::malloc_device<float>(n, q);
    float* b1 = sycl::malloc_device<float>(n, q);
    float* b2 = sycl::malloc_device<float>(n, q);
    std::vector<float> ha(n), h1(n), h2(n);
    for (int i = 0; i < n; ++i) ha[i] = 0.5f * i;
    q.memcpy(a, ha.data(), n * 4).wait();
    q.memset(b1, 0, n * 4).wait(); q.memset(b2, 0, n * 4).wait();
    const xe::dim3 grid(n / 256), block(256);

    /* 1a. the SYCL launch */
    xe::launch_as<probe_kernel>(probe_kernel, q, grid, block, 0, n, (const float*)a, b1, 1.5f, 7).wait();

    /* 1b. the same kernel natively: its id by name, its ze handle, the tuple as argument 0 */
    std::optional<sycl::kernel_id> kid;
    for (const sycl::kernel_id& id : sycl::get_kernel_ids()) {
        std::string nm = id.get_name();
        if (nm.find("probe_kernel") != std::string::npos && nm.find("launch_as") != std::string::npos) { kid = id; break; }
    }
    if (!kid) { std::printf("FAIL: no kernel id named after probe_kernel/launch_as\n"); return 1; }
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev}, {*kid});
    sycl::kernel k = kb.get_kernel(*kid);
    ze_kernel_handle_t zk = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(k);
    ze_context_handle_t zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    std::printf("kernel id: %s\n", kid->get_name());

    /* 3. the SYCL queue's native handle: a variant of a command queue or an immediate command list in
     * this SYCL version */
    try {
        auto nat = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
        if (std::holds_alternative<ze_command_queue_handle_t>(nat))
            std::printf("get_native(queue): a ze_command_queue_handle_t\n");
        else if (std::holds_alternative<ze_command_list_handle_t>(nat))
            std::printf("get_native(queue): an immediate ze_command_list_handle_t (the backend would append lists to it or keep its own queue + events)\n");
        else
            std::printf("get_native(queue): another variant\n");
    } catch (const sycl::exception& e) {
        std::printf("get_native(queue): throws (%s) -> the backend needs its own command queue\n", e.what());
    }
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_handle_t own = nullptr;
    if (ze.zeCommandQueueCreate(zctx, zdev, &qd, &own) != ZE_RESULT_SUCCESS) { std::printf("FAIL: zeCommandQueueCreate\n"); return 1; }

    using Args = std::tuple<int, const float*, float*, float, int>;
    Args args(n, a, b2, 1.5f, 7);
    ze_result_t r = ze.zeKernelSetArgumentValue(zk, 0, sizeof(Args), &args);
    if (r != ZE_RESULT_SUCCESS) { std::printf("FAIL: zeKernelSetArgumentValue(0, tuple): 0x%x -- the kernel's parameters are not the one tuple\n", (unsigned)r); return 1; }
    r = ze.zeKernelSetGroupSize(zk, block.x, block.y, block.z);
    if (r != ZE_RESULT_SUCCESS) { std::printf("FAIL: zeKernelSetGroupSize 0x%x\n", (unsigned)r); return 1; }
    ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    ze_command_list_handle_t one = nullptr;
    if (ze.zeCommandListCreate(zctx, zdev, &ld, &one) != ZE_RESULT_SUCCESS) { std::printf("FAIL: zeCommandListCreate\n"); return 1; }
    ze_group_count_t gc{grid.x, grid.y, grid.z};
    if (ze.zeCommandListAppendLaunchKernel(one, zk, &gc, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS) { std::printf("FAIL: AppendLaunchKernel\n"); return 1; }
    ze.zeCommandListClose(one);
    if (ze.zeCommandQueueExecuteCommandLists(own, 1, &one, nullptr) != ZE_RESULT_SUCCESS) { std::printf("FAIL: ExecuteCommandLists\n"); return 1; }
    ze.zeCommandQueueSynchronize(own, UINT64_MAX);
    q.memcpy(h1.data(), b1, n * 4).wait();
    q.memcpy(h2.data(), b2, n * 4).wait();
    int bad = 0;
    for (int i = 0; i < n; ++i) if (h1[i] != h2[i]) ++bad;
    std::printf("1. native launch of the launch_as kernel: %s (%d of %d elements differ; b[1] sycl %g native %g)\n",
                bad ? "FAIL" : "PASS, identical", bad, n, h1[1], h2[1]);
    if (bad) return 1;

    /* 2. the cost of a submission: N one-kernel SYCL graphs one by one; N one-kernel lists in one execute;
     * one list with N kernels. Tiny kernel (one work-group) so the device time is the gap. */
    namespace sx = sycl::ext::oneapi::experimental;
    const xe::dim3 g1(1), b64(64);
    std::vector<sx::command_graph<sx::graph_state::executable>> execs;
    execs.reserve((size_t)N);
    for (int i = 0; i < N; ++i) {
        sx::command_graph<sx::graph_state::modifiable> gr(ctx, dev);
        gr.begin_recording(q);
        xe::launch_as<probe_kernel>(probe_kernel, q, g1, b64, 0, 64, (const float*)a, b1, 1.0f, i);
        gr.end_recording(q);
        execs.push_back(gr.finalize());
    }
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) q.ext_oneapi_graph(execs[(size_t)i]);
    q.wait();
    const double ms_graphs = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) q.ext_oneapi_graph(execs[(size_t)i]);
    q.wait();
    const double ms_graphs2 = ms_since(t0);

    std::vector<ze_command_list_handle_t> lists((size_t)N, nullptr);
    ze.zeKernelSetGroupSize(zk, 64, 1, 1);
    ze_group_count_t gc1{1, 1, 1};
    for (int i = 0; i < N; ++i) {
        Args ai(64, a, b2, 1.0f, i);
        ze.zeKernelSetArgumentValue(zk, 0, sizeof(Args), &ai);
        ze.zeCommandListCreate(zctx, zdev, &ld, &lists[(size_t)i]);
        ze.zeCommandListAppendLaunchKernel(lists[(size_t)i], zk, &gc1, nullptr, 0, nullptr);
        ze.zeCommandListClose(lists[(size_t)i]);
    }
    t0 = std::chrono::steady_clock::now();
    ze.zeCommandQueueExecuteCommandLists(own, (uint32_t)N, lists.data(), nullptr);
    ze.zeCommandQueueSynchronize(own, UINT64_MAX);
    const double ms_lists = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    ze.zeCommandQueueExecuteCommandLists(own, (uint32_t)N, lists.data(), nullptr);
    ze.zeCommandQueueSynchronize(own, UINT64_MAX);
    const double ms_lists2 = ms_since(t0);

    ze_command_list_handle_t big = nullptr;
    ze.zeCommandListCreate(zctx, zdev, &ld, &big);
    for (int i = 0; i < N; ++i) {
        Args ai(64, a, b2, 1.0f, i);
        ze.zeKernelSetArgumentValue(zk, 0, sizeof(Args), &ai);
        ze.zeCommandListAppendLaunchKernel(big, zk, &gc1, nullptr, 0, nullptr);
    }
    ze.zeCommandListClose(big);
    t0 = std::chrono::steady_clock::now();
    ze.zeCommandQueueExecuteCommandLists(own, 1, &big, nullptr);
    ze.zeCommandQueueSynchronize(own, UINT64_MAX);
    const double ms_big = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    ze.zeCommandQueueExecuteCommandLists(own, 1, &big, nullptr);
    ze.zeCommandQueueSynchronize(own, UINT64_MAX);
    const double ms_big2 = ms_since(t0);
    std::printf("2. N=%d one-kernel submissions: SYCL graphs one by one %.1f / %.1f ms (%.1f us a graph); "
                "native lists in one execute %.1f / %.1f ms (%.1f us a list); one list of N kernels %.1f / %.1f ms (%.1f us a kernel)\n",
                N, ms_graphs, ms_graphs2, ms_graphs2 * 1000 / N, ms_lists, ms_lists2, ms_lists2 * 1000 / N, ms_big, ms_big2, ms_big2 * 1000 / N);
    for (auto l : lists) ze.zeCommandListDestroy(l);
    ze.zeCommandListDestroy(big); ze.zeCommandListDestroy(one);
    ze.zeCommandQueueDestroy(own);
    sycl::free(a, q); sycl::free(b1, q); sycl::free(b2, q);
    std::printf("native_launch_test: PASS\n");
    return 0;
}
