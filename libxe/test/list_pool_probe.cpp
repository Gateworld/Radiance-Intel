/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, owner note 59 phase 0: the xe-direct design brief's measurements).
 *
 * list_pool_probe.cpp -- what the Level Zero driver holds per recorded command list, and how it submits, measured
 * from this process's own DRM client (/proc/self/fdinfo: drm-total-vram0 / drm-total-gtt, drm-cycles-ccs), no
 * model. Read-only toward the engine: its own context, its own one-work-group kernel, nothing shared.
 *   A  VRAM per regular list: N lists of K launches each (the recorder's in-order lists, l0.cpp n_list_create),
 *      created, then destroyed (does the memory come back?), then created again (is it reused?)
 *   B  GPU time per command inside a played list: K dependent launches in an in-order list against K launches with
 *      no dependence (flags 0, no barriers), played on an in-order immediate list as the engine plays (the Exp append)
 *   C  host round trip of one immediate append + host synchronize, and the host cost of an append alone
 *   D  the CCS engine's busy share while the process idles after C. Run 75 (X1-X3): 0.5 % idle with direct
 *      submission active -- the driver parks its ring when idle, so this does not show direct submission; count
 *      DRM_IOCTL_XE_EXEC with strace from the host instead (m3/intel-host/x-strace.sh: 3 execs for ~4,000 submissions)
 *
 *   xe_list_pool_probe [device-index=0] [lists=256]
 */
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {
class Tiny;
using AppendFn = ze_result_t (*)(ze_command_list_handle_t, uint32_t, ze_command_list_handle_t*, ze_event_handle_t,
                                 uint32_t, ze_event_handle_t*);

/* This process's xe DRM clients from fdinfo, summed once per client id: memory in MiB and the CCS cycle counters. */
struct Fd { double vram = 0, gtt = 0; unsigned long long ccs = 0, ccs_total = 0; int clients = 0; };
double to_mib(const char* s) {
    double v = std::strtod(s, nullptr);
    if (std::strstr(s, "GiB")) return v * 1024;
    if (std::strstr(s, "MiB")) return v;
    if (std::strstr(s, "KiB")) return v / 1024;
    return v / 1048576.0;
}
Fd fdinfo() {
    Fd r;
    std::map<std::string, Fd> by_client;
    DIR* d = opendir("/proc/self/fdinfo");
    if (!d) return r;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string p = std::string("/proc/self/fdinfo/") + e->d_name;
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) continue;
        char ln[256];
        bool xe = false;
        std::string id;
        Fd c;
        while (std::fgets(ln, sizeof ln, f)) {
            const char* v = std::strchr(ln, ':');
            if (!v) continue;
            ++v;
            while (*v == ' ' || *v == '\t') ++v;
            if (!std::strncmp(ln, "drm-driver:", 11)) xe = !std::strncmp(v, "xe", 2);
            else if (!std::strncmp(ln, "drm-client-id:", 14)) id = v;
            else if (!std::strncmp(ln, "drm-total-vram0:", 16)) c.vram = to_mib(v);
            else if (!std::strncmp(ln, "drm-total-gtt:", 14)) c.gtt = to_mib(v);
            else if (!std::strncmp(ln, "drm-cycles-ccs:", 15)) c.ccs = std::strtoull(v, nullptr, 10);
            else if (!std::strncmp(ln, "drm-total-cycles-ccs:", 21)) c.ccs_total = std::strtoull(v, nullptr, 10);
        }
        std::fclose(f);
        if (xe && !id.empty()) by_client[id] = c;
    }
    closedir(d);
    for (auto& [id, c] : by_client) { r.vram += c.vram; r.gtt += c.gtt; r.ccs += c.ccs; r.ccs_total += c.ccs_total; ++r.clients; }
    return r;
}
double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool ok(ze_result_t rc, const char* what) {
    if (rc != ZE_RESULT_SUCCESS) std::printf("probe: %s: 0x%x\n", what, (unsigned)rc);
    return rc == ZE_RESULT_SUCCESS;
}
}  // namespace

int main(int argc, char** argv) {
    const int want = argc > 1 ? std::atoi(argv[1]) : 0;
    const int nlists = argc > 2 ? std::atoi(argv[2]) : 256;
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (want >= (int)devs.size()) { std::printf("probe: no device %d\n", want); return 2; }
    sycl::device dev = devs[(size_t)want];
    if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) { std::printf("probe: not a Level Zero device\n"); return 2; }
    sycl::context sctx(dev);
    sycl::queue q(sctx, dev, sycl::property::queue::in_order());
    float* buf = sycl::malloc_device<float>(64, q);
    q.memset(buf, 0, 64 * sizeof(float)).wait();
    /* the kernel: one work-group of 64, one store a lane (submitted once through SYCL so its bundle exists) */
    q.parallel_for<Tiny>(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> it) { buf[it.get_global_id(0)] += 1.0f; }).wait();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(sctx, {dev}, {sycl::get_kernel_id<Tiny>()});
    sycl::kernel sk = kb.get_kernel(sycl::get_kernel_id<Tiny>());
    ze_context_handle_t ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sctx);
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_kernel_handle_t k = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sk);
    std::printf("probe: %s, %d list(s) a set\n", dev.get_info<sycl::info::device::name>().c_str(), nlists);

    ze_kernel_properties_t kp{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, nullptr};
    if (!ok(zeKernelGetProperties(k, &kp), "zeKernelGetProperties")) return 1;
    for (uint32_t a = 0; a < kp.numKernelArgs; ++a)
        if (zeKernelSetArgumentValue(k, a, sizeof buf, &buf) != ZE_RESULT_SUCCESS) {
            int zero = 0;
            if (!ok(zeKernelSetArgumentValue(k, a, sizeof zero, &zero), "zeKernelSetArgumentValue")) return 1;
        }
    if (!ok(zeKernelSetGroupSize(k, 64, 1, 1), "zeKernelSetGroupSize")) return 1;
    const ze_group_count_t one{1, 1, 1};
    std::printf("probe: kernel args %u, spill %u B\n", kp.numKernelArgs, kp.spillMemSize);

    auto make = [&](bool inorder, int launches) -> ze_command_list_handle_t {
        ze_command_list_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, inorder ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0u};
        ze_command_list_handle_t l = nullptr;
        if (zeCommandListCreate(ctx, zd, &d, &l) != ZE_RESULT_SUCCESS) return nullptr;
        for (int i = 0; i < launches; ++i)
            if (zeCommandListAppendLaunchKernel(l, k, &one, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS) { zeCommandListDestroy(l); return nullptr; }
        if (zeCommandListClose(l) != ZE_RESULT_SUCCESS) { zeCommandListDestroy(l); return nullptr; }
        return l;
    };

    /* A: VRAM per list */
    const Fd f0 = fdinfo();
    std::printf("A clients %d, start: vram %.1f MiB, gtt %.1f MiB\n", f0.clients, f0.vram, f0.gtt);
    const int ks[] = {0, 1, 16, 128};
    for (int pass = 0; pass < 2; ++pass)
        for (int kk : ks) for (int io = 1; io >= 0; --io) {
            if (!io && kk != 16) continue;
            const Fd a = fdinfo();
            std::vector<ze_command_list_handle_t> ls;
            for (int i = 0; i < nlists; ++i) {
                ze_command_list_handle_t l = make(io, kk);
                if (!l) { std::printf("A list %d of %d failed\n", i, nlists); break; }
                ls.push_back(l);
            }
            const Fd b = fdinfo();
            for (auto l : ls) zeCommandListDestroy(l);
            const Fd c = fdinfo();
            std::printf("A pass %d, %s, %3d launches: %zu lists +%.1f MiB vram (%.3f MiB a list), +%.1f MiB gtt (%.3f a list); "
                        "after destroy %+.1f MiB vram, %+.1f gtt from before\n",
                        pass, io ? "in-order" : "plain   ", kk, ls.size(), b.vram - a.vram, (b.vram - a.vram) / (ls.empty() ? 1 : ls.size()),
                        b.gtt - a.gtt, (b.gtt - a.gtt) / (ls.empty() ? 1 : ls.size()), c.vram - a.vram, c.gtt - a.gtt);
        }
    const Fd f1 = fdinfo();
    std::printf("A end: vram %.1f MiB (%+.1f from start), gtt %.1f MiB (%+.1f)\n", f1.vram, f1.vram - f0.vram, f1.gtt, f1.gtt - f0.gtt);

    /* the engine's play path: an in-order immediate list on the compute group, the Exp append */
    void* ze = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
    AppendFn append = ze ? (AppendFn)dlsym(ze, "zeCommandListImmediateAppendCommandListsExp") : nullptr;
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t imm = nullptr;
    if (!ok(zeCommandListCreateImmediate(ctx, zd, &qd, &imm), "zeCommandListCreateImmediate")) return 1;

    /* B: GPU time a command inside a played list */
    if (append) {
        for (int io = 1; io >= 0; --io) {
            double per[2] = {0, 0};
            const int sizes[2] = {1, 512};
            for (int s = 0; s < 2; ++s) {
                ze_command_list_handle_t l = make(io, sizes[s]);
                if (!l) { std::printf("B list failed\n"); break; }
                for (int w = 0; w < 3; ++w) { append(imm, 1, &l, nullptr, 0, nullptr); zeCommandListHostSynchronize(imm, UINT64_MAX); }
                const int reps = 40;
                const double t0 = now_us();
                for (int r = 0; r < reps; ++r) { append(imm, 1, &l, nullptr, 0, nullptr); zeCommandListHostSynchronize(imm, UINT64_MAX); }
                per[s] = (now_us() - t0) / reps;
                zeCommandListDestroy(l);
            }
            std::printf("B %s list: play of 1 launch %.1f us, of 512 launches %.1f us -> %.3f us a launch in the list\n",
                        io ? "in-order" : "plain   ", per[0], per[1], (per[1] - per[0]) / 511.0);
        }
    } else {
        std::printf("B skipped: the loader has no zeCommandListImmediateAppendCommandListsExp\n");
    }

    /* C: host round trip of one immediate launch + synchronize, and an append alone */
    {
        for (int w = 0; w < 50; ++w) { zeCommandListAppendLaunchKernel(imm, k, &one, nullptr, 0, nullptr); zeCommandListHostSynchronize(imm, UINT64_MAX); }
        const int reps = 2000;
        double t0 = now_us();
        for (int r = 0; r < reps; ++r) { zeCommandListAppendLaunchKernel(imm, k, &one, nullptr, 0, nullptr); zeCommandListHostSynchronize(imm, UINT64_MAX); }
        const double rt = (now_us() - t0) / reps;
        t0 = now_us();
        for (int r = 0; r < reps; ++r) zeCommandListAppendLaunchKernel(imm, k, &one, nullptr, 0, nullptr);
        const double app = (now_us() - t0) / reps;
        zeCommandListHostSynchronize(imm, UINT64_MAX);
        const double all = (now_us() - t0) / reps;
        std::printf("C immediate: launch + synchronize %.2f us a round trip; append alone %.2f us host; %d queued launches drain at %.2f us each\n",
                    rt, app, reps, all);
    }

    /* D: the CCS engine's busy share at idle (a polled ring keeps the engine busy with nothing submitted) */
    {
        const Fd a = fdinfo();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        const Fd b = fdinfo();
        const unsigned long long busy = b.ccs - a.ccs, total = b.ccs_total - a.ccs_total;
        std::printf("D idle 2 s after submitting: ccs busy %llu of %llu cycles (%.1f %%)%s\n", busy, total,
                    total ? 100.0 * busy / total : 0.0, total ? "" : " -- no drm-cycles-ccs in fdinfo");
    }
    {
        const Fd a = fdinfo();
        for (int r = 0; r < 200; ++r) { zeCommandListAppendLaunchKernel(imm, k, &one, nullptr, 0, nullptr); zeCommandListHostSynchronize(imm, UINT64_MAX); }
        const Fd b = fdinfo();
        const unsigned long long busy = b.ccs - a.ccs, total = b.ccs_total - a.ccs_total;
        std::printf("D busy control (200 round trips): ccs busy %llu of %llu cycles (%.1f %%)\n", busy, total, total ? 100.0 * busy / total : 0.0);
    }
    zeCommandListDestroy(imm);
    sycl::free(buf, q);
    const Fd f2 = fdinfo();
    std::printf("probe: done; vram %.1f MiB, gtt %.1f MiB\n", f2.vram, f2.gtt);
    return 0;
}
