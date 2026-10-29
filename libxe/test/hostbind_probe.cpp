/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, runs 88-89, owner notes 84 and 87: the two-card first step's bind).
 *
 * hostbind_probe.cpp -- the two-card first-step hang without the model. Run 88's prebind (R15) put it at the first
 * copy on card1 from a 16,991 MiB host allocation of the context both cards share: NEO binds the buffer for card1's
 * copy engine, and xe answers DRM_IOCTL_XE_VM_BIND with EBUSY without end, while card0 binds the same allocation in
 * milliseconds. This probe (HB1) then showed 1.5 and 4 GiB binding on card1, and 8 GiB crashing the kernel
 * (xe_res_next, m3 fault 22). Owner note 87: never again bind a host allocation of more than 4 GiB on the importing
 * card, so this probe refuses one (over 4 GiB a piece is refused before anything is allocated).
 *
 * It makes the engine's copy (64 bytes with queue::memcpy on one card's in-order queue) and the engine's zero-copy
 * read (a kernel on the card reading host memory) from host memory made one of two ways:
 *   usm     sycl::aligned_alloc_host on a context over every GPU, CHUNK_MIB a piece (what l0.cpp does today; the
 *           second card imports each piece from the first)
 *   import  one contiguous mmap of GIB, registered with the driver in CHUNK_MIB pieces by NEO's
 *           zexDriverImportExternalPointer (each card maps the pages itself; the range stays contiguous)
 *   own1    as usm, but allocated through a second context over the same cards with card 1 first (run 93: does
 *           card 1 then own them instead of importing card 0's?), and used from the first context's queues. HB4: no,
 *           card 0 still owns them.
 *   solo1   allocated through a context of card 1 alone, used from the first context's queue on card 1 (ORDER "1").
 *           HB5: works, but card 0 still holds the pages: NEO places host USM on its first device whatever the context.
 *   shared1 sycl::aligned_alloc_shared on card 1 in the shared context: a buffer card 1 creates itself (does it stay in
 *           system memory, and does card 0 hold none of it?)
 *   xe_hostbind_probe GIB ORDER CHUNK_MIB [FILL=1] [MODE=usm]
 * At the end, each xe DRM client's fdinfo (drm-total-system = imports not bound; drm-total-gtt) says who owns what.
 *     GIB        host memory in all, in GiB (fractions allowed)
 *     ORDER      the cards that touch it, in order: "1" (card1 only), "01" (card0 then card1, the engine's), "10"
 *     CHUNK_MIB  the piece size, at most 4096
 *     FILL       1 = write every page from the host first, as the engine's load fills its pool
 * For each card in ORDER: one copy from each piece, then one kernel summing every piece's first byte (piece i holds
 * i + 1 there), checked against the host's sum. Each step is printed before it starts and timed. One that takes
 * longer than RI_HB_STUCK_S (30) seconds prints STUCK and the process exits 3. Exit 0 when everything completes and
 * every sum is right, 4 when a sum is wrong. A two-card probe; no CPU test.
 */
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <dirent.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;

std::atomic<long long> g_touch_start{0};   /* ns since t0 of the step in progress, 0 = none */
std::string g_what;                         /* what it is doing (written before g_touch_start) */
const clk::time_point t0 = clk::now();

long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count(); }

void begin(const std::string& what) {
    g_what = what;
    std::printf("hostbind: %s\n", what.c_str());
    std::fflush(stdout);
    g_touch_start.store(now_ns());
}
double end_ms() {
    const long long t = g_touch_start.exchange(0);
    return (now_ns() - t) * 1e-6;
}

using ImportFn = ze_result_t (*)(ze_driver_handle_t, void*, size_t);
using ReleaseFn = ze_result_t (*)(ze_driver_handle_t, void*);
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: xe_hostbind_probe GIB ORDER CHUNK_MIB [FILL=1] [MODE=usm|import]\n");
        return 2;
    }
    const double gib = std::strtod(argv[1], nullptr);
    const std::string order = argv[2];
    const long long chunk_mib = std::strtoll(argv[3], nullptr, 10);
    const bool fill = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    const std::string mode = argc > 5 ? argv[5] : "usm";
    const char* st = std::getenv("RI_HB_STUCK_S");
    const double stuck_s = st ? std::strtod(st, nullptr) : 30.0;
    if (chunk_mib <= 0 || chunk_mib > 4096) {
        std::fprintf(stderr, "hostbind: CHUNK_MIB %lld refused: pieces are 1-4096 MiB (owner note 87: no host "
                             "allocation over 4 GiB on the importing card)\n", chunk_mib);
        return 2;
    }
    if (mode != "usm" && mode != "import" && mode != "own1" && mode != "solo1" && mode != "shared1" && mode != "pair") {
        std::fprintf(stderr, "hostbind: MODE usm, import, own1, solo1, shared1 or pair\n");
        return 2;
    }
    /* pair (run 93): two sets of GIB each in the shared context -- A, standing for the other rank's pool, never
     * copied from or read here, then B, the set the cards in ORDER use. If a kernel's indirect host access makes
     * every host allocation resident, a kernel on card 1 binds A too. RI_HB_RESIDENT=1 makes B resident on each card
     * in ORDER first (zeContextMakeMemoryResident), for use with NEO's DisableIndirectAccess=1. */
    const bool pair = mode == "pair";

    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 2) {
        std::fprintf(stderr, "hostbind: %zu GPU(s); the probe needs two\n", gpus.size());
        return 2;
    }
    sycl::context ctx(gpus);   /* one context over every card, as l0.cpp makes for two ranks */
    /* own1 (run 93, owner note 91 item 3): the pieces come from a second context over the same cards with card 1
     * first, as if card 1 owned them, and are used from `ctx`'s queues as before. Does card 1 then bind them as its
     * own (no import), and do its kernels read them? */
    sycl::context ctx1(std::vector<sycl::device>{gpus[1], gpus[0]});
    /* solo1 (run 93, HB4: own1 still left card 0 the owner): the pieces from a context of card 1 alone, used from the
     * shared context's queue on card 1 -- memory only card 1 ever maps, as a rank's private pool could be. */
    sycl::context ctx_solo1(gpus[1]);
    std::vector<sycl::queue> qs;
    for (auto& d : gpus) qs.emplace_back(ctx, d, sycl::property::queue::in_order{});
    for (size_t i = 0; i < gpus.size(); ++i)
        std::printf("hostbind: card %zu = %s\n", i, gpus[i].get_info<sycl::info::device::name>().c_str());

    const long long total = (long long)(gib * 1073741824.0) / 4096 * 4096;
    const long long piece = chunk_mib << 20;
    std::vector<std::pair<char*, long long>> allocs;
    char* range = nullptr;
    ze_driver_handle_t drv = nullptr;
    ImportFn import_fn = nullptr;
    ReleaseFn release_fn = nullptr;
    if (mode == "import") {
        drv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0].get_platform());
        if (zeDriverGetExtensionFunctionAddress(drv, "zexDriverImportExternalPointer", (void**)&import_fn) !=
                ZE_RESULT_SUCCESS || !import_fn ||
            zeDriverGetExtensionFunctionAddress(drv, "zexDriverReleaseImportedPointer", (void**)&release_fn) !=
                ZE_RESULT_SUCCESS || !release_fn) {
            std::printf("hostbind: the driver has no zexDriverImportExternalPointer\n");
            return 1;
        }
        void* m = mmap(nullptr, (size_t)total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (m == MAP_FAILED) { std::printf("hostbind: mmap of %lld bytes failed\n", total); return 1; }
        range = (char*)m;
    }
    std::vector<std::pair<char*, long long>> other;   /* pair: set A */
    if (pair)
        for (long long off = 0; off < total; off += piece) {
            const long long n = std::min(piece, total - off);
            char* p = sycl::aligned_alloc_host<char>(4096, (size_t)n, ctx);
            if (!p) { std::printf("hostbind: set A allocation failed\n"); return 1; }
            if (fill) for (long long b = 0; b < n; b += 4096) p[b] = 1;
            other.emplace_back(p, n);
        }
    for (long long off = 0; off < total; off += piece) {
        const long long n = std::min(piece, total - off);
        char* p = nullptr;
        if (mode == "usm" || pair) {
            p = sycl::aligned_alloc_host<char>(4096, (size_t)n, ctx);
        } else if (mode == "own1") {
            p = sycl::aligned_alloc_host<char>(4096, (size_t)n, ctx1);
        } else if (mode == "solo1") {
            p = sycl::aligned_alloc_host<char>(4096, (size_t)n, ctx_solo1);
        } else if (mode == "shared1") {
            p = sycl::aligned_alloc_shared<char>(4096, (size_t)n, gpus[1], ctx);
        } else {
            p = range + off;
            const ze_result_t r = import_fn(drv, p, (size_t)n);
            if (r != ZE_RESULT_SUCCESS) {
                std::printf("hostbind: zexDriverImportExternalPointer(%p, %.2f MiB) = 0x%x after %zu piece(s)\n",
                            (void*)p, n / 1048576.0, (unsigned)r, allocs.size());
                return 1;
            }
        }
        if (!p) {
            std::printf("hostbind: allocation of %.2f MiB failed after %zu\n", n / 1048576.0, allocs.size());
            return 1;
        }
        if (fill) for (long long b = 0; b < n; b += 4096) p[b] = (char)(b >> 12);
        p[0] = (char)(allocs.size() + 1);
        allocs.emplace_back(p, n);
    }
    long long want = 0;
    for (size_t i = 0; i < allocs.size(); ++i) want += (signed char)(char)(i + 1);   /* what the kernel reads */
    std::printf("hostbind: %s, %zu piece(s) of at most %lld MiB, %.2f GiB in all%s, order %s\n", mode.c_str(),
                allocs.size(), chunk_mib, total / 1073741824.0, fill ? ", every page written" : "", order.c_str());
    std::fflush(stdout);

    std::vector<char*> scratch;
    std::vector<long long*> sums;
    std::vector<char**> ptrs;
    /* Only the cards in ORDER are touched at all (HB3, run 92: a setup copy on card0 queued behind its stuck copy job
     * though only card1 was asked for). */
    for (size_t i = 0; i < gpus.size(); ++i) {
        scratch.push_back(nullptr);
        sums.push_back(nullptr);
        ptrs.push_back(nullptr);
        if (order.find((char)('0' + i)) == std::string::npos) continue;
        scratch[i] = sycl::malloc_device<char>(4096, gpus[i], ctx);
        sums[i] = sycl::malloc_device<long long>(1, gpus[i], ctx);
        ptrs[i] = sycl::malloc_device<char*>(allocs.size(), gpus[i], ctx);
        std::vector<char*> hp;
        for (auto& a : allocs) hp.push_back(a.first);
        qs[i].memcpy(ptrs[i], hp.data(), hp.size() * sizeof(char*)).wait();
    }

    if (pair) {
        std::printf("hostbind: pair: set A %zu piece(s), %.2f GiB, never used here; set B is the pieces above\n",
                    other.size(), total / 1073741824.0);
        const char* res = std::getenv("RI_HB_RESIDENT");
        if (res && res[0] == '1') {
            ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
            for (char c : order) {
                const int d = c - '0';
                ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[(size_t)d]);
                for (size_t i = 0; i < allocs.size(); ++i) {
                    const auto t = now_ns();
                    const ze_result_t r = zeContextMakeMemoryResident(zc, zd, allocs[i].first, (size_t)allocs[i].second);
                    std::printf("hostbind: resident on card %d: piece %zu = 0x%x in %.1f ms\n", d, i + 1, (unsigned)r,
                                (now_ns() - t) * 1e-6);
                }
            }
        }
        std::fflush(stdout);
    }

    std::thread watch([&] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const long long s = g_touch_start.load();
            if (s > 0 && (now_ns() - s) * 1e-9 > stuck_s) {
                std::printf("hostbind: STUCK %.0f s in %s\n", stuck_s, g_what.c_str());
                std::fflush(stdout);
                _exit(3);
            }
        }
    });
    watch.detach();

    int bad = 0;
    for (char c : order) {
        const int d = c - '0';
        if (d < 0 || d >= (int)gpus.size()) { std::fprintf(stderr, "hostbind: no card %c\n", c); return 2; }
        for (size_t i = 0; i < allocs.size(); ++i) {
            char what[160];
            std::snprintf(what, sizeof what, "copy on card %d from piece %zu of %zu (%.2f MiB at %p)", d, i + 1,
                          allocs.size(), allocs[i].second / 1048576.0, (void*)allocs[i].first);
            begin(what);
            qs[(size_t)d].memcpy(scratch[(size_t)d], allocs[i].first, 64).wait();
            std::printf("hostbind:   done in %.1f ms\n", end_ms());
        }
        char what[96];
        std::snprintf(what, sizeof what, "kernel on card %d reading all %zu piece(s)", d, allocs.size());
        begin(what);
        long long* out = sums[(size_t)d];
        char** pp = ptrs[(size_t)d];
        const size_t n = allocs.size();
        qs[(size_t)d].single_task([=] {
            long long s = 0;
            for (size_t i = 0; i < n; ++i) s += (signed char)pp[i][0];
            *out = s;
        }).wait();
        long long got = 0;
        qs[(size_t)d].memcpy(&got, out, sizeof got).wait();
        const double ms = end_ms();
        std::printf("hostbind:   done in %.1f ms: sum %lld, want %lld%s\n", ms, got, want, got == want ? "" : "  WRONG");
        if (got != want) ++bad;
    }
    std::printf(bad ? "hostbind: every step completed, %d sum(s) WRONG\n" : "hostbind: every touch completed\n", bad);
    /* Who owns the pieces: each DRM client's memory as the driver counts it. A card imports what shows in its
     * "system" placement (not resident until bound); a card's own host buffers are its GTT. */
    if (DIR* dd = opendir("/proc/self/fdinfo")) {
        while (dirent* e = readdir(dd)) {
            if (e->d_name[0] == '.') continue;
            std::string fp = std::string("/proc/self/fdinfo/") + e->d_name;
            FILE* f = std::fopen(fp.c_str(), "r");
            if (!f) continue;
            char ln[256];
            std::string out;
            bool xe = false;
            while (std::fgets(ln, sizeof ln, f)) {
                if (!std::strncmp(ln, "drm-driver:", 11) && std::strstr(ln, "xe")) xe = true;
                if (!std::strncmp(ln, "drm-pdev:", 9) || !std::strncmp(ln, "drm-total-system:", 17) ||
                    !std::strncmp(ln, "drm-resident-system:", 20) || !std::strncmp(ln, "drm-total-gtt:", 14) ||
                    !std::strncmp(ln, "drm-total-vram0:", 16)) {
                    ln[std::strcspn(ln, "\n")] = 0;
                    out += std::string("  ") + ln;
                }
            }
            std::fclose(f);
            if (xe) std::printf("hostbind: fd %s:%s\n", e->d_name, out.c_str());
        }
        closedir(dd);
    }
    std::fflush(stdout);
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (scratch[i]) sycl::free(scratch[i], ctx);
        if (sums[i]) sycl::free(sums[i], ctx);
        if (ptrs[i]) sycl::free(ptrs[i], ctx);
    }
    if (mode == "usm" || mode == "shared1" || pair) {
        for (auto& a : allocs) sycl::free(a.first, ctx);
        for (auto& a : other) sycl::free(a.first, ctx);
    } else if (mode == "own1") {
        for (auto& a : allocs) sycl::free(a.first, ctx1);
    } else if (mode == "solo1") {
        for (auto& a : allocs) sycl::free(a.first, ctx_solo1);
    } else {
        for (auto& a : allocs) release_fn(drv, a.first);
        munmap(range, (size_t)total);
    }
    return bad ? 4 : 0;
}
