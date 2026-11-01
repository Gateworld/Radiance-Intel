/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 114, owner note 145 item 1).
 *
 * vram_fill_probe.cpp -- how much device memory one card really allocates under the runtime it runs on: blocks of
 * sycl::malloc_device, each filled (so the driver commits it), until an allocation or its fill fails, the cap, or
 * the runtime's free figure falls under the floor; the runtime's own figures (global_mem_size, and
 * ext_intel_free_memory where the runtime offers it) before, every 4 GiB, past 28 GiB every block, and at the end;
 * then everything freed. The 26.05 / 26.35 comparison of note 69's finding: NEO 26.35 leaves 1.6 GiB less for
 * the experts than 26.05 -- is it allocatable?
 * The xe driver never refuses an allocation: it over-commits past physical VRAM and the card resets when its VM
 * worker runs out of memory (run 114's VF1, m3 fault 34: 31.75 GiB filled on card1, then device lost). Hence the
 * cap (default 30.25 GiB, 1.4 GiB under VF1's last free reading) and the floor (default 1 GiB).
 *
 *   xe_vram_fill_probe [block_mib=256] [cap_mib=30976] [floor_mib=1024]
 */
#include <sycl/sycl.hpp>

#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
double gib(uint64_t b) { return (double)b / (double)(1ull << 30); }
// the kernel's own count of this process's VRAM: drm-total-vram0 over the process's DRM fds (fdinfo, KiB)
uint64_t kernel_vram() {
    uint64_t kib = 0;
    DIR* dir = opendir("/proc/self/fdinfo");
    if (!dir) return 0;
    while (dirent* e = readdir(dir)) {
        if (e->d_name[0] == '.') continue;
        char path[64];
        std::snprintf(path, sizeof path, "/proc/self/fdinfo/%s", e->d_name);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        char line[256];
        unsigned long long v;
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "drm-total-vram0: %llu KiB", &v) == 1) kib += v;
        std::fclose(f);
    }
    closedir(dir);
    return kib << 10;
}
// the runtime's free figure (-1 where it offers none), printed with global_mem_size and the kernel's count
long long report(sycl::queue& q, const char* when, uint64_t held) {
    const sycl::device d = q.get_device();
    const uint64_t total = d.get_info<sycl::info::device::global_mem_size>();
    long long fr = -1;
    if (d.has(sycl::aspect::ext_intel_free_memory)) fr = (long long)d.get_info<sycl::ext::intel::info::device::free_memory>();
    std::printf("vram %s: held %.3f GiB; global_mem_size %.3f GiB; free %s%.3f GiB; kernel vram0 %.3f GiB\n", when,
                gib(held), gib(total), fr < 0 ? "(not offered) " : "", fr < 0 ? 0.0 : gib((uint64_t)fr),
                gib(kernel_vram()));
    return fr;
}
}  // namespace

int main(int argc, char** argv) {
    const uint64_t block = (uint64_t)(argc > 1 ? std::atoi(argv[1]) : 256) << 20;
    const uint64_t cap = (uint64_t)(argc > 2 ? std::atoi(argv[2]) : 30976) << 20;
    const long long floor = (long long)(argc > 3 ? std::atoi(argv[3]) : 1024) << 20;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("vram_fill_probe on %s, blocks of %.0f MiB, cap %.3f GiB, floor %.3f GiB free\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), (double)block / (1 << 20), gib(cap),
                gib((uint64_t)floor));
    if (report(q, "start", 0) < 0) { std::printf("RESULT vram_fill: no free figure, so no floor; not run\n"); return 1; }
    std::vector<void*> held;
    uint64_t total = 0;
    const char* why = "the cap";
    while (total + block <= cap) {
        if (total + block > (28ull << 30) || total % (4ull << 30) == 0) {
            if (report(q, "before block", total) < floor + (long long)block) { why = "the floor"; break; }
        }
        void* p = nullptr;
        try {
            p = sycl::malloc_device(block, q);
            if (!p) { why = "malloc_device returned null"; break; }
            q.memset(p, 0x5a, block).wait_and_throw();
        } catch (const sycl::exception& e) {
            std::printf("vram: block %zu failed: %s\n", held.size(), e.what());
            if (p) sycl::free(p, q);
            why = "an exception";
            break;
        }
        held.push_back(p);
        total += block;
    }
    report(q, "end", total);
    std::printf("RESULT vram_fill: %.3f GiB allocated and filled in %zu blocks; stopped by %s\n", gib(total), held.size(), why);
    for (void* p : held) sycl::free(p, q);
    report(q, "after free", 0);
    return 0;
}
