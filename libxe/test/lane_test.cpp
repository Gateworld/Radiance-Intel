/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * lane_test.cpp -- the lane shim (xe_lanes.h) against tests/r4d_emu's CPU semantics.
 *
 * Runs every case of lane_cases.h through the shim on a SYCL device (RI_DEVICE=cpu: the OpenCL CPU
 * device; otherwise the first Level Zero GPU) and through the emulator (lane_emu.cpp), and compares
 * every output word. Outputs carry 4 KiB canaries on both sides, checked after each case.
 * Exit 0 only if every case matches and every canary holds.
 */
#include "xe_lanes.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define LANE_SHARED(type, name, n) XE_SHARED(type, name, n)
#include "lane_cases.h"

extern "C" void lane_emu_run(int c, const int* in, int* out, int* counter);

namespace {

constexpr int kThreads = LANE_GRID * LANE_BLOCK;
constexpr int kOut = kThreads * LANE_NOUT;
constexpr size_t kCanary = 4096;            /* bytes each side */
constexpr uint32_t kPattern = 0xC0DEFA11u;

struct Case { const char* name; const char* builtins; const char* spec; bool ulps; };
const Case kCases[LANE_NCASES] = {
    {"dpp", "update_dpp row_shl/row_shr/row_xmask, bound_ctrl on/off", "emulator", false},
    {"xlane", "permlanex16, readlane, readfirstlane, ds_bpermute", "emulator", false},
    {"ballot", "ballot_w32, __ballot, __popc (ISA), __popcll, __ffs (ISA), __ffsll (ISA)", "emulator", false},
    {"shfl", "__shfl, __shfl_xor, __shfl_down, __shfl_up (ISA) at widths 32/16/8", "emulator", false},
    {"perm", "v_perm_b32, __byte_perm", "emulator", false},
    {"fp8enc", "cvt_pk_fp8_f32", "emulator", false},
    {"fp8dec", "cvt_pk_f32_fp8, cvt_f32_fp8", "ISA reference", false},
    {"f16", "cvt_pkrtz, fdot2", "ISA reference", false},
    {"math", "exp2f, rcpf, __expf, rsqrtf", "emulator / ISA, <= 4 ulp", true},
    {"barrier", "__syncthreads, __syncthreads_or, s_barrier, s_barrier_signal/wait, wave_barrier, LDS atomicAdd", "emulator", false},
    {"atomic", "__hip_atomic_fetch_add/load/store, fence x4, sched_barrier, sched_group_barrier, s_sleep, __threadfence(_system)", "emulator", false},
    {"wmma", "wmma_f32_16x16x16 bf16 / f16 / fp8 (gfx12 w32)", "emulator", false},
};

template <auto K, class... A>
void run_xe(sycl::queue& q, A... a) {
    /* the launch returns no event unless profiled (run 187): wait on the queue */
    XE_LAUNCH(&q, K, dim3(LANE_GRID), dim3(LANE_BLOCK), 0, a...);
    q.wait_and_throw();
}

void run_case(sycl::queue& q, int c, const int* in, int* out, int* counter) {
    switch (c) {
    case 0:  run_xe<lane_case_dpp>(q, in, out); break;
    case 1:  run_xe<lane_case_xlane>(q, in, out); break;
    case 2:  run_xe<lane_case_ballot>(q, in, out); break;
    case 3:  run_xe<lane_case_shfl>(q, in, out); break;
    case 4:  run_xe<lane_case_perm>(q, in, out); break;
    case 5:  run_xe<lane_case_fp8enc>(q, in, out); break;
    case 6:  run_xe<lane_case_fp8dec>(q, in, out); break;
    case 7:  run_xe<lane_case_f16>(q, in, out); break;
    case 8:  run_xe<lane_case_math>(q, in, out); break;
    case 9:  run_xe<lane_case_barrier>(q, in, out); break;
    case 10: run_xe<lane_case_atomic>(q, in, out, counter); break;
    case 11: run_xe<lane_case_wmma>(q, in, out); break;
    }
}

int64_t ulp_diff(int a, int b) {
    auto key = [](int x) { return x < 0 ? (int64_t)INT32_MIN - x : (int64_t)x; };
    return std::llabs(key(a) - key(b));
}

}  // namespace

int main(int argc, char** argv) {
    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && std::strcmp(dev, "cpu") == 0;
    sycl::device d;
    try {
        d = cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception& e) {
        std::printf("lane_test: no %s device: %s\n", cpu ? "cpu" : "gpu", e.what());
        return 2;
    }
    std::printf("lane_test: device %s (%s), driver %s\n", d.get_info<sycl::info::device::name>().c_str(),
                d.get_platform().get_info<sycl::info::platform::name>().c_str(),
                d.get_info<sycl::info::device::driver_version>().c_str());
    std::string sgs;
    bool has32 = false;
    for (size_t s : d.get_info<sycl::info::device::sub_group_sizes>()) {
        sgs += " " + std::to_string(s);
        has32 |= s == 32;
    }
    std::printf("lane_test: sub-group sizes%s\n", sgs.c_str());
    if (!has32) { std::printf("lane_test: FAIL no sub-group size 32\n"); return 1; }
    sycl::queue q(d, sycl::property::queue::in_order());

    /* inputs: an LCG, with a few edge values */
    std::vector<int> hin(kThreads);
    uint32_t r = 0x1234567u;
    for (int i = 0; i < kThreads; ++i) { r = r * 1664525u + 1013904223u; hin[i] = (int)r; }
    hin[3] = 0; hin[4] = -1; hin[5] = INT32_MIN; hin[6] = 0x7fffffff; hin[40] = 1; hin[77] = 0x80;

    const size_t cw = kCanary / 4;
    int* din = sycl::malloc_device<int>(kThreads, q);
    int* dbuf = sycl::malloc_device<int>(kOut + 2 * cw, q);
    int* dcnt = sycl::malloc_device<int>(2 + 2 * cw, q);
    int* dout = dbuf + cw;
    q.memcpy(din, hin.data(), sizeof(int) * kThreads).wait();

    std::vector<int> hbuf(kOut + 2 * cw), ref(kOut), hcnt(2 + 2 * cw);
    int fails = 0, canary_bad = 0;
    for (int c = 0; c < LANE_NCASES; ++c) {
        for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] = (i < cw || i >= cw + kOut) ? (int)kPattern : 0;
        for (size_t i = 0; i < hcnt.size(); ++i) hcnt[i] = (i < cw || i >= cw + 2) ? (int)kPattern : 0;
        q.memcpy(dbuf, hbuf.data(), sizeof(int) * hbuf.size());
        q.memcpy(dcnt, hcnt.data(), sizeof(int) * hcnt.size()).wait();
        run_case(q, c, din, dout, dcnt + cw);
        q.memcpy(hbuf.data(), dbuf, sizeof(int) * hbuf.size());
        q.memcpy(hcnt.data(), dcnt, sizeof(int) * hcnt.size()).wait();
        int bad_canary = 0;
        for (size_t i = 0; i < cw; ++i) {
            bad_canary += hbuf[i] != (int)kPattern;
            bad_canary += hbuf[cw + kOut + i] != (int)kPattern;
            bad_canary += hcnt[i] != (int)kPattern;
            bad_canary += hcnt[cw + 2 + i] != (int)kPattern;
        }
        canary_bad += bad_canary;

        std::fill(ref.begin(), ref.end(), 0);
        int rcnt[2] = {0, 0};
        lane_emu_run(c, hin.data(), ref.data(), rcnt);

        int mism = 0, first = -1;
        int64_t worst = 0;
        for (int i = 0; i < kOut; ++i) {
            const int a = hbuf[cw + i], b = ref[i];
            const bool ok = kCases[c].ulps ? ulp_diff(a, b) <= 4 : a == b;
            if (kCases[c].ulps) worst = std::max(worst, ulp_diff(a, b));
            if (!ok) { if (first < 0) first = i; ++mism; }
        }
        if (c == 10 && (hcnt[cw] != rcnt[0] || hcnt[cw + 1] != rcnt[1])) {
            ++mism;
            std::printf("  atomic counters: xe %d %d, emu %d %d\n", hcnt[cw], hcnt[cw + 1], rcnt[0], rcnt[1]);
        }
        const bool pass = mism == 0 && bad_canary == 0;
        fails += !pass;
        std::printf("%s case %-8s [%s] %s", pass ? "PASS" : "FAIL", kCases[c].name, kCases[c].spec,
                    kCases[c].builtins);
        if (kCases[c].ulps) std::printf(" (worst %lld ulp)", (long long)worst);
        std::printf("; canaries %s\n", bad_canary ? "BROKEN" : "ok");
        if (first >= 0)
            std::printf("  %d mismatches; first at thread %d out %d: xe 0x%08x emu 0x%08x (in 0x%08x)\n", mism,
                        first / LANE_NOUT, first % LANE_NOUT, (unsigned)hbuf[cw + first], (unsigned)ref[first],
                        (unsigned)hin[first / LANE_NOUT]);
    }
    sycl::free(din, q);
    sycl::free(dbuf, q);
    sycl::free(dcnt, q);
    std::printf("lane_test: %d of %d cases pass, canary words broken %d -> %s\n", LANE_NCASES - fails,
                LANE_NCASES, canary_bad, fails ? "FAIL" : "PASS");
    (void)argc; (void)argv;
    return fails ? 1 : 0;
}
