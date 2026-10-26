/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * native_tape_test.cpp -- owner note 28: recorded passes as direct Level Zero command lists, through
 * the backend's tape API (core/device/l0.cpp, "native tapes") and libxe's launch path (xe::launch_as,
 * which records into the open list through rad_l0_rec.h). On a card:
 *  1. the mode: the first tape must be native (rad_l0_ntape_stats), else FAIL;
 *  2. a kernel with dynamic work-group memory (XE_DYN_LDS) and a plain one, recorded and replayed,
 *     against their plain launches: identical bytes;
 *  3. the order between the SYCL runtime's own commands and the appended lists, many rounds with no
 *     host sync: a plain launch writes the round's pattern, the tape (copy + transform) plays, a
 *     plain launch folds the result into a checksum, then everything is checked on the host;
 *  4. a tape with an event record: another stream waits on the event and reads what the tape wrote;
 *  5. a direct SYCL submission inside a recording is refused (the tape would replay without it);
 *  6. the counters: every list created is destroyed once the tapes are gone and the stream drained.
 * On the CPU device (RADIANCE_L0_DEVICE=cpu) the backend records SYCL graphs: prints so and passes.
 *
 *   xe_native_tape_test [ROUNDS=400]
 */
#include "device/device.h"
#include "rad_device.h"
#include "xe_lanes.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void rad_l0_ntape_stats(int64_t out[6]);
extern "C" void rad_l0_tape_stats(int64_t out[4]);

namespace {

constexpr int kN = 65536;      /* elements */
constexpr int kBlock = 256;

/* a plain kernel: y = x * m + add */
__global__ void scale_kernel(const uint32_t* x, uint32_t* y, int n, uint32_t m, uint32_t add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * m + add;
}
/* block sums through dynamic work-group memory: out[b] = sum of x over block b, staged in LDS */
__global__ void block_sum_kernel(const uint32_t* x, uint32_t* out, int n) {
    XE_DYN_LDS(uint32_t, lds);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    lds[threadIdx.x] = i < n ? x[i] : 0u;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = lds[0];
}
/* a slow pattern writer (the round's pattern), so an ordering bug has time to show */
__global__ void pattern_kernel(uint32_t* x, int n, uint32_t seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = (uint32_t)i * 2654435761u + seed;
    for (int k = 0; k < 400; ++k) v = v * 1664525u + 1013904223u;
    x[i] = v;
}
/* folds y into the checksum slot of the round: sum[r] = sum of y */
__global__ void fold_kernel(const uint32_t* y, int n, uint32_t* sum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xe::atomic_add(sum, y[i]);
}

uint32_t pattern_host(int i, uint32_t seed) {
    uint32_t v = (uint32_t)i * 2654435761u + seed;
    for (int k = 0; k < 400; ++k) v = v * 1664525u + 1013904223u;
    return v;
}
const xe::dim3 grid(kN / kBlock), block(kBlock);

}  // namespace

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 400;
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (rad_dev_count() <= 0 || rad_dev_set(0) != RAD_OK) { std::printf("native_tape: no device\n"); return 77; }
    RadStream s = nullptr, s2 = nullptr;
    if (rad_stream_create(&s, 0) != RAD_OK || rad_stream_create(&s2, 0) != RAD_OK) { std::printf("native_tape: no stream\n"); return 1; }
    sycl::queue& q = *(sycl::queue*)s;
    sycl::queue& q2 = *(sycl::queue*)s2;
    const bool cpu = !q.get_device().is_gpu();
    int fails = 0;
    auto check = [&](bool ok, const char* what) { std::printf("  %s: %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++fails; };

    uint32_t* x = (uint32_t*)rad_dev_alloc(kN * 4, RAD_MEM_DEVICE);
    uint32_t* y = (uint32_t*)rad_dev_alloc(kN * 4, RAD_MEM_DEVICE);
    uint32_t* y2 = (uint32_t*)rad_dev_alloc(kN * 4, RAD_MEM_DEVICE);
    uint32_t* bs = (uint32_t*)rad_dev_alloc(kN / kBlock * 4, RAD_MEM_DEVICE);
    uint32_t* bs2 = (uint32_t*)rad_dev_alloc(kN / kBlock * 4, RAD_MEM_DEVICE);
    uint32_t* sums = (uint32_t*)rad_dev_alloc((size_t)(rounds + 1) * 4, RAD_MEM_DEVICE);
    if (!x || !y || !y2 || !bs || !bs2 || !sums) { std::printf("native_tape: no memory\n"); return 1; }
    std::vector<uint32_t> hx(kN), hy(kN), hy2(kN), hb(kN / kBlock), hb2(kN / kBlock);
    for (int i = 0; i < kN; ++i) hx[i] = pattern_host(i, 7);
    rad_memcpy_async(x, hx.data(), kN * 4, s);
    rad_memset_async(sums, 0, (size_t)(rounds + 1) * 4, s);
    rad_stream_sync(s);

    /* 1 + 2: the plain launches, then the same two kernels recorded (they run while recorded) and replayed */
    XE_LAUNCH(&q, scale_kernel, grid, block, 0, (const uint32_t*)x, y, kN, 3u, 11u);
    XE_LAUNCH(&q, block_sum_kernel, grid, block, kBlock * sizeof(uint32_t), (const uint32_t*)x, bs, kN);
    rad_stream_sync(s);
    rad_memcpy_async(hy.data(), y, kN * 4, s);
    rad_memcpy_async(hb.data(), bs, kN / kBlock * 4, s);
    rad_stream_sync(s);
    if (rad::tape_begin() < 0) { std::printf("native_tape: tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
    XE_LAUNCH(&q, scale_kernel, grid, block, 0, (const uint32_t*)x, y2, kN, 3u, 11u);
    rad::tape_mark();
    XE_LAUNCH(&q, block_sum_kernel, grid, block, kBlock * sizeof(uint32_t), (const uint32_t*)x, bs2, kN);
    rad::tape_mark();
    rad::RadTape t = rad::tape_end(true);
    if (!t) { std::printf("native_tape: tape_end refused: %s\n", rad_dev_last_error()); return 1; }
    int64_t ns[6];
    rad_l0_ntape_stats(ns);
    std::printf("native_tape: device %s, mode %s (lists created %lld, kernels %lld)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                ns[5] == 2 ? "native, in-order lists" : ns[5] == 1 ? "native, barrier-separated lists" : "SYCL graphs",
                (long long)ns[0], (long long)ns[4]);
    if (cpu) {
        std::printf("native_tape: the CPU device records SYCL graphs; nothing native to test -- PASS\n");
        rad::tape_destroy(t);
        rad_stream_sync(s);
        return 0;
    }
    check(ns[5] > 0, "1. the first tape is native");
    auto fetch = [&] {
        rad_memcpy_async(hy2.data(), y2, kN * 4, s);
        rad_memcpy_async(hb2.data(), bs2, kN / kBlock * 4, s);
        rad_stream_sync(s);
    };
    fetch();
    check(hy == hy2 && hb == hb2, "2a. recorded run: scale and LDS block-sum identical to plain");
    rad_memset_async(y2, 0, kN * 4, s);
    rad_memset_async(bs2, 0, kN / kBlock * 4, s);
    rad::tape_play(t, 0, rad::tape_len(t));
    fetch();
    check(hy == hy2 && hb == hb2, "2b. replay: identical to plain");
    rad::tape_destroy(t);

    /* 3. ordering against the runtime's own commands: pattern (plain) -> tape: copy x->y, y = y*5+r (recorded
     *    with r frozen at 0, so the expected value uses 0) -> fold (plain) into sums[r]; no sync in between */
    if (rad::tape_begin() < 0) { std::printf("tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
    rad_memcpy_async(y, x, kN * 4, s);
    rad::tape_mark();
    XE_LAUNCH(&q, scale_kernel, grid, block, 0, (const uint32_t*)y, y, kN, 5u, 0u);
    rad::tape_mark();
    rad::RadTape t3 = rad::tape_end(true);
    if (!t3) { std::printf("tape_end refused: %s\n", rad_dev_last_error()); return 1; }
    for (int r = 1; r <= rounds; ++r) {
        XE_LAUNCH(&q, pattern_kernel, grid, block, 0, x, kN, (uint32_t)r);
        rad::tape_play(t3, 0, rad::tape_len(t3));
        XE_LAUNCH(&q, fold_kernel, grid, block, 0, (const uint32_t*)y, kN, sums + r);
    }
    rad_stream_sync(s);
    std::vector<uint32_t> hs((size_t)rounds + 1);
    rad_memcpy_async(hs.data(), sums, (size_t)(rounds + 1) * 4, s);
    rad_stream_sync(s);
    int wrong = 0;
    for (int r = 1; r <= rounds; ++r) {
        uint32_t want = 0;
        for (int i = 0; i < kN; ++i) want += pattern_host(i, (uint32_t)r) * 5u;
        if (hs[(size_t)r] != want) ++wrong;
    }
    std::printf("  3. %d rounds of plain -> replay -> plain with no host sync: %d wrong checksum(s)\n", rounds, wrong);
    check(wrong == 0, "3. order held");
    rad::tape_destroy(t3);

    /* 4. an event record inside the tape; stream 2 waits on it and reads the tape's output */
    RadEvent ev = nullptr;
    rad_event_create(&ev);
    rad_memset_async(y2, 0, kN * 4, s);
    rad_stream_sync(s);
    if (rad::tape_begin() < 0) { std::printf("tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
    XE_LAUNCH(&q, pattern_kernel, grid, block, 0, x, kN, 99u);
    XE_LAUNCH(&q, scale_kernel, grid, block, 0, (const uint32_t*)x, y, kN, 1u, 1u);
    rad_event_record(ev, s);
    rad::tape_mark();
    rad::RadTape t4 = rad::tape_end(true);
    if (!t4) { std::printf("tape_end refused: %s\n", rad_dev_last_error()); return 1; }
    int bad4 = 0;
    for (int rep = 0; rep < 3; ++rep) {
        rad_memset_async(y, 0, kN * 4, s);
        rad_stream_sync(s);
        rad::tape_play(t4, 0, rad::tape_len(t4));
        rad_event_wait(s2, ev);
        XE_LAUNCH(&q2, scale_kernel, grid, block, 0, (const uint32_t*)y, y2, kN, 1u, 0u);
        rad_stream_sync(s2);
        rad_memcpy_async(hy2.data(), y2, kN * 4, s2);
        rad_stream_sync(s2);
        for (int i = 0; i < kN; ++i) bad4 += hy2[(size_t)i] != pattern_host(i, 99u) + 1u;
        rad_stream_sync(s);
    }
    check(bad4 == 0, "4. an event recorded in the tape orders another stream");
    rad::tape_destroy(t4);
    rad_event_destroy(ev);

    /* 5. a direct SYCL submission inside the recording: refused */
    if (rad::tape_begin() < 0) { std::printf("tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
    XE_LAUNCH(&q, scale_kernel, grid, block, 0, (const uint32_t*)x, y, kN, 1u, 0u);
    q.single_task([=] { y[0] = 1u; });
    rad::tape_mark();
    rad::RadTape t5 = rad::tape_end(true);
    const char* why5 = rad_dev_last_error();
    std::printf("  5. tape_end after a direct submission: %s (%s)\n", t5 ? "KEPT" : "refused", why5 ? why5 : "");
    check(t5 == nullptr && why5 && std::strstr(why5, "bypassed"), "5. the bypass is refused");
    if (t5) rad::tape_destroy(t5);

    /* 6. the counters */
    rad_stream_sync(s);
    rad_stream_sync(s2);
    int64_t ts[4];
    rad_l0_tape_stats(ts);
    rad_l0_ntape_stats(ns);
    std::printf("  6. lists created %lld destroyed %lld, plays %lld (lists appended %lld), kernel objects %lld; tapes freed %lld, graveyard %lld\n",
                (long long)ns[0], (long long)ns[1], (long long)ns[2], (long long)ns[3], (long long)ns[4], (long long)ts[2], (long long)ts[3]);
    check(ns[0] == ns[1] && ts[3] == 0, "6. every list destroyed, graveyard empty");

    rad_dev_free(x, RAD_MEM_DEVICE); rad_dev_free(y, RAD_MEM_DEVICE); rad_dev_free(y2, RAD_MEM_DEVICE);
    rad_dev_free(bs, RAD_MEM_DEVICE); rad_dev_free(bs2, RAD_MEM_DEVICE); rad_dev_free(sums, RAD_MEM_DEVICE);
    rad_stream_destroy(s2);
    rad_stream_destroy(s);
    std::printf("native_tape: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
