/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * tape_lifetime_test.cpp -- GPU faults 3 and 4 (owner note 10): a recorded pass freed while a play of
 * it still ran left the copy engine fetching commands from unmapped memory. Many tapes, each
 * recorded against per-request buffers that are freed and reallocated between rounds, played and
 * destroyed at once with no sync -- as the engine does when a pass key changes. Each tape writes a
 * pattern with a slow kernel and copies it (memcpy and a pitched 2D copy, both recorded) into that
 * round's destination buffers; after each round every copied byte is checked. The backend's tape
 * counters (rad_l0_tape_stats) must show tapes deferred at destroy (the hazard was real), a
 * graveyard that never passes its bound, and every tape freed at the end. 4 KiB canaries around
 * the destinations.
 *
 *   xe_tape_lifetime_test [ROUNDS=200] [TAPES=16]   RADIANCE_L0_DEVICE=cpu for the CPU device
 * Exit 0 when every byte matches, every canary holds and the counters are as above.
 */
#include "device/device.h"
#include "rad_device.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void rad_l0_tape_stats(int64_t out[4]);

namespace {

constexpr size_t kN = 16384;   /* bytes a destination */
constexpr size_t kCanary = 4096;
constexpr uint8_t kPat = 0xC5;
constexpr int64_t kW = 96, kH = 32, kSP = 128, kDP = 160;   /* the 2D copy */

uint8_t pat(int round, int k, size_t i) { return (uint8_t)(i * 7 + (size_t)k * 31 + (size_t)round * 13 + 1); }

}  // namespace

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 200;
    const int tapes = argc > 2 ? std::atoi(argv[2]) : 16;
    if (rad_dev_count() <= 0 || rad_dev_set(0) != RAD_OK) { std::printf("tape_lifetime: no device\n"); return 77; }
    RadStream s = nullptr;
    if (rad_stream_create(&s, 0) != RAD_OK) { std::printf("tape_lifetime: no stream\n"); return 1; }
    sycl::queue& q = *(sycl::queue*)s;
    /* the source each tape fills: kN bytes, and a kSP-pitched block for the 2D copy */
    uint8_t* src = (uint8_t*)rad_dev_alloc(kN + kSP * kH, RAD_MEM_DEVICE);
    if (!src) { std::printf("tape_lifetime: no memory\n"); return 1; }
    std::vector<uint8_t> host(kCanary * 2 + kN + kDP * kH);
    long long bad = 0, canary_bad = 0, played = 0;
    int64_t st[4], max_dead = 0;

    for (int r = 0; r < rounds; ++r) {
        /* this round's per-request buffers: fresh allocations (the previous round's are freed) */
        std::vector<uint8_t*> dst((size_t)tapes);
        for (int k = 0; k < tapes; ++k) {
            dst[(size_t)k] = (uint8_t*)rad_dev_alloc(kCanary * 2 + kN + kDP * kH, RAD_MEM_DEVICE);
            if (!dst[(size_t)k]) { std::printf("tape_lifetime: no memory at round %d\n", r); return 1; }
            rad_memset_async(dst[(size_t)k], kPat, kCanary * 2 + kN + kDP * kH, s);
        }
        for (int k = 0; k < tapes; ++k) {
            uint8_t* d = dst[(size_t)k] + kCanary;
            if (rad::tape_begin() < 0) { std::printf("tape_begin refused: %s\n", rad_dev_last_error()); return 1; }
            const int rr = r, kk = k;
            /* a slow fill, so the play is still running when the tape is destroyed */
            q.parallel_for(sycl::range<1>(kN + kSP * kH), [=](sycl::id<1> id) {
                const size_t i = id[0];
                uint32_t x = (uint32_t)i;
                for (int spin = 0; spin < 2000; ++spin) x = x * 1664525u + 1013904223u;
                /* x is never 0x9E3779B9 for these seeds in practice; the test checks the bytes anyway */
                src[i] = (uint8_t)(i * 7 + (size_t)kk * 31 + (size_t)rr * 13 + 1) ^ (uint8_t)(x == 0x9E3779B9u);
            });
            rad_memcpy_async(d, src, kN, s);
            rad_memcpy_2d_async(d + kN, kDP, src + kN, kSP, kW, kH, s);
            rad::tape_mark();
            rad::RadTape t = rad::tape_end(true);
            if (!t) { std::printf("tape_end refused: %s\n", rad_dev_last_error()); return 1; }
            rad::tape_play(t, 0, rad::tape_len(t));   /* held, then submitted by the next call */
            rad::tape_destroy(t);                      /* no sync: the play may still be running */
            ++played;
            rad_l0_tape_stats(st);
            if (st[3] > max_dead) max_dead = st[3];
        }
        rad_stream_sync(s);
        for (int k = 0; k < tapes; ++k) {
            q.memcpy(host.data(), dst[(size_t)k], host.size()).wait();
            for (size_t i = 0; i < kCanary; ++i)
                canary_bad += (host[i] != kPat) + (host[kCanary + kN + kDP * kH + i] != kPat);
            for (size_t i = 0; i < kN; ++i) bad += host[kCanary + i] != pat(r, k, i);
            for (int64_t y = 0; y < kH; ++y)
                for (int64_t x = 0; x < kDP; ++x) {
                    const uint8_t got = host[kCanary + kN + (size_t)(y * kDP + x)];
                    bad += got != (x < kW ? pat(r, k, (size_t)(kN + y * kSP + x)) : kPat);
                }
            rad_dev_free(dst[(size_t)k], RAD_MEM_DEVICE);
        }
        if (r % 50 == 49 || r == rounds - 1) {
            rad_l0_tape_stats(st);
            std::printf("  round %4d: tapes %lld, deferred %lld, waited %lld, freed %lld, graveyard %lld (max %lld); "
                        "wrong bytes %lld, canary %s\n", r + 1, played, (long long)st[0], (long long)st[1],
                        (long long)st[2], (long long)st[3], (long long)max_dead, bad, canary_bad ? "BROKEN" : "ok");
            std::fflush(stdout);
        }
    }
    rad_stream_sync(s);
    rad_l0_tape_stats(st);
    const bool ok = bad == 0 && canary_bad == 0 && st[2] == played && st[3] == 0 && max_dead <= 64 + 1;
    std::printf("tape_lifetime: %s -- %lld tapes, %lld deferred at destroy (a play still running), %lld freed, "
                "graveyard max %lld, wrong bytes %lld\n", ok ? "PASS" : "FAIL", played, (long long)st[0],
                (long long)st[2], (long long)max_dead, bad);
    rad_dev_free(src, RAD_MEM_DEVICE);
    rad_stream_destroy(s);
    return ok ? 0 : 1;
}
