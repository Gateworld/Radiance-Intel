/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * event_stress.cpp -- owner note 7: rad_event_record / rad_event_wait far past run 24's ~74,000
 * expert moves with no stream sync, through the Level Zero backend (core/device/l0.cpp). The live
 * events must stay flat and every record must still order the work before it.
 *
 * Phase 1 (the mover's pattern): a ring of K events made once and re-recorded. For record i, stream A
 * stores i + 1 into a device counter, records ev[i % K]; stream B waits on it and counts a read of
 * the counter below i + 1 as an ordering error. Every 1024 records B records a host-wait "lead"
 * event and the host syncs on it (the mover's rad_event_sync(old)): the queues stay shallow and no
 * stream is synced. Phase 2: the same on a stream that has had a gate (its records are barriers).
 * Every 100,000 records: rad_l0_event_stats (RadEvents alive, records reusing the queue's last
 * event, barrier / tag records) and the process's resident memory.
 *
 *   xe_event_stress [N=1000000] [K=64] [timed]     RADIANCE_L0_DEVICE=cpu for the CPU device
 * `timed` asks for an elapsed time first, which on the card switches records to profiling tags
 * (bounded per stream between host syncs). Exit 0 when no ordering error, the RadEvents alive never
 * change, records == reused + barriers + tags, resident memory and the card's free memory at the end
 * are within 64 MiB of the sample after 100,000 records, and no stream holds more tags than the
 * bound.
 */
#include "device/device.h"
#include "rad_device.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" void rad_l0_event_stats(int64_t out[6]);

namespace {

using aref = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                              sycl::access::address_space::global_space>;

double rss_mib() {
    long pages = 0, res = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (f) { if (std::fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0; std::fclose(f); }
    return (double)res * 4096.0 / (1 << 20);
}

struct Sample { long long i; int64_t st[6]; double rss; double free_mib; };

/* The card's free memory (sysman: ZES_ENABLE_SYSMAN=1), -1 where the device cannot say. */
double free_mib(sycl::queue& q) {
    const sycl::device d = q.get_device();
    if (!d.has(sycl::aspect::ext_intel_free_memory)) return -1.0;
    return (double)d.get_info<sycl::ext::intel::info::device::free_memory>() / (1 << 20);
}

}  // namespace

int main(int argc, char** argv) {
    const long long N = argc > 1 ? std::atoll(argv[1]) : 1000000;
    const int K = argc > 2 ? std::atoi(argv[2]) : 64;
    if (rad_dev_count() <= 0 || rad_dev_set(0) != RAD_OK) { std::printf("event_stress: no device\n"); return 77; }
    RadStream a = nullptr, b = nullptr, c = nullptr;
    if (rad_stream_create(&a, 0) || rad_stream_create(&b, 0) || rad_stream_create(&c, 0)) {
        std::printf("event_stress: no stream\n");
        return 1;
    }
    sycl::queue& qa = *(sycl::queue*)a;
    sycl::queue& qb = *(sycl::queue*)b;
    sycl::queue& qc = *(sycl::queue*)c;
    uint64_t* d = (uint64_t*)rad_dev_alloc(4096, RAD_MEM_DEVICE);   /* [0] counter A, [1] errors, [2] counter C */
    if (!d) { std::printf("event_stress: no memory\n"); return 1; }
    rad_memset_async(d, 0, 4096, a);
    rad_stream_sync(a);

    std::vector<RadEvent> ev((size_t)K);
    for (RadEvent& e : ev) if (rad_event_create_local(&e) != RAD_OK) { std::printf("event_stress: no event\n"); return 1; }
    RadEvent lead = nullptr;
    if (rad_event_create(&lead) != RAD_OK) { std::printf("event_stress: no lead event\n"); return 1; }

    long long records = 0;
    int fails = 0;
    std::vector<Sample> samples;
    auto sample = [&](long long i) {
        Sample s{i, {}, rss_mib(), free_mib(qa)};
        rad_l0_event_stats(s.st);
        samples.push_back(s);
        std::printf("  %8lld records: RadEvents alive %lld, reused %lld, barriers %lld, tags %lld, untimed %lld, "
                    "max tags/stream %lld, RSS %.1f MiB, device free %.1f MiB\n", (long long)records,
                    (long long)s.st[0], (long long)s.st[1], (long long)s.st[2], (long long)s.st[3],
                    (long long)s.st[4], (long long)s.st[5], s.rss, s.free_mib);
        std::fflush(stdout);
    };
    /* `timed`: ask for a time first, so the card records profiling tags (bounded a stream) */
    const bool timed = argc > 3 && std::string(argv[3]) == "timed";
    if (timed) {
        float t0 = 0;
        rad_event_record(ev[0], a);
        rad_event_record(ev[1], a);
        rad_stream_sync(a);
        records += 2;
        const int trc = rad_event_elapsed_ms(ev[0], ev[1], &t0);
        std::printf("  timed: first elapsed rc %d (an untimed record turns timing on)\n", trc);
    }
    sample(0);
    auto step = [&](sycl::queue& qp, RadStream sp, sycl::queue& qw, RadStream sw, int slot, long long i) {
        const uint64_t v = (uint64_t)i + 1;
        qp.single_task([=] { aref(d[slot]).store(v); });
        RadEvent e = ev[(size_t)(i % K)];
        if (rad_event_record(e, sp) != RAD_OK) return false;
        ++records;
        if (rad_event_wait(sw, e) != RAD_OK) return false;
        qw.single_task([=] { if (aref(d[slot]).load() < v) aref(d[1]).fetch_add(1); });
        if (i % 1024 == 1023) {
            if (rad_event_record(lead, sw) != RAD_OK) return false;
            ++records;
            if (rad_event_sync(lead) != RAD_OK) return false;
        }
        return true;
    };

    std::printf("event_stress: %lld records on a ring of %d events, no stream sync\n", N, K);
    for (long long i = 0; i < N; ++i) {
        if (!step(qa, a, qb, b, 0, i)) { std::printf("  record %lld failed\n", i); ++fails; break; }
        if (i % 100000 == 99999) sample(i + 1);
    }
    /* phase 2: a stream that has had a gate (opened at once) records with barriers */
    void* g = nullptr;
    if (rad_stream_gate(c, &g) != RAD_OK || rad_gate_open(g) != RAD_OK) { std::printf("  gate failed\n"); ++fails; }
    const long long N2 = N / 20;
    for (long long j = 0; j < N2; ++j)
        if (!step(qc, c, qa, a, 2, j)) { std::printf("  gated record %lld failed\n", j); ++fails; break; }
    sample(N + N2);

    rad_stream_sync(a);
    rad_stream_sync(b);
    rad_stream_sync(c);
    std::vector<uint64_t> h(512, 0);
    rad_memcpy_async(h.data(), d, 4096, a);
    rad_stream_sync(a);
    /* canary: the kernels write words 0-2 only; the rest of the zeroed 4 KiB must stay zero */
    int canary_bad = 0;
    for (size_t w = 3; w < h.size(); ++w) canary_bad += h[w] != 0;
    std::printf("  counter A %llu (want %lld), counter C %llu (want %lld), ordering errors %llu, canary %s\n",
                (unsigned long long)h[0], N, (unsigned long long)h[2], N2, (unsigned long long)h[1],
                canary_bad ? "BROKEN" : "ok");
    if (h[1] != 0 || h[0] != (uint64_t)N || h[2] != (uint64_t)N2 || canary_bad) ++fails;

    /* the baseline is the first sample after 100,000 records (the runtime has warmed up by then) */
    const Sample& f = samples.size() > 1 ? samples[1] : samples.front();
    const Sample& l = samples.back();
    for (const Sample& s : samples) if (s.st[0] != f.st[0]) ++fails;
    if (l.st[1] + l.st[2] + l.st[3] != records) {
        std::printf("  records %lld != reused + barriers + tags %lld\n", records, (long long)(l.st[1] + l.st[2] + l.st[3]));
        ++fails;
    }
    if (l.rss > f.rss + 64.0) { std::printf("  resident memory grew %.1f MiB\n", l.rss - f.rss); ++fails; }
    if (f.free_mib >= 0 && l.free_mib < f.free_mib - 64.0) {
        std::printf("  device free memory fell %.1f MiB\n", f.free_mib - l.free_mib);
        ++fails;
    }
    if (l.st[5] > 32768) { std::printf("  a stream holds %lld tags, past the bound\n", (long long)l.st[5]); ++fails; }

    /* timing still works where records are stamped (the CPU device profiles every command) */
    float ms = -1;
    qa.single_task([=] { aref(d[0]).store(1); });
    rad_event_record(ev[0], a);
    qa.single_task([=] { aref(d[0]).store(2); });
    rad_event_record(ev[1], a);
    rad_stream_sync(a);
    const int erc = rad_event_elapsed_ms(ev[0], ev[1], &ms);
    std::printf("  elapsed between two records: rc %d, %.4f ms\n", erc, ms);

    for (RadEvent e : ev) rad_event_destroy(e);
    rad_event_destroy(lead);
    rad_dev_free(d, RAD_MEM_DEVICE);
    std::printf("event_stress: %s (%lld records)\n", fails ? "FAIL" : "PASS", records);
    return fails ? 1 : 0;
}
