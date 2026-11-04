/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 2).
 *
 * xe_allreduce.cpp -- all_reduce for exactly two ranks on the Level Zero backend, libr4d's
 * one-shot design (r4d_ar_oneshot_2rank_exact) on SYCL:
 *
 *   xe_ar_oneshot_2rank_exact       peer stores: each rank writes its payload into the PEER's device
 *                                   scratch (both cards share one context, l0.cpp discover; peer
 *                                   access enabled by the engine). Round 2 M2b: 24 us an exchange.
 *   xe_ar_oneshot_2rank_exact_host  the same through a pinned host-memory mailbox (no peer access
 *                                   needed; the CPU device and a card pair without P2P). 79 us.
 *
 * One launch per all-reduce and rank. Work-group b owns a contiguous chunk of the message: it
 * copies its chunk of x into the peer's slot, publishes flag[b] = seq with a system-scope release,
 * waits (bounded) for the peer's flag[b] >= seq, then writes dst = f32(rank 0's chunk) + f32(rank
 * 1's chunk) in ascending rank order, so both ranks hold the same bits (the host backend's rule).
 * Slots are double-buffered by the launch's parity: all-reduce k+2 reuses k's slot, and it cannot
 * start before k+1 completed, which needed the peer's k+1 flag, set after the peer finished reading
 * slot k. The sequence number is a device-resident counter the kernel advances (one per instance),
 * so a recorded pass replays the same launch and stays in lockstep with the peer: graph-safe.
 *
 * An instance (init) is one per (rank, world, dtype, message bound): scratch for the bound in each
 * slot, one flag word per block per slot, the seq counter, allocated on the rank's device through
 * the backend (rad_l0k_device) -- or in pinned host memory for the _host form -- and exchanged with
 * the peer through a process-wide rendezvous (one process, one thread per rank, as libref's). A
 * stalled handshake ends after kSpin polls with err set; the launch's next call reports it.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "r4d_plugin.h"

#include <sycl/ext/intel/experimental/cache_control_properties.hpp>

#include <atomic>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

extern "C" int rad_l0k_device(int index, const sycl::device** dev, const sycl::context** ctx);

namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int      kWG = 256;            /* work-items a block */
constexpr int      kMaxBlocks = 512;       /* flag words a slot */
constexpr int      kResidentBlocks = 32;   /* the launch's block cap (see launch); 32 since run 20: two launches
                                            * overlapping on a card (M12: both engines spinning) must both fit */
constexpr uint32_t kSpin = 400000000u;   /* bounded wait: minutes at ~0.3 us an uncached poll (XE_AR_SPIN lowers it), never a hung card */
constexpr int      kMaxRanks = 2;

/* uncached stores and loads for flags and payload (M2a/M2b's working mailbox) */
using uc_u32 = syclex::annotated_ptr<uint32_t, decltype(syclex::properties(
    intelex::write_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>,
    intelex::read_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>))>;
inline void uc_st(uint32_t* p, uint32_t v) { uc_u32 a(p); a[0] = v; }
inline uint32_t uc_ld(const uint32_t* p) { uc_u32 a((uint32_t*)p); return a[0]; }
/* the same for 8-byte words (run 102's LL form: a payload word and its sequence number in one aligned store) */
using uc_u64 = syclex::annotated_ptr<uint64_t, decltype(syclex::properties(
    intelex::write_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>,
    intelex::read_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>))>;
inline void uc_st64(uint64_t* p, uint64_t v) { uc_u64 a(p); a[0] = v; }
inline uint64_t uc_ld64(const uint64_t* p) { uc_u64 a((uint64_t*)p); return a[0]; }
/* and for two of them in one aligned 16-byte store (run 161's LL form, RADIANCE_XE_AR_PROTO=5: PEER2 measured 16-byte
 * tagged stores at ~28 % under the 8-byte form's exchange at the decode sizes, and anything wider a slow path) */
using v2u64 = sycl::vec<uint64_t, 2>;
using uc_u128 = syclex::annotated_ptr<v2u64, decltype(syclex::properties(
    intelex::write_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>,
    intelex::read_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>))>;
inline void uc_st128(uint64_t* p, v2u64 v) { uc_u128 a((v2u64*)p); a[0] = v; }
inline v2u64 uc_ld128(const uint64_t* p) { uc_u128 a((v2u64*)p); return a[0]; }
#ifdef __SYCL_DEVICE_ONLY__
#define XE_AR_CONSTANT __attribute__((opencl_constant))
#else
#define XE_AR_CONSTANT
#endif
/* A block that exhausted its budget: counted in seq_ctr[3], the first two blocks of the launch say so
 * from the device (run 18: a stall was silent, and its skipped arrival desynchronized every later
 * launch of the rank). */
inline void ar_gave_up(uint32_t* seq_ctr, uint32_t rank, uint32_t s, uint32_t b, uint32_t nblk, uint32_t t) {
    if (t != 0) return;
    uc_st(seq_ctr + 1, s);
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> stalls(seq_ctr[3]);
    const uint32_t n = stalls.fetch_add(1u) + 1u;
    if (b < 2u) {
        static const XE_AR_CONSTANT char fmt[] = "W xe all-reduce: rank %u gave up waiting for the peer at sequence %u (block %u of %u; stalled blocks so far %u)\n";
        sycl::ext::oneapi::experimental::printf(fmt, rank, s, b, nblk, n);
    }
}

/* RADIANCE_XE_AR_SKEW (run 122): block 0's failed polls for this launch into the ring (a kernel argument, null when
 * off). One thread, one store a launch. (m3 fault 35: the first form loaded the ring's address from the counter block
 * -- an indirect access, so under NEO's DisableIndirectAccess=1 the ring was not resident and its first store
 * page-faulted on the card; the CPU device has no residency.) */
constexpr uint32_t kSkewRing = 1u << 20;
/* the entry: s << 32 | total << 16 | first -- `first` the polls before the peer's first word (the arrival skew plus
 * the link's latency), `total` thread 0's polls over all its words (the payload's streaming too); each saturates at
 * 65535 (run 123: M207's first-word record alone could not tell the skew from the transfer) */
inline void ar_skew(uint64_t* ring, uint32_t s, uint32_t first, uint32_t total) {
    if (!ring) return;
    first = first < 65535u ? first : 65535u;
    total = total < 65535u ? total : 65535u;
    ring[s & (kSkewRing - 1u)] = ((uint64_t)s << 32) | (total << 16) | first;
}

struct Inst {
    int       rank = 0, world = 2, dtype = 0;   /* dtype: 0 bf16, 1 fp16, 2 fp32 */
    int       host = 0;
    int64_t   numel = 0;                       /* the message bound, in elements */
    int64_t   slot_bytes = 0;                  /* one slot: numel * element bytes, rounded */
    uint8_t*  my_scratch = nullptr;            /* 2 slots */
    uint32_t* my_flags = nullptr;              /* 2 * kMaxBlocks words */
    uint32_t* my_seq = nullptr;                /* [0] the launch counter, [1] the stall flag */
    uint8_t*  peer_scratch = nullptr;
    uint32_t* peer_flags = nullptr;
    const sycl::device*  dev = nullptr;
    const sycl::context* ctx = nullptr;
    int refs = 0;
    /* RADIANCE_XE_AR_SKEW=<dir> (owner note 151 A, run 122): the wait record -- block 0's polls before the peer's
     * data arrived, one (sequence << 32 | polls) a launch in a device ring (a kernel argument), appended to
     * <dir>/ar-skew-r<rank>-dt<dtype>-n<numel>.bin by xe_ar2_stalls, which the engine calls once a step: records of
     * (call count, entry), 16 bytes. */
    uint64_t* ring = nullptr;
    uint32_t  ring_mask = 0, last_s = 0;
    uint64_t  calls = 0;
    FILE*     skew_f = nullptr;
};
const char* skew_dir() {
    static const char* v = [] { const char* e = std::getenv("RADIANCE_XE_AR_SKEW"); return e && *e ? e : nullptr; }();
    return v;
}

/* the rendezvous: one process, the rank threads publish their instance pointers and pair up */
struct World {
    std::mutex m;
    std::condition_variable cv;
    Inst* in[kMaxRanks] = { nullptr, nullptr };
    int generation = 0;
};
World& world_for(int key) {
    static World w[8];
    return w[key & 7];
}

std::mutex& cache_mutex() { static std::mutex m; return m; }
std::vector<Inst*>& cache() { static std::vector<Inst*> v; return v; }

int dtype_of(const char* s) {
    if (!s) return -1;
    if (!std::strcmp(s, "bf16")) return 0;
    if (!std::strcmp(s, "fp16") || !std::strcmp(s, "f16")) return 1;
    if (!std::strcmp(s, "fp32") || !std::strcmp(s, "f32") || !std::strcmp(s, "i32")) return 2;
    return -1;
}
int64_t elem_bytes(int dt) { return dt == 2 ? 4 : 2; }

template <class T> T* alloc(const Inst& in, size_t n) {
    if (in.host) return sycl::aligned_alloc_host<T>(4096, n, *in.ctx);
    return sycl::aligned_alloc_device<T>(4096, n, *in.dev, *in.ctx);
}

/* The LL form (see ar_kernel) is the default since run 105 (owner note 121, a correctness fix: the flag-and-payload
 * form's large messages raced, and served answers changed from load to load); RADIANCE_XE_AR_PROTO=0 keeps the old
 * form as an opt-out. (Forms 1-3, the old handshake with lighter fences, were measured in AB1 -- no gain, and the
 * same large-message divergence -- and removed.) */
int ar_proto() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_XE_AR_PROTO");
        /* 5, the LL form with 16-byte stores, was the default from run 161 (owner note 163 item 2: 25.1 -> 20.2 us a
         * launch, the rotation +1.6 % median at the same texts); =4 the 8-byte form, =0 the old flag form;
         * 6 (run 170, owner note 172 d) is the default since run 174 (owner note 176 item 1): the LL form 5 below
         * ar_bw_min_bytes() and the bandwidth form (ar_kernel's PR 6: the payload once, a read-back before the
         * flag) at or above it; =5 keeps the LL form for every size */
        return !e ? 6 : (e[0] == '0' ? 0 : (e[0] == '4' ? 4 : (e[0] == '5' ? 5 : 6)));
    }();
    return v;
}
/* RADIANCE_XE_AR_BW_MIN_KB (run 170): the message size from which proto 6 takes the bandwidth form; 256 KiB by
 * default (decode's 2-20 KB messages keep the LL form's latency; a 4,096-row chunk's 20 MB takes the bandwidth form) */
int64_t ar_bw_min_bytes() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_XE_AR_BW_MIN_KB");
        const long n = e ? std::strtol(e, nullptr, 10) : 256L;
        return (int64_t)(n > 0 ? n : 256L) * 1024;
    }();
    return v;
}
/* XE_AR_TEST_TEAR=N (the all-reduce test only): under proto 6 every block's thread 0 holds its first quad back
 * for N polls after the control words, so the reader's check must fail first and the re-read path is exercised;
 * the result must still be right (allreduce_test prints the re-reads it caused). 0 outside the test. */
uint32_t ar_test_tear() {
    static const uint32_t v = [] {
        const char* e = std::getenv("XE_AR_TEST_TEAR");
        const long n = e ? std::strtol(e, nullptr, 10) : 0L;
        return (uint32_t)(n > 0 ? (n < 1000000L ? n : 1000000L) : 0L);
    }();
    return v;
}

int build(Inst* in) {
    const int64_t eb = elem_bytes(in->dtype);
    in->slot_bytes = ((in->numel * eb + 4095) / 4096) * 4096;
    /* the LL form carries a sequence number with every payload word: twice the bytes, so every message up to the
     * bound takes it (the shipped form's large-message divergence is what it removes) */
    if (ar_proto() >= 4) in->slot_bytes *= 2;
    in->my_scratch = alloc<uint8_t>(*in, (size_t)(2 * in->slot_bytes));
    in->my_flags = alloc<uint32_t>(*in, 2 * kMaxBlocks);
    in->my_seq = sycl::aligned_alloc_device<uint32_t>(4096, 1024, *in->dev, *in->ctx);
    if (!in->my_scratch || !in->my_flags || !in->my_seq) return RAD_E_NOMEM;
    /* zero the flags and the counter before anyone polls them */
    sycl::queue q(*in->ctx, *in->dev, sycl::property::queue::in_order());
    if (in->host) std::memset(in->my_flags, 0, 2 * kMaxBlocks * sizeof(uint32_t));
    else q.memset(in->my_flags, 0, 2 * kMaxBlocks * sizeof(uint32_t));
    q.memset(in->my_seq, 0, 1024 * sizeof(uint32_t));
    if (skew_dir() && !in->host) {
        in->ring = sycl::aligned_alloc_device<uint64_t>(4096, kSkewRing, *in->dev, *in->ctx);
        if (!in->ring) return RAD_E_NOMEM;
        in->ring_mask = kSkewRing - 1u;
        q.memset(in->ring, 0, kSkewRing * sizeof(uint64_t));
        char path[512];
        std::snprintf(path, sizeof path, "%s/ar-skew-r%d-dt%d-n%lld.bin", skew_dir(), in->rank, in->dtype, (long long)in->numel);
        in->skew_f = std::fopen(path, "wb");
    }
    q.wait_and_throw();
    /* rendezvous: publish, wait for the peer, take its pointers */
    World& w = world_for(in->dtype * 2 + in->host);
    std::unique_lock<std::mutex> lk(w.m);
    if (w.in[in->rank]) return RAD_E_STATE;   /* a second init for this rank before the pair finished */
    w.in[in->rank] = in;
    w.cv.notify_all();
    const int peer = in->rank ^ 1;
    if (!w.cv.wait_for(lk, std::chrono::seconds(120), [&] { return w.in[peer] != nullptr && w.in[peer]->numel == in->numel; }))
        return RAD_E_DEVICE;   /* no peer init within 2 minutes: a lone rank, or unequal bounds */
    in->peer_scratch = w.in[peer]->my_scratch;
    in->peer_flags = w.in[peer]->my_flags;
    /* both ranks have read the other's pointers once both have passed here: the second to arrive
     * clears the table for the next instance and wakes the first (the first run of this code forgot
     * the wake-up and every first arriver slept out its timeout: run 5's hung CPU test) */
    const int gen = ++w.generation;
    if (gen % 2 == 0) { w.in[0] = w.in[1] = nullptr; w.cv.notify_all(); return RAD_OK; }
    if (!w.cv.wait_for(lk, std::chrono::seconds(120), [&] { return w.generation >= gen + 1; })) return RAD_E_DEVICE;
    return RAD_OK;
}

int ar_init(const RadParam* p, int n_p, int rank, int world, int host, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (world != 2 || rank < 0 || rank > 1) return RAD_E_SHAPE;
    const long long ws = rad_param_geti(p, n_p, "world_size", world);
    if (ws != world) return RAD_E_SHAPE;
    /* the message bound: the ranged `numel` collapsed to its ceiling at init (all_reduce,
     * all_gather), or M's ceiling times n (ar_hc_write, which declares M and n instead) */
    long long numel = rad_param_getdim(p, n_p, "numel", 0);
    if (numel <= 0) numel = rad_param_getdim(p, n_p, "M", 0) * rad_param_getdim(p, n_p, "n", 0);
    if (numel <= 0) return RAD_E_INVAL;
    const int dt = dtype_of(rad_param_gets(p, n_p, "dtype", nullptr));
    if (dt < 0) return RAD_E_DTYPE;
    {
        std::lock_guard<std::mutex> lk(cache_mutex());
        for (Inst* c : cache())
            if (c->rank == rank && c->world == world && c->dtype == dt && c->numel == numel && c->host == host) {
                ++c->refs;
                *out = c;
                return RAD_OK;
            }
    }
    Inst* in = new (std::nothrow) Inst();
    if (!in) return RAD_E_NOMEM;
    in->rank = rank; in->world = world; in->dtype = dt; in->host = host; in->numel = numel; in->refs = 1;
    if (rad_l0k_device(rank, &in->dev, &in->ctx) != 0 || !in->dev || !in->ctx) { delete in; return RAD_E_DEVICE; }
    int rc;
    try { rc = build(in); } catch (const sycl::exception&) { rc = RAD_E_DEVICE; }
    if (rc != RAD_OK) { delete in; return rc; }
    std::lock_guard<std::mutex> lk(cache_mutex());
    cache().push_back(in);
    *out = in;
    return RAD_OK;
}

void ar_fini(void* instance) {
    Inst* in = (Inst*)instance;
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(cache_mutex());
        if (--in->refs > 0) return;
        for (size_t i = 0; i < cache().size(); ++i) if (cache()[i] == in) { cache().erase(cache().begin() + (long)i); break; }
    }
    try {
        if (in->my_scratch) sycl::free(in->my_scratch, *in->ctx);
        if (in->my_flags) sycl::free(in->my_flags, *in->ctx);
        if (in->my_seq) sycl::free(in->my_seq, *in->ctx);
        if (in->ring) sycl::free(in->ring, *in->ctx);
    } catch (const sycl::exception&) {}
    if (in->skew_f) std::fclose(in->skew_f);
    delete in;
}

/* the sum of two elements in f32, ascending rank order: r0 + r1 */
template <int DT> inline float to_f(uint32_t bits16or32);
template <> inline float to_f<0>(uint32_t b) { return __uint_as_float(b << 16); }
template <> inline float to_f<1>(uint32_t b) { return (float)sycl::bit_cast<sycl::half>((uint16_t)b); }
template <> inline float to_f<2>(uint32_t b) { return __uint_as_float(b); }
template <int DT> inline uint32_t from_f(float f);
template <> inline uint32_t from_f<0>(float f) { const uint32_t u = __float_as_uint(f); return ((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u) : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <> inline uint32_t from_f<1>(float f) { return (uint32_t)sycl::bit_cast<uint16_t>((sycl::half)f); }
template <> inline uint32_t from_f<2>(float f) { return __float_as_uint(f); }

/* Run 99 (owner note 105 item 4, RADIANCE_XE_AR_HC_FUSE=1): ar_hc_write's residual write inside the bf16 all-reduce,
 * as libr4d's ar_hc_write does it -- h[r][c][i] += inj[r][c] * y[r][i] on the two elements of each reduced word, with
 * hc_write_kernel's arithmetic (bf16 operands to f32, one product and one add, narrowed by libr4d's f2bf: round half
 * away, u + 0x8000), so the same bytes as the all-reduce then hc_write, one launch fewer. A word never straddles a
 * row (n is even). */
struct HcW {
    const uint16_t* inj = nullptr;   /* [M][irow] bf16 gains; null: no write */
    uint16_t* h = nullptr;           /* [M][hrow] bf16 streams, hc runs of n */
    uint32_t n = 0, hc = 0, irow = 0, hrow = 0;
};
inline float hcw_bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
inline uint16_t hcw_f2bf(float f) { return (uint16_t)((__float_as_uint(f) + 0x8000u) >> 16); }

/* One all-reduce: words of 32 bits (two bf16/fp16 or one f32); `n32` words in the message. */
template <int DT, bool HW = false, int PR = 0>
void ar_kernel(const uint32_t* __restrict__ x, uint32_t* __restrict__ y, uint32_t* peer_slots, uint32_t* my_slots,
               uint32_t* peer_flags, uint32_t* my_flags, uint32_t* seq_ctr, uint32_t n32, uint32_t slot_words,
               uint32_t rank, uint32_t nblk, uint32_t spin, HcW hw, uint64_t* ring, uint32_t tear) {
    const uint32_t b = blockIdx.x, t = threadIdx.x;
    /* the launch's sequence number: every block reads the same counter; block 0 advances it last */
    const uint32_t s = uc_ld(seq_ctr) + 1u;
    const uint32_t par = s & 1u;
    uint32_t* out = peer_slots + par * slot_words;
    const uint32_t* in = my_slots + par * slot_words;
    const uint32_t per0 = (n32 + nblk - 1) / nblk;
    const uint32_t per = PR == 5 ? ((per0 + 1u) & ~1u)          /* PR 5: a block's words in pairs, 16-byte aligned */
                       : PR == 6 ? ((per0 + 3u) & ~3u) : per0;  /* PR 6: in quads, a block's start 16-byte aligned */
    const uint32_t w0 = b * per, w1 = (w0 + per < n32) ? w0 + per : n32;
    /* the sum of one word, rank 0's first, into y (and, with HW, hc_write's update of the streams) */
    auto emit = [&](uint32_t k, uint32_t mine, uint32_t theirs) {
        const uint32_t r0 = rank == 0 ? mine : theirs, r1 = rank == 0 ? theirs : mine;
        uint32_t o;
        if constexpr (DT == 2) o = from_f<2>(to_f<2>(r0) + to_f<2>(r1));
        else {
            const uint32_t lo = from_f<DT>(to_f<DT>(r0 & 0xffffu) + to_f<DT>(r1 & 0xffffu));
            const uint32_t hi = from_f<DT>(to_f<DT>(r0 >> 16) + to_f<DT>(r1 >> 16));
            o = lo | (hi << 16);
        }
        y[k] = o;
        if constexpr (HW && DT == 0) {
            const uint32_t j0 = 2u * k, r = j0 / hw.n, i = j0 - r * hw.n;
            const float v0 = hcw_bf2f(o & 0xffffu), v1 = hcw_bf2f(o >> 16);
            const uint16_t* gr = hw.inj + (size_t)r * hw.irow;
            uint16_t* hp = hw.h + (size_t)r * hw.hrow + i;
            for (uint32_t c = 0; c < hw.hc; ++c) {
                const float g = hcw_bf2f(gr[c]);
                uint16_t* p = hp + (size_t)c * hw.n;
                p[0] = hcw_f2bf(hcw_bf2f(p[0]) + g * v0);
                p[1] = hcw_f2bf(hcw_bf2f(p[1]) + g * v1);
            }
        }
    };
    if constexpr (PR == 5) {
        /* RADIANCE_XE_AR_PROTO=5 (run 161, owner note 163 item 2): the LL form below with 16-byte stores -- a lane
         * carries two tagged words in one aligned store (PEER2: ~28 % off the 8-byte form's exchange at the decode
         * sizes; 32 and 64-byte stores were a slow path). The tags are still checked a word, so a store that lands
         * in pieces is still read right. `per` is even: a pair never straddles a block, and the slots are
         * page-aligned, so every pair's store is 16-byte aligned. An odd last word of the message goes alone. */
        uint64_t* out64 = (uint64_t*)peer_slots + par * (slot_words / 2);
        const uint64_t* in64 = (const uint64_t*)my_slots + par * (slot_words / 2);
        for (uint32_t k = w0 + 2u * t; k < w1; k += 2u * kWG) {
            if (k + 1u < w1) uc_st128(out64 + k, v2u64{((uint64_t)s << 32) | x[k], ((uint64_t)s << 32) | x[k + 1u]});
            else uc_st64(out64 + k, ((uint64_t)s << 32) | x[k]);
        }
        uint32_t ok = 1, polls0 = 0, polls_all = 0;
        bool first = true;
        for (uint32_t k = w0 + 2u * t; k < w1 && ok; k += 2u * kWG) {
            uint32_t budget = spin;
            const bool pair = k + 1u < w1;
            v2u64 v;
            for (;;) {
                if (pair) v = uc_ld128(in64 + k);
                else { v[0] = uc_ld64(in64 + k); v[1] = v[0]; }
                if ((uint32_t)(v[0] >> 32) == s && (uint32_t)(v[1] >> 32) == s) break;
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                if (--budget == 0) { ok = 0; break; }
            }
            if (first) { polls0 = spin - budget; first = false; }
            polls_all += spin - budget;
            if (ok) { emit(k, x[k], (uint32_t)v[0]); if (pair) emit(k + 1u, x[k + 1u], (uint32_t)v[1]); }
        }
        if (b == 0 && t == 0) ar_skew(ring, s, polls0, polls_all);
        if (!ok) ar_gave_up(seq_ctr, rank, s, b, nblk, t);
    } else if constexpr (PR == 4) {
        /* RADIANCE_XE_AR_PROTO=4 (run 102): the LL form (NCCL's low-latency protocol). Each payload word goes to the
         * peer with this launch's sequence number in the same aligned 8-byte store, and the reader polls each word
         * until its number matches -- no flag word, no system-scope fence, nothing assumed about the order in which
         * the peer's stores land (the shipped form's flag can arrive before its payload: the large-message rank
         * divergence of X11-X16 and AB1-AB2). Twice the slot bytes; the launch takes this form only when 2 x n32
         * words fit a parity's slot. Stale words carry s - 2 (the other parity) or 0 (never written): s >= 1. */
        uint64_t* out64 = (uint64_t*)peer_slots + par * (slot_words / 2);
        const uint64_t* in64 = (const uint64_t*)my_slots + par * (slot_words / 2);
        for (uint32_t k = w0 + t; k < w1; k += kWG) uc_st64(out64 + k, ((uint64_t)s << 32) | x[k]);
        uint32_t ok = 1, polls0 = 0, polls_all = 0;
        bool first = true;
        for (uint32_t k = w0 + t; k < w1 && ok; k += kWG) {
            uint32_t budget = spin;
            uint64_t v;
            for (;;) {
                v = uc_ld64(in64 + k);
                if ((uint32_t)(v >> 32) == s) break;
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                if (--budget == 0) { ok = 0; break; }
            }
            if (first) { polls0 = spin - budget; first = false; }
            polls_all += spin - budget;
            if (ok) emit(k, x[k], (uint32_t)v);
        }
        if (b == 0 && t == 0) ar_skew(ring, s, polls0, polls_all);
        if (!ok) ar_gave_up(seq_ctr, rank, s, b, nblk, t);
    } else if constexpr (PR == 6) {
        /* RADIANCE_XE_AR_PROTO=6 (run 170, owner note 168 item 3 / note 172 d): the bandwidth form for prefill's
         * large messages -- "flag + checksum". The payload crosses ONCE, untagged, in 16-byte uncached stores (a
         * sub-group's 256 contiguous bytes an instruction, PR 5's width): 1x the bytes where the LL forms carry 2x.
         * Nothing is assumed about the order the stores land in (the old flag form's flag overtook its payload on
         * large messages, AB1 / AB2; one thread's read-back behind the block's stores did not fence them either,
         * AR1): instead the block's 32-bit wrapping sum and 32-bit xor travel as two 8-byte TAGGED words (the LL
         * form's proven atom: the launch's sequence number in the same aligned store), at the tail of the parity's
         * slot. The reader polls both tags, then reads the block from its OWN slot (the peer's stores land in the
         * reader's memory, so these reads are local), reduces the sum and xor, and only when both match reads the
         * block once more and emits; a stale or partially landed block fails the 64-bit check (2^-64 a block) and
         * is read again after the next poll, bounded like every wait here. One emit pass, as the fused hc write
         * accumulates. Exact: emit is the same sum in the same order. */
        /* The check is position-dependent (the supervisor's review, run 170): three 32-bit wrapping sums over the
         * block's words w_i (i the word's index in the block) -- c1 = sum w_i, c2 = sum (i + 1) * w_i, c3 = sum
         * rotl(w_i, i mod 32) -- so two words exchanging places, or a stale word and a fresh one whose difference
         * cancels in a plain sum, fail it; each sum reduces across the block in any order. */
        XE_SHARED(uint32_t, red_a, kWG);
        XE_SHARED(uint32_t, red_b, kWG);
        XE_SHARED(uint32_t, red_c, kWG);
        /* the control words: the last 3 x kMaxBlocks 8-byte words of the parity's slot (the LL sizing doubles the
         * slot, and the bandwidth form takes messages of 256 KiB or more, so the payload never reaches them) */
        uint64_t* ctl_out = (uint64_t*)out + (slot_words / 2) - 3u * kMaxBlocks + 3u * b;
        const uint64_t* ctl_in = (const uint64_t*)in + (slot_words / 2) - 3u * kMaxBlocks + 3u * b;
        auto rotl = [](uint32_t v, uint32_t r) { r &= 31u; return r ? ((v << r) | (v >> (32u - r))) : v; };
        uint32_t c1 = 0, c2 = 0, c3 = 0;
        /* XE_AR_TEST_TEAR (the test's forced tear): thread 0 holds its first quad back until after the control
         * words, so the reader must see a mismatch first; never set outside the test */
        const bool hold = tear != 0u && t == 0u && w0 + 3u < w1;
        for (uint32_t k = w0 + 4u * t; k < w1; k += 4u * kWG) {
            if (k + 3u < w1) {
                const uint32_t a0 = x[k], a1 = x[k + 1u], a2 = x[k + 2u], a3 = x[k + 3u];
                if (!(hold && k == w0))
                    uc_st128((uint64_t*)(out + k), v2u64{(uint64_t)a0 | ((uint64_t)a1 << 32), (uint64_t)a2 | ((uint64_t)a3 << 32)});
                const uint32_t i = k - w0;
                c1 += a0 + a1 + a2 + a3;
                c2 += (i + 1u) * a0 + (i + 2u) * a1 + (i + 3u) * a2 + (i + 4u) * a3;
                c3 += rotl(a0, i) + rotl(a1, i + 1u) + rotl(a2, i + 2u) + rotl(a3, i + 3u);
            } else {
                for (uint32_t j = k; j < w1; ++j) {
                    uc_st(out + j, x[j]);
                    const uint32_t i = j - w0;
                    c1 += x[j]; c2 += (i + 1u) * x[j]; c3 += rotl(x[j], i);
                }
            }
        }
        red_a[t] = c1;
        red_b[t] = c2;
        red_c[t] = c3;
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        __syncthreads();
        if (t == 0) {
            uint32_t A = 0, B = 0, C = 0;
            for (uint32_t i = 0; i < (uint32_t)kWG; ++i) { A += red_a[i]; B += red_b[i]; C += red_c[i]; }
            uc_st64(ctl_out, ((uint64_t)s << 32) | A);
            uc_st64(ctl_out + 1, ((uint64_t)s << 32) | B);
            uc_st64(ctl_out + 2, ((uint64_t)s << 32) | C);
            if (hold) {
                for (uint32_t d = 0; d < tear; ++d) sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                uc_st128((uint64_t*)(out + w0), v2u64{(uint64_t)x[w0] | ((uint64_t)x[w0 + 1u] << 32),
                                                      (uint64_t)x[w0 + 2u] | ((uint64_t)x[w0 + 3u] << 32)});
            }
        }
        /* the poll: thread 0 waits for the three control words of this launch, then hands the wanted sums on */
        uint32_t budget = spin, ok = 1;
        if (t == 0) {
            for (;;) {
                const uint64_t v0 = uc_ld64(ctl_in), v1 = uc_ld64(ctl_in + 1), v2 = uc_ld64(ctl_in + 2);
                if ((uint32_t)(v0 >> 32) == s && (uint32_t)(v1 >> 32) == s && (uint32_t)(v2 >> 32) == s) {
                    red_a[0] = (uint32_t)v0; red_b[0] = (uint32_t)v1; red_c[0] = (uint32_t)v2;
                    break;
                }
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                if (--budget == 0) { ok = 0; break; }
            }
            red_a[1] = ok;
        }
        __syncthreads();
        const uint32_t A_want = red_a[0], B_want = red_b[0], C_want = red_c[0];
        ok = red_a[1];
        __syncthreads();
        if (b == 0 && t == 0) ar_skew(ring, s, spin - budget, spin - budget);
        /* the check: the block's words from this rank's own slot, until the three sums are the writer's */
        uint32_t tries = 0;
        while (ok) {
            c1 = 0; c2 = 0; c3 = 0;
            for (uint32_t k = w0 + 4u * t; k < w1; k += 4u * kWG) {
                if (k + 3u < w1) {
                    const v2u64 v = uc_ld128((const uint64_t*)(in + k));
                    const uint32_t b0 = (uint32_t)v[0], b1 = (uint32_t)(v[0] >> 32), b2 = (uint32_t)v[1], b3 = (uint32_t)(v[1] >> 32);
                    const uint32_t i = k - w0;
                    c1 += b0 + b1 + b2 + b3;
                    c2 += (i + 1u) * b0 + (i + 2u) * b1 + (i + 3u) * b2 + (i + 4u) * b3;
                    c3 += rotl(b0, i) + rotl(b1, i + 1u) + rotl(b2, i + 2u) + rotl(b3, i + 3u);
                } else {
                    for (uint32_t j = k; j < w1; ++j) {
                        const uint32_t v = uc_ld(in + j), i = j - w0;
                        c1 += v; c2 += (i + 1u) * v; c3 += rotl(v, i);
                    }
                }
            }
            red_a[t] = c1;
            red_b[t] = c2;
            red_c[t] = c3;
            __syncthreads();
            if (t == 0) {
                uint32_t A = 0, B = 0, C = 0;
                for (uint32_t i = 0; i < (uint32_t)kWG; ++i) { A += red_a[i]; B += red_b[i]; C += red_c[i]; }
                red_a[0] = (A == A_want && B == B_want && C == C_want) ? 1u : 0u;
            }
            __syncthreads();
            const uint32_t good = red_a[0];
            __syncthreads();
            if (good) break;
            if (++tries > (spin >> 8)) { ok = 0; break; }   /* bounded: minutes of re-reads, never a hung card */
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        }
        if (t == 0 && tries) {   /* the re-reads before a block passed: seq_ctr[4], read by xe_ar2_retries */
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> rt(seq_ctr[4]);
            rt.fetch_add(tries);
        }
        if (!ok) ar_gave_up(seq_ctr, rank, s, b, nblk, t);
        else {
            for (uint32_t k = w0 + 4u * t; k < w1; k += 4u * kWG) {
                if (k + 3u < w1) {
                    const v2u64 v = uc_ld128((const uint64_t*)(in + k));
                    emit(k, x[k], (uint32_t)v[0]);
                    emit(k + 1u, x[k + 1u], (uint32_t)(v[0] >> 32));
                    emit(k + 2u, x[k + 2u], (uint32_t)v[1]);
                    emit(k + 3u, x[k + 3u], (uint32_t)(v[1] >> 32));
                } else {
                    for (uint32_t j = k; j < w1; ++j) emit(j, x[j], uc_ld(in + j));
                }
            }
        }
    } else {
    for (uint32_t k = w0 + t; k < w1; k += kWG) uc_st(out + k, x[k]);
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    __syncthreads();
    if (t == 0) { uc_st(peer_flags + par * kMaxBlocks + b, s); sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); }
    /* The poll: an uncached load of this rank's own flag word, with a system-scope acquire fence in
     * the loop. The fence keeps the compiler from hoisting the load out of the loop (run 5's CPU test:
     * a plain load was read once and whichever rank polled first always stalled), and the uncached
     * hint keeps the GPU from serving a stale cached line (M2a/M2b's mailbox). Not an atomic load: a
     * system-scope atomic on the host mailbox hung the card pair (run 6, B3's host form). */
    const uint32_t* fp = my_flags + par * kMaxBlocks + b;
    uint32_t budget = spin, ok = 1;
    for (;;) {
        const uint32_t f = uc_ld(fp);
        if ((f - s) <= 0x7fffffffu) break;
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        if (--budget == 0) { ok = 0; break; }
    }
    if (b == 0 && t == 0) ar_skew(ring, s, spin - budget, spin - budget);
    if (!ok) ar_gave_up(seq_ctr, rank, s, b, nblk, t);   /* no return: the block still arrives below */
    else {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        for (uint32_t k = w0 + t; k < w1; k += kWG) emit(k, x[k], uc_ld(in + k));
    }
    }
    /* the counter advances once per launch, after every block has read it: the last block to
     * finish does it (an atomic arrival count in seq_ctr[2]); a block that gave up arrives too, so
     * the next launch's sequence stays in step with the peer's (run 18: the skipped arrival left the
     * counter behind for the rest of the run, a stall at every later collective) */
    __syncthreads();
    if (t == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> arr(seq_ctr[2]);
        if (arr.fetch_add(1u) + 1u == nblk) { arr.store(0u); uc_st(seq_ctr, s); }
    }
}

/* XE_AR_SPIN: the poll budget (default kSpin); a probe lowers it so a stalled pair fails in seconds */
uint32_t spin_budget() {
    static const uint32_t v = [] { const char* e = std::getenv("XE_AR_SPIN"); const long n = e ? std::strtol(e, nullptr, 10) : 0; return n > 0 ? (uint32_t)(n < 0x7fffffffL ? n : 0x7fffffffL) : kSpin; }();
    return v;
}

int launch(const RadArgs* a, RadStream s, int host_form, const HcW* hw = nullptr) {
    Inst* in = (Inst*)a->instance;
    if (!in || in->host != host_form) return R4D_NO(RAD_E_STATE);
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !x->data) return R4D_NO(RAD_E_INVAL);
    if (!y || !y->data) y = x;   /* in place */
    /* the message is the operand; `numel` is the band's ceiling (libref's band_rows): the engine
     * hands the collapsed geometry (numel 10,485,760) with a 38,400-element row slice (T1, run 8) */
    const long long nel = r4d_numel(x);
    if (nel <= 0 || nel > in->numel) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(y) < nel) return R4D_NO(RAD_E_SHAPE);
    const int64_t bytes = nel * elem_bytes(in->dtype);
    if (bytes % 4) return R4D_NO(RAD_E_SHAPE);   /* 32-bit words: an odd bf16 count is not served */
    if (((uintptr_t)x->data | (uintptr_t)y->data) % 4) return R4D_NO(RAD_E_STRIDE);
    const uint32_t n32 = (uint32_t)(bytes / 4);
    uint32_t nblk = (n32 + 4 * kWG - 1) / (4 * kWG);   /* 4 words a work-item */
    if (nblk < 1) nblk = 1;
    /* Every block of a launch must be resident at once: block b spins until the peer's block b has
     * stored its chunk, so a block that cannot start before another finishes deadlocks the pair. A
     * B70 holds ~80 work-groups of 256 at SIMD16 (1,280 hardware threads); 64 leaves room for the
     * compute the all-reduce overlaps. The CPU device (16 cores) needs XE_AR_BLOCKS=4 or so. */
    if (nblk > (uint32_t)kResidentBlocks) nblk = kResidentBlocks;
    /* XE_AR_BLOCKS: a fixed block count (the CPU-device probe: its runtime schedules the two ranks'
     * kernels one after the other unless each is one work-group, round 2 M2b's shape) */
    static const long env_blocks = [] { const char* v = std::getenv("XE_AR_BLOCKS"); return v ? std::strtol(v, nullptr, 10) : 0L; }();
    if (env_blocks > 0 && env_blocks <= kMaxBlocks) nblk = (uint32_t)env_blocks;
    try {
        sycl::queue* q = (sycl::queue*)s;
        const dim3 grid(nblk), block(kWG);
#define XE_ARP(DT, HW_, PR_) XE_LAUNCH(q, (ar_kernel<DT, HW_, PR_>), grid, block, 0, (const uint32_t*)x->data,       \
                            (uint32_t*)y->data, (uint32_t*)in->peer_scratch, (uint32_t*)in->my_scratch, in->peer_flags, \
                            in->my_flags, in->my_seq, n32, (uint32_t)(in->slot_bytes / 4), (uint32_t)in->rank, nblk,     \
                            spin_budget(), hw ? *hw : HcW{}, in->ring, ar_test_tear())
        /* the LL form (a sequence number in every payload word; the default since run 105, RADIANCE_XE_AR_PROTO=0 the
         * old form), taken when the message's 8-byte words fit a parity's slot -- build() doubles the slots, so always */
        const uint32_t slot_words = (uint32_t)(in->slot_bytes / 4);
        const bool ll = ar_proto() >= 4 && 2ull * n32 <= slot_words;
        /* proto 6: the bandwidth form for a message of ar_bw_min_bytes() or more (its 1x payload always fits a
         * parity's slot), the LL form 5 below it */
        const bool bw = ar_proto() == 6 && bytes >= ar_bw_min_bytes() && n32 + 6u * kMaxBlocks <= slot_words;   /* the payload + the control tail fit */
#define XE_AR(DT, HW_) do { if (bw) XE_ARP(DT, HW_, 6); else if (!ll) XE_ARP(DT, HW_, 0); else if (ar_proto() >= 5) XE_ARP(DT, HW_, 5); else XE_ARP(DT, HW_, 4); } while (0)
        if (hw && in->dtype == 0) XE_AR(0, true);
        else switch (in->dtype) { case 0: XE_AR(0, false); break; case 1: XE_AR(1, false); break; default: XE_AR(2, false); break; }
#undef XE_AR
#undef XE_ARP
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}


/* all_gather for two ranks, in 32-bit words: rank r's contribution lands at y[r * nel + i] (plain), or
 * at y[(i / row) * (2 * row) + r * row + i % row] (row-interleaved, libref's ref_all_gather), for
 * i in [0, nel). The peer's contribution comes through the same double-buffered slots and flags
 * as the all-reduce. */
void gather_kernel(const uint32_t* __restrict__ x, uint32_t* __restrict__ y, uint32_t* peer_slots,
                   uint32_t* my_slots, uint32_t* peer_flags, uint32_t* my_flags, uint32_t* seq_ctr,
                   uint32_t n32, uint32_t row32, uint32_t slot_words, uint32_t rank, uint32_t nblk,
                   uint32_t spin, uint64_t* ring) {
    const uint32_t b = blockIdx.x, t = threadIdx.x;
    const uint32_t s = uc_ld(seq_ctr) + 1u;
    const uint32_t par = s & 1u;
    uint32_t* out = peer_slots + par * slot_words;
    const uint32_t* in = my_slots + par * slot_words;
    const uint32_t per = (n32 + nblk - 1) / nblk;
    const uint32_t w0 = b * per, w1 = (w0 + per < n32) ? w0 + per : n32;
    for (uint32_t k = w0 + t; k < w1; k += kWG) uc_st(out + k, x[k]);
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    __syncthreads();
    if (t == 0) { uc_st(peer_flags + par * kMaxBlocks + b, s); sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); }
    const uint32_t* fp = my_flags + par * kMaxBlocks + b;
    uint32_t budget = spin, ok = 1;
    for (;;) {
        const uint32_t f = uc_ld(fp);
        if ((f - s) <= 0x7fffffffu) break;
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        if (--budget == 0) { ok = 0; break; }
    }
    if (b == 0 && t == 0) ar_skew(ring, s, spin - budget, spin - budget);
    if (!ok) ar_gave_up(seq_ctr, rank, s, b, nblk, t);   /* no return: the block still arrives below */
    else {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        const uint32_t peer = rank ^ 1u;
        const bool inter = row32 > 0 && row32 < n32;
        for (uint32_t k = w0 + t; k < w1; k += kWG) {
            const uint32_t om = inter ? (k / row32) * (2u * row32) + rank * row32 + (k % row32) : rank * n32 + k;
            const uint32_t op = inter ? (k / row32) * (2u * row32) + peer * row32 + (k % row32) : peer * n32 + k;
            y[om] = x[k];
            y[op] = uc_ld(in + k);
        }
    }
    __syncthreads();
    if (t == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> arr(seq_ctr[2]);
        if (arr.fetch_add(1u) + 1u == nblk) { arr.store(0u); uc_st(seq_ctr, s); }
    }
}

int gather_launch(const RadArgs* a, RadStream s, int host_form) {
    Inst* in = (Inst*)a->instance;
    if (!in || in->host != host_form) return R4D_NO(RAD_E_STATE);
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !x->data || !y || !y->data) return R4D_NO(RAD_E_INVAL);
    const long long nel = r4d_numel(x);   /* the contribution is the operand; `numel` the band */
    if (nel <= 0 || nel > in->numel) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(y) < 2 * nel) return R4D_NO(RAD_E_SHAPE);
    const int64_t esz = rad_dtype_bits(x->dtype) / 8;
    if (esz <= 0 || x->dtype != y->dtype) return R4D_NO(RAD_E_DTYPE);
    const long long row = r4d_geti_or(a, "row", 0);
    if (row < 0 || (row > 0 && nel % row)) return R4D_NO(RAD_E_SHAPE);
    if ((nel * esz) % 4 || (row * esz) % 4) return R4D_NO(RAD_E_SHAPE);   /* 32-bit words */
    if (((uintptr_t)x->data | (uintptr_t)y->data) % 4) return R4D_NO(RAD_E_STRIDE);
    const uint32_t n32 = (uint32_t)(nel * esz / 4), row32 = (uint32_t)(row * esz / 4);
    uint32_t nblk = (n32 + 4 * kWG - 1) / (4 * kWG);
    if (nblk < 1) nblk = 1;
    if (nblk > (uint32_t)kResidentBlocks) nblk = kResidentBlocks;
    static const long env_blocks = [] { const char* v = std::getenv("XE_AR_BLOCKS"); return v ? std::strtol(v, nullptr, 10) : 0L; }();
    if (env_blocks > 0 && env_blocks <= kMaxBlocks) nblk = (uint32_t)env_blocks;
    try {
        sycl::queue* q = (sycl::queue*)s;
        XE_LAUNCH(q, gather_kernel, dim3(nblk), dim3(kWG), 0, (const uint32_t*)x->data, (uint32_t*)y->data,
                  (uint32_t*)in->peer_scratch, (uint32_t*)in->my_scratch, in->peer_flags, in->my_flags, in->my_seq,
                  n32, row32, (uint32_t)(in->slot_bytes / 4), (uint32_t)in->rank, nblk, spin_budget(), in->ring);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

}  // namespace
/* libxe's hc_write (ports/r4d_hc_bf16.cpp): y, inj, h with M, n, hc, dtype */
extern "C" int r4d_rad_hc_write(const RadArgs* a, RadStream stream);
namespace {

/* ar_hc_write (libr4d's fused all-reduce + gated residual write, OPS.md): the all-reduce of y in
 * place over M * n elements, then libxe's hc_write on the reduced y. Two launches on the stream; the
 * same bytes as the pair the architecture would otherwise issue. */
int ar_hc_write_launch(const RadArgs* a, RadStream s, int host_form) {
    Inst* in = (Inst*)a->instance;
    if (!in || in->host != host_form) return R4D_NO(RAD_E_STATE);
    const RadTensor* y = r4d_opt(a, 0);
    const RadTensor* inj = r4d_opt(a, 1);
    const RadTensor* h = r4d_opt(a, 2);
    if (!y || !y->data || !inj || !h) return R4D_NO(RAD_E_INVAL);
    if (r4d_geti_or(a, "wire", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (n <= 0 || hc <= 0 || M <= 0 || y_ld != n) return R4D_NO(RAD_E_STRIDE);   /* a dense message */
    RadTensor xt = *y;
    RadParam p[3]{};
    p[0].key = "world_size"; p[0].kind = RAD_P_INT; p[0].ival = 2;
    p[1].key = "numel"; p[1].kind = RAD_P_INT; p[1].ival = M * n;
    p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = r4d_gets(a, "dtype");
    RadArgs ar{};
    ar.t = &xt; ar.n_t = 1; ar.p = p; ar.n_p = 3; ar.instance = in; ar.rank = a->rank; ar.world_size = a->world_size;
    /* RADIANCE_XE_AR_HC_FUSE=1 (owner note 105 item 4): the residual write inside the bf16 all-reduce (ar_kernel's
     * HW form); any shape it does not take keeps the two launches */
    static const bool fuse = [] { const char* v = std::getenv("RADIANCE_XE_AR_HC_FUSE"); return v && v[0] == '1'; }();
    if (fuse && in->dtype == 0 && n % 2 == 0 && inj->dtype == RAD_BF16 && h->dtype == RAD_BF16 && inj->data && h->data) {
        long long irows = 0, hrows = 0;
        const long long irow = r4d_rows_pitch(inj, hc, &irows), hrow = r4d_rows_pitch(h, hc * n, &hrows);
        if (irow >= 0 && hrow >= 0 && irows >= M && hrows >= M) {
            HcW hw;
            hw.inj = (const uint16_t*)inj->data; hw.h = (uint16_t*)h->data;
            hw.n = (uint32_t)n; hw.hc = (uint32_t)hc; hw.irow = (uint32_t)irow; hw.hrow = (uint32_t)hrow;
            return launch(&ar, s, host_form, &hw);
        }
    }
    const int rc = launch(&ar, s, host_form);
    if (rc != RAD_OK) return rc;
    RadTensor t3[3] = { *y, *inj, *h };
    RadParam q[4]{};
    q[0].key = "M"; q[0].kind = RAD_P_INT; q[0].ival = M;
    q[1].key = "n"; q[1].kind = RAD_P_INT; q[1].ival = n;
    q[2].key = "hc"; q[2].kind = RAD_P_INT; q[2].ival = hc;
    q[3].key = "dtype"; q[3].kind = RAD_P_STR; q[3].sval = r4d_gets(a, "dtype");
    RadArgs hw{};
    hw.t = t3; hw.n_t = 3; hw.p = q; hw.n_p = 4; hw.rank = a->rank; hw.world_size = a->world_size;
    return r4d_rad_hc_write(&hw, s);
}

}  // namespace
/* libxe's moe_gather (ports/r4d_moe.cpp): y_expert, expert_w, sorted_tok, shared?, shared_gate?, y with
 * M, n, top_k, dtype, act; scratch M * top_k * 4 bytes */
extern "C" int r4d_rad_moe_gather(const RadArgs* a, RadStream s);
extern "C" int64_t r4d_rad_moe_gather_scratch(const RadArgs* a);
namespace {

/* ar_gather_hc_write (libr4d's fusion, OPS.md): the routed experts' rows summed back to token order
 * (moe_gather, with the shared arm folded when present) ARE the all-reduce's message, and the gated
 * residual write follows. Here: libxe's moe_gather into y, then ar_hc_write on y -- three launches. */
int ar_gather_hc_write_launch(const RadArgs* a, RadStream s, int host_form) {
    Inst* in = (Inst*)a->instance;
    if (!in || in->host != host_form) return R4D_NO(RAD_E_STATE);
    if (a->n_t < 8) return R4D_NO(RAD_E_INVAL);
    const RadTensor* y = r4d_opt(a, 5);
    const RadTensor* inj = r4d_opt(a, 6);
    const RadTensor* h = r4d_opt(a, 7);
    if (!y || !y->data || !inj || !h) return R4D_NO(RAD_E_INVAL);
    if (r4d_geti_or(a, "wire", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0), top_k = r4d_geti_or(a, "top_k", 0);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (n <= 0 || hc <= 0 || top_k <= 0 || M <= 0 || y_ld != n) return R4D_NO(RAD_E_STRIDE);
    const char* dt = r4d_gets(a, "dtype");
    const char* act = r4d_gets(a, "act");
    /* 1. the gather: operands 0..4 and y, with the fused op's scratch */
    RadTensor tg[6] = { a->t[0], a->t[1], a->t[2], a->t[3], a->t[4], *y };
    RadParam pg[5]{};
    int ng = 0;
    pg[ng].key = "M"; pg[ng].kind = RAD_P_INT; pg[ng].ival = M; ++ng;
    pg[ng].key = "n"; pg[ng].kind = RAD_P_INT; pg[ng].ival = n; ++ng;
    pg[ng].key = "top_k"; pg[ng].kind = RAD_P_INT; pg[ng].ival = top_k; ++ng;
    pg[ng].key = "dtype"; pg[ng].kind = RAD_P_STR; pg[ng].sval = dt; ++ng;
    if (act) { pg[ng].key = "act"; pg[ng].kind = RAD_P_STR; pg[ng].sval = act; ++ng; }
    RadArgs ga{};
    ga.t = tg; ga.n_t = 6; ga.p = pg; ga.n_p = ng; ga.scratch = a->scratch; ga.scratch_bytes = a->scratch_bytes;
    ga.rank = a->rank; ga.world_size = a->world_size;
    int rc = r4d_rad_moe_gather(&ga, s);
    if (rc != RAD_OK) return rc;
    /* 2. the all-reduce of y in place, 3. hc_write */
    RadTensor th[3] = { *y, *inj, *h };
    RadParam ph[5]{};
    ph[0].key = "n"; ph[0].kind = RAD_P_INT; ph[0].ival = n;
    ph[1].key = "hc"; ph[1].kind = RAD_P_INT; ph[1].ival = hc;
    ph[2].key = "world_size"; ph[2].kind = RAD_P_INT; ph[2].ival = 2;
    ph[3].key = "dtype"; ph[3].kind = RAD_P_STR; ph[3].sval = dt;
    ph[4].key = "wire"; ph[4].kind = RAD_P_INT; ph[4].ival = 0;
    RadArgs ha{};
    ha.t = th; ha.n_t = 3; ha.p = ph; ha.n_p = 5; ha.instance = in; ha.rank = a->rank; ha.world_size = a->world_size;
    return ar_hc_write_launch(&ha, s, host_form);
}

}  // namespace

extern "C" int xe_rad_ar_gather_hc_write_2rank(const RadArgs* a, RadStream s) { return ar_gather_hc_write_launch(a, s, 0); }
extern "C" int xe_rad_ar_gather_hc_write_2rank_host(const RadArgs* a, RadStream s) { return ar_gather_hc_write_launch(a, s, 1); }
extern "C" int64_t xe_ar_gather_hc_write_scratch(const RadArgs* a) { return r4d_rad_moe_gather_scratch(a); }
extern "C" int xe_rad_ar_gather_2rank(const RadArgs* a, RadStream s) { return gather_launch(a, s, 0); }
extern "C" int xe_rad_ar_gather_2rank_host(const RadArgs* a, RadStream s) { return gather_launch(a, s, 1); }
extern "C" int xe_rad_ar_hc_write_2rank(const RadArgs* a, RadStream s) { return ar_hc_write_launch(a, s, 0); }
extern "C" int xe_rad_ar_hc_write_2rank_host(const RadArgs* a, RadStream s) { return ar_hc_write_launch(a, s, 1); }

extern "C" int xe_ar2_init(const RadParam* p, int n_p, int rank, int world, void** out) { return ar_init(p, n_p, rank, world, 0, out); }
extern "C" int xe_ar2_host_init(const RadParam* p, int n_p, int rank, int world, void** out) { return ar_init(p, n_p, rank, world, 1, out); }
extern "C" void xe_ar2_fini(void* instance) { ar_fini(instance); }
extern "C" int xe_rad_ar_oneshot_2rank_exact(const RadArgs* a, RadStream s) { return launch(a, s, 0); }
extern "C" int xe_rad_ar_oneshot_2rank_exact_host(const RadArgs* a, RadStream s) { return launch(a, s, 1); }
/* the tail skip's stats (xe_moe_w4nl_i8.cpp, owner note 162 item 1): this is the engine's once-a-step call into libxe */
extern "C" void xe_tskip_report(int rank, void* stream, const void* key);
/* the blocks that gave up so far (seq_ctr[3]), for tests */
extern "C" uint32_t xe_ar2_stalls(void* instance, RadStream s) {
    Inst* in = (Inst*)instance;
    if (!in) return 0;
    uint32_t v = 0;
    ((sycl::queue*)s)->memcpy(&v, in->my_seq + 3, 4).wait();
    xe_tskip_report(in->rank, (void*)s, instance);
    /* RADIANCE_XE_AR_SKEW: the engine calls this once a step -- append the launches completed since the last call
     * (their sequence numbers up to the counter, my_seq[0]) with the call count, the step's index for this rank */
    if (in->skew_f && in->ring) {
        ++in->calls;
        uint32_t cur = 0;
        ((sycl::queue*)s)->memcpy(&cur, in->my_seq, 4).wait();
        uint32_t from = in->last_s + 1u;
        if (cur - in->last_s > kSkewRing) from = cur - kSkewRing + 1u;   /* lost to the wrap: keep the newest */
        if ((int32_t)(cur - in->last_s) > 0) {
            const uint32_t n = cur - from + 1u;
            std::vector<uint64_t> e(n);
            const uint32_t i0 = from & in->ring_mask, first = n < kSkewRing - i0 ? n : kSkewRing - i0;
            ((sycl::queue*)s)->memcpy(e.data(), in->ring + i0, (size_t)first * 8);
            if (first < n) ((sycl::queue*)s)->memcpy(e.data() + first, in->ring, (size_t)(n - first) * 8);
            ((sycl::queue*)s)->wait();
            for (uint64_t x : e) {
                const uint64_t rec[2] = { in->calls, x };
                std::fwrite(rec, 8, 2, in->skew_f);
            }
            in->last_s = cur;
        }
        if ((in->calls & 255u) == 0) std::fflush(in->skew_f);
    }
    return v;
}
/* a stalled handshake, for tests: the launch's sequence at which a block gave up, 0 if none */
extern "C" uint32_t xe_ar2_stalled(void* instance, RadStream s) {
    Inst* in = (Inst*)instance;
    if (!in) return 0;
    uint32_t v = 0;
    ((sycl::queue*)s)->memcpy(&v, in->my_seq + 1, 4).wait();
    return v;
}
/* run 170 (proto 6): the checksum re-reads this instance's blocks made before passing, summed over its launches
 * (seq_ctr[4]); zero means the bandwidth form never met a stale block */
extern "C" uint32_t xe_ar2_retries(void* instance, RadStream s) {
    Inst* in = (Inst*)instance;
    if (!in) return 0;
    uint32_t v = 0;
    ((sycl::queue*)s)->memcpy(&v, in->my_seq + 4, 4).wait();
    return v;
}
