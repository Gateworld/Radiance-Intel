/* xprefetch.cpp -- the expert prefetch's host half: Ctx::xp_*. See xprefetch.h for what it does. */
#include "ctx.h"
#include "xp_state.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#if !(defined(RAD_HAVE_L0) && RAD_HAVE_L0)
/* The gather is the Level Zero backend's (core/device/l0_kernels.sycl); elsewhere gather mode is never on. */
extern "C" int rad_xp_gather_launch(const rad::XpBox*, uint64_t, uint32_t*, uint32_t*, uint32_t, void*) { return -1; }
extern "C" int rad_xp_plan2_launch(const float*, uint32_t, uint32_t, uint32_t, void** const*, const int32_t*,
                                   const int32_t*, rad::XpPools, uint64_t, uint64_t, rad::XpRec*, int32_t*,
                                   rad::XpBox*, uint32_t*, rad::XpStats*, void*) {
    return -1;
}
extern "C" int rad_xp_step2_launch(const rad::XpRec*, const int32_t*, int, int, const uint32_t*, const uint32_t*,
                                   void*) {
    return -1;
}
#endif
#if !(defined(RAD_HAVE_HIP) && RAD_HAVE_HIP) && !(defined(RAD_HAVE_L0) && RAD_HAVE_L0)
/* A build without HIP or Level Zero has no kernels to launch: rad_xp_supported is 0 there, so nothing
 * calls these. */
extern "C" int rad_xp_logits_launch(const uint16_t*, const uint16_t*, float*, uint32_t, uint32_t, uint32_t, void*) {
    return -1;
}
extern "C" int rad_xp_plan_launch(const float*, uint32_t, uint32_t, uint32_t, void** const*, const int32_t*,
                                  const int32_t*, rad::XpPools, uint64_t, uint64_t, rad::XpRec*, int32_t*,
                                  rad::XpBox*, uint32_t, rad::XpStats*, void*) {
    return -1;
}
extern "C" int rad_xp_step_launch(const rad::XpRec*, const int32_t*, const rad::XpRec*, const int32_t*,
                                  const uint32_t*, uint32_t, void*) {
    return -1;
}
#endif

namespace rad {

namespace {

/* A pool class wider than this holds whole tables (the token embedding read in place), not experts. */
constexpr uint64_t kXpMaxUnit = (uint64_t)64 << 20;
/* Lists the queue may be behind before a request is dropped. Nothing else throttles a lagging
 * queue: its copies would land too late anyway, and an ever-growing queue would end up polling
 * marks already returned to the pool. */
constexpr int64_t kXpBehind = 3;
/* Dispatches after a dropped request's mark before its signal may be reused; far more than the
 * pass that completes it. */
constexpr int64_t kXpMarkHold = 4096;
/* Payoff. A used copy saves link time under the GEMM; an unused one costs link time the GEMMs
 * wanted, and the prediction always costs a little. On Qwen3.8-Flash-Next break-even is near a
 * quarter of copies used (--tp 1 used half and decode was 3.6% faster; --tp 2, where the pool holds
 * rarely routed experts, used a sixth and was 4-6% slower). Every kXpWindow predictions the share
 * used over the last window is judged; below kXpMinUse the prefetch rests for kXpRest passes. */
constexpr int64_t  kXpWindow    = 2048;
constexpr double   kXpMinUse    = 0.30;
constexpr uint32_t kXpMinJudged = 512;
constexpr int64_t  kXpRest      = 8192;

}  // namespace

void Ctx::xp_free(XpState& x) {
    if (x.stream) rad_stream_sync(x.stream);
    if (x.ring) rad_dev_free_uncached(x.ring);
    for (void* p : {(void*)x.rec, (void*)x.nrec, (void*)x.stats, (void*)x.logits, (void*)x.maps,
                    (void*)x.places})
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    for (void* p : {(void*)x.box, (void*)x.done})
        if (p) rad_dev_free(p, x.gather ? RAD_MEM_DEVICE : RAD_MEM_HOST_PINNED);
    if (x.stats_h) rad_dev_free(x.stats_h, RAD_MEM_HOST_PINNED);
    if (x.ctr) rad_dev_free(x.ctr, RAD_MEM_DEVICE);
    if (x.tick) rad_dev_free(x.tick, RAD_MEM_DEVICE);
    if (x.mark_ev) rad_event_destroy(x.mark_ev);
    if (x.tail) rad_event_destroy(x.tail);
    if (x.sync) rad_event_destroy(x.sync);
    if (x.stream) rad_stream_destroy(x.stream);
}

/* At bringup, after placement has made its pinned pools: allocates everything, or nothing and logs
 * why the prefetch is off. */
void Ctx::xp_open(const uint64_t* lo, const uint64_t* hi, const uint64_t* stride, int n) {
    if (xp_ || device_is_host() || route_.empty()) return;
    if (!rad_xp_supported(device_)) {
        RAD_INFO("rank %d: expert prefetch off: the device backend has no DMA engine to spare for it",
                 rank_);
        return;
    }
    auto x = std::make_unique<XpState>();
    x->device = device_;
    for (int i = 0; i < n && x->pools.n < kXpMaxPools; ++i) {
        if (stride[i] == 0 || stride[i] > kXpMaxUnit || hi[i] <= lo[i]) continue;
        x->pools.lo[x->pools.n] = lo[i];
        x->pools.hi[x->pools.n] = hi[i];
        x->pools.stride[x->pools.n] = stride[i];
        x->slot = std::max(x->slot, stride[i]);
        ++x->pools.n;
    }
    if (x->pools.n == 0) return;
    const int64_t nl = (int64_t)route_.size();
    const int64_t ring = (int64_t)(kXpGroups * kXpSlots * x->slot);
    if (rad_dev_set(device_) < 0) return;
    x->ring   = (char*)rad_dev_alloc_uncached(ring);
    x->rec    = (XpRec*)rad_dev_alloc((int64_t)sizeof(XpRec) * kXpGroups * kXpSlots, RAD_MEM_DEVICE);
    x->nrec   = (int32_t*)rad_dev_alloc(4 * kXpGroups, RAD_MEM_DEVICE);
    x->stats  = (XpStats*)rad_dev_alloc((int64_t)sizeof(XpStats), RAD_MEM_DEVICE);
    x->logits = (float*)rad_dev_alloc(4 * (int64_t)kXpGroups * kXpMaxRows * kXpMaxNe, RAD_MEM_DEVICE);
    x->maps   = (void**)rad_dev_alloc(nl * kXpMaxNe * kXpParts * (int64_t)sizeof(void*), RAD_MEM_DEVICE);
    x->places = (int32_t*)rad_dev_alloc(nl * kXpMaxNe * 4, RAD_MEM_DEVICE);
    /* Level Zero (owner note 139 stage 1): RADIANCE_XE_EXPERT_PREFETCH=1 is v2 (rec2: recorded kernels on the
     * compute stream, copies on the copy engine by the host thread); =2 the first form, gather mode (host steps a
     * layer, a gather kernel on the prefetch queue; exact, but the tape cuts cost 22 %, M175). */
#if defined(RAD_HAVE_L0) && RAD_HAVE_L0
    {
        const char* e = std::getenv("RADIANCE_XE_EXPERT_PREFETCH");
        if (e && e[0] == '2') x->gather = true;
        else x->rec2 = true;
    }
#endif
    const int bk = x->gather ? RAD_MEM_DEVICE : RAD_MEM_HOST_PINNED;
    x->box    = (XpBox*)rad_dev_alloc((int64_t)sizeof(XpBox) * kXpGroups, bk);
    x->done   = (uint32_t*)rad_dev_alloc(4 * kXpGroups, bk);
    x->stats_h = (XpStats*)rad_dev_alloc((int64_t)sizeof(XpStats), RAD_MEM_HOST_PINNED);
    if (x->gather) x->ctr = (uint32_t*)rad_dev_alloc(4 * kXpGroups, RAD_MEM_DEVICE);
    if (x->rec2) x->tick = (uint32_t*)rad_dev_alloc(4 * kXpGroups, RAD_MEM_DEVICE);
    const bool ok = x->ring && x->rec && x->nrec && x->stats && x->logits && x->maps && x->places && x->box &&
                    x->done && x->stats_h && (!x->gather || x->ctr) && (!x->rec2 || x->tick) &&
                    (!x->rec2 || rad_memset_async(x->tick, 0, 4 * kXpGroups, stream_) >= 0) &&
                    rad_xp_stream_create(&x->stream) >= 0 &&
                    rad_event_create_local(&x->tail) >= 0 && rad_event_create_local(&x->sync) >= 0 &&
                    (!x->gather || rad_event_create_local(&x->mark_ev) >= 0) &&
                    rad_memset_async(x->nrec, 0, 4 * kXpGroups, stream_) >= 0 &&
                    (!x->gather || (rad_memset_async(x->box, 0, (int64_t)sizeof(XpBox) * kXpGroups, stream_) >= 0 &&
                                    rad_memset_async(x->done, 0, 4 * kXpGroups, stream_) >= 0 &&
                                    rad_memset_async(x->ctr, 0, 4 * kXpGroups, stream_) >= 0)) &&
                    rad_memset_async(x->stats, 0, sizeof(XpStats), stream_) >= 0 && rad_stream_sync(stream_) >= 0;
    if (!ok) {
        const char* why = rad_dev_last_error();
        RAD_WARN("rank %d: expert prefetch off: %s", rank_, why && *why ? why : "no memory for it");
        xp_free(*x);
        return;
    }
    if (!x->gather) {
        std::memset((void*)x->box, 0, sizeof(XpBox) * kXpGroups);
        std::memset(x->done, 0, 4 * kXpGroups);
    }
    x->layers.resize((size_t)nl);
    XpState* xs = x.get();
    if (x->rec2) x->thr = std::thread([this, xs] { xp_poll(*xs); });
    else if (!x->gather) x->thr = std::thread([this, xs] { xp_serve(*xs); });
    RAD_INFO("rank %d: expert prefetch on: %d copies a routed layer by %s, ring %s, %d pool ranges", rank_, kXpSlots,
             x->rec2 ? "the copy engine, the plan recorded with the pass (v2)"
                     : x->gather ? "a gather kernel on its own queue" : "DMA",
             humanb(ring).c_str(), x->pools.n);
    xp_ = std::move(x);
}

void Ctx::xp_close() {
    if (!xp_) return;
    XpState& x = *xp_;
    /* Nothing may touch the ring when it is freed: issued compute work can read it through entries
     * still pointing at it, and queued DMA lists write it until they finish. */
    if (stream_) rad_stream_sync(stream_);
    if (lane1_) rad_stream_sync(lane1_);
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.stop = true;
    }
    x.cv.notify_all();
    if (x.thr.joinable()) x.thr.join();
    const auto t0 = std::chrono::steady_clock::now();
    while (rad_xp_behind(x.device) > 0) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            RAD_WARN("rank %d: the prefetch DMA queue still has %lld lists after 5 s; its ring is freed anyway",
                     rank_, (long long)rad_xp_behind(x.device));
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    /* Summary: copies planned, and how many restored copies the step routed. */
    if (x.stream && rad_memcpy_async(x.stats_h, x.stats, sizeof(XpStats), x.stream) >= 0 &&
        rad_stream_sync(x.stream) >= 0 && x.stats_h->planned)
        RAD_INFO("rank %d: expert prefetch: %u copies planned, %u of %u looked back on routed (%.1f%%), %lld "
                 "requests dropped behind a slow queue, %lld rests", rank_, x.stats_h->planned, x.stats_h->routed,
                 x.stats_h->copied, x.stats_h->copied ? 100.0 * x.stats_h->routed / x.stats_h->copied : 0.0,
                 (long long)x.dropped.load(), (long long)x.rests);
    xp_free(x);
    xp_.reset();
}

/* Host thread: hands each request's mailbox to the DMA queue once the plan has written it. */
void Ctx::xp_serve(XpState& x) {
    for (;;) {
        XpReq r;
        {
            std::unique_lock<std::mutex> lk(x.mu);
            x.cv.wait(lk, [&] { return x.stop.load() || !x.q.empty(); });
            if (x.stop) return;
            r = x.q.front();
            x.q.pop_front();
        }
        /* Return the marks of finished lists to the pool. */
        const int64_t behind = rad_xp_behind(x.device);
        while (!x.marks.empty() && x.marks.front().second <= x.lists - behind) {
            rad_ride_release(x.marks.front().first, nullptr, 0);
            x.marks.pop_front();
        }
        /* The mailbox reaches the request number once the plan has run; a higher number means a
         * later request overwrote it and this one is stale. */
        volatile XpBox* b = x.box + r.g;
        const auto t0 = std::chrono::steady_clock::now();
        bool got = true;
        while ((int32_t)(b->seq - r.seq) < 0) {
            _mm_pause();
            if (x.stop || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
                got = false;
                break;
            }
        }
        if (got && b->seq != r.seq) got = false;
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t src[kXpSlots], dst[kXpSlots], bytes[kXpSlots];
        int n = 0;
        if (got) {
            n = std::min<int>((int)b->n, kXpSlots);
            for (int i = 0; i < n && got; ++i) {
                /* A job seen before its own stores landed carries an older tag. */
                const auto t1 = std::chrono::steady_clock::now();
                while (b->jobs[i].tag != r.seq &&
                       std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(100))
                    _mm_pause();
                if (b->jobs[i].tag != r.seq) { got = false; break; }
                std::atomic_thread_fence(std::memory_order_acquire);
                src[i] = b->jobs[i].src;
                dst[i] = b->jobs[i].dst;
                bytes[i] = b->jobs[i].bytes;
            }
        }
        if (!got || n == 0 || behind > kXpBehind) {
            if (got && n > 0) ++x.dropped;
            if (r.mark) rad_ride_release(r.mark, r.ms, kXpMarkHold);
            continue;
        }
        if (rad_xp_list(x.device, r.mark, n, src, dst, bytes, x.done + r.g, r.seq) == RAD_OK) {
            ++x.lists;
            if (r.mark) x.marks.emplace_back(r.mark, x.lists);
        } else {
            if (r.mark) rad_ride_release(r.mark, r.ms, kXpMarkHold);
            RAD_WARN("rank %d: the prefetch DMA queue refused a list", rank_);
        }
    }
}

/* Builds the layer's expert -> table-entry map; false until all its tables are bound. An expert
 * whose weights are not each in exactly one table is left out. */
bool Ctx::xp_bind(XpLayer& L, int layer) {
    const RadPrefetchNext& nx = L.next;
    std::vector<void**> hm((size_t)kXpMaxNe * kXpParts, nullptr);
    std::vector<int32_t> hp((size_t)kXpMaxNe, -1);
    for (int64_t e = 0; e < nx.n_expert; ++e) {
        void** ent[kXpParts] = {};
        bool usable = true;
        for (int64_t k = 0; k < nx.n_parts && usable; ++k) {
            const rad_weight w = nx.parts[e * nx.n_parts + k];
            if (!w) continue;
            if ((size_t)w >= wmemb_head_.size() || wmemb_head_[w] < 0) { usable = false; break; }
            const WMemb& m = wmemb_[(size_t)wmemb_head_[w]];
            if (m.next >= 0) { usable = false; break; }
            const WeightTable& t = wtabs_[(size_t)m.slot];
            if (!t.dev) return false;
            ent[k] = (void**)t.dev + m.entry;
        }
        if (usable)
            for (int k = 0; k < kXpParts; ++k) hm[(size_t)(e * kXpParts + k)] = ent[k];
        hp[(size_t)e] = nx.count_at ? nx.count_at[e] : (int32_t)e;
    }
    XpState& x = *xp_;
    void** map = x.maps + (size_t)layer * kXpMaxNe * kXpParts;
    int32_t* place = x.places + (size_t)layer * kXpMaxNe;
    /* Runs once per layer at first issue, so this host wait is off the step path. v2 binds while a pass may be
     * recording natively on the compute stream, so it uploads on the prefetch's own queue, which never records. */
    const RadStream bs = x.rec2 ? x.stream : stream_;
    if (rad_memcpy_async(map, hm.data(), (int64_t)hm.size() * (int64_t)sizeof(void*), bs) < 0 ||
        rad_memcpy_async(place, hp.data(), (int64_t)hp.size() * 4, bs) < 0 || rad_stream_sync(bs) < 0)
        return false;
    L.map = map;
    L.place = place;
    L.bound = true;
    return true;
}

/* Judges the share of copies used from the counters sent to the host one window ago, then sends the
 * next window's. The first window after a rest is not judged (its counters predate the rest), so
 * each rest is followed by one full fresh window. */
void Ctx::xp_judge(XpState& x) {
    if (x.have_snap) {
        const volatile XpStats* h = x.stats_h;
        const uint32_t copied = h->copied, routed = h->routed;
        const uint32_t dc = copied - x.snap_copied, dr = routed - x.snap_routed;
        x.snap_copied = copied;
        x.snap_routed = routed;
        if (x.judge && dc >= kXpMinJudged && (double)dr < kXpMinUse * (double)dc) {
            x.rest_until = x.passes + kXpRest;
            x.judge = false;
            if (x.rests++ == 0)
                RAD_INFO("rank %d: expert prefetch resting: the steps used %.0f%% of its last %u copies, under the "
                         "%.0f%% that pays for them; it measures again every %lld passes", rank_,
                         100.0 * dr / dc, dc, 100.0 * kXpMinUse, (long long)kXpRest);
        } else {
            x.judge = true;
        }
    }
    if (rad_memcpy_async(x.stats_h, x.stats, sizeof(XpStats), x.stream) >= 0) x.have_snap = true;
}

/* When recording, each of the three calls becomes its own tape step and is re-run at every play:
 * its stream work and rides are issued afresh, not recorded. */
int Ctx::xp_predict(const RadPrefetchNext* nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    if (!xp_ && nx && rows > 0 && rows <= kXpMaxRows && xg_on()) return xg_guess(*nx, h, r0, rows, n_embd);
    /* v2: kernels on the compute stream, recorded with the pass when it records -- never a tape step. */
    if (xp_ && xp_->rec2)
        return nx && rows > 0 && rows <= kXpMaxRows ? xp_predict_rec(*nx, h, r0, rows, n_embd) : RAD_OK;
    /* Too wide to predict; keep the tape free of an empty step. */
    if (!xp_ || !nx || rows <= 0 || rows > kXpMaxRows) return RAD_OK;
    if (!rec_) return xp_predict_run(*nx, h, r0, rows, n_embd);
    TapeXp p;
    p.next = *nx;
    p.h = h;
    p.r0 = r0;
    p.rows = rows;
    p.n_embd = n_embd;
    TapeStep st;
    st.kind = kTapeXp;
    st.a = (int32_t)rec_->xp.size();
    rec_->xp.push_back(p);
    return tape_host(st, [&] { return xp_predict_run(*nx, h, r0, rows, n_embd); });
}

int Ctx::xp_mark() {
    if (!xp_ || xp_->rec2) return RAD_OK;   /* v2: the host thread starts the copies as soon as it sees the plan */
    if (!rec_) return xp_mark_run();
    TapeStep st;
    st.kind = kTapeXpMark;
    return tape_host(st, [&] { return xp_mark_run(); });
}

int Ctx::xp_step(int layer) {
    if (!xp_ && xg_on()) xg_judge(layer);
    if (xp_ && xp_->rec2) return xp_step_rec(layer);
    if (!xp_) return RAD_OK;
    if (!rec_) return xp_step_run(layer);
    TapeStep st;
    st.kind = kTapeXpStep;
    st.a = layer;
    return tape_host(st, [&] { return xp_step_run(layer); });
}

int Ctx::xp_predict_run(const RadPrefetchNext& nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    XpState& x = *xp_;
    const int layer = nx.layer;
    if (rows <= 0 || rows > kXpMaxRows || nx.top_k <= 0 || nx.top_k > kXpMaxK || nx.n_expert <= 0 ||
        nx.n_expert > kXpMaxNe || nx.n_parts <= 0 || nx.n_parts > kXpParts || !nx.parts || n_embd % 256 ||
        n_embd > 4096 || layer < 0 || layer >= (int)x.layers.size())
        return RAD_OK;
    XpLayer& L = x.layers[(size_t)layer];
    if (!L.bound) {
        L.next = nx;
        if (!xp_bind(L, layer)) return RAD_OK;
    }
    const void* w = weight_ptr(nx.router);
    const char* hb = (const char*)buf_ptr(h);
    if (!w || !hb) return RAD_OK;
    const RadStream cs = cur_stream();
    /* A new pass waits for the prefetch stream's work from the last pass, so that stream is never a
     * pass behind: the records and mailboxes a request reuses are three layers old. */
    if (layer <= x.last_layer) {
        if (x.tail_live) {
            RAD_TRY(rad_event_record(x.tail, x.stream));
            RAD_TRY(rad_event_wait(cs, x.tail));
            x.tail_live = false;
        }
        ++x.passes;
    }
    x.last_layer = layer;
    if (x.passes < x.rest_until) return RAD_OK;
    /* The router input is read in place, riding behind the next compute dispatch (the router, which
     * reads it) rather than via a barrier packet between two kernels. */
    void* rt = nullptr;
    if (rad_ride(cs, &rt) == RAD_OK) {
        RAD_TRY(rad_ride_wait(x.stream, rt));
    } else {
        RAD_TRY(rad_event_record(x.sync, cs));
        RAD_TRY(rad_event_wait(x.stream, x.sync));
    }
    /* An earlier request still waiting for its mark is sent now, unmarked. */
    if (x.pend.valid && x.gather) {
        if (rad_xp_gather_launch(x.box + x.pend.g, x.slot, x.done + x.pend.g, x.ctr + x.pend.g, x.pend.seq,
                                 (void*)x.stream))
            return RAD_E_DEVICE;
        x.pend.valid = false;
    } else if (x.pend.valid) {
        {
            std::lock_guard<std::mutex> lk(x.mu);
            x.q.push_back(x.pend);
        }
        x.cv.notify_one();
        x.pend.valid = false;
    }
    const int g = layer % kXpGroups;
    const uint32_t seq = ++x.seq[g];
    const uint16_t* hg = (const uint16_t*)(hb + r0 * n_embd * 2);
    float* lg = x.logits + (size_t)g * kXpMaxRows * kXpMaxNe;
    const int32_t* cnt = route_n_expert_ > 0 && nx.n_expert <= route_n_expert_
                             ? route_dev_ + (int64_t)layer * route_n_expert_ : nullptr;
    const uint64_t ring = (uint64_t)(uintptr_t)(x.ring + (size_t)g * kXpSlots * x.slot);
    if (rad_xp_logits_launch(hg, (const uint16_t*)w, lg, (uint32_t)rows, (uint32_t)n_embd, (uint32_t)nx.n_expert,
                             (void*)x.stream) ||
        rad_xp_plan_launch(lg, (uint32_t)rows, (uint32_t)nx.n_expert, (uint32_t)nx.top_k, (void** const*)L.map,
                           L.place, cnt, x.pools, ring, x.slot, x.rec + (size_t)g * kXpSlots, x.nrec + g,
                           x.gather ? x.box + g : (XpBox*)rad_dev_device_ptr(x.box + g), seq, x.stats,
                           (void*)x.stream))
        return RAD_E_DEVICE;
    x.tail_live = true;
    if (++x.calls % kXpWindow == 0) xp_judge(x);
    x.pend.g = g;
    x.pend.seq = seq;
    x.pend.mark = nullptr;
    x.pend.ms = nullptr;
    x.pend.valid = true;
    L.want = true;
    L.g = g;
    L.seq = seq;
    return RAD_OK;
}

int Ctx::xp_mark_run() {
    XpState& x = *xp_;
    if (!x.pend.valid) return RAD_OK;
    /* Gather mode: the copies start once the compute stream reaches this point (this layer's streamed
     * experts are read), on the prefetch stream beside the next layer's mixer. */
    if (x.gather) {
        x.pend.valid = false;
        RAD_TRY(rad_event_record(x.mark_ev, cur_stream()));
        RAD_TRY(rad_event_wait(x.stream, x.mark_ev));
        return rad_xp_gather_launch(x.box + x.pend.g, x.slot, x.done + x.pend.g, x.ctr + x.pend.g, x.pend.seq,
                                    (void*)x.stream) ? RAD_E_DEVICE : RAD_OK;
    }
    /* Refused only while another ride waits on the stream; the list then starts unmarked. */
    void* mark = nullptr;
    if (rad_ride(cur_stream(), &mark) == RAD_OK) {
        x.pend.mark = mark;
        x.pend.ms = cur_stream();
    }
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.q.push_back(x.pend);
    }
    x.cv.notify_one();
    x.pend.valid = false;
    return RAD_OK;
}

int Ctx::xp_step_run(int layer) {
    XpState& x = *xp_;
    XpLayer* L = layer >= 0 && (size_t)layer < x.layers.size() ? &x.layers[(size_t)layer] : nullptr;
    const bool patch = L && L->want;
    if (!patch && x.restore_g < 0) return RAD_OK;
    const XpRec* rrec = x.restore_g >= 0 ? x.rec + (size_t)x.restore_g * kXpSlots : nullptr;
    const int32_t* rn = x.restore_g >= 0 ? x.nrec + x.restore_g : nullptr;
    const XpRec* prec = patch ? x.rec + (size_t)L->g * kXpSlots : nullptr;
    const int32_t* pn = patch ? x.nrec + L->g : nullptr;
    const uint32_t* done = !patch ? nullptr
                         : x.gather ? x.done + L->g : (const uint32_t*)rad_dev_device_ptr(x.done + L->g);
    if (rad_xp_step_launch(rrec, rn, prec, pn, done, patch ? L->seq : 0, (void*)cur_stream())) return RAD_E_DEVICE;
    x.restore_g = patch ? L->g : -1;
    if (L) L->want = false;
    return RAD_OK;
}

/* ---- v2 (owner note 139 stage 1, run 113): recorded kernels and the copy engine ---- */

/* At this layer's router input, on the compute stream: the next layer's logits and the plan, which numbers the
 * request from the group's device ticket and writes the pinned mailbox. Plain launches, so a pass being recorded
 * keeps them and every play repeats them; nothing here waits or cuts the tape. */
int Ctx::xp_predict_rec(const RadPrefetchNext& nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    XpState& x = *xp_;
    const int layer = nx.layer;
    if (rows <= 0 || rows > kXpMaxRows || nx.top_k <= 0 || nx.top_k > kXpMaxK || nx.n_expert <= 0 ||
        nx.n_expert > kXpMaxNe || nx.n_parts <= 0 || nx.n_parts > kXpParts || !nx.parts || n_embd % 256 ||
        n_embd > 4096 || layer < 0 || layer >= (int)x.layers.size())
        return RAD_OK;
    XpLayer& L = x.layers[(size_t)layer];
    if (!L.bound) {
        L.next = nx;
        if (!xp_bind(L, layer)) return RAD_OK;
    }
    const void* w = weight_ptr(nx.router);
    const char* hb = (const char*)buf_ptr(h);
    if (!w || !hb) return RAD_OK;
    const RadStream cs = cur_stream();
    const int g = layer % kXpGroups;
    const uint16_t* hg = (const uint16_t*)(hb + r0 * n_embd * 2);
    float* lg = x.logits + (size_t)g * kXpMaxRows * kXpMaxNe;
    const int32_t* cnt = route_n_expert_ > 0 && nx.n_expert <= route_n_expert_
                             ? route_dev_ + (int64_t)layer * route_n_expert_ : nullptr;
    const uint64_t ring = (uint64_t)(uintptr_t)(x.ring + (size_t)g * kXpSlots * x.slot);
    if (rad_xp_logits_launch(hg, (const uint16_t*)w, lg, (uint32_t)rows, (uint32_t)n_embd, (uint32_t)nx.n_expert,
                             (void*)cs) ||
        rad_xp_plan2_launch(lg, (uint32_t)rows, (uint32_t)nx.n_expert, (uint32_t)nx.top_k, (void** const*)L.map,
                            L.place, cnt, x.pools, ring, x.slot, x.rec + (size_t)g * kXpSlots, x.nrec + g,
                            (XpBox*)rad_dev_device_ptr(x.box + g), x.tick + g, x.stats, (void*)cs))
        return RAD_E_DEVICE;
    if (x.pred.size() < x.layers.size()) x.pred.assign(x.layers.size(), 0);
    x.pred[(size_t)layer] = 1;
    return RAD_OK;
}

/* Ahead of a layer's routed GEMMs, on the compute stream: one kernel that puts back the other groups' patches
 * and, when this pass predicted for this layer, patches its group where the copies landed. Its arguments are fixed
 * per layer, so the recording repeats the same decision at every play. */
int Ctx::xp_step_rec(int layer) {
    XpState& x = *xp_;
    if (layer < 0 || layer >= (int)x.layers.size()) return RAD_OK;
    const int g = layer % kXpGroups;
    /* The prediction for this layer came at the previous layer's router input, earlier in the same pass. */
    const bool patch = (size_t)layer < x.pred.size() && x.pred[(size_t)layer] && x.layers[(size_t)layer].bound;
    if ((size_t)layer < x.pred.size()) x.pred[(size_t)layer] = 0;
    const uint32_t* done = (const uint32_t*)rad_dev_device_ptr(x.done + g);
    return rad_xp_step2_launch(x.rec, x.nrec, g, patch ? 1 : 0, done, x.tick + g, (void*)cur_stream())
               ? RAD_E_DEVICE : RAD_OK;
}

/* The host thread: watches every group's mailbox number (the plan bumps it), and hands each new request's jobs to
 * the copy engine with the number as the done value. Sleeps between looks: 50 us while plans keep coming, 1 ms
 * after half a second without one. */
void Ctx::xp_poll(XpState& x) {
    uint32_t last[kXpGroups];
    for (int g = 0; g < kXpGroups; ++g) last[g] = x.box[g].seq;
    int64_t idle = 0;
    bool warned = false;
    while (!x.stop.load()) {
        bool any = false;
        for (int g = 0; g < kXpGroups; ++g) {
            volatile XpBox* b = x.box + g;
            const uint32_t s = b->seq;
            if (s == last[g]) continue;
            last[g] = s;
            any = true;
            std::atomic_thread_fence(std::memory_order_acquire);
            const int n = std::min<int>((int)b->n, kXpSlots);
            uint64_t src[kXpSlots], dst[kXpSlots], bytes[kXpSlots];
            int m = 0;
            for (int i = 0; i < n; ++i) {
                if ((uint32_t)b->jobs[i].tag != s) { m = 0; break; }   /* a later plan is writing this group */
                src[m] = b->jobs[i].src;
                dst[m] = b->jobs[i].dst;
                bytes[m] = b->jobs[i].bytes;
                ++m;
            }
            if (m == 0) continue;
            if (rad_xp_behind(x.device) > kXpBehind) { ++x.dropped; continue; }
            if (rad_xp_list(x.device, nullptr, m, src, dst, bytes, x.done + g, s) == RAD_OK) {
                ++x.lists;
            } else if (!warned) {
                warned = true;
                const char* why = rad_dev_last_error();
                RAD_WARN("rank %d: the prefetch copy queue refused a list: %s", rank_, why && *why ? why : "?");
            }
        }
        idle = any ? 0 : idle + 1;
        std::this_thread::sleep_for(std::chrono::microseconds(idle > 10000 ? 1000 : 50));
    }
}

/* ---- Owner note 139 stage 0: the guess measured (see ctx.h: XgLayer) ---- */

int64_t Ctx::xg_on() {
    /* The decode passes to score; "1" means 4096. */
    static const int64_t n = [] {
        const char* v = std::getenv("RADIANCE_XP_GUESS");
        const long long x = v ? std::atoll(v) : 0;
        return (int64_t)(x == 1 ? 4096 : x > 0 ? x : 0);
    }();
    return n;
}

void Ctx::xg_open(const uint64_t* lo, const uint64_t* hi, int n) {
    xg_pools_.clear();
    for (int i = 0; i < n; ++i)
        if (hi[i] > lo[i]) xg_pools_.emplace_back(lo[i], hi[i]);
    xg_cap_ = xg_on();
    RAD_INFO("rank %d: expert guess tap on (owner note 139 stage 0): %lld decode passes, %d pinned pool ranges; "
             "no fetches", rank_, (long long)xg_cap_, (int)xg_pools_.size());
}

int Ctx::xg_guess(const RadPrefetchNext& nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    /* Rank 0 only: every rank routes the same rows over the same experts here, and the host products heat the
     * CPU (H87 stopped at 75 C). */
    if (rank_ != 0 || last_nondecode_ || rec_ || xg_steps_ >= xg_cap_ || nx.layer < 0 || nx.n_expert <= 0 || nx.top_k <= 0 ||
        nx.top_k > nx.n_expert || !nx.parts || nx.n_parts <= 0)
        return RAD_OK;
    if (xg_.size() < route_.size()) xg_.resize(route_.size());
    if ((size_t)nx.layer >= xg_.size()) return RAD_OK;
    XgLayer& L = xg_[(size_t)nx.layer];
    const RadStream s = cur_stream();
    if (L.router.empty()) {
        const void* w = weight_ptr(nx.router);
        if (!w) return RAD_OK;
        std::vector<uint16_t> wb((size_t)(nx.n_expert * n_embd));
        RAD_TRY(rad_memcpy_async(wb.data(), w, (int64_t)wb.size() * 2, s));
        RAD_TRY(rad_stream_sync(s));
        L.router.resize(wb.size());
        for (size_t i = 0; i < wb.size(); ++i) {
            const uint32_t u = (uint32_t)wb[i] << 16;
            std::memcpy(&L.router[i], &u, 4);
        }
        L.next = nx;
    }
    const char* hb = (const char*)buf_ptr(h);
    if (!hb) return RAD_OK;
    std::vector<uint16_t> xb((size_t)(rows * n_embd));
    RAD_TRY(rad_memcpy_async(xb.data(), hb + r0 * n_embd * 2, (int64_t)xb.size() * 2, s));
    RAD_TRY(rad_stream_sync(s));
    std::vector<float> x((size_t)n_embd), lg((size_t)nx.n_expert);
    std::vector<int32_t> ix((size_t)nx.n_expert);
    std::vector<uint8_t> g((size_t)nx.n_expert, 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n_embd; ++i) {
            const uint32_t u = (uint32_t)xb[(size_t)(r * n_embd + i)] << 16;
            std::memcpy(&x[(size_t)i], &u, 4);
        }
        for (int64_t e = 0; e < nx.n_expert; ++e) {
            const float* we = L.router.data() + e * n_embd;
            const float* xv = x.data();
            float a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };   /* eight sums, so the loop vectorises; n_embd % 8 == 0 */
            for (int64_t i = 0; i + 8 <= n_embd; i += 8)
                for (int j = 0; j < 8; ++j) a[j] += we[i + j] * xv[i + j];
            lg[(size_t)e] = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
            ix[(size_t)e] = (int32_t)e;
        }
        std::partial_sort(ix.begin(), ix.begin() + nx.top_k, ix.end(),
                          [&](int32_t a, int32_t b) { return lg[(size_t)a] > lg[(size_t)b]; });
        for (int64_t k = 0; k < nx.top_k; ++k) g[(size_t)ix[(size_t)k]] = 1;
    }
    L.guess.push_back(std::move(g));
    return RAD_OK;
}

void Ctx::xg_judge(int layer) {
    if (layer < 0 || (size_t)layer >= xg_.size() || last_nondecode_ || rec_) return;
    XgLayer& L = xg_[(size_t)layer];
    if (L.guess.empty()) return;
    const RadPrefetchNext& nx = L.next;
    const int32_t* cd = route_counts(layer, nx.n_expert);
    if (!cd || route_n_expert_ <= 0) { L.guess.clear(); return; }
    std::vector<int32_t> cnt((size_t)route_n_expert_);
    const RadStream s = cur_stream();
    if (rad_memcpy_async(cnt.data(), cd, (int64_t)cnt.size() * 4, s) < 0 || rad_stream_sync(s) < 0) {
        L.guess.clear();
        return;
    }
    /* In call order: the mixer input when the arch asked for it, then the router input. */
    const std::vector<uint8_t>* v0 = L.guess.size() >= 2 ? &L.guess[0] : nullptr;
    const std::vector<uint8_t>* v1 = &L.guess.back();
    const bool have_prev = !L.prev.empty();
    std::vector<uint8_t> routed_now((size_t)nx.n_expert, 0);
    for (int64_t e = 0; e < nx.n_expert; ++e) {
        const int32_t j = nx.count_at ? nx.count_at[e] : (int32_t)e;
        const rad_weight w = nx.parts[e * nx.n_parts];
        if (j < 0 || j >= route_n_expert_ || !w) continue;
        const bool routed = cnt[(size_t)j] > 0;
        const WeightSlot* ws = residency_ ? residency_->lookup(w) : nullptr;
        const uint64_t p = (uint64_t)(uintptr_t)(ws ? ws->ptr : nullptr);
        bool res = true;
        for (const auto& r : xg_pools_)
            if (p >= r.first && p < r.second) { res = false; break; }
        routed_now[(size_t)e] = routed;
        L.routed += routed;
        L.missed += routed && !res;
        const bool gs[4] = { v0 && (*v0)[(size_t)e], (bool)(*v1)[(size_t)e], have_prev && L.prev[(size_t)e],
                             (*v1)[(size_t)e] || (have_prev && L.prev[(size_t)e]) };
        for (int k = 0; k < 4; ++k) {
            if (!gs[k]) continue;
            L.v[k][0] += routed;
            L.v[k][1] += routed && !res;
            L.v[k][2] += !res;
            L.v[k][3] += !res && !routed;
            L.v[k][4] += 1;
        }
    }
    L.prev.swap(routed_now);
    L.guess.clear();
    ++L.passes;
    if (layer <= xg_last_) {
        ++xg_steps_;
        if (xg_steps_ % 128 == 0 || xg_steps_ == xg_cap_) xg_report(xg_steps_ % 1024 == 0 || xg_steps_ == xg_cap_);
    }
    xg_last_ = layer;
}

void Ctx::xg_report(bool layers) {
    uint64_t routed = 0, missed = 0, v[4][5] = {};
    for (const XgLayer& L : xg_) {
        routed += L.routed;
        missed += L.missed;
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < 5; ++i) v[k][i] += L.v[k][i];
    }
    const double st = xg_steps_ > 0 ? (double)xg_steps_ : 1.0;
    auto pc = [](uint64_t a, uint64_t b) { return b ? 100.0 * (double)a / (double)b : 0.0; };
    static const char* nm[4] = { "mixer-in", "router-in", "last-step", "router-in|last" };
    RAD_INFO("rank %d: guess over %lld decode steps: routed %.2f, missed %.2f a step (local experts)", rank_,
             (long long)xg_steps_, routed / st, missed / st);
    for (int k = 0; k < 4; ++k)
        RAD_INFO("rank %d: guess %-14s hit %.1f%% of routed, covers %.1f%% of misses, %.2f fetches a step "
                 "(%.1f%% unused), %.2f guessed a step", rank_, nm[k], pc(v[k][0], routed), pc(v[k][1], missed),
                 v[k][2] / st, pc(v[k][3], v[k][2]), v[k][4] / st);
    if (!layers) return;
    for (size_t l = 0; l < xg_.size(); ++l) {
        const XgLayer& L = xg_[l];
        if (!L.passes) continue;
        RAD_INFO("rank %d: guess layer %zu: routed %.2f missed %.2f | hit mixer-in %.1f%% router-in %.1f%% last %.1f%% "
                 "| covers %.1f%% %.1f%% %.1f%% %.1f%%", rank_, l, L.routed / (double)L.passes,
                 L.missed / (double)L.passes, pc(L.v[0][0], L.routed), pc(L.v[1][0], L.routed),
                 pc(L.v[2][0], L.routed), pc(L.v[0][1], L.missed), pc(L.v[1][1], L.missed), pc(L.v[2][1], L.missed),
                 pc(L.v[3][1], L.missed));
    }
}

}  /* namespace rad */
