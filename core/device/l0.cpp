/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * l0.cpp -- the Level Zero device backend.
 *
 * The card is driven through the SYCL runtime on its Level Zero plugin: a stream is an in-order
 * sycl::queue (an immediate command list underneath), memory is USM (device, host-pinned), an event
 * is the barrier a record submits, and a kernel library launches on the queue a RadStream points to.
 * The contract with libxe: RadStream points to a `sycl::queue` (in order, on the device's one
 * context); the backend's per-stream state follows it in L0Stream, so the cast works both ways.
 *
 * Devices: every Level Zero GPU, in the runtime's order, or the one RADIANCE_L0_PCI names
 * ("0000:0f:00.0"). RADIANCE_L0_DEVICE=cpu offers the OpenCL CPU device instead, so the backend and
 * the kernels are checked on a machine with no card (it is a device to the engine, not the host
 * backend: device rows run on it). RADIANCE_L0=0 turns the backend off.
 *
 * Compiled only in the Level Zero build (RAD_HAVE_L0, cmake/RadL0.cmake); elsewhere this file is empty.
 * Tapes are SYCL command graphs (see "recorded passes"). Left null (supported, see device.h): elastic memory, the expert prefetch, gates, link holds,
 * code-object checks.
 */
#if defined(RAD_HAVE_L0) && RAD_HAVE_L0

#include "device.h"
#include "../rad_internal.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <map>
#include <set>
#include <thread>
#include <vector>

/* core/device/l0_kernels.sycl (librad_l0k) */
extern "C" int rad_l0k_fill(void* stream, void* dst, int value, int64_t bytes);

namespace rad {
namespace {

constexpr int64_t kAlign = 4096;
constexpr int kMaxDevices = 16;
constexpr int64_t kRoomGuard = 64ll << 20;   /* the room check keeps this much of the card free */

struct Dev {
    sycl::device  d;
    sycl::context ctx;
    std::string   pci;     /* "" when the device has none (the CPU device) */
    bool          cpu = false;
    /* Event timestamps: profiling tags where the device has them (no profiling queue, whose launch
     * cost is ~20 us on the B70, round 2 M3); else profiling queues (the CPU device). */
    bool          tag = false;
};
std::vector<Dev>&  devs() { static std::vector<Dev> v; return v; }
std::atomic<int64_t> g_used[kMaxDevices];
thread_local int   t_device = 0;

struct AllocRec { int64_t bytes; int kind; int device; };
std::mutex& alloc_mutex() { static std::mutex m; return m; }
/* By base address, so an interior pointer finds its allocation (a pinned range handed out in pieces). */
std::map<uintptr_t, AllocRec>& alloc_table() {
    static std::map<uintptr_t, AllocRec> t;
    return t;
}

/* A stream. `q` first: a kernel library reads RadStream as the sycl::queue* it is. */
struct L0Stream {
    sycl::queue      q;
    std::atomic<int> gates{0};   /* closed gates on it */
    /* Whether any thread's backend call has named it yet; see t_touched. */
    std::atomic<bool> used{false};
    uint64_t          id = 0;      /* serial, never reused */
    explicit L0Stream(sycl::queue&& x) : q(std::move(x)) {}
};
std::mutex& stream_mutex() { static std::mutex m; return m; }
std::vector<L0Stream*>& streams() { static std::vector<L0Stream*> v; return v; }
std::atomic<uint64_t> g_stream_ids{0};
/* The streams this thread's backend calls have ever named, by serial: a tape records those (and
 * streams nobody has used yet), not a stream another thread drives (the mover's). Run 8 on card1:
 * keeping only the streams this thread named LAST lost the engine's own stream, which the load had
 * named from another thread, and the tape came out empty. */
thread_local std::set<uint64_t> t_touched;
thread_local L0Stream* t_last = nullptr;
void touch(L0Stream& s) {
    s.used.store(true, std::memory_order_relaxed);
    t_touched.insert(s.id);
    t_last = &s;
}

/* A gate: a host task on the queue that returns once the host opens it. */
struct Gate {
    std::mutex              m;
    std::condition_variable cv;
    bool                    open = false;
};
struct GateHandle {
    std::shared_ptr<Gate> g;
    L0Stream*             st;
};

struct L0Event {
    std::mutex  m;
    sycl::event ev;
    bool        recorded = false;
    unsigned    flags = 0;
    /* Recorded inside a tape: the graph-internal barrier, its tape and segment (see tapes). */
    sycl::event gev;
    uint64_t    gtape = 0;
    int         gseg = -1;
};
std::mutex& event_mutex() { static std::mutex m; return m; }
std::set<L0Event*>& events_alive() { static std::set<L0Event*> s; return s; }

/* SYCL reports asynchronous errors here; an in-order queue with no handler would lose them. */
void async_errors(sycl::exception_list el) {
    for (auto& e : el) {
        try { std::rethrow_exception(e); }
        catch (const sycl::exception& x) { RAD_WARN("device: Level Zero asynchronous error: %s", x.what()); }
    }
}

int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

bool dev_ok(int device, const char* who) {
    if (device >= 0 && device < (int)devs().size()) return true;
    device_set_error("%s(%d): the Level Zero backend has %d device(s)", who, device, (int)devs().size());
    return false;
}

int check_stream(RadStream s, const char* who) {
    if (s) return RAD_OK;
    device_set_error("%s: null stream -- radiance does not use the default stream, create one with "
                     "rad_stream_create", who);
    return RAD_E_INVAL;
}
L0Stream&    S(RadStream s) { return *reinterpret_cast<L0Stream*>(s); }
sycl::queue& Q(RadStream s) { return S(s).q; }

/* Runs `f`, turning a SYCL exception into the error slot and RAD_E_DEVICE. */
template <class F> int guarded(const char* who, F&& f) {
    try { f(); return RAD_OK; }
    catch (const sycl::exception& e) { device_set_error("%s: %s", who, e.what()); return RAD_E_DEVICE; }
}

/* ------------------------------------------------------------------ devices */
int z_count(void) { return (int)devs().size(); }

int z_set(int device) {
    if (!dev_ok(device, "rad_dev_set")) return RAD_E_INVAL;
    t_device = device;
    return RAD_OK;
}

int z_props(int device, RadDeviceProps* out) {
    if (!out) return RAD_E_INVAL;
    if (!dev_ok(device, "rad_dev_props")) return RAD_E_INVAL;
    const Dev& D = devs()[device];
    std::memset(out, 0, sizeof *out);
    try {
        std::snprintf(out->name, sizeof out->name, "%s", D.d.get_info<sycl::info::device::name>().c_str());
        std::snprintf(out->arch, sizeof out->arch, "%s", D.cpu ? "sycl-cpu" : "xe2-bmg-g31");
        out->device_id = device;
        out->n_cu = (int)D.d.get_info<sycl::info::device::max_compute_units>();
        out->warp_size = 32;   /* every libxe kernel runs at sub-group 32 */
        out->vram_bytes = (int64_t)D.d.get_info<sycl::info::device::global_mem_size>();
        if (D.cpu) {
            /* The CPU device's "VRAM" is this machine's RAM: offer a bounded budget, 16 GiB or
             * RADIANCE_L0_CPU_MIB, so a planner never sizes pools from the whole of it. */
            const char* mib = std::getenv("RADIANCE_L0_CPU_MIB");
            const int64_t cap = (mib && *mib ? std::atoll(mib) : 16384) << 20;
            if (out->vram_bytes > cap) out->vram_bytes = cap;
        }
        const int64_t used = g_used[device].load(std::memory_order_relaxed);
        out->vram_free = out->vram_bytes > used ? out->vram_bytes - used : 0;
        if (!D.cpu && D.d.has(sycl::aspect::ext_intel_free_memory))
            out->vram_free = (int64_t)D.d.get_info<sycl::ext::intel::info::device::free_memory>();
        out->lds_bytes = (int64_t)D.d.get_info<sycl::info::device::local_mem_size>();
        out->supports_p2p = 0;
        out->is_host_backend = 0;
    } catch (const sycl::exception& e) {
        device_set_error("rad_dev_props(%d): %s", device, e.what());
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

int z_enable_peer(int self, int peer) {
    if (!dev_ok(self, "rad_dev_enable_peer") || !dev_ok(peer, "rad_dev_enable_peer")) return RAD_E_INVAL;
    if (self == peer) return RAD_OK;
    device_set_error("rad_dev_enable_peer(%d, %d): no peer access in milestone 1 (one card)", self, peer);
    return RAD_E_UNSUPPORTED;
}
int z_peer_access(int self, int peer) { return self == peer ? 1 : 0; }

/* ------------------------------------------------------------------ memory */
void* z_alloc(int64_t bytes, int kind) {
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    if (kind < RAD_MEM_DEVICE || kind > RAD_MEM_HOST_MAPPED) {
        device_set_error("rad_dev_alloc: unknown memory kind %d", kind);
        return nullptr;
    }
    const int dev = t_device;
    if (!dev_ok(dev, "rad_dev_alloc")) return nullptr;
    const Dev& D = devs()[dev];
    /* Whole pages, so DirectFile can read an aligned superset into any allocation. */
    const size_t n = (size_t)align_up(bytes, kAlign);
    if (kind == RAD_MEM_DEVICE) {
        /* The room check: refuse what the card cannot hold here, with the numbers, rather than
         * leave it to the driver (which may overcommit and evict, round 2 M5). */
        RadDeviceProps pr;
        if (z_props(dev, &pr) == RAD_OK && (int64_t)n + kRoomGuard > pr.vram_free) {
            device_set_error("rad_dev_alloc(%lld) on card %d: the driver has %lld MiB free of %lld MiB, and "
                             "this process holds %lld MiB of it", (long long)bytes, dev,
                             (long long)(pr.vram_free >> 20), (long long)(pr.vram_bytes >> 20),
                             (long long)(g_used[dev].load() >> 20));
            return nullptr;
        }
    }
    void* p = nullptr;
    try {
        if (kind == RAD_MEM_DEVICE)   p = sycl::aligned_alloc_device(kAlign, n, D.d, D.ctx);
        else if (kind == RAD_MEM_HOST) { if (posix_memalign(&p, (size_t)kAlign, n) != 0) p = nullptr; }
        else                          p = sycl::aligned_alloc_host(kAlign, n, D.ctx);   /* pinned + mapped */
    } catch (const sycl::exception& e) {
        device_set_error("rad_dev_alloc(%lld, kind=%d): %s", (long long)bytes, kind, e.what());
        return nullptr;
    }
    if (!p) {
        device_set_error("rad_dev_alloc(%lld, kind=%d): out of %s memory", (long long)bytes, kind,
                         kind == RAD_MEM_DEVICE ? "device" : "host");
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(alloc_mutex());
        alloc_table()[(uintptr_t)p] = AllocRec{bytes, kind, dev};
    }
    if (kind == RAD_MEM_DEVICE) g_used[dev].fetch_add(bytes, std::memory_order_relaxed);
    return p;
}

void z_free(void* p, int kind) {
    if (!p) return;
    AllocRec r;
    {
        std::lock_guard<std::mutex> lk(alloc_mutex());
        auto it = alloc_table().find((uintptr_t)p);
        if (it == alloc_table().end()) {
            RAD_WARN("rad_dev_free(%p, kind=%d): pointer was not allocated by the device layer", p, kind);
            return;
        }
        r = it->second;
        alloc_table().erase(it);
    }
    if (r.kind != kind) RAD_WARN("rad_dev_free(%p): freed as kind %d, allocated as kind %d", p, kind, r.kind);
    if (r.kind == RAD_MEM_HOST) { std::free(p); return; }
    if (r.kind == RAD_MEM_DEVICE) g_used[r.device].fetch_sub(r.bytes, std::memory_order_relaxed);
    try { sycl::free(p, devs()[r.device].ctx); }
    catch (const sycl::exception& e) { RAD_WARN("rad_dev_free(%p): %s", p, e.what()); }
}

/* The allocation holding `p`, interior pointers included. */
bool find_alloc(void* p, AllocRec* out) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    auto it = alloc_table().upper_bound((uintptr_t)p);
    if (it == alloc_table().begin()) return false;
    --it;
    if ((uintptr_t)p >= it->first + (uintptr_t)it->second.bytes) return false;
    *out = it->second;
    return true;
}
/* USM host memory has one address on both sides; device memory has no host view (on the CPU device
 * it does, but nothing may rely on that). */
void* z_host_ptr(void* p) {
    AllocRec r;
    if (!p || !find_alloc(p, &r)) return nullptr;
    return r.kind == RAD_MEM_DEVICE ? nullptr : p;
}
void* z_device_ptr(void* p) {
    AllocRec r;
    if (!p || !find_alloc(p, &r)) return nullptr;
    return r.kind == RAD_MEM_HOST ? nullptr : p;
}

/* ------------------------------------------------------------------ recorded passes (tapes)
 * A tape is a list of segments, each a SYCL command graph recorded (record mode) from the streams
 * this thread submits on, kernel libraries' launches included. tape_mark closes the open segment:
 * finalize, submit (the pass runs while it is recorded, as on the AQL backend) and keep the
 * executable graph; a record count is a segment count. tape_play(a, b) submits one graph holding
 * segments [a, b) as sub-graphs, built at the stretch's first play and kept: one submission a
 * stretch. Every segment runs on the home stream (the thread's last), fenced against the other
 * recorded streams before and after, so the segments of a tape are in order. An event record in a
 * segment is a graph barrier for waits inside the segment and is recorded for real after it (and
 * after each played stretch holding it): later than its place, which only delays a waiter.
 * Refused (tape_end returns null, the reason in the error slot, and the engine issues that pass
 * in full): a gate, a wait on an event recorded outside the pass, a stream outside the recording. */
namespace sx = sycl::ext::oneapi::experimental;
using Graph = sx::command_graph<sx::graph_state::modifiable>;
using Exec  = sx::command_graph<sx::graph_state::executable>;

struct Seg {
    std::shared_ptr<Exec> ex;          /* null when the segment holds only event records */
    std::vector<L0Event*> evs;         /* recorded for real after it */
    size_t nodes = 0;
};
struct Tape {
    std::vector<Seg> segs;
    std::map<std::pair<int, int>, std::shared_ptr<Exec>> stretch;
    std::vector<L0Stream*> qs;         /* the recorded streams, the home stream first */
    std::string refused;
    uint64_t id = 0;                   /* serial, never reused (L0Event::gtape) */
};
std::atomic<uint64_t> g_tape_ids{0};
struct Rec {
    Tape* t = nullptr;
    std::optional<Graph> g;            /* the open segment */
    std::vector<L0Event*> pend;        /* its event records */
    int pause = 0;
};
thread_local Rec t_rec;

bool recording() { return t_rec.t && t_rec.g && t_rec.pause == 0; }
bool recorded_stream(const L0Stream& s) {
    for (L0Stream* x : t_rec.t->qs) if (x == &s) return true;
    return false;
}
void refuse(const std::string& why) {
    if (t_rec.t && t_rec.t->refused.empty()) t_rec.t->refused = why;
}

/* A real record of `e` on `q`: a profiling tag where the device has them, else a barrier. */
void real_record(L0Event* e, sycl::queue& q) {
    sycl::event b = devs()[0].tag ? sx::submit_profiling_tag(q) : q.ext_oneapi_submit_barrier();
    std::lock_guard<std::mutex> lk(e->m);
    e->ev = b;
    e->recorded = true;
}
/* Submits `ex` on the home stream, fenced against the other recorded streams. */
void run_on_home(Tape& t, Exec& ex) {
    sycl::queue& H = t.qs[0]->q;
    if (t.qs.size() > 1) {
        std::vector<sycl::event> pre;
        for (size_t k = 1; k < t.qs.size(); ++k) pre.push_back(t.qs[k]->q.ext_oneapi_submit_barrier());
        H.ext_oneapi_submit_barrier(pre);
    }
    sycl::event ge = H.ext_oneapi_graph(ex);
    for (size_t k = 1; k < t.qs.size(); ++k) t.qs[k]->q.ext_oneapi_submit_barrier({ge});
}
void rerecord(Tape& t, const std::vector<L0Event*>& evs) {
    std::lock_guard<std::mutex> lk(event_mutex());
    for (L0Event* e : evs)
        if (events_alive().count(e)) real_record(e, t.qs[0]->q);
}

void seg_open() {
    Tape& t = *t_rec.t;
    const Dev& D = devs()[t_device];
    t_rec.g.emplace(D.ctx, D.d);
    std::vector<sycl::queue> qs;
    for (L0Stream* s : t.qs) qs.push_back(s->q);
    t_rec.g->begin_recording(qs);
}
/* Closes the open segment: finalize, run, keep. */
void seg_close() {
    Tape& t = *t_rec.t;
    std::vector<sycl::queue> qs;
    for (L0Stream* s : t.qs) qs.push_back(s->q);
    t_rec.g->end_recording(qs);
    Seg sg;
    sg.nodes = t_rec.g->get_nodes().size();
    sg.evs.swap(t_rec.pend);
    if (sg.nodes) {
        sg.ex = std::make_shared<Exec>(t_rec.g->finalize());
        run_on_home(t, *sg.ex);
    }
    t_rec.g.reset();
    if (sg.nodes || !sg.evs.empty()) {
        rerecord(t, sg.evs);
        t.segs.push_back(std::move(sg));
    }
}
/* A host wait inside a recording: what is recorded so far runs, then a new segment opens. */
template <class F> int seg_cut(const char* who, F&& f) {
    return guarded(who, [&] { seg_close(); f(); seg_open(); });
}

int z_tape_begin(void) {
    if (t_rec.t) { device_set_error("rad tape_begin: a tape is already being recorded on this thread"); return RAD_E_STATE; }
    if (!dev_ok(t_device, "tape_begin")) return RAD_E_INVAL;
    const Dev& D = devs()[t_device];
    /* RADIANCE_L0_TAPE=0: every pass is issued in full (owner note 1's plain-launch measurement) */
    static const bool off = [] { const char* v = std::getenv("RADIANCE_L0_TAPE"); return v && std::strcmp(v, "0") == 0; }();
    if (off) { device_set_error("tape_begin: RADIANCE_L0_TAPE=0"); return RAD_E_UNSUPPORTED; }
    if (!D.d.has(sycl::aspect::ext_oneapi_graph) && !D.d.has(sycl::aspect::ext_oneapi_limited_graph)) {
        device_set_error("tape_begin: the device takes no SYCL command graphs");
        return RAD_E_UNSUPPORTED;
    }
    auto t = std::make_unique<Tape>();
    t->id = ++g_tape_ids;
    {
        std::lock_guard<std::mutex> lk(stream_mutex());
        for (L0Stream* s : streams())
            if (t_touched.count(s->id) || !s->used.load(std::memory_order_relaxed)) t->qs.push_back(s);
        for (size_t k = 0; k < t->qs.size(); ++k)
            if (t->qs[k] == t_last) std::swap(t->qs[0], t->qs[k]);
    }
    if (t->qs.empty()) { device_set_error("tape_begin: no stream to record"); return RAD_E_STATE; }
    t_rec = Rec{};
    t_rec.t = t.release();
    const int rc = guarded("tape_begin", [] { seg_open(); });
    if (rc != RAD_OK) { delete t_rec.t; t_rec = Rec{}; }
    return rc;
}
int z_tape_mark(void) {
    if (!t_rec.t) return 0;
    if (recording()) {
        const int rc = guarded("tape_mark", [] { seg_close(); seg_open(); });
        if (rc != RAD_OK) refuse(std::string("a segment could not be closed: ") + rad_dev_last_error());
    }
    return (int)t_rec.t->segs.size();
}
void z_tape_pause(int on) {
    if (!t_rec.t) return;
    if (on) {
        if (t_rec.pause++ == 0 && t_rec.g && guarded("tape_pause", [] { seg_close(); }) != RAD_OK)
            refuse("a segment could not be closed at a pause");
    } else if (t_rec.pause > 0 && --t_rec.pause == 0) {
        if (guarded("tape_pause", [] { seg_open(); }) != RAD_OK) refuse("no segment after a pause");
    }
}
RadTape z_tape_end(int keep) {
    Tape* t = t_rec.t;
    if (!t) return nullptr;
    if (t_rec.g && guarded("tape_end", [] { seg_close(); }) != RAD_OK) refuse("the last segment could not be closed");
    t_rec = Rec{};
    /* A pass always puts work on the card; an empty recording means it went to a stream outside
     * the recording, and replaying nothing would be silently wrong. */
    if (t->segs.empty() && t->refused.empty())
        t->refused = "nothing was recorded (the pass's work went to a stream outside the recording)";
    if (!keep || !t->refused.empty()) {
        if (keep) device_set_error("the pass cannot be replayed from a SYCL graph: %s", t->refused.c_str());
        delete t;
        return nullptr;
    }
    return (RadTape)t;
}
int z_tape_len(RadTape p) { return p ? (int)static_cast<Tape*>(p)->segs.size() : RAD_E_INVAL; }
int z_tape_play(RadTape p, int from, int to) {
    auto* t = static_cast<Tape*>(p);
    if (!t || from < 0 || to > (int)t->segs.size() || from > to) { device_set_error("tape_play: bad range"); return RAD_E_INVAL; }
    if (from == to) return RAD_OK;
    return guarded("tape_play", [&] {
        std::shared_ptr<Exec>& ex = t->stretch[{from, to}];
        if (!ex) {
            const Dev& D = devs()[t_device];
            Graph g(D.ctx, D.d);
            std::optional<sx::node> prev;
            for (int i = from; i < to; ++i) {
                if (!t->segs[(size_t)i].ex) continue;
                Exec& sub = *t->segs[(size_t)i].ex;
                auto add = [&](sycl::handler& h) { h.ext_oneapi_graph(sub); };
                prev = prev ? g.add(add, sycl::property_list{sx::property::node::depends_on(*prev)}) : g.add(add);
            }
            if (prev) ex = std::make_shared<Exec>(g.finalize());
        }
        if (ex) run_on_home(*t, *ex);
        std::vector<L0Event*> evs;
        for (int i = from; i < to; ++i) evs.insert(evs.end(), t->segs[(size_t)i].evs.begin(), t->segs[(size_t)i].evs.end());
        rerecord(*t, evs);
    });
}
/* Segment by segment: node and event-record counts (SYCL keeps a graph's arguments to itself, so
 * this is weaker than the AQL tape's packet comparison; the pass key fixes the buffers). */
int z_tape_diff(RadTape pa, RadTape pb, char* why, int cap) {
    auto* a = static_cast<Tape*>(pa);
    auto* b = static_cast<Tape*>(pb);
    if (!a || !b) return RAD_E_INVAL;
    const size_t n = std::min(a->segs.size(), b->segs.size());
    for (size_t i = 0; i < n; ++i)
        if (a->segs[i].nodes != b->segs[i].nodes || a->segs[i].evs.size() != b->segs[i].evs.size()) {
            if (why && cap > 0)
                std::snprintf(why, (size_t)cap, "segment %zu: %zu nodes, %zu event records against %zu, %zu", i,
                              a->segs[i].nodes, a->segs[i].evs.size(), b->segs[i].nodes, b->segs[i].evs.size());
            return (int)i;
        }
    if (a->segs.size() != b->segs.size()) {
        if (why && cap > 0) std::snprintf(why, (size_t)cap, "%zu segments against %zu", a->segs.size(), b->segs.size());
        return (int)n;
    }
    return -1;
}
void z_tape_destroy(RadTape p) { delete static_cast<Tape*>(p); }   /* graphs in flight hold their own references */
/* A segment is ordered behind everything before it: an event-kind record, so no overlap is asked. */
int z_tape_record(RadTape p, int i, int* kind, uint32_t* queue) {
    auto* t = static_cast<Tape*>(p);
    if (!t || i < 0 || i >= (int)t->segs.size()) return RAD_E_INVAL;
    if (kind) *kind = 1;
    if (queue) *queue = 0;
    return RAD_OK;
}
int z_tape_set_overlap(RadTape, int, int) { return RAD_OK; }

/* ------------------------------------------------------------------ streams */
int z_stream_create(RadStream* out, int high_priority) {
    if (!out) return RAD_E_INVAL;
    const int dev = t_device;
    if (!dev_ok(dev, "rad_stream_create")) return RAD_E_INVAL;
    const Dev& D = devs()[dev];
    try {
        using sycl::property::queue::in_order;
        using sycl::property::queue::enable_profiling;
        using sycl::ext::oneapi::property::queue::priority_high;
        sycl::property_list pl =
            D.tag ? (high_priority ? sycl::property_list{in_order(), priority_high()} : sycl::property_list{in_order()})
                  : (high_priority ? sycl::property_list{in_order(), priority_high(), enable_profiling()}
                                   : sycl::property_list{in_order(), enable_profiling()});
        auto* st = new L0Stream(sycl::queue(D.ctx, D.d, async_errors, pl));
        st->id = ++g_stream_ids;
        if ((void*)&st->q != (void*)st) { delete st; device_set_error("rad_stream_create: layout"); return RAD_E_STATE; }
        {
            std::lock_guard<std::mutex> lk(stream_mutex());
            streams().push_back(st);
        }
        *out = (RadStream)st;
    } catch (const sycl::exception& e) {
        device_set_error("rad_stream_create: %s", e.what());
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

void z_stream_destroy(RadStream s) {
    if (!s) return;
    try { Q(s).wait_and_throw(); }   /* drains rather than discards, as on the host backend */
    catch (const sycl::exception& e) { RAD_WARN("rad_stream_destroy: %s", e.what()); }
    {
        std::lock_guard<std::mutex> lk(stream_mutex());
        auto& v = streams();
        v.erase(std::remove(v.begin(), v.end(), &S(s)), v.end());
    }
    if (t_last == &S(s)) t_last = nullptr;
    delete &S(s);
}

/* Waits for the stream's work. */
int z_stream_sync(RadStream s) {
    RAD_TRY(check_stream(s, "rad_stream_sync"));
    if (S(s).gates.load() > 0) {
        device_set_error("rad_stream_sync: the stream waits at a closed gate, which only rad_gate_open releases");
        return RAD_E_STATE;
    }
    /* queue::wait: it may also wait for work other threads issue meanwhile (HIP waits only for the
     * work before the call), the simplest form that is never short. Run 3 tried barriers that name
     * the queue's last event: ext_oneapi_get_last_event misbehaved behind a gate's host task on the
     * CPU device (an early return, then a runtime abort). device_test's concurrent-sync case still
     * passes on card1 (run b2). */
    touch(S(s));
    if (recording()) return seg_cut("rad_stream_sync", [&] { Q(s).wait_and_throw(); });
    return guarded("rad_stream_sync", [&] { Q(s).wait_and_throw(); });
}

/* ------------------------------------------------------------------ gates */
int z_stream_gate(RadStream s, void** out) {
    RAD_TRY(check_stream(s, "rad_stream_gate"));
    if (!out) return RAD_E_INVAL;
    auto g = std::make_shared<Gate>();
    auto* h = new (std::nothrow) GateHandle{g, &S(s)};
    if (!h) return RAD_E_NOMEM;
    touch(S(s));
    auto submit = [&] {
        Q(s).submit([&](sycl::handler& cgh) {
            cgh.host_task([g] {
                std::unique_lock<std::mutex> lk(g->m);
                g->cv.wait(lk, [&] { return g->open; });
            });
        });
    };
    /* Not inside a tape (device.h): the gate goes on the queue outside the graph, and the tape is refused. */
    if (recording()) refuse("a gate inside the pass");
    const int rc = recording() ? seg_cut("rad_stream_gate", submit) : guarded("rad_stream_gate", submit);
    if (rc != RAD_OK) { delete h; return rc; }
    S(s).gates.fetch_add(1);
    *out = h;
    return RAD_OK;
}
int z_gate_open(void* p) {
    auto* h = static_cast<GateHandle*>(p);
    if (!h || !h->g) return RAD_E_INVAL;
    {
        std::lock_guard<std::mutex> lk(h->g->m);
        h->g->open = true;
    }
    h->g->cv.notify_all();
    h->st->gates.fetch_sub(1);
    delete h;   /* single use; the host task keeps the gate itself alive */
    return RAD_OK;
}
/* An immediate command list has no small ring to fill. */
int64_t z_gate_room(RadStream s) { return s ? (int64_t)1 << 20 : 0; }

/* ------------------------------------------------------------------ events */
int z_event_create(RadEvent* out, unsigned flags) {
    if (!out) return RAD_E_INVAL;
    auto* e = new (std::nothrow) L0Event();
    if (!e) { device_set_error("rad_event_create: out of memory"); return RAD_E_NOMEM; }
    e->flags = flags;
    {
        std::lock_guard<std::mutex> lk(event_mutex());
        events_alive().insert(e);
    }
    *out = (RadEvent)e;
    return RAD_OK;
}
/* The sycl::event keeps the runtime's own reference; the handle can go at once. */
void z_event_destroy(RadEvent ev) {
    {
        std::lock_guard<std::mutex> lk(event_mutex());
        events_alive().erase(static_cast<L0Event*>(ev));
    }
    delete static_cast<L0Event*>(ev);
}

/* Inside a recording: an event recorded in the open segment is a graph barrier there; a host wait
 * on it first runs what is recorded (seg_cut). True when `e` is pending in the open segment. */
bool pending_here(L0Event* e) {
    std::lock_guard<std::mutex> lk(e->m);
    return recording() && e->gtape == t_rec.t->id && e->gseg == (int)t_rec.t->segs.size();
}

int z_event_record(RadEvent ev, RadStream s) {
    if (!ev || !s) {
        device_set_error("rad_event_record: null %s", ev ? "stream" : "event");
        return RAD_E_INVAL;
    }
    auto* e = static_cast<L0Event*>(ev);
    touch(S(s));
    if (recording()) {
        if (!recorded_stream(S(s))) { refuse("an event record on a stream outside the recording"); }
        else return guarded("rad_event_record", [&] {
            sycl::event b = Q(s).ext_oneapi_submit_barrier();   /* a graph node */
            std::lock_guard<std::mutex> lk(e->m);
            e->gev = b;
            e->gtape = t_rec.t->id;
            e->gseg = (int)t_rec.t->segs.size();
            t_rec.pend.push_back(e);
        });
    }
    /* everything before it on the queue; a profiling tag carries a timestamp as well */
    return guarded("rad_event_record", [&] { real_record(e, Q(s)); });
}

int z_event_wait(RadStream s, RadEvent ev) {
    if (!ev || !s) {
        device_set_error("rad_event_wait: null %s", ev ? "stream" : "event");
        return RAD_E_INVAL;
    }
    auto* e = static_cast<L0Event*>(ev);
    touch(S(s));
    sycl::event b;
    if (recording()) {
        bool here = false, earlier = false, never = false;
        {
            std::lock_guard<std::mutex> lk(e->m);
            here = e->gtape == t_rec.t->id && e->gseg == (int)t_rec.t->segs.size();
            earlier = e->gtape == t_rec.t->id && e->gseg < (int)t_rec.t->segs.size();
            never = !e->recorded && e->gtape != t_rec.t->id;
            b = e->gev;
        }
        if (here) return guarded("rad_event_wait", [&] { Q(s).ext_oneapi_submit_barrier({b}); });
        if (earlier || never) return RAD_OK;   /* the segments run in order; a never-recorded event is no wait */
        /* a graph cannot depend on outside work: refused, and this pass waits for real between segments */
        refuse("a wait on an event recorded outside the pass");
        {
            std::lock_guard<std::mutex> lk(e->m);
            b = e->ev;
        }
        return seg_cut("rad_event_wait", [&] { Q(s).ext_oneapi_submit_barrier({b}); });
    }
    {
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_OK;   /* a never-recorded event is no wait, as hipStreamWaitEvent */
        b = e->ev;
    }
    return guarded("rad_event_wait", [&] { Q(s).ext_oneapi_submit_barrier({b}); });
}

int z_event_query(RadEvent ev) {
    if (!ev) return RAD_E_INVAL;
    auto* e = static_cast<L0Event*>(ev);
    if (pending_here(e)) RAD_TRY(seg_cut("rad_event_query", [] {}));
    std::lock_guard<std::mutex> lk(e->m);
    if (!e->recorded) return 1;
    try {
        return e->ev.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete ? 1 : 0;
    } catch (const sycl::exception& x) {
        device_set_error("rad_event_query: %s", x.what());
        return RAD_E_DEVICE;
    }
}

int z_event_sync(RadEvent ev) {
    if (!ev) return RAD_E_INVAL;
    auto* e = static_cast<L0Event*>(ev);
    if (!(e->flags & RAD_EVENT_HOST_WAIT)) {
        device_set_error("rad_event_sync: the event was made without RAD_EVENT_HOST_WAIT");
        return RAD_E_INVAL;
    }
    if (pending_here(e)) RAD_TRY(seg_cut("rad_event_sync", [] {}));
    sycl::event b;
    {
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_OK;
        b = e->ev;
    }
    return guarded("rad_event_sync", [&] { b.wait_and_throw(); });
}

/* From the records' end timestamps (profiling tags, or the CPU device's profiling queues). */
int z_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    if (!a || !b || !out) return RAD_E_INVAL;
    auto stamp = [](RadEvent ev, uint64_t* t) {
        auto* e = static_cast<L0Event*>(ev);
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_E_STATE;
        if (e->ev.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) return RAD_E_STATE;
        *t = e->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        return RAD_OK;
    };
    uint64_t ta = 0, tb = 0;
    try {
        RAD_TRY(stamp(a, &ta));
        RAD_TRY(stamp(b, &tb));
    } catch (const sycl::exception& e) {
        device_set_error("rad_event_elapsed_ms: %s", e.what());
        return RAD_E_DEVICE;
    }
    *out = (float)((double)((int64_t)(tb - ta)) * 1e-6);
    return RAD_OK;
}

/* Host writes to USM host memory are visible to work submitted after them. */
void z_host_wrote(void) {}

/* ------------------------------------------------------------------ copies and fills */
int z_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_async"));
    touch(S(s));
    if (bytes < 0 || (bytes > 0 && (!dst || !src))) {
        device_set_error("rad_memcpy_async: bad arguments (%p, %p, %lld)", dst, src, (long long)bytes);
        return RAD_E_INVAL;
    }
    if (bytes == 0) return RAD_OK;
    return guarded("rad_memcpy_async", [&] { Q(s).memcpy(dst, src, (size_t)bytes); });
}

int z_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch, int64_t width,
                      int64_t height, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_2d_async"));
    touch(S(s));
    if (width < 0 || height < 0 || dpitch < width || spitch < width || (width && height && (!dst || !src))) {
        device_set_error("rad_memcpy_2d_async: bad arguments");
        return RAD_E_INVAL;
    }
    if (!width || !height) return RAD_OK;
    return guarded("rad_memcpy_2d_async", [&] {
        Q(s).ext_oneapi_memcpy2d(dst, (size_t)dpitch, src, (size_t)spitch, (size_t)width, (size_t)height);
    });
}

int z_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memset_async"));
    touch(S(s));
    if (bytes < 0 || (bytes > 0 && !dst)) { device_set_error("rad_memset_async: bad arguments"); return RAD_E_INVAL; }
    if (bytes == 0) return RAD_OK;
    /* As on a HIP card (device.h, contract note 1): the destination must be device-accessible. */
    if (sycl::get_pointer_type(dst, Q(s).get_context()) == sycl::usm::alloc::unknown) {
        device_set_error("rad_memset_async(%p): pageable host memory; a card fills only device-accessible "
                         "memory (RAD_MEM_DEVICE, RAD_MEM_HOST_PINNED or RAD_MEM_HOST_MAPPED)", dst);
        return RAD_E_UNSUPPORTED;
    }
    /* Only the low byte, as hipMemsetAsync and memset. Host memory is filled by a kernel
     * (l0_kernels.sycl rad_l0k_fill): queue::memset left the tail of a pinned range unwritten on
     * card1. Device memory keeps the runtime's fill. */
    if (sycl::get_pointer_type(dst, Q(s).get_context()) != sycl::usm::alloc::device)
        return rad_l0k_fill(s, dst, value & 0xff, bytes) == 0 ? RAD_OK : RAD_E_DEVICE;
    return guarded("rad_memset_async", [&] { Q(s).memset(dst, value & 0xff, (size_t)bytes); });
}

/* The ranges are read here, at the call: the contract keeps the list unchanged until the stream
 * passes the copy, so reading it early is equivalent. One copy a range. */
int z_memcpy_ranges_async(void* dst, const void* src, int64_t prefix, const RadCopyRange* ranges, int n,
                          RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_ranges_async"));
    touch(S(s));
    if (!dst || !src || prefix < 0 || prefix % 16 || n < 0 || (n > 0 && !ranges)) {
        device_set_error("rad_memcpy_ranges_async: bad arguments (the prefix is a multiple of 16)");
        return RAD_E_INVAL;
    }
    /* The same refusals as the card's 16-byte copy kernel (aql_blit.hip), so a caller bug fails
     * here as there. */
    if (((uintptr_t)dst | (uintptr_t)src) % 16) {
        device_set_error("rad_memcpy_ranges_async: both bases must be 16-byte aligned");
        return RAD_E_UNSUPPORTED;
    }
    return guarded("rad_memcpy_ranges_async", [&] {
        auto& q = Q(s);
        if (prefix) q.memcpy(dst, src, (size_t)prefix);
        for (int i = 0; i < n; ++i)
            if (ranges[i].bytes > 0)
                q.memcpy((char*)dst + ranges[i].dst, (const char*)src + ranges[i].src, (size_t)ranges[i].bytes);
    });
}

int z_memcpy_list_async(const RadCopySeg* segs, int n, unsigned, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_list_async"));
    touch(S(s));
    if (n < 0 || (n > 0 && !segs)) { device_set_error("rad_memcpy_list_async: bad arguments"); return RAD_E_INVAL; }
    return guarded("rad_memcpy_list_async", [&] {
        for (int i = 0; i < n; ++i)
            if (segs[i].bytes > 0) Q(s).memcpy(segs[i].dst, segs[i].src, (size_t)segs[i].bytes);
    });
}

/* ------------------------------------------------------------------ identity */
int z_pci_address(int device, char* out, int cap) {
    if (!dev_ok(device, "rad_dev_pci_address") || !out || cap <= 0) return RAD_E_INVAL;
    if (devs()[device].pci.empty()) return RAD_E_UNSUPPORTED;
    std::snprintf(out, (size_t)cap, "%s", devs()[device].pci.c_str());
    return RAD_OK;
}
int z_runtime_version(char* out, int cap) {
    if (!out || cap <= 0) return RAD_E_INVAL;
    if (devs().empty()) return RAD_E_UNSUPPORTED;
    const Dev& D = devs()[0];
    try {
        std::snprintf(out, (size_t)cap, "%s %s, driver %s",
                      D.cpu ? "SYCL/OpenCL" : "SYCL/Level Zero",
                      D.d.get_platform().get_info<sycl::info::platform::version>().c_str(),
                      D.d.get_info<sycl::info::device::driver_version>().c_str());
    } catch (const sycl::exception&) { return RAD_E_DEVICE; }
    return RAD_OK;
}

DeviceBackend make_backend() {
    DeviceBackend b{};
    b.name = "l0";
    b.count = z_count;
    b.set = z_set;
    b.props = z_props;
    b.enable_peer = z_enable_peer;
    b.alloc = z_alloc;
    b.free = z_free;
    b.host_ptr = z_host_ptr;
    b.device_ptr = z_device_ptr;
    b.stream_create = z_stream_create;
    b.stream_destroy = z_stream_destroy;
    b.stream_sync = z_stream_sync;
    b.event_create = z_event_create;
    b.event_destroy = z_event_destroy;
    b.event_record = z_event_record;
    b.event_wait = z_event_wait;
    b.event_query = z_event_query;
    b.event_sync = z_event_sync;
    b.event_elapsed_ms = z_event_elapsed_ms;
    b.host_wrote = z_host_wrote;
    b.memcpy_async = z_memcpy_async;
    b.memcpy_2d_async = z_memcpy_2d_async;
    b.memset_async = z_memset_async;
    b.memcpy_ranges_async = z_memcpy_ranges_async;
    b.memcpy_list_async = z_memcpy_list_async;
    b.peer_access = z_peer_access;
    b.pci_address = z_pci_address;
    b.runtime_version = z_runtime_version;
    b.stream_gate = z_stream_gate;
    b.gate_open = z_gate_open;
    b.gate_room = z_gate_room;
    b.tape_begin = z_tape_begin;
    b.tape_mark = z_tape_mark;
    b.tape_pause = z_tape_pause;
    b.tape_end = z_tape_end;
    b.tape_len = z_tape_len;
    b.tape_play = z_tape_play;
    b.tape_diff = z_tape_diff;
    b.tape_destroy = z_tape_destroy;
    b.tape_record = z_tape_record;
    b.tape_set_overlap = z_tape_set_overlap;
    return b;
}

/* Fills devs(); false when there is nothing to offer. */
bool discover() {
    const char* want_cpu = std::getenv("RADIANCE_L0_DEVICE");
    const char* want_pci = std::getenv("RADIANCE_L0_PCI");
    const bool cpu = want_cpu && std::strcmp(want_cpu, "cpu") == 0;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (cpu ? p.get_backend() != sycl::backend::opencl : p.get_backend() != sycl::backend::ext_oneapi_level_zero)
            continue;
        for (const auto& d : p.get_devices(cpu ? sycl::info::device_type::cpu : sycl::info::device_type::gpu)) {
            Dev D;
            D.d = d;
            D.cpu = cpu;
            if (!cpu && d.has(sycl::aspect::ext_intel_pci_address))
                D.pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
            if (want_pci && *want_pci && D.pci != want_pci) continue;
            D.tag = !cpu && d.has(sycl::aspect::ext_oneapi_queue_profiling_tag);
            D.ctx = sycl::context(d);
            if ((int)devs().size() < kMaxDevices) devs().push_back(D);
        }
        if (cpu && !devs().empty()) break;   /* one CPU device */
    }
    return !devs().empty();
}

}  /* namespace */

const DeviceBackend* l0_backend_if_present(void) {
    const char* env = std::getenv("RADIANCE_L0");
    if (env && std::strcmp(env, "0") == 0) return nullptr;
    static DeviceBackend g;
    static bool ok = false;
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            ok = discover();
        } catch (const sycl::exception& e) {
            RAD_INFO("device: Level Zero discovery failed: %s", e.what());
            ok = false;
        }
        if (ok) g = make_backend();
    });
    return ok ? &g : nullptr;
}

}  /* namespace rad */

#endif /* RAD_HAVE_L0 */
