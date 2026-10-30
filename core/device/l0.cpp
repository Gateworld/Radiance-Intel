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
 * Tapes are direct Level Zero command lists on a card (see "native tapes", owner note 28) and SYCL
 * command graphs on the CPU device or with RADIANCE_L0_TAPE=graph (see "recorded passes"). Left null
 * (supported, see device.h): elastic memory, the expert prefetch, gates, link holds, code-object checks.
 */
#if defined(RAD_HAVE_L0) && RAD_HAVE_L0

#include "device.h"
#include "../rad_internal.h"
#include "rad_l0_rec.h"

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <thread>
#include <utility>
#include <vector>

/* core/device/l0_kernels.sycl (librad_l0k) */
extern "C" int rad_l0k_fill(void* stream, void* dst, int value, int64_t bytes);
extern "C" void rad_l0k_set_flush(void (*f)(void));
extern "C" void rad_l0k_set_devices(int n, const sycl::device* (*dev_of)(int), const sycl::context* (*ctx_of)(int));

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
    /* Profiling tags recorded on it since its last host sync (see real_record's bound). */
    std::atomic<int64_t> tags{0};
    /* Ever gated: its last command may be a gate's host task, so records on it are barriers. */
    std::atomic<bool> gated{false};
    /* Whether any thread's backend call has named it yet; see t_touched. */
    std::atomic<bool> used{false};
    uint64_t          id = 0;      /* serial, never reused */
    int               dev = 0;     /* the device it was created on (a tape records one device's streams) */
    /* The queue's native handle, an immediate command list (native tapes append to it); null until
     * asked for, and stays null when the runtime gives a command queue instead. */
    ze_command_list_handle_t zimm = nullptr;
    bool zimm_tried = false;
    /* A native play this stream's work must follow (the play ran on the backend's own list): the
     * next submission to the runtime's queue host-waits for it first (settle). */
    int npend_dev = -1, npend_slot = -1;
    uint64_t npend_serial = 0;
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
struct L0Stream;
bool native_on(const L0Stream& s);
void settle(L0Stream& s);
void touch(L0Stream& s) {
    s.used.store(true, std::memory_order_relaxed);
    t_touched.insert(s.id);
    t_last = &s;
}
int flush_held();   /* "recorded passes": submits a held tape play */
/* Every backend entry that may submit on `s`: name it, submit the held play first, then wait on
 * the host for a native play the stream's work must follow (settle) -- in that order, so a
 * submission right after a held play follows it (N9: settling before the flush let the bench's sync
 * return before the play). */
int enter(L0Stream& s) {
    touch(s);
    RAD_TRY(flush_held());
    if (!native_on(s)) settle(s);
    return RAD_OK;
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
    bool        stamped = false;   /* `ev` carries a timestamp (rad_event_elapsed_ms can read it) */
    unsigned    flags = 0;
    /* Recorded inside a tape: the graph-internal barrier, its tape and segment (see tapes). */
    sycl::event gev;
    uint64_t    gtape = 0;
    int         gseg = -1;
    /* Recorded for real after a native play: that play's ring slot (its `done` event and serial, see
     * ZeDev::Ring), so a wait on this event is a native barrier and a query reads the event. */
    int         zdev = -1, zslot = -1;
    uint64_t    zserial = 0;
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

/* Peer access (milestone 2, two cards): SYCL's peer-access extension over Level Zero. M2b measured
 * the B70 pair at 24 us an exchange with peer stores against 79 us through a host mailbox. The
 * devices must share a context (discover) for a peer pointer to be valid. */
int z_enable_peer(int self, int peer) {
    if (!dev_ok(self, "rad_dev_enable_peer") || !dev_ok(peer, "rad_dev_enable_peer")) return RAD_E_INVAL;
    if (self == peer) return RAD_OK;
    Dev& a = devs()[self];
    Dev& b = devs()[peer];
    if (a.cpu || b.cpu) { device_set_error("rad_dev_enable_peer(%d, %d): the CPU device has no peers", self, peer); return RAD_E_UNSUPPORTED; }
    if (a.ctx != b.ctx) { device_set_error("rad_dev_enable_peer(%d, %d): the cards do not share a context", self, peer); return RAD_E_UNSUPPORTED; }
    try {
        if (!a.d.ext_oneapi_can_access_peer(b.d)) {
            device_set_error("rad_dev_enable_peer(%d, %d): the driver reports no peer access", self, peer);
            return RAD_E_UNSUPPORTED;
        }
        a.d.ext_oneapi_enable_peer_access(b.d);
    } catch (const sycl::exception& e) {
        device_set_error("rad_dev_enable_peer(%d, %d): %s", self, peer, e.what());
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
int z_peer_access(int self, int peer) {
    if (self == peer) return 1;
    if (!dev_ok(self, "rad_dev_peer_access") || !dev_ok(peer, "rad_dev_peer_access")) return -1;
    Dev& a = devs()[self];
    Dev& b = devs()[peer];
    if (a.cpu || b.cpu || a.ctx != b.ctx) return 0;
    try { return a.d.ext_oneapi_can_access_peer(b.d) ? 1 : 0; } catch (const sycl::exception&) { return -1; }
}

/* ------------------------------------------------------------------ memory */
void vtrace_alloc(int d, int64_t bytes, int kind, bool freed);   /* RADIANCE_L0_VRAM_TRACE's ledger (below) */
/* RADIANCE_L0_EXPLICIT_RESIDENCY=1 (the Intel port, run 93), for use with NEO's DisableIndirectAccess=1.
 * SYCL launches kernels with indirect access to USM, for which the driver makes every allocation of the
 * context resident on the launching card: on two B70s a kernel on card1 binds both ranks' host pools and
 * card0's device memory as imports, and some of those binds never return (xe answers VM_BIND with EBUSY,
 * runs 87-93; the probe's `pair`, HB7). With indirect access off, a kernel makes only its argument
 * allocations resident, so everything a kernel reaches through a pointer table -- the weights, the expert
 * pools -- is made resident here, at allocation: on its own card always, and on every card when it is at
 * most kResidentEverywhere (mailboxes, staging, small shared buffers). The other rank's large host pieces
 * and the other card's large device pools are not bound on a card that never reads them. */
bool explicit_residency() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_L0_EXPLICIT_RESIDENCY"); return e && e[0] == '1'; }();
    return v;
}
constexpr int64_t kResidentEverywhere = 1ll << 30;
void make_resident_where_needed(void* p, int64_t bytes, int kind, int dev) {
    if (!explicit_residency() || kind == RAD_MEM_HOST || !p) return;
    static std::once_flag said;
    std::call_once(said, [] {
        const char* k = std::getenv("DisableIndirectAccess");
        const char* r = std::getenv("NEOReadDebugKeys");
        const bool on = k && k[0] == '1' && r && r[0] == '1';
        RAD_INFO("l0 residency: each allocation made resident at allocation on its own card, and on every card up "
                 "to %lld MiB (RADIANCE_L0_EXPLICIT_RESIDENCY); NEO's DisableIndirectAccess %s",
                 (long long)(kResidentEverywhere >> 20),
                 on ? "is set" : "is NOT set: kernels still make every allocation resident on their card");
    });
    for (int d = 0; d < (int)devs().size(); ++d) {
        const Dev& D = devs()[d];
        if (D.cpu || (d != dev && bytes > kResidentEverywhere)) continue;
        ze_result_t r = ZE_RESULT_ERROR_UNKNOWN;
        try {
            ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(D.ctx);
            ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(D.d);
            r = zeContextMakeMemoryResident(zc, zd, p, (size_t)bytes);
        } catch (const sycl::exception&) {}
        if (r != ZE_RESULT_SUCCESS)
            RAD_WARN("l0 residency: %.2f MiB at %p (kind %d, card %d's) not made resident on card %d: 0x%x",
                     bytes / 1048576.0, p, kind, dev, d, (unsigned)r);
    }
}

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
    make_resident_where_needed(p, (int64_t)n, kind, dev);
    vtrace_alloc(dev, (int64_t)n, kind, false);
    return p;
}

void z_free(void* p, int kind) {
    (void)flush_held();
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
    vtrace_alloc(r.device, r.bytes, r.kind, true);
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

/* A native segment's command, kept so a stretch of segments can be composed into one list. */
struct NCmd {
    int kind = 0;                      /* 0 launch, 1 copy, 2 2D copy */
    ze_kernel_handle_t k = nullptr;
    const std::string* name = nullptr; /* the kernel's name (owned by the device's kernel table) */
    uint32_t i_arg0 = 0;
    int i_lds = -1;
    std::vector<uint8_t> arg0;
    size_t lds = 0;
    uint32_t gc[3] = {1, 1, 1}, bs[3] = {1, 1, 1};
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0, dp = 0, sp = 0, w = 0, h = 0;
};
struct Seg {
    std::shared_ptr<Exec> ex;          /* null when the segment holds only event records, or released (below) */
    std::optional<Graph>  g;           /* the recorded (modifiable) graph: re-finalized if a stretch needs it */
    std::vector<L0Event*> evs;         /* recorded for real after it */
    size_t nodes = 0;                  /* graph nodes, or a native segment's commands */
    ze_command_list_handle_t list = nullptr;   /* native tapes: the segment's closed regular command list */
    std::vector<NCmd> cmds;            /* native tapes: the segment's commands, for a composed stretch */
};
/* Owner note 27 (run 12): a recorded decode pass is ~1,036 segment graphs, each an executable with its
 * own command list and -- for the 375 kernels that spill -- its own scratch surface (the largest spill a
 * thread x the hardware threads): ~10 GB a card for one pass (T2). A play composes the segments into
 * one stretch graph, and in a steady decode step that is the whole pass. Once the full-pass stretch
 * exists and the recording's own submissions have completed, the segment executables are released
 * (their modifiable graphs stay, host-side, so another stretch can still be composed by re-finalizing).
 * RADIANCE_L0_TAPE_KEEP_SEGS=1 keeps them (the A/B). */
bool keep_segs() {
    static const bool k = [] { const char* v = std::getenv("RADIANCE_L0_TAPE_KEEP_SEGS"); return v && std::strcmp(v, "1") == 0; }();
    return k;
}
std::atomic<int64_t> g_segs_released{0}, g_segs_refinalized{0};
struct Tape {
    std::vector<Seg> segs;
    std::map<std::pair<int, int>, std::shared_ptr<Exec>> stretch;
    std::vector<L0Stream*> qs;         /* the recorded streams, the home stream first */
    std::string refused;
    uint64_t id = 0;                   /* serial, never reused (L0Event::gtape) */
    int dev = 0;                       /* the device recorded (its graphs and held plays belong to it) */
    /* Fence the other recorded streams around each submission. Cleared at tape_end for a tape
     * with no event records: its lanes are not ordered against each other in a plain pass either,
     * so the fences only cost (two barriers a stretch; b12b: replay 1.66 ms a step against 1.30). */
    bool fence = true;
    /* The event of its latest graph submission (all of them go through run_on_home, in order on
     * the home stream): when it is complete, no command buffer of this tape is still running. */
    sycl::event last;
    bool submitted = false;
    bool segs_released = false;        /* note 27: the segment executables are gone, the stretches stand */
    std::vector<char> covered;         /* note 27: which segments some composed stretch already holds */
    int plays_told = 0;
    /* the bounded stretch cache (hybrid mode): last-played clock per range, and the evicted executables
     * with the tape's `last` event at eviction, freed once it has completed */
    std::map<std::pair<int, int>, uint64_t> stretch_used;
    uint64_t stretch_clock = 0;
    std::deque<std::pair<std::shared_ptr<Exec>, sycl::event>> dead_stretch;
    /* native tapes (owner note 28): the segments are regular Level Zero command lists */
    bool native = false;
    size_t ncmds = 0;                  /* commands over every segment */
    int nslot = -1;                    /* the ring slot of the latest play and its serial (tape_idle) */
    uint64_t nserial = 0;
    std::map<ze_command_list_handle_t, std::pair<int, uint64_t>> nlast;   /* each replayed list's latest play (slot, serial): a
                                        * list must not be in flight twice; the tape's other lists may (run 20) */
    /* composed stretches: one list per played range (owner note 28's shape), the hybrid's bounded cache;
     * an evicted list is destroyed once the tape's play that followed its eviction has completed */
    std::map<std::pair<int, int>, ze_command_list_handle_t> nstretch;
    std::map<std::pair<int, int>, uint64_t> nstretch_used;
    uint64_t nclock = 0;
    struct NDead { ze_command_list_handle_t list; int slot; uint64_t serial; };
    std::deque<NDead> ndead;
    /* RADIANCE_L0_TAPE_EARLY=1 (owner note 65): segment lists released while recording, each once its own
     * play has completed (and the recording made to wait when too many are in flight), blocks composed as
     * they fill, the rest released at tape_end -- instead of every segment list waiting for the first play */
    std::deque<NDead> nseg_dead;
    bool segs_dropped = false;         /* the segment lists went at the first play (the pool's phase 2) */
    bool oom_told = false;             /* owner note 58: an out-of-memory append retried, said once a tape */
    /* owner note 42: the recorded waits whose signaller is inside the tape (the event, the waiter's segment),
     * checked at tape_end; the event-only segments closed after native work, whose records follow the
     * tape's latest play instead of the runtime queue */
    std::vector<std::pair<L0Event*, int>> rec_waits;
    int dangling_fixed = 0;
    ~Tape();
};
std::atomic<uint64_t> g_tape_ids{0};
std::atomic<int64_t> g_nlists_created{0}, g_nlists_destroyed{0}, g_nplays{0}, g_nlists_played{0}, g_nkernels{0};
std::atomic<int64_t> g_sig_early{0}, g_bar_early{0};
std::atomic<int64_t> g_same_tape_waits{0};   /* plays that waited for the same tape's previous play (its lists were still in flight) */   /* N4's diagnostic: the ring event / the barrier already complete right after the play's submission */
std::atomic<int64_t> g_nstretch_composed{0}, g_nstretch_evicted{0};
std::atomic<int64_t> g_append_retries{0};   /* owner note 58: appends retried after an out-of-memory result */
void vtrace_list(int d, int kind, int delta);   /* RADIANCE_L0_VRAM_TRACE's list counts (below) */
Tape::~Tape() {
    for (Seg& s : segs)
        if (s.list) { zeCommandListDestroy(s.list); s.list = nullptr; g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); vtrace_list(dev, 0, -1); }
    for (auto& kv : nstretch)
        if (kv.second) { zeCommandListDestroy(kv.second); g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); vtrace_list(dev, 1, -1); }
    for (NDead& d : ndead)
        if (d.list) { zeCommandListDestroy(d.list); g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); vtrace_list(dev, 1, -1); }
    for (NDead& d : nseg_dead)   /* a tape is deleted idle (release_tape), so these plays have completed */
        if (d.list) { zeCommandListDestroy(d.list); g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); vtrace_list(dev, 0, -1); }
}
struct Rec {
    Tape* t = nullptr;
    std::optional<Graph> g;            /* the open segment (SYCL-graph tapes) */
    std::vector<L0Event*> pend;        /* its event records */
    int pause = 0;
    bool stopped = false;              /* recording given up: the rest of the pass runs plainly */
    bool lost = false;                 /* owner note 58: a segment's close failed, so its commands never ran */
    /* native tapes: the open regular command list and its command count */
    bool native = false;
    ze_command_list_handle_t nlist = nullptr;
    size_t ncmds = 0;
    std::vector<NCmd> cmds;            /* the open segment's commands (descriptors) */
};
/* A pass longer than this many segments is not kept. Past the bound the recording stops, the rest
 * of the pass is issued plainly, and the tape is refused (the engine runs that pass plain from then
 * on: owner note 1's per-pass fallback, counted in /stats passes.issued_refused). History: H4
 * (milestone 1b) set this to 256 because each new prompt length recorded a ~1,083-segment prefill
 * pass and their command buffers filled device memory ("tape_mark:
 * UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"); but a decode pass with the mover's table refreshes has
 * ~1,083 segments as well (one per refresh point), so that bound refused every decode pass while the
 * heat engine was on (milestone 2 K1: played 0, refused 1 a step). The engine now records only
 * decode-sized passes (ctx.cpp record_pass, n_tok <= 64), and this is a safety bound. */
constexpr size_t kTapeMaxSegs = 2048;
thread_local Rec t_rec;

/* RADIANCE_L0_VRAM_TRACE=1 (owner note 61 item 0): where the recordings' VRAM goes, from this process's own DRM
 * client per card (/proc/self/fdinfo: drm-total-vram0 of the xe clients whose drm-pdev is the card), read at each
 * tape's begin and end and every 256 plays, against this card's lists alive by kind (the recordings' segment lists,
 * the composed stretches) and their peaks, and the device layer's own allocations (g_used). The driver keeps a
 * destroyed list's memory and reuses it (X1, run 75), so the peaks are what the VRAM follows. Run 76's V1 charged
 * growth to each list's build window instead: most of it landed outside the windows, and two fdinfo reads a segment
 * (the driver walks every buffer object on a read) cost enough host CPU to reach the heat stop. */
bool vram_trace() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_L0_VRAM_TRACE"); return e && e[0] == '1'; }();
    return v;
}
struct VTrace {
    std::mutex m;
    std::vector<std::string> paths;    /* this card's xe client fdinfo files, one a client */
    int64_t base = -1, used_base = 0;  /* at the first reading */
    std::atomic<int64_t> seg_alive{0}, seg_peak{0}, seg_made{0}, str_alive{0}, str_peak{0}, str_made{0};
};
VTrace& vtrace(int d) { static VTrace v[kMaxDevices]; return v[d]; }
int64_t fdinfo_bytes(const char* v) {
    while (*v == ' ' || *v == '\t') ++v;
    const double x = std::strtod(v, nullptr);
    if (std::strstr(v, "GiB")) return (int64_t)(x * 1073741824.0);
    if (std::strstr(v, "MiB")) return (int64_t)(x * 1048576.0);
    if (std::strstr(v, "KiB")) return (int64_t)(x * 1024.0);
    return (int64_t)x;
}
/* Finds this card's xe clients among the process's open files (on first use and at each tape_begin). */
void vtrace_scan(int d) {
    VTrace& V = vtrace(d);
    const std::string& pci = devs()[d].pci;
    std::map<std::string, std::string> by_client;
    for (int fd = 0; fd < 4096; ++fd) {
        char p[64];
        std::snprintf(p, sizeof p, "/proc/self/fdinfo/%d", fd);
        FILE* f = std::fopen(p, "r");
        if (!f) continue;
        char ln[256];
        bool xe = false, mine = false;
        std::string id;
        while (std::fgets(ln, sizeof ln, f)) {
            if (!std::strncmp(ln, "drm-driver:", 11)) xe = std::strstr(ln + 11, "xe") != nullptr;
            else if (!std::strncmp(ln, "drm-pdev:", 9)) mine = !pci.empty() && std::strstr(ln + 9, pci.c_str()) != nullptr;
            else if (!std::strncmp(ln, "drm-client-id:", 14)) id = ln + 14;
        }
        std::fclose(f);
        if (xe && mine && !id.empty() && !by_client.count(id)) by_client[id] = p;
    }
    std::lock_guard<std::mutex> lk(V.m);
    V.paths.clear();
    for (auto& kv : by_client) V.paths.push_back(kv.second);
}
int64_t vram_proc(int d) {
    VTrace& V = vtrace(d);
    if (V.paths.empty()) vtrace_scan(d);
    std::lock_guard<std::mutex> lk(V.m);
    int64_t sum = 0;
    for (const std::string& p : V.paths) {
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) continue;
        char ln[256];
        while (std::fgets(ln, sizeof ln, f))
            if (!std::strncmp(ln, "drm-total-vram0:", 16)) sum += fdinfo_bytes(ln + 16);
        std::fclose(f);
    }
    if (V.base < 0) { V.base = sum; V.used_base = g_used[d].load(std::memory_order_relaxed); }
    return sum;
}
/* A list of this card's made (+1) or destroyed (-1): kind 0 a recording's segment list, 1 a composed stretch. */
void vtrace_list(int d, int kind, int delta) {
    if (!vram_trace()) return;
    VTrace& V = vtrace(d);
    std::atomic<int64_t>& alive = kind ? V.str_alive : V.seg_alive;
    std::atomic<int64_t>& peak = kind ? V.str_peak : V.seg_peak;
    const int64_t a = alive.fetch_add(delta, std::memory_order_relaxed) + delta;
    if (delta > 0) {
        (kind ? V.str_made : V.seg_made).fetch_add(delta, std::memory_order_relaxed);
        int64_t p = peak.load(std::memory_order_relaxed);
        while (a > p && !peak.compare_exchange_weak(p, a, std::memory_order_relaxed)) {}
    }
}
/* The driver's free VRAM on card d (what the floor and the planner read), -1 when it cannot say. */
int64_t driver_free(int d) {
    const Dev& D = devs()[d];
    try {
        if (!D.cpu && D.d.has(sycl::aspect::ext_intel_free_memory))
            return (int64_t)D.d.get_info<sycl::ext::intel::info::device::free_memory>();
    } catch (const sycl::exception&) {}
    return -1;
}
void vtrace_report(int d, const char* at, unsigned long long id) {
    if (!vram_trace()) return;
    VTrace& V = vtrace(d);
    const int64_t now = vram_proc(d), used = g_used[d].load(std::memory_order_relaxed);
    const double M = 1048576.0;
    RAD_INFO("l0 vram trace dev %d %s %llu: process VRAM %.0f MiB (+%.0f since the first reading); driver free %.0f MiB; "
             "engine allocations %.0f MiB (+%.0f); "
             "segment lists alive %lld (peak %lld, made %lld); composed lists alive %lld (peak %lld, made %lld)",
             d, at, id, now / M, (now - V.base) / M, driver_free(d) / M, used / M, (used - V.used_base) / M,
             (long long)V.seg_alive.load(),
             (long long)V.seg_peak.load(), (long long)V.seg_made.load(), (long long)V.str_alive.load(),
             (long long)V.str_peak.load(), (long long)V.str_made.load());
}
/* Run 96 (owner note 95 item 1): the ledger -- each device or host-mapped allocation (and free) of at least 16 MiB,
 * with the card's process VRAM and the driver's free VRAM right after it. The engine logs what it allocated just
 * after (its arena, KV, pools), so the log's order names each line's owner. */
void vtrace_alloc(int d, int64_t bytes, int kind, bool freed) {
    if (!vram_trace() || bytes < (16ll << 20) || kind == RAD_MEM_HOST || devs()[d].cpu) return;
    const double M = 1048576.0;
    RAD_INFO("l0 vram ledger dev %d %s %.1f MiB %s: process VRAM %.0f MiB; driver free %.0f MiB; engine allocations %.0f MiB",
             d, freed ? "free" : "alloc", bytes / M, kind == RAD_MEM_DEVICE ? "device" : "host-mapped",
             vram_proc(d) / M, driver_free(d) / M, g_used[d].load(std::memory_order_relaxed) / M);
}

bool recording() { return t_rec.t && (t_rec.g || t_rec.nlist) && t_rec.pause == 0; }
/* The native recorder is open on this thread: commands on its streams go into the command list. */
bool native_recording() { return t_rec.t && t_rec.native && t_rec.nlist && t_rec.pause == 0 && !t_rec.stopped; }
bool recorded_stream(const L0Stream& s) {
    for (L0Stream* x : t_rec.t->qs) if (x == &s) return true;
    return false;
}
bool native_on(const L0Stream& s) { return native_recording() && recorded_stream(s); }
void refuse(const std::string& why) {
    if (t_rec.t && t_rec.t->refused.empty()) t_rec.t->refused = why;
}

/* Event records without growth (run 24: card1 ran out of device memory after ~74,000 expert moves,
 * each recorded with a new profiling tag on a stream nobody host-syncs; the runtime keeps a
 * timestamp event until its queue syncs).
 *
 * A record submits nothing: on an in-order queue the last command's event is complete exactly when
 * everything before the record is, so the RadEvent takes that event (the runtime recycles it like
 * any kernel's). The events the backend holds are therefore one per RadEvent, a set the engine
 * creates up front. On the CPU device a reused event's time is the end of the command before the
 * record, not the moment the queue reached it (a per-op profile there shows no gaps). Three cases
 * submit a command:
 *  - a queue with no command yet: one barrier (at most once a stream);
 *  - a stream that has had a gate: a barrier (the last event could be the gate's host task, which
 *    misbehaved on the CPU device, run 3);
 *  - timestamps: profiling tags, only once rad_event_elapsed_ms has asked for one (g_timing; the
 *    per-op profile) and at most kTagsPerSync on a stream between its host syncs. Past that bound a
 *    record is untimed, never a new tag; rad_event_elapsed_ms says so once.
 * The CPU device's queues profile every command, so there every record is stamped. */
constexpr int64_t kTagsPerSync = 32768;   /* the per-op profile's 8192 pairs a step, twice over */
std::atomic<bool> g_timing{false};
std::atomic<int64_t> g_rec_reused{0}, g_rec_barriers{0}, g_rec_tags{0}, g_rec_untimed{0};

void real_record(L0Event* e, L0Stream& st) {
    sycl::queue& q = st.q;
    const bool tags = devs()[0].tag;
    sycl::event b;
    bool stamped;
    if (tags && g_timing.load(std::memory_order_relaxed) && st.tags.load() < kTagsPerSync) {
        b = sx::submit_profiling_tag(q);
        st.tags.fetch_add(1);
        g_rec_tags.fetch_add(1, std::memory_order_relaxed);
        stamped = true;
    } else {
        if (tags && g_timing.load(std::memory_order_relaxed)) g_rec_untimed.fetch_add(1, std::memory_order_relaxed);
        if (st.gated.load()) {
            b = q.ext_oneapi_submit_barrier();
            g_rec_barriers.fetch_add(1, std::memory_order_relaxed);
            stamped = !tags;
        } else {
            std::optional<sycl::event> last = q.ext_oneapi_get_last_event();
            if (last) {
                b = *last;
                g_rec_reused.fetch_add(1, std::memory_order_relaxed);
            } else {   /* nothing on the queue yet: one barrier, so the record still has a time */
                b = q.ext_oneapi_submit_barrier();
                g_rec_barriers.fetch_add(1, std::memory_order_relaxed);
            }
            stamped = !tags;
        }
    }
    std::lock_guard<std::mutex> lk(e->m);
    e->ev = b;
    e->recorded = true;
    e->stamped = stamped;
    e->zdev = -1;
}
/* Submits `ex` on the home stream, fenced against the other recorded streams. */
void run_on_home(Tape& t, Exec& ex) {
    sycl::queue& H = t.qs[0]->q;
    const bool fence = t.fence && t.qs.size() > 1;
    if (fence) {
        std::vector<sycl::event> pre;
        for (size_t k = 1; k < t.qs.size(); ++k) pre.push_back(t.qs[k]->q.ext_oneapi_submit_barrier());
        H.ext_oneapi_submit_barrier(pre);
    }
    sycl::event ge = H.ext_oneapi_graph(ex);
    t.last = ge;
    t.submitted = true;
    if (fence)
        for (size_t k = 1; k < t.qs.size(); ++k) t.qs[k]->q.ext_oneapi_submit_barrier({ge});
}

/* Freeing a tape (GPU faults 3 and 4, owner note 10). Deleting a Tape releases its executable
 * graphs and with them their command buffers; the runtime does not keep a graph alive for a
 * submission it enqueued straight to the device, so a tape freed while a play of it still runs left
 * the copy engine fetching commands from unmapped memory (fault 4's coredump: RING_BBADDR / DMA_FADD
 * inside a second-level batch, "Engine memory CAT error, class=bcs"). A tape whose last submission
 * has not completed goes to a graveyard instead, freed once it has; past kDeadMax pending, the
 * oldest is waited for. Never grows without limit. */
constexpr size_t kDeadMax = 64;
std::mutex& dead_mutex() { static std::mutex m; return m; }
std::deque<Tape*>& dead_tapes() { static std::deque<Tape*> d; return d; }
std::atomic<int64_t> g_tapes_deferred{0}, g_tapes_waited{0}, g_tapes_freed{0};
bool n_slot_done(int dev, int slot, uint64_t serial);
void n_slot_wait(int dev, int slot, uint64_t serial, const char* who = "");
bool tape_idle(Tape* t) {
    if (!t->submitted) return true;
    if (t->native) return n_slot_done(t->dev, t->nslot, t->nserial);
    try {
        return t->last.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (const sycl::exception&) {
        return true;   /* a failed submission's event: nothing of it runs */
    }
}
/* Frees the graveyard's completed tapes, oldest first; `all` waits for every one. */
void reap_tapes(bool all = false) {
    std::lock_guard<std::mutex> lk(dead_mutex());
    auto& d = dead_tapes();
    while (!d.empty()) {
        Tape* t = d.front();
        if (!tape_idle(t)) {
            if (!all && d.size() <= kDeadMax) break;
            if (t->native) n_slot_wait(t->dev, t->nslot, t->nserial, "the graveyard");
            else { try { t->last.wait(); } catch (const sycl::exception&) {} }
            g_tapes_waited.fetch_add(1, std::memory_order_relaxed);
        }
        d.pop_front();
        delete t;
        g_tapes_freed.fetch_add(1, std::memory_order_relaxed);
    }
}
void release_tape(Tape* t) {
    if (!t) return;
    if (tape_idle(t)) {
        delete t;
        g_tapes_freed.fetch_add(1, std::memory_order_relaxed);
    } else {
        std::lock_guard<std::mutex> lk(dead_mutex());
        dead_tapes().push_back(t);
        g_tapes_deferred.fetch_add(1, std::memory_order_relaxed);
    }
    reap_tapes();
}
void rerecord(Tape& t, const std::vector<L0Event*>& evs) {
    std::lock_guard<std::mutex> lk(event_mutex());
    for (L0Event* e : evs)
        if (events_alive().count(e)) real_record(e, *t.qs[0]);
}

/* ------------------------------------------------------------------ native tapes (owner note 28)
 * On a card a tape's segments are regular Level Zero command lists, not SYCL graphs: a segment is
 * recorded into one in-order list (every launch, copy and fill appended with the arguments set once;
 * libxe's launch_as and the core's kernels come in through rad_l0_rec.h), closed at tape_mark and run
 * at once by appending it to the home stream's immediate command list, and a play appends the
 * stretch's lists in one call (N1: 1.9 us a list, no per-graph scratch surface or command-list copy).
 * The SYCL runtime orders its own commands through its last event, so a play waits on that event and
 * signals one of the device's ring events, which the home stream then barriers on (imported as a
 * sycl::event): both directions of the order hold whether or not the runtime's list is driver-in-order.
 * The recorder owns its kernel objects (zeKernelCreate on the kernel's module, by name), so the
 * runtime's kernel argument state is never touched. Refused, with the segment so far run: a kernel the
 * recorder cannot create and an append the driver rejects. A direct SYCL submission on a recorded
 * stream is not caught (see n_seg_close): it runs once and never replays. Everything else -- the pass
 * key, the held play, the graveyard -- is shared with the graphs. */
void sleepy_wait(const sycl::event& e);
bool sleepy_wait_on();
bool event_done(const sycl::event& e);
void stop_recording(const std::string& why);
int ntape_sync();
/* the sleeping wait's constants (owner note 25; defined here for the native waits above their use) */
constexpr int64_t kWaitSpinUs = 100;
constexpr int64_t kWaitSleepMinUs = 20, kWaitSleepMaxUs = 200;

constexpr uint32_t kRingEvents = 256;
struct NKernel {
    ze_kernel_handle_t k = nullptr;
    uint32_t nargs = 0;
    bool own = false;                  /* created here (else the runtime's handle, shared) */
    bool lds_told = false;             /* the local-memory argument refused a size once (said once) */
    std::string name;                  /* the kernel's function name (the tape audit's message) */
};
struct ZeDev {
    bool tried = false, ok = false;
    std::string why;
    ze_context_handle_t ctx = nullptr;
    ze_device_handle_t  dev = nullptr;
    ze_event_pool_handle_t pool = nullptr;
    /* A play's slot: `ev` is signalled when the appended lists complete, `done` by the native barrier
     * that waits on `ev` (the command that joins the lists to the driver's in-order chain). A slot is
     * reused only once `done` is signalled -- N6: reusing it on a SYCL marker's completion reset `ev`
     * under a barrier that had not consumed it yet (the runtime elides a plain barrier at times), which
     * lost the order (246 wrong rounds of 2000) and hung the lifetime test. `serial` tells a holder of
     * the slot (a tape, a RadEvent) that the slot has moved on, i.e. its play completed long ago. */
    struct Consumer { int dev; int slot; uint64_t serial; };
    struct Ring {
        ze_event_handle_t ev = nullptr;
        std::atomic<uint64_t> serial{0};   /* read without the lock (n_slot_done): a device never holds its lock while blocking on another's */
        bool used = false;
        std::vector<sycl::event> keep;   /* the runtime's events the play waited on: alive until the slot's reuse, so the runtime cannot recycle them under the device-side wait */
        std::vector<Consumer> consumers;  /* slots of other devices whose lists wait on this one's event (rad_event_wait across streams): reuse waits for them too */
        /* what the slot carries, for the 10 s wait warnings (owner note 42): a play's tape, range and list
         * count, or a cross-stream wait (kind 1); the slot it followed; the runtime events it waited on */
        int kind = 0;
        uint64_t tape = 0;
        int from = -1, to = -1;
        uint32_t nlists = 0, nwaits = 0;
        int dep_dev = -1, dep_slot = -1;
        uint64_t dep_serial = 0;
    };
    std::unique_ptr<Ring[]> ring;
    size_t next = 0;
    uint64_t serials = 0;
    /* the backend's own immediate command list: every native play runs there, chained by events */
    ze_command_list_handle_t imm = nullptr;
    int prev_slot = -1;                /* the previous play's slot: the next play waits on it */
    uint64_t prev_serial = 0;
    bool inorder = true;               /* ZE_COMMAND_LIST_FLAG_IN_ORDER accepted; else a barrier before each command */
    std::mutex m;                      /* the ring and the kernel table (plays may come from another thread) */
    std::map<ze_kernel_handle_t, NKernel> kernels;   /* by the runtime's handle */
    double timer_ns = 0;               /* the device timer's resolution (ns a tick) and valid bits (the tape profiler) */
    uint32_t ts_bits = 64;
};
/* RADIANCE_L0_NTAPE_PROFILE=1 (run 27): composed stretches time every command with device timestamps. */
bool ntape_profile() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_L0_NTAPE_PROFILE"); return e && e[0] == '1'; }();
    return v;
}
struct NProf {
    uint64_t* ts = nullptr;            /* host-visible: [2k] before, [2k + 1] after command k */
    std::vector<const std::string*> names;
    std::vector<int> kinds;
    uint32_t n = 0;
    bool played = false;               /* the timestamps hold a completed play */
};
std::mutex& nprof_mutex() { static std::mutex m; return m; }
std::unordered_map<ze_command_list_handle_t, NProf>& nprof_lists() { static std::unordered_map<ze_command_list_handle_t, NProf> m; return m; }
struct NProfAcc { uint64_t ticks = 0, count = 0; };
std::unordered_map<std::string, NProfAcc>& nprof_acc() { static std::unordered_map<std::string, NProfAcc> m; return m; }
uint64_t g_nprof_harvested = 0;
ZeDev& zedev(int d) { static ZeDev z[kMaxDevices]; return z[d]; }
using AppendFn = ze_result_t (*)(ze_command_list_handle_t, uint32_t, ze_command_list_handle_t*, ze_event_handle_t,
                                 uint32_t, ze_event_handle_t*);
AppendFn append_fn() {
    static AppendFn f = (AppendFn)dlsym(RTLD_DEFAULT, "zeCommandListImmediateAppendCommandListsExp");
    return f;
}
int tape_mode() {   /* 0 off, 1 graphs, 2 native where the device allows */
    static const int m = [] {
        const char* v = std::getenv("RADIANCE_L0_TAPE");
        if (!v) return 2;
        if (!std::strcmp(v, "0")) return 0;
        if (!std::strcmp(v, "graph")) return 1;
        return 2;
    }();
    return m;
}
ze_command_list_handle_t n_list_create(ZeDev& Z, bool inorder) {
    ze_command_list_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, inorder ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0u};
    ze_command_list_handle_t l = nullptr;
    if (zeCommandListCreate(Z.ctx, Z.dev, &d, &l) != ZE_RESULT_SUCCESS) return nullptr;
    g_nlists_created.fetch_add(1, std::memory_order_relaxed);
    return l;
}
/* The stream's immediate command list, or null (a command queue, or no native handle). */
ze_command_list_handle_t imm_of(L0Stream& s) {
    if (!s.zimm_tried) {
        s.zimm_tried = true;
        try {
            auto nat = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(s.q);
            if (auto* l = std::get_if<ze_command_list_handle_t>(&nat)) s.zimm = *l;
        } catch (const sycl::exception&) {}
    }
    return s.zimm;
}
/* The device's native state, made on first use: the handles, the ring of signal events, the
 * in-order list check. False (with `why`) when native tapes cannot run here. */
bool ze_ready(int d) {
    ZeDev& Z = zedev(d);
    if (Z.tried) return Z.ok;
    Z.tried = true;
    const Dev& D = devs()[d];
    if (D.cpu) { Z.why = "the CPU device has no Level Zero handles"; return false; }
    if (!append_fn()) { Z.why = "the loader has no zeCommandListImmediateAppendCommandListsExp"; return false; }
    try {
        Z.ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(D.ctx);
        Z.dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(D.d);
    } catch (const sycl::exception& e) { Z.why = std::string("no native handles: ") + e.what(); return false; }
    /* the pool over every card of the context, so another rank's stream can wait on a play's event */
    std::vector<ze_device_handle_t> all;
    for (const Dev& X : devs()) {
        if (X.cpu || X.ctx != D.ctx) continue;
        try { all.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(X.d)); } catch (const sycl::exception&) {}
    }
    if (all.empty()) all.push_back(Z.dev);
    {   /* the device timer, for the tape profiler (ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: timerResolution in ns a tick) */
        ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
        if (zeDeviceGetProperties(Z.dev, &dp) == ZE_RESULT_SUCCESS) { Z.timer_ns = (double)dp.timerResolution; Z.ts_bits = dp.timestampValidBits ? dp.timestampValidBits : 64; }
    }
    ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, kRingEvents};
    if (zeEventPoolCreate(Z.ctx, &pd, (uint32_t)all.size(), all.data(), &Z.pool) != ZE_RESULT_SUCCESS) { Z.why = "zeEventPoolCreate failed"; return false; }
    Z.ring = std::make_unique<ZeDev::Ring[]>(kRingEvents);
    for (uint32_t i = 0; i < kRingEvents; ++i) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
        if (zeEventCreate(Z.pool, &ed, &Z.ring[i].ev) != ZE_RESULT_SUCCESS) { Z.why = "zeEventCreate failed"; return false; }
    }
    /* the backend's own immediate list on the compute queue group */
    /* in order (run 20, M12): the appended batches execute one after another, so two plays never overlap
     * on a device whatever the event chain says (a cross-stream wait slot fires with an older play) */
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    if (zeCommandListCreateImmediate(Z.ctx, Z.dev, &qd, &Z.imm) != ZE_RESULT_SUCCESS) {
        qd.flags = 0;
        Z.imm = nullptr;
        if (zeCommandListCreateImmediate(Z.ctx, Z.dev, &qd, &Z.imm) != ZE_RESULT_SUCCESS) { Z.why = "zeCommandListCreateImmediate failed"; return false; }
        RAD_WARN("l0 tape: the driver refused an in-order immediate list on device %d; plays are ordered by their events alone", d);
    }
    ze_command_list_handle_t probe = n_list_create(Z, true);
    if (!probe) {
        Z.inorder = false;
        probe = n_list_create(Z, false);
        if (!probe) { Z.why = "zeCommandListCreate failed"; return false; }
    }
    {   /* Owner note 68 item 4: what the driver takes for a command list here, read from this process's own
         * VRAM (fdinfo) across eight empty lists, so the log says whether NEO's OverrideCmdListCmdBufferSizeInKb
         * (a debug key: run 80's X9 measured 0.25 MiB a list with 64 against 1.19 MiB by default, NEO 1.14.37020)
         * took effect. The driver reuses this memory for the recordings' lists, so it costs nothing. */
        const char* ovr = std::getenv("OverrideCmdListCmdBufferSizeInKb");
        const char* rdk = std::getenv("NEOReadDebugKeys");
        const int64_t v0 = vram_proc(d);
        ze_command_list_handle_t ls[8] = {};
        int made = 0;
        for (int i = 0; i < 8; ++i) {
            ls[i] = n_list_create(Z, Z.inorder);
            if (!ls[i]) break;
            zeCommandListClose(ls[i]);
            ++made;
        }
        const int64_t v1 = vram_proc(d);
        for (int i = 0; i < made; ++i) { zeCommandListDestroy(ls[i]); g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); }
        if (made == 8 && v0 > 0) {
            const double per = (double)(v1 - v0) / 8.0 / 1048576.0;
            const bool asked = ovr && rdk && rdk[0] == '1';
            RAD_INFO("l0 tape: the driver takes %.2f MiB a command list on device %d (NEO OverrideCmdListCmdBufferSizeInKb %s%s%s)",
                     per, d, asked ? "set to " : "not set", asked ? ovr : "",
                     asked ? (per < 0.6 ? ": in effect" : ": NOT in effect -- the recordings cost the default size; the floor still guards the card") : "");
        }
    }
    zeCommandListDestroy(probe);
    g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed);
    Z.ok = true;
    RAD_INFO("l0 tape: native Level Zero command lists on device %d (%s lists, the backend's own immediate list, a ring of %u signal events)",
             d, Z.inorder ? "in-order" : "barrier-separated", kRingEvents);
    return true;
}
/* Waits for a slot's `done` event on the host: a short spin, then sleeps (as the sleeping wait). */
void done_wait(ze_event_handle_t done) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int64_t sleep_us = kWaitSleepMinUs;
    bool told = false;
    while (zeEventQueryStatus(done) != ZE_RESULT_SUCCESS) {
        const int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
        if (us < kWaitSpinUs) { std::this_thread::yield(); continue; }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min<int64_t>(sleep_us * 2, kWaitSleepMaxUs);
        if (!told && us > 10000000) {
            told = true;
            RAD_WARN("l0 tape: a thread on device %d has waited 10 s for a ring slot's last play to complete (slot reuse; plays so far %lld)",
                     t_device, (long long)g_nplays.load());
        }
    }
}
bool n_slot_done(int dev, int slot, uint64_t serial);
void n_slot_wait(int dev, int slot, uint64_t serial, const char* who);
/* Takes the next ring slot: once its last play has completed -- and every list of another device that
 * waited on its event has passed that wait -- the event is reset. */
ZeDev::Ring& ring_take(ZeDev& Z) {
    ZeDev::Ring& r = Z.ring[Z.next];
    Z.next = (Z.next + 1) % kRingEvents;
    if (r.used) {
        done_wait(r.ev);
        for (const ZeDev::Consumer& c : r.consumers) n_slot_wait(c.dev, c.slot, c.serial, "a slot's consumer");
        r.consumers.clear();
        zeEventHostReset(r.ev);
        r.used = false;
        r.keep.clear();
    }
    r.serial.store(++Z.serials, std::memory_order_release);
    return r;
}
/* A play's slot has completed: the slot moved on to a later play (its reuse needed the event), or
 * the event is signalled. Lock-free (a reset racing the query reads "not yet", never "done"). */
bool n_slot_done(int dev, int slot, uint64_t serial) {
    ZeDev& Z = zedev(dev);
    if (slot < 0 || slot >= (int)kRingEvents || !Z.ring) return true;
    const ZeDev::Ring& r = Z.ring[(size_t)slot];
    return r.serial.load(std::memory_order_acquire) != serial || zeEventQueryStatus(r.ev) == ZE_RESULT_SUCCESS;
}
/* The slot's event for a device-side wait elsewhere, or null when the play has completed. */
ze_event_handle_t n_slot_event(int dev, int slot, uint64_t serial) {
    ZeDev& Z = zedev(dev);
    if (slot < 0 || slot >= (int)kRingEvents || !Z.ring) return nullptr;
    const ZeDev::Ring& r = Z.ring[(size_t)slot];
    if (r.serial.load(std::memory_order_acquire) != serial || zeEventQueryStatus(r.ev) == ZE_RESULT_SUCCESS) return nullptr;
    return r.ev;
}
/* What a slot carries, for the 10 s warnings (owner note 42): read without the device's lock (plain fields
 * written before the slot's serial was published; a stale read only misdescribes a slot that moved on). */
std::string slot_story(int dev, int slot, uint64_t serial) {
    ZeDev& Z = zedev(dev);
    if (slot < 0 || slot >= (int)kRingEvents || !Z.ring) return "no slot";
    const ZeDev::Ring& r = Z.ring[(size_t)slot];
    char buf[320];
    if (r.serial.load(std::memory_order_acquire) != serial) {
        std::snprintf(buf, sizeof buf, "the slot moved on (serial %llu now)", (unsigned long long)r.serial.load());
        return buf;
    }
    const char* ev = zeEventQueryStatus(r.ev) == ZE_RESULT_SUCCESS ? "signalled" : "pending";
    const char* dep = r.dep_slot < 0 ? "none" : n_slot_done(r.dep_dev, r.dep_slot, r.dep_serial) ? "completed" : "pending";
    if (r.kind == 1)
        std::snprintf(buf, sizeof buf, "a cross-stream wait on device %d slot %d serial %llu (%s), its own signal %p %s",
                      r.dep_dev, r.dep_slot, (unsigned long long)r.dep_serial, dep, (void*)r.ev, ev);
    else
        std::snprintf(buf, sizeof buf, "tape %llu range [%d, %d) in %u list(s), signal event %p %s; after slot %d serial %llu of device %d (%s) and %u runtime event(s)",
                      (unsigned long long)r.tape, r.from, r.to, r.nlists, (void*)r.ev, ev, r.dep_slot, (unsigned long long)r.dep_serial,
                      r.dep_dev, dep, r.nwaits);
    return buf;
}
void n_slot_wait(int dev, int slot, uint64_t serial, const char* who) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int64_t sleep_us = kWaitSleepMinUs;
    bool told = false;
    while (!n_slot_done(dev, slot, serial)) {
        const int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
        if (us < kWaitSpinUs) { std::this_thread::yield(); continue; }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min<int64_t>(sleep_us * 2, kWaitSleepMaxUs);
        if (!told && us > 10000000) {
            told = true;
            RAD_WARN("l0 tape: a thread on device %d is waiting 10 s for a play on device %d, slot %d, serial %llu (%s; plays so far %lld): %s",
                     t_device, dev, slot, (unsigned long long)serial, who, (long long)g_nplays.load(), slot_story(dev, slot, serial).c_str());
        }
    }
}
/* The stream's next submission to the runtime follows its pending native play: the host waits here. */
void settle(L0Stream& s) {
    if (s.npend_slot < 0) return;
    n_slot_wait(s.npend_dev, s.npend_slot, s.npend_serial, "settle before a runtime submission");
    s.npend_slot = -1;
}
/* How a play is ordered against the SYCL runtime's own commands (N2-N10, card1). A play runs on the
 * backend's OWN immediate command list, never the SYCL queue's: appending there perturbed the
 * runtime (the v2 adapter's queue wait never returned: the lifetime test hung at its first stream
 * sync) and the runtime's next command did not wait for a native barrier or an imported event under
 * either adapter (164 wrong rounds of 2000, bursty: the fold saw the previous round or a mix). No
 * event handle crosses between the runtime and the backend in either direction: before the append
 * the host waits for the runtime's queue(s) (the sleeping wait; N10: a device-side wait on the
 * runtime's last event hung once the v1 adapter recycled the event under it), the plays chain on the
 * backend's own events, and afterwards the stream's next submission to the runtime host-waits for
 * the play's event, deferred to that submission (settle, from enter and the library launch hook).
 * That is the host oracle's semantics at the latest moments; in a decode step the queue is idle at
 * the play (the engine synced to read the token) and the engine syncs again after it.
 * RADIANCE_L0_NTAPE_SYNC=host waits at once after the play instead (the oracle). */
int ntape_sync() {   /* 0 the design, 3 the host oracle, 4 the host wait before the play only */
    static const int m = [] {
        const char* v = std::getenv("RADIANCE_L0_NTAPE_SYNC");
        if (!v) return 0;
        if (!std::strcmp(v, "host")) return 3;
        if (!std::strcmp(v, "hostbefore")) return 4;
        return 0;
    }();
    return m;
}
/* The tape profiler's harvest: the list's previous play has completed, so its timestamps are final;
 * every command's device time goes to its kernel's total. Every 8192 plays the table is printed. */
void n_prof_harvest(ZeDev& Z, ze_command_list_handle_t l, bool played_before) {
    std::lock_guard<std::mutex> lk(nprof_mutex());
    auto it = nprof_lists().find(l);
    if (it == nprof_lists().end()) return;
    NProf& p = it->second;
    if (played_before && p.played) {
        const uint64_t mask = Z.ts_bits >= 64 ? ~0ull : ((1ull << Z.ts_bits) - 1);
        for (uint32_t k = 0; k < p.n; ++k) {
            const uint64_t a = p.ts[2 * k] & mask, b = p.ts[2 * k + 1] & mask;
            const uint64_t d = b >= a ? b - a : (mask - a) + b + 1;
            const std::string& nm = p.names[k] ? *p.names[k] : (p.kinds[k] == 1 ? std::string("(copy)") : std::string("(2D copy)"));
            NProfAcc& acc = nprof_acc()[nm];
            acc.ticks += d;
            acc.count += 1;
        }
        if (++g_nprof_harvested % 8192 == 0) {
            std::vector<std::pair<std::string, NProfAcc>> rows(nprof_acc().begin(), nprof_acc().end());
            std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.second.ticks > y.second.ticks; });
            uint64_t tot = 0;
            for (const auto& r : rows) tot += r.second.ticks;
            RAD_INFO("l0 tape profile: %llu lists harvested, %zu kernels, %.1f ms of device time in all (timer %.3f ns a tick); the top 30:",
                     (unsigned long long)g_nprof_harvested, rows.size(), tot * Z.timer_ns * 1e-6, Z.timer_ns);
            for (size_t i = 0; i < rows.size() && i < 30; ++i) {
                const NProfAcc& a = rows[i].second;
                std::string nm = rows[i].first;
                const size_t z = nm.find("_Z");   /* the mangled kernel after launch_as's own prefix */
                if (z != std::string::npos && z + 2 < nm.size() && nm.find("_Z", z + 2) != std::string::npos) nm = nm.substr(nm.find("_Z", z + 2));
                if (nm.size() > 96) nm.resize(96);
                RAD_INFO("  %6.1f%%  %8.1f ms  %8llu x  %7.1f us  %s", 100.0 * a.ticks / (tot ? tot : 1), a.ticks * Z.timer_ns * 1e-6,
                         (unsigned long long)a.count, a.count ? a.ticks * Z.timer_ns * 1e-3 / a.count : 0.0, nm.c_str());
            }
        }
    }
    p.played = true;
}
/* Appends `n` closed lists to the backend's own immediate list, after the home stream's last
 * command (and the other recorded streams' when fenced) and after the previous play; marks the
 * streams so their next runtime submission waits for it. */
void n_prof_harvest(ZeDev& Z, ze_command_list_handle_t l, bool played_before);
void n_play_lists(Tape& t, ze_command_list_handle_t* lists, uint32_t n, bool replay = true, int from = -1, int to = -1) {
    if (!n) return;
    ZeDev& Z = zedev(t.dev);
    sycl::queue& H = t.qs[0]->q;
    const bool host = ntape_sync() == 3;
    /* this tape's lists must not be in flight twice (an in-order list's counter state is one
     * submission's): its previous play first */
    if (replay)
        for (uint32_t i = 0; i < n; ++i) {   /* this list's own previous play (one step ago, as a rule: no wait) */
            auto it = t.nlast.find(lists[i]);
            if (it != t.nlast.end() && !n_slot_done(t.dev, it->second.first, it->second.second)) {
                g_same_tape_waits.fetch_add(1, std::memory_order_relaxed);
                n_slot_wait(t.dev, it->second.first, it->second.second, "this list's previous play");
            }
            if (ntape_profile()) n_prof_harvest(Z, lists[i], it != t.nlast.end());
        }
    std::vector<sycl::event> keep;
    std::vector<ze_event_handle_t> waits;
    /* the runtime's queue(s) first: a device-side wait on the queue's last event, the sycl::event
     * kept alive in the slot until its reuse (so the handle cannot be recycled under the wait). T6:
     * a host wait here blocked a rank inside a cycle between the ranks' plays (the N10 hypothesis
     * behind it -- recycled handles -- was the same list in flight twice, fixed since).
     * RADIANCE_L0_NTAPE_SYNC=hostbefore keeps the host form for the A/B. */
    const bool host_before = host || ntape_sync() == 4;
    auto add_wait = [&](sycl::queue& q) {
        if (host_before) {
            if (!host && sleepy_wait_on() && !devs()[t.dev].cpu) {
                std::optional<sycl::event> e = q.ext_oneapi_get_last_event();
                if (e) sleepy_wait(*e);
            }
            q.wait_and_throw();
            return;
        }
        std::optional<sycl::event> e = q.ext_oneapi_get_last_event();
        if (!e) return;
        try {
            waits.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*e));
            keep.push_back(*e);
        } catch (const sycl::exception&) {
            sleepy_wait(*e);       /* no native handle (a host event): waited for here */
            e->wait_and_throw();
        }
    };
    /* every recorded stream, fence or not: the play moves every lane's commands onto the backend's
     * list, so each lane's own queue must come before it and (below) follow it. M4: a draft pass
     * records two lanes and no event, so it was "played unfenced" and only the home lane was ordered;
     * the second lane's next plain submission (the draft head's sampler) ran ahead of the replay */
    add_wait(H);
    for (size_t k = 1; k < t.qs.size(); ++k) add_wait(t.qs[k]->q);
    std::lock_guard<std::mutex> lk(Z.m);
    const int dep_slot = Z.prev_slot;          /* named by the 10 s warning (owner note 42) */
    const uint64_t dep_serial = Z.prev_serial;
    if (Z.prev_slot >= 0) {   /* the previous play (or cross-stream wait), unless its slot has moved on (then it completed) */
        const ZeDev::Ring& pr = Z.ring[(size_t)Z.prev_slot];
        if (pr.serial.load(std::memory_order_acquire) == Z.prev_serial && zeEventQueryStatus(pr.ev) != ZE_RESULT_SUCCESS) waits.push_back(pr.ev);
    }
    ZeDev::Ring& r = ring_take(Z);
    /* Owner note 58 (m3 fault 14): at the VRAM ceiling the append can fail out of device memory -- a segment
     * closed while recording (H21) or a play -- and the commands of that list then never ran on this rank while
     * the other rank ran its own: the ranks' collectives no longer matched and the engine hung. Issuing the pass
     * live from here would run the stretches already played twice, so the same append is retried instead, after
     * draining this device's immediate list (its completed work returns the driver's command buffers and the
     * graveyard's tapes can go): the same commands then run on both ranks. RADIANCE_L0_FAKE_APPEND_FAIL=N makes
     * the first N appends fail as out of device memory once each (the test). Retries are logged, once a tape. */
    static std::atomic<int> fake_left{[] {
        const char* v = std::getenv("RADIANCE_L0_FAKE_APPEND_FAIL");
        return v ? std::atoi(v) : 0;
    }()};
    auto do_append = [&]() -> ze_result_t {
        if (fake_left.load(std::memory_order_relaxed) > 0 && fake_left.fetch_sub(1) > 0)
            return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        return append_fn()(Z.imm, n, lists, r.ev, (uint32_t)waits.size(), waits.empty() ? nullptr : waits.data());
    };
    ze_result_t rc = do_append();
    for (int attempt = 1; attempt <= 3 && (rc == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
                                           rc == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY); ++attempt) {
        if (!t.oom_told) {
            t.oom_told = true;
            RAD_WARN("l0 tape %llu: appending %u list(s) failed out of %s memory (0x%x); draining device %d's "
                     "immediate list and retrying (attempt %d of 3; owner note 58)", (unsigned long long)t.id, n,
                     rc == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ? "device" : "host", (unsigned)rc, t.dev, attempt);
        }
        g_append_retries.fetch_add(1, std::memory_order_relaxed);
        zeCommandListHostSynchronize(Z.imm, UINT64_MAX);
        rc = do_append();
    }
    if (rc != ZE_RESULT_SUCCESS) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "zeCommandListImmediateAppendCommandListsExp: 0x%x (%u lists)", (unsigned)rc, n);
        RAD_ERR("l0 tape %llu: %s after 3 retries -- this rank's step fails rather than waiting for commands that "
                "never ran (owner note 58)", (unsigned long long)t.id, msg);
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), msg);
    }
    r.used = true;
    r.keep = std::move(keep);
    r.kind = 0; r.tape = t.id; r.from = from; r.to = to; r.nlists = n; r.nwaits = (uint32_t)waits.size();
    r.dep_dev = t.dev; r.dep_slot = dep_slot; r.dep_serial = dep_serial;
    const int64_t np = g_nplays.fetch_add(1, std::memory_order_relaxed) + 1;
    g_nlists_played.fetch_add(n, std::memory_order_relaxed);
    if (vram_trace() && np % 256 == 0) vtrace_report(t.dev, "after play", (unsigned long long)np);
    if (np % 8192 == 0)   /* the pool's shape over a long run (run 23: the replays' host memory growth) */
        RAD_INFO("l0 tape: %lld plays; lists created %lld, destroyed %lld (alive %lld); stretches composed %lld, evicted %lld; kernel objects %lld",
                 (long long)np, (long long)g_nlists_created.load(), (long long)g_nlists_destroyed.load(),
                 (long long)(g_nlists_created.load() - g_nlists_destroyed.load()), (long long)g_nstretch_composed.load(),
                 (long long)g_nstretch_evicted.load(), (long long)g_nkernels.load());
    const int slot = (int)(&r - Z.ring.get());
    Z.prev_slot = slot;
    Z.prev_serial = r.serial.load(std::memory_order_relaxed);
    t.nslot = slot;
    t.nserial = Z.prev_serial;
    t.submitted = true;
    if (replay) for (uint32_t i = 0; i < n; ++i) t.nlast[lists[i]] = {slot, Z.prev_serial};
    if (zeEventQueryStatus(r.ev) == ZE_RESULT_SUCCESS) g_sig_early.fetch_add(1, std::memory_order_relaxed);
    if (host) zeEventHostSynchronize(r.ev, UINT64_MAX);
    for (size_t k = 0; k < t.qs.size(); ++k) {
        t.qs[k]->npend_dev = t.dev;
        t.qs[k]->npend_slot = slot;
        t.qs[k]->npend_serial = Z.prev_serial;
    }
}
/* Events recorded in a native tape's segments: recorded for real after a play as the play's slot. */
void n_rerecord(Tape& t, const std::vector<L0Event*>& evs) {
    std::lock_guard<std::mutex> lk(event_mutex());
    for (L0Event* e : evs) {
        if (!events_alive().count(e)) continue;
        std::lock_guard<std::mutex> le(e->m);
        e->ev = sycl::event();
        e->recorded = true;
        e->stamped = false;
        e->zdev = t.dev;
        e->zslot = t.nslot;
        e->zserial = t.nserial;
    }
}
/* The recorder's kernel object for a library's sycl::kernel: its own, from the kernel's module. */
NKernel& nkernel(ZeDev& Z, const sycl::kernel& k) {
    const ze_kernel_handle_t rk = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(k);
    std::lock_guard<std::mutex> lk(Z.m);
    auto it = Z.kernels.find(rk);
    if (it != Z.kernels.end()) return it->second;
    NKernel nk;
    try { nk.name = k.get_info<sycl::info::kernel::function_name>(); } catch (const sycl::exception&) { nk.name = "?"; }
    static const bool own = [] { const char* v = std::getenv("RADIANCE_L0_NTAPE_OWN_KERNELS"); return !(v && !std::strcmp(v, "0")); }();
    if (own) try {
        const std::string& name = nk.name;
        auto kb = k.get_kernel_bundle();
        std::vector<ze_module_handle_t> mods = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kb);
        for (ze_module_handle_t m : mods) {
            ze_kernel_desc_t d{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name.c_str()};
            if (zeKernelCreate(m, &d, &nk.k) == ZE_RESULT_SUCCESS) { nk.own = true; break; }
            nk.k = nullptr;
        }
    } catch (const sycl::exception&) { nk.k = nullptr; }
    if (!nk.k) {
        static std::atomic<int> told{0};
        if (own && told.fetch_add(1) < 2)
            RAD_WARN("l0 tape: no kernel object of my own for a library kernel; recording with the runtime's handle");
        nk.k = rk;
    } else {
        g_nkernels.fetch_add(1, std::memory_order_relaxed);
    }
    ze_kernel_properties_t p{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, nullptr};
    if (zeKernelGetProperties(nk.k, &p) == ZE_RESULT_SUCCESS) nk.nargs = p.numKernelArgs;
    return Z.kernels[rk] = nk;
}
/* The native recording's own refusal: what the open list holds runs, then the rest of the pass is
 * issued plainly. */
void n_stop(const std::string& why) { stop_recording(why); }
/* A barrier before each command when the driver gave no in-order list. */
void n_sep(ZeDev& Z) {
    if (!Z.inorder && t_rec.ncmds) zeCommandListAppendBarrier(t_rec.nlist, nullptr, 0, nullptr);
}
/* ---- the hooks (rad_l0_rec.h) */
int n_hook_device(const void* queue) {
    if (!native_recording() || !queue) {
        /* a library launch reaches the queue without a backend call: a held tape play goes first
         * (N3: a play held past the next libxe launch ran after it) */
        if (flush_held() != RAD_OK) RAD_WARN("device: a held tape play failed: %s", rad_dev_last_error());
        if (queue) settle(S((RadStream)queue));
        return -1;
    }
    return recorded_stream(S((RadStream)queue)) ? t_rec.t->dev : -1;
}
int n_hook_launch(const void* queue, const RadL0RecKernel* kernel, const uint32_t grid[3], const uint32_t block[3],
                  const void* arg0, size_t arg0_size, size_t lds) {
    if (!queue || !kernel || !kernel->kernel || !native_on(S((RadStream)queue))) return -1;
    ZeDev& Z = zedev(t_rec.t->dev);
    NKernel& nk = nkernel(Z, *static_cast<const sycl::kernel*>(kernel->kernel));
    if (!nk.k) { n_stop("a library kernel has no Level Zero kernel object"); return -1; }
    /* the descriptor's layout against the kernel object's: a mismatch is refused, never guessed. A
     * launch_config with work_group_scratch_size gives the kernel one implicit trailing argument the
     * descriptor does not list (N2 on card1: 2 arguments, the descriptor's one by-value parameter):
     * that is the dynamic local memory, set by size. */
    int i_lds = kernel->i_lds;
    uint32_t want_args = 1u + (i_lds >= 0 ? 1u : 0u);
    if (i_lds < 0 && nk.nargs == 2) { i_lds = 1; want_args = 2; }
    if ((nk.nargs && nk.nargs != want_args) || (kernel->arg0_size && kernel->arg0_size != arg0_size) ||
        kernel->i_arg0 < 0 || kernel->i_arg0 > 1 || i_lds > 1 || i_lds == kernel->i_arg0 || (lds && i_lds < 0)) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "a kernel's layout does not match its descriptor (%u arguments, by-value at %d of %u bytes, "
                      "local memory at %d, %zu bytes asked, %zu given)", nk.nargs, kernel->i_arg0, kernel->arg0_size,
                      i_lds, lds, arg0_size);
        n_stop(msg);
        return -1;
    }
    ze_result_t rc = zeKernelSetArgumentValue(nk.k, (uint32_t)kernel->i_arg0, arg0_size, arg0);
    if (rc != ZE_RESULT_SUCCESS) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "zeKernelSetArgumentValue(%d, %zu bytes): 0x%x (%u arguments)", kernel->i_arg0, arg0_size,
                      (unsigned)rc, nk.nargs);
        n_stop(msg);
        return -1;
    }
    if (i_lds >= 0) {
        /* the dynamic work-group memory (kind_dynamic_work_group_memory): a local argument of `lds` bytes */
        rc = zeKernelSetArgumentValue(nk.k, (uint32_t)i_lds, lds, nullptr);
        if (rc != ZE_RESULT_SUCCESS) {
            if (!nk.lds_told) {
                nk.lds_told = true;
                static std::atomic<int> told{0};
                if (told.fetch_add(1) < 2)
                    RAD_WARN("l0 tape: a kernel's local-memory argument %d refused %zu bytes (0x%x)", i_lds, lds, (unsigned)rc);
            }
            if (lds) { n_stop("the dynamic local memory argument was refused"); return -1; }
        }
    }
    rc = zeKernelSetGroupSize(nk.k, block[0], block[1], block[2]);
    if (rc != ZE_RESULT_SUCCESS) { n_stop("zeKernelSetGroupSize refused the block"); return -1; }
    n_sep(Z);
    ze_group_count_t gc{grid[0], grid[1], grid[2]};
    rc = zeCommandListAppendLaunchKernel(t_rec.nlist, nk.k, &gc, nullptr, 0, nullptr);
    if (rc != ZE_RESULT_SUCCESS) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "zeCommandListAppendLaunchKernel: 0x%x", (unsigned)rc);
        n_stop(msg);
        return -1;
    }
    ++t_rec.ncmds;
    NCmd c;
    c.kind = 0; c.k = nk.k; c.name = &nk.name; c.i_arg0 = (uint32_t)kernel->i_arg0; c.i_lds = i_lds; c.lds = lds;
    c.arg0.assign((const uint8_t*)arg0, (const uint8_t*)arg0 + arg0_size);
    for (int d = 0; d < 3; ++d) { c.gc[d] = grid[d]; c.bs[d] = block[d]; }
    t_rec.cmds.push_back(std::move(c));
    return 0;
}
int n_hook_copy2d(const void* queue, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    if (!queue || !native_on(S((RadStream)queue))) return -1;
    if (!width || !height) return 0;
    ZeDev& Z = zedev(t_rec.t->dev);
    n_sep(Z);
    ze_copy_region_t dr{0, 0, 0, (uint32_t)width, (uint32_t)height, 1};
    ze_copy_region_t sr{0, 0, 0, (uint32_t)width, (uint32_t)height, 1};
    const ze_result_t rc = zeCommandListAppendMemoryCopyRegion(t_rec.nlist, dst, &dr, (uint32_t)dpitch, 0, src, &sr,
                                                               (uint32_t)spitch, 0, nullptr, 0, nullptr);
    if (rc != ZE_RESULT_SUCCESS) { n_stop("zeCommandListAppendMemoryCopyRegion refused the copy"); return -1; }
    ++t_rec.ncmds;
    NCmd c;
    c.kind = 2; c.dst = dst; c.src = src; c.dp = dpitch; c.sp = spitch; c.w = width; c.h = height;
    t_rec.cmds.push_back(std::move(c));
    return 0;
}
/* The backend's own copies and fills while recording natively. */
int n_copy(void* dst, const void* src, size_t bytes) {
    n_sep(zedev(t_rec.t->dev));
    const ze_result_t rc = zeCommandListAppendMemoryCopy(t_rec.nlist, dst, src, bytes, nullptr, 0, nullptr);
    if (rc != ZE_RESULT_SUCCESS) { n_stop("zeCommandListAppendMemoryCopy refused the copy"); return RAD_E_DEVICE; }
    ++t_rec.ncmds;
    NCmd c;
    c.kind = 1; c.dst = dst; c.src = src; c.bytes = bytes;
    t_rec.cmds.push_back(std::move(c));
    return RAD_OK;
}
const RadL0RecHooks g_rec_hooks{n_hook_device, n_hook_launch, n_hook_copy2d};
/* One copy on stream `s`: appended while the native recording is open on it, else -- and after a
 * refusal, which has run the segment so far -- submitted plainly, so the pass never drops a copy. */
int copy_on(RadStream s, void* dst, const void* src, size_t bytes) {
    if (native_on(S(s)) && n_copy(dst, src, bytes) == RAD_OK) return RAD_OK;
    return guarded("rad_memcpy_async", [&] { Q(s).memcpy(dst, src, bytes); });
}

/* RADIANCE_L0_HOST_PREBIND=1 (the Intel port, owner note 84, run 88): rad_host_prebind. Run 87 found
 * the two-card first step stuck in the mover's first copy on card1: NEO binding the copy's host
 * buffer, which the shared context imported from card0, and xe answering VM_BIND with EBUSY forever.
 * This makes the same kind of copy (64 bytes from each host allocation, on the caller's stream, so
 * the same engine) at startup, one allocation at a time, logged before each: a bind that still
 * fails names its allocation, and the step loop's copies find their buffers bound. Off by default. */
int z_host_prebind(RadStream s, const char* tag) {
    /* 1: every host allocation; own (run 93, owner note 91 item 4): only those made with the stream's card current,
     * what that card's rank uses -- R19 showed `1` making card1 bind both ranks' pools and stop at ~26 GiB. */
    static const int mode = [] {
        const char* e = std::getenv("RADIANCE_L0_HOST_PREBIND");
        if (!e) return 0;
        if (!std::strcmp(e, "own")) return 2;
        return e[0] == '1' ? 1 : 0;
    }();
    if (!mode) return RAD_OK;
    RAD_TRY(check_stream(s, "rad_host_prebind"));
    L0Stream& st = S(s);
    RAD_TRY(enter(st));
    std::vector<std::pair<uintptr_t, AllocRec>> host;
    {
        std::lock_guard<std::mutex> lk(alloc_mutex());
        for (const auto& kv : alloc_table())
            if ((kv.second.kind == RAD_MEM_HOST_PINNED || kv.second.kind == RAD_MEM_HOST_MAPPED) &&
                (mode == 1 || kv.second.device == st.dev))
                host.push_back(kv);
    }
    const Dev& D = devs()[st.dev];
    void* scratch = nullptr;
    RAD_TRY(guarded("rad_host_prebind", [&] { scratch = sycl::malloc_device(kAlign, D.d, D.ctx); }));
    if (!scratch) {
        device_set_error("rad_host_prebind: no %lld-byte scratch on card %d", (long long)kAlign, st.dev);
        return RAD_E_NOMEM;
    }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    int64_t total = 0;
    int rc = RAD_OK;
    for (size_t i = 0; i < host.size(); ++i) {
        const AllocRec& r = host[i].second;
        RAD_INFO("l0 prebind %s (card %d): host allocation %zu of %zu at 0x%llx, %.2f MiB, kind %d, allocated with card %d current",
                 tag, st.dev, i + 1, host.size(), (unsigned long long)host[i].first, r.bytes / 1048576.0, r.kind, r.device);
        const auto t = clk::now();
        rc = guarded("rad_host_prebind", [&] {
            st.q.memcpy(scratch, (const void*)host[i].first, (size_t)std::min<int64_t>(r.bytes, 64)).wait();
        });
        if (rc < 0) break;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t).count();
        if (ms > 50) RAD_WARN("l0 prebind %s (card %d): allocation %zu took %.1f ms", tag, st.dev, i + 1, ms);
        total += r.bytes;
    }
    (void)guarded("rad_host_prebind", [&] { sycl::free(scratch, D.ctx); });
    if (rc >= 0)
        RAD_INFO("l0 prebind %s (card %d): %zu host allocation(s), %.2f GiB, touched in %.1f ms", tag, st.dev, host.size(),
                 total / 1073741824.0, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    return rc;
}

void seg_open() {
    Tape& t = *t_rec.t;
    if (t_rec.native) {
        ZeDev& Z = zedev(t.dev);
        t_rec.nlist = n_list_create(Z, Z.inorder);
        if (!t_rec.nlist) throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), "zeCommandListCreate failed");
        t_rec.ncmds = 0;
        return;
    }
    const Dev& D = devs()[t_device];
    t_rec.g.emplace(D.ctx, D.d);
    std::vector<sycl::queue> qs;
    for (L0Stream* s : t.qs) qs.push_back(s->q);
    t_rec.g->begin_recording(qs);
}
void n_early_after_seg(Tape& t, int si);   /* RADIANCE_L0_TAPE_EARLY (owner note 65), below n_compose */
void n_early_end(Tape& t);
/* Closes the open native segment: close the list, run it (a play of one list), keep it. */
void n_seg_close() {
    Tape& t = *t_rec.t;
    ze_command_list_handle_t l = t_rec.nlist;
    t_rec.nlist = nullptr;
    Seg sg;
    sg.nodes = t_rec.ncmds;
    sg.evs.swap(t_rec.pend);
    sg.cmds.swap(t_rec.cmds);
    /* Not detected: a direct SYCL submission on a recorded stream inside the recording. It runs at
     * once and is absent from the replay (native_tape_test case 5). N2 attempt 2 tried to catch it by
     * the stream's last event moving between a segment's open and close, but
     * ext_oneapi_get_last_event on an in-order queue makes a new event (a barrier) on each call, so
     * the comparison always differed. The engine and libxe submit only through the recorder. */
    if (sg.nodes) {
        const ze_result_t rc = zeCommandListClose(l);
        if (rc != ZE_RESULT_SUCCESS) {
            zeCommandListDestroy(l);
            g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed);
            throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), "zeCommandListClose failed");
        }
        sg.list = l;
        t.ncmds += sg.nodes;
        vtrace_list(t.dev, 0, 1);
    } else {
        zeCommandListDestroy(l);
        g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed);
    }
    if (sg.nodes || !sg.evs.empty()) {
        /* kept before it runs: a failed append leaves the list to the tape's destructor, not leaked */
        t.segs.push_back(std::move(sg));
        Seg& kept = t.segs.back();
        const int si = (int)t.segs.size() - 1;
        if (kept.list) n_play_lists(t, &kept.list, 1, false, si, si + 1);   /* a recording's segments are distinct lists: no same-tape wait */
        if (kept.list) n_rerecord(t, kept.evs);
        else if (t.nslot >= 0) { n_rerecord(t, kept.evs); ++t.dangling_fixed; }   /* owner note 42: event-only after native work: a runtime-queue record would not follow the lists */
        else rerecord(t, kept.evs);
        n_early_after_seg(t, si);
    }
}
/* Closes the open segment: finalize, run, keep. */
void seg_close() {
    if (t_rec.native) { n_seg_close(); return; }
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
        sg.g.emplace(*t_rec.g);        /* the handle to the recorded graph, for a later re-finalize */
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
    RAD_TRY(flush_held());
    reap_tapes();
    if (t_rec.t) { device_set_error("rad tape_begin: a tape is already being recorded on this thread"); return RAD_E_STATE; }
    if (!dev_ok(t_device, "tape_begin")) return RAD_E_INVAL;
    const Dev& D = devs()[t_device];
    /* RADIANCE_L0_TAPE=0: every pass is issued in full (owner note 1's plain-launch measurement);
     * =graph: SYCL command graphs; else native command lists where the device allows (owner note 28) */
    if (tape_mode() == 0) { device_set_error("tape_begin: RADIANCE_L0_TAPE=0"); return RAD_E_UNSUPPORTED; }
    const bool native = tape_mode() == 2 && ze_ready(t_device);
    if (!native) {
        static std::atomic<int> told{0};
        if (tape_mode() == 2 && told.fetch_add(1) < 1)
            RAD_INFO("l0 tape: SYCL command graphs on device %d (%s)", t_device, zedev(t_device).why.c_str());
        if (!D.d.has(sycl::aspect::ext_oneapi_graph) && !D.d.has(sycl::aspect::ext_oneapi_limited_graph)) {
            device_set_error("tape_begin: the device takes no SYCL command graphs");
            return RAD_E_UNSUPPORTED;
        }
    }
    auto t = std::make_unique<Tape>();
    t->id = ++g_tape_ids;
    t->dev = t_device;
    t->native = native;
    {
        std::lock_guard<std::mutex> lk(stream_mutex());
        /* This device's streams only (TP2, run 9: with two rank threads, each recording its own card,
         * the other rank's not-yet-used streams were swept into this tape's graph). */
        for (L0Stream* s : streams())
            if (s->dev == t_device && (t_touched.count(s->id) || !s->used.load(std::memory_order_relaxed)))
                t->qs.push_back(s);
        for (size_t k = 0; k < t->qs.size(); ++k)
            if (t->qs[k] == t_last) std::swap(t->qs[0], t->qs[k]);
        /* RADIANCE_L0_TAPE_HOME_ONLY=1 (step 5 experiment): record the home stream alone. Valid only
         * where nothing else is used in the pass -- the tape-vs-plain token check says so. */
        static const bool home = [] { const char* v = std::getenv("RADIANCE_L0_TAPE_HOME_ONLY");
                                      return v && std::strcmp(v, "1") == 0; }();
        if (home && t->qs.size() > 1) t->qs.resize(1);
    }
    if (t->qs.empty()) { device_set_error("tape_begin: no stream to record"); return RAD_E_STATE; }
    if (native && !imm_of(*t->qs[0])) {
        device_set_error("tape_begin: the home stream's native handle is not an immediate command list");
        return RAD_E_UNSUPPORTED;
    }
    if (vram_trace() && native) {
        vtrace_scan(t_device);
        static std::atomic<int> told{0};
        if (told.fetch_add(1) < 4) vtrace_report(t_device, "before tape", (unsigned long long)t->id);
    }
    t_rec = Rec{};
    t_rec.t = t.release();
    t_rec.native = native;
    const int rc = guarded("tape_begin", [] { seg_open(); });
    if (rc != RAD_OK) { delete t_rec.t; t_rec = Rec{}; }
    return rc;
}
/* Gives up the recording: the queues leave record mode and the rest of the pass is issued plainly.
 * A native segment's commands so far run first (they were appended, not submitted). */
void stop_recording(const std::string& why) {
    refuse(why);
    if (t_rec.native && t_rec.nlist) {
        try { n_seg_close(); }
        catch (const sycl::exception& e) { RAD_WARN("l0 tape: the refused segment could not be run: %s", e.what()); }
        t_rec.nlist = nullptr;
    }
    if (t_rec.g) {
        std::vector<sycl::queue> qs;
        for (L0Stream* s : t_rec.t->qs) qs.push_back(s->q);
        try { t_rec.g->end_recording(qs); } catch (const sycl::exception&) {}
        t_rec.g.reset();
    }
    t_rec.stopped = true;
}
int z_tape_mark(void) {
    if (!t_rec.t) return 0;
    if (recording()) {
        const bool cap = t_rec.t->segs.size() + 1 >= kTapeMaxSegs;
        const int rc = guarded("tape_mark", [cap] { seg_close(); if (!cap) seg_open(); });
        if (rc != RAD_OK) t_rec.lost = true;
        if (rc != RAD_OK)
            stop_recording(std::string("a segment could not be closed: ") + rad_dev_last_error());
        else if (cap)
            stop_recording("the pass has more than " + std::to_string(kTapeMaxSegs) +
                           " segments (a prefill-sized pass): it runs plainly, so recorded graphs stay bounded");
    }
    return (int)t_rec.t->segs.size();
}
void z_tape_pause(int on) {
    if (!t_rec.t || t_rec.stopped) return;
    if (on) {
        if (t_rec.pause++ == 0 && (t_rec.g || t_rec.nlist) && guarded("tape_pause", [] { seg_close(); }) != RAD_OK) {
            t_rec.lost = true;
            stop_recording("a segment could not be closed at a pause");
        }
    } else if (t_rec.pause > 0 && --t_rec.pause == 0) {
        if (guarded("tape_pause", [] { seg_open(); }) != RAD_OK) stop_recording("no segment after a pause");
    }
}
RadTape z_tape_end(int keep) {
    Tape* t = t_rec.t;
    if (!t) return nullptr;
    if ((t_rec.g || t_rec.nlist) && guarded("tape_end", [] { seg_close(); }) != RAD_OK) {
        t_rec.lost = true;
        refuse("the last segment could not be closed");
    }
    const bool lost = t_rec.lost;
    t_rec = Rec{};
    if (vram_trace() && t->native) vtrace_report(t->dev, "after tape", (unsigned long long)t->id);
    /* Owner note 58 (E11): a segment whose close failed never ran, so this pass is incomplete on this rank --
     * reported with a fixed prefix the context turns into a failed step, never a silent refusal (the rest of
     * the pass ran plainly without that segment's commands and gave a different answer). */
    if (lost) {
        device_set_error("l0 tape lost: a recorded segment's commands never ran (its list could not be "
                         "appended): the pass is incomplete on this rank");
        release_tape(t);
        return nullptr;
    }
    /* A pass always puts work on the card; an empty recording means it went to a stream outside
     * the recording, and replaying nothing would be silently wrong. */
    if (t->segs.empty() && t->refused.empty())
        t->refused = "nothing was recorded (the pass's work went to a stream outside the recording)";
    /* owner note 42 (run 38): every recorded wait must have its signaller in the tape -- the event's record kept
     * in a segment at or before the waiter's -- or in the runtime's prior work (an event recorded before the
     * pass, which z_event_wait does not list). One without is refused: the pass runs live, and the warning
     * says which tape. */
    if (t->native && t->refused.empty() && !t->rec_waits.empty()) {
        int lost = 0;
        for (const auto& w : t->rec_waits) {
            bool found = false;
            for (int k = 0; k <= w.second && k < (int)t->segs.size() && !found; ++k)
                for (L0Event* e : t->segs[(size_t)k].evs) if (e == w.first) { found = true; break; }
            if (!found) ++lost;
        }
        if (lost) {
            RAD_WARN("l0 tape %llu: %d of %zu recorded wait(s) have no signaller in the tape (the event's record is in no segment at or before the waiter's): the pass runs live",
                     (unsigned long long)t->id, lost, t->rec_waits.size());
            t->refused = "a recorded wait has no signaller in the tape";
        }
    }
    if (t->native && t->dangling_fixed) {
        static std::atomic<int> told_fix{0};
        if (told_fix.fetch_add(1) < 8)
            RAD_INFO("l0 tape %llu: %d event-only segment(s) closed after native work: their event records follow the tape's latest play, not the runtime queue",
                     (unsigned long long)t->id, t->dangling_fixed);
    }
    if (!keep || !t->refused.empty()) {
        if (keep) device_set_error("the pass cannot be replayed from a SYCL graph: %s", t->refused.c_str());
        release_tape(t);   /* its segments ran while recorded and may still be running */
        return nullptr;
    }
    /* Step 5's per-launch figure: what one replay submits (graph nodes are the pass's dispatches,
     * copies and barriers), for the first few tapes kept. */
    size_t nodes = 0, evs = 0;
    for (const Seg& s : t->segs) { nodes += s.nodes; evs += s.evs.size(); }
    t->fence = evs > 0;
    static std::atomic<int> told{0};
    if (told.fetch_add(1) < 8)
        RAD_INFO("l0 tape: %zu segment(s), %zu %s, %zu event record(s), %zu stream(s) recorded%s",
                 t->segs.size(), nodes, t->native ? "command(s) in native lists" : "graph node(s)", evs, t->qs.size(),
                 t->fence ? "" : ", played unfenced");
    if (t->native) {
        const int rc = guarded("tape_end", [t] { n_early_end(*t); });
        if (rc != RAD_OK) { release_tape(t); return nullptr; }
    }
    return (RadTape)t;
}

/* ---- held-back plays. The engine plays a pass as stretches cut at its host work (table
 * refreshes, routing reports); in a steady decode step that work submits nothing, so a play is
 * held back and the next play of the same tape that starts where it ends extends it: one graph
 * submission a step instead of one a stretch. Anything else that reaches a queue -- every backend
 * call that submits, waits or frees, and the core's own kernels (l0_kernels.sycl, through
 * rad_l0k_set_flush) -- submits the held play first. A held play's error surfaces at that flush.
 * One held play per device: a flush submits the current device's. With two rank threads (TP2, run 9)
 * a single slot had rank 1's first backend call play rank 0's held tape -- composing rank 0's
 * executable graphs into a graph of rank 1's device, on rank 1's thread -- and the server died of a
 * SIGSEGV after the first recorded pass (T1 attempt 3). */
int play_now(Tape& t, int from, int to);
/* One mutex a device (run 19, M11): a process-wide one, held across play_now's host waits, let rank 0's
 * wait for its own device block rank 1's next play -- the play whose collective rank 0's device was
 * waiting for (M4/M9/M11: the first decode step of a request's second run stalled for good). */
std::mutex& held_mutex(int d) { static std::mutex m[kMaxDevices]; return m[d]; }
struct Held { Tape* t = nullptr; int a = 0, b = 0; };
Held g_held[kMaxDevices];
std::atomic<bool> g_have_held[kMaxDevices];
int flush_held() {
    const int d = t_device;
    if (!g_have_held[d].load(std::memory_order_acquire)) return RAD_OK;
    std::lock_guard<std::mutex> lk(held_mutex(d));
    const Held h = g_held[d];
    g_held[d] = Held{};
    g_have_held[d].store(false, std::memory_order_release);
    return h.t ? play_now(*h.t, h.a, h.b) : RAD_OK;
}
/* librad_l0k's hook (it cannot return an error to its caller's caller, so it warns) */
void flush_held_hook(void) {
    if (flush_held() != RAD_OK) RAD_WARN("device: a held tape play failed: %s", rad_dev_last_error());
}
int z_tape_len(RadTape p) { return p ? (int)static_cast<Tape*>(p)->segs.size() : RAD_E_INVAL; }
int z_tape_play(RadTape p, int from, int to) {
    auto* t = static_cast<Tape*>(p);
    if (!t || from < 0 || to > (int)t->segs.size() || from > to) { device_set_error("tape_play: bad range"); return RAD_E_INVAL; }
    if (from == to) return RAD_OK;
    std::lock_guard<std::mutex> lk(held_mutex(t->dev));
    Held& held = g_held[t->dev];
    if (held.t == t && held.b == from) { held.b = to; return RAD_OK; }   /* the next stretch */
    const Held prev = held;
    held = Held{t, from, to};
    g_have_held[t->dev].store(true, std::memory_order_release);
    return prev.t ? play_now(*prev.t, prev.a, prev.b) : RAD_OK;
}
/* Owner note 27, G2 (run 12): a step plays the pass as dozens of stretches whose ranges change with the
 * mover's refresh points, so composing a stretch graph per range grew the pool without bound (one
 * executable per distinct range; +8 GB of host memory over two requests), and releasing the segment
 * executables made every new range re-finalize its segments (minutes a request). G3 then showed the
 * other side: playing the ~1,036 segment executables directly bounds the pool (+0.9 GB a card) but costs
 * ~13 us of device-side gap a submission (40.7 ms a step against 27.2 composed).
 * Default now, "hybrid": a stretch of kComposeMin segments or more is composed and cached as before, but
 * the cache is bounded (kMaxStretch entries a tape, least recently played out, freed once its last
 * submission has completed); shorter stretches play their segment executables directly; the segment
 * executables stay (they are what a composition reads). RADIANCE_L0_TAPE_COMPOSE=direct|stretch|record
 * = the other modes (stretch = the old unbounded cache, with the segment release above);
 * RADIANCE_L0_TAPE_STRETCHES=N sets the bound. */
constexpr int kComposeMin = 8;
constexpr int kBlock = 16;   /* a composed block's segments (run 23): a 1,036-segment pass is 65 blocks */
int compose_mode() {   /* 0 direct, 1 stretch (unbounded), 2 record, 3 hybrid */
    static const int m = [] {
        const char* v = std::getenv("RADIANCE_L0_TAPE_COMPOSE");
        if (!v) return 3;
        if (!std::strcmp(v, "record")) return 2;
        if (!std::strcmp(v, "stretch")) return 1;
        if (!std::strcmp(v, "direct")) return 0;
        return 3;
    }();
    return m;
}
size_t max_stretches() {
    /* 512: with fixed blocks (run 23) a tape holds its blocks and the recurring partials, a few hundred at most */
    static const size_t n = [] { const char* v = std::getenv("RADIANCE_L0_TAPE_STRETCHES"); long x = v ? std::atol(v) : 0; return x > 0 ? (size_t)x : (size_t)512; }();
    return n;
}
std::atomic<int64_t> g_stretch_composed{0}, g_stretch_evicted{0}, g_plays_direct{0};
void run_on_home_direct(Tape& t, int from, int to) {
    sycl::queue& H = t.qs[0]->q;
    const bool fence = t.fence && t.qs.size() > 1;
    if (fence) {
        std::vector<sycl::event> pre;
        for (size_t k = 1; k < t.qs.size(); ++k) pre.push_back(t.qs[k]->q.ext_oneapi_submit_barrier());
        H.ext_oneapi_submit_barrier(pre);
    }
    bool any = false;
    for (int i = from; i < to; ++i) {
        Seg& s = t.segs[(size_t)i];
        if (!s.ex) continue;
        t.last = H.ext_oneapi_graph(*s.ex);
        any = true;
    }
    if (any) t.submitted = true;
    if (fence && any)
        for (size_t k = 1; k < t.qs.size(); ++k) t.qs[k]->q.ext_oneapi_submit_barrier({t.last});
}
/* Frees the evicted stretch executables whose last submission has completed (the tape's `last` at the
 * time of eviction: everything of the stretch ran before it on the in-order home stream). Past kDeadMax
 * pending, the oldest is waited for. */
void reap_stretches(Tape& t, bool all = false) {
    auto& d = t.dead_stretch;
    while (!d.empty()) {
        bool done = true;
        try { done = d.front().second.get_info<sycl::info::event::command_execution_status>() ==
                     sycl::info::event_command_status::complete; }
        catch (const sycl::exception&) { done = true; }
        if (!done) {
            if (!all && d.size() <= kDeadMax) break;
            try { d.front().second.wait(); } catch (const sycl::exception&) {}
        }
        d.pop_front();
    }
}
/* Composes segments [from, to) into one in-order list from their descriptors (owner note 28's "one
 * command list per recorded stretch"). Null when the range holds no command. */
ze_command_list_handle_t n_compose(Tape& t, int from, int to) {
    ZeDev& Z = zedev(t.dev);
    size_t n = 0;
    for (int i = from; i < to; ++i) n += t.segs[(size_t)i].cmds.size();
    if (!n) return nullptr;
    ze_command_list_handle_t l = n_list_create(Z, Z.inorder);
    if (!l) throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), "zeCommandListCreate failed (a composed stretch)");
    NProf prof;
    if (ntape_profile()) {
        ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
        void* pm = nullptr;
        if (zeMemAllocHost(Z.ctx, &hd, 16 * n, 64, &pm) == ZE_RESULT_SUCCESS && pm) {
            prof.ts = (uint64_t*)pm;
            std::memset(pm, 0, 16 * n);
            prof.n = (uint32_t)n;
            prof.names.reserve(n);
            prof.kinds.reserve(n);
        }
    }
    size_t k = 0;
    auto fail = [&](const char* what, ze_result_t rc) {
        zeCommandListDestroy(l);
        g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed);
        char msg[128];
        std::snprintf(msg, sizeof msg, "%s: 0x%x (composing a stretch)", what, (unsigned)rc);
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), msg);
    };
    for (int i = from; i < to; ++i)
        for (const NCmd& c : t.segs[(size_t)i].cmds) {
            if (!Z.inorder && k) zeCommandListAppendBarrier(l, nullptr, 0, nullptr);
            ze_result_t rc = ZE_RESULT_SUCCESS;
            if (prof.ts) {
                zeCommandListAppendWriteGlobalTimestamp(l, prof.ts + 2 * k, nullptr, 0, nullptr);
                prof.names.push_back(c.name);
                prof.kinds.push_back(c.kind);
            }
            if (c.kind == 0) {
                rc = zeKernelSetArgumentValue(c.k, c.i_arg0, c.arg0.size(), c.arg0.data());
                if (rc == ZE_RESULT_SUCCESS && c.i_lds >= 0) {
                    const ze_result_t rl = zeKernelSetArgumentValue(c.k, (uint32_t)c.i_lds, c.lds, nullptr);
                    if (rl != ZE_RESULT_SUCCESS && c.lds) rc = rl;
                }
                if (rc == ZE_RESULT_SUCCESS) rc = zeKernelSetGroupSize(c.k, c.bs[0], c.bs[1], c.bs[2]);
                ze_group_count_t gc{c.gc[0], c.gc[1], c.gc[2]};
                if (rc == ZE_RESULT_SUCCESS) rc = zeCommandListAppendLaunchKernel(l, c.k, &gc, nullptr, 0, nullptr);
                if (rc != ZE_RESULT_SUCCESS) fail("a launch", rc);
            } else if (c.kind == 1) {
                rc = zeCommandListAppendMemoryCopy(l, c.dst, c.src, c.bytes, nullptr, 0, nullptr);
                if (rc != ZE_RESULT_SUCCESS) fail("a copy", rc);
            } else {
                ze_copy_region_t dr{0, 0, 0, (uint32_t)c.w, (uint32_t)c.h, 1};
                ze_copy_region_t sr{0, 0, 0, (uint32_t)c.w, (uint32_t)c.h, 1};
                rc = zeCommandListAppendMemoryCopyRegion(l, c.dst, &dr, (uint32_t)c.dp, 0, c.src, &sr, (uint32_t)c.sp, 0, nullptr, 0, nullptr);
                if (rc != ZE_RESULT_SUCCESS) fail("a 2D copy", rc);
            }
            if (prof.ts) zeCommandListAppendWriteGlobalTimestamp(l, prof.ts + 2 * k + 1, nullptr, 0, nullptr);
            ++k;
        }
    const ze_result_t rc = zeCommandListClose(l);
    if (rc != ZE_RESULT_SUCCESS) fail("zeCommandListClose", rc);
    if (prof.ts) { std::lock_guard<std::mutex> lk(nprof_mutex()); nprof_lists()[l] = std::move(prof); }
    g_nstretch_composed.fetch_add(1, std::memory_order_relaxed);
    vtrace_list(t.dev, 1, 1);
    static std::atomic<int> told{0};
    if (told.fetch_add(1) < 4)
        RAD_INFO("l0 tape %llu: segments [%d, %d) composed into one native list of %zu command(s)", (unsigned long long)t.id, from, to, n);
    return l;
}
/* Destroys the evicted stretch lists whose follow-up play has completed. */
void n_reap_stretches(Tape& t) {
    while (!t.ndead.empty() && n_slot_done(t.dev, t.ndead.front().slot, t.ndead.front().serial)) {
        if (t.ndead.front().list) { zeCommandListDestroy(t.ndead.front().list); g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed); vtrace_list(t.dev, 1, -1); }
        t.ndead.pop_front();
    }
}
/* RADIANCE_L0_TAPE_EARLY=1 (owner note 65). The driver keeps the high-water mark of lists alive at once
 * (X1, V2), and with segment lists waiting for their tape's first play, the recordings of new shapes stack
 * up before any is replayed (H26: 3,636 segment lists at once, +4.9 GiB a card, where the steady state
 * plays 542 composed lists). Here a recording releases its segment lists as it goes: each segment's list is
 * queued with its own play's slot and destroyed once that play has completed, the recording waiting for
 * the oldest when more than kEarlyWindow are in flight; each full block of kBlock segments is composed as it
 * fills (from the kept descriptors, as a play would); tape_end waits for the recording's last play, releases
 * the rest and marks the tape so every play composes (the partial ranges on demand, as today). Off: the
 * shipped behaviour (segment lists live until the first play). Kept opt-in after owner note 68's M-pair (run
 * 90): text identical both ways, but the MTP-on single prompt 48.9 tok/s against 51.5 (M54 / M56 against M55 /
 * M57), the plays' composing costing host time; NEO's 64 KB command buffer per list (the service's setting,
 * m3/service/run.sh) takes most of the recordings' VRAM back without it (H56: +1.9 GiB against +4.9). */
bool tape_early() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_L0_TAPE_EARLY"); return e && e[0] == '1'; }();
    return v;
}
constexpr size_t kEarlyWindow = 64;   /* segment lists of the current recording in flight at most */
void n_early_drop(Tape& t, const Tape::NDead& d) {
    if (!d.list) return;
    t.nlast.erase(d.list);
    zeCommandListDestroy(d.list);
    g_nlists_destroyed.fetch_add(1, std::memory_order_relaxed);
    vtrace_list(t.dev, 0, -1);
}
void n_early_after_seg(Tape& t, int si) {
    if (!tape_early() || !t.native || compose_mode() == 0) return;
    Seg& s = t.segs[(size_t)si];
    if (s.list) {
        t.nseg_dead.push_back(Tape::NDead{s.list, t.nslot, t.nserial});   /* its play is the tape's latest */
        s.list = nullptr;
    }
    if ((si + 1) % kBlock == 0) {
        const std::pair<int, int> key{si + 1 - kBlock, si + 1};
        if (!t.nstretch_used.count(key)) {
            t.nstretch[key] = n_compose(t, key.first, key.second);   /* null when the block holds no command */
            t.nstretch_used[key] = ++t.nclock;
        }
    }
    while (!t.nseg_dead.empty() && n_slot_done(t.dev, t.nseg_dead.front().slot, t.nseg_dead.front().serial)) {
        n_early_drop(t, t.nseg_dead.front());
        t.nseg_dead.pop_front();
    }
    while (t.nseg_dead.size() > kEarlyWindow) {
        n_slot_wait(t.dev, t.nseg_dead.front().slot, t.nseg_dead.front().serial, "an early segment release");
        n_early_drop(t, t.nseg_dead.front());
        t.nseg_dead.pop_front();
    }
}
void n_early_end(Tape& t) {
    if (!tape_early() || !t.native || compose_mode() == 0) return;
    for (Seg& s : t.segs)   /* none expected: each segment's list was queued at its close */
        if (s.list) { t.nseg_dead.push_back(Tape::NDead{s.list, t.nslot, t.nserial}); s.list = nullptr; }
    if (!t.nseg_dead.empty()) {
        /* the immediate list is in order: the last queued play complete means all of them are */
        n_slot_wait(t.dev, t.nseg_dead.back().slot, t.nseg_dead.back().serial, "the recording's segment release");
        for (const Tape::NDead& d : t.nseg_dead) n_early_drop(t, d);
        t.nseg_dead.clear();
    }
    t.segs_dropped = true;
}
int play_now(Tape& tr, int from, int to) {
    Tape* t = &tr;
    if (t->native) {
        return guarded("tape_play", [&] {
            std::vector<ze_command_list_handle_t> lists;
            /* phase 2 of the pool: from the first play on, every range is a composed list; the segment
             * lists (one a segment, 1,036 a decode pass) go once the recording's own plays have completed */
            if (compose_mode() != 0 && !t->segs_dropped) {
                if (t->submitted && !n_slot_done(t->dev, t->nslot, t->nserial)) n_slot_wait(t->dev, t->nslot, t->nserial, "the segment lists' release");
                int64_t n = 0;
                for (Seg& s : t->segs)
                    if (s.list) { zeCommandListDestroy(s.list); s.list = nullptr; ++n; }
                g_nlists_destroyed.fetch_add(n, std::memory_order_relaxed);
                vtrace_list(t->dev, 0, -(int)n);
                t->segs_dropped = true;
                static std::atomic<int> told{0};
                if (told.fetch_add(1) < 4)
                    RAD_INFO("l0 tape %llu: %lld segment list(s) released at the first play; every played range is a composed list from here",
                             (unsigned long long)t->id, (long long)n);
            }
            const bool compose = compose_mode() != 0 && (t->segs_dropped || to - from >= kComposeMin);
            if (compose) {
                /* fixed blocks + the partial head and tail (run 23, H2c): the ranges the mover cuts move every
                 * step, so one list a range was 160 composed lists a second under the rotation; the blocks
                 * are composed once and the partials recur */
                auto take = [&](int a, int b) {
                    if (a >= b) return;
                    ze_command_list_handle_t& l = t->nstretch[{a, b}];
                    if (!l && !t->nstretch_used.count({a, b})) l = n_compose(*t, a, b);
                    t->nstretch_used[{a, b}] = ++t->nclock;
                    if (l) lists.push_back(l);
                };
                const int b0 = (from + kBlock - 1) / kBlock * kBlock, b1 = to / kBlock * kBlock;
                if (b0 >= b1) take(from, to);
                else {
                    take(from, b0);
                    for (int a = b0; a < b1; a += kBlock) take(a, a + kBlock);
                    take(b1, to);
                }
            } else {
                lists.reserve((size_t)(to - from));
                for (int i = from; i < to; ++i)
                    if (t->segs[(size_t)i].list) lists.push_back(t->segs[(size_t)i].list);
            }
            n_play_lists(*t, lists.data(), (uint32_t)lists.size(), true, from, to);
            if (compose) {
                /* the bounded cache: the least recently played range out, its list destroyed once this play
                 * (which the tape's next play waits for) has completed */
                while (t->nstretch.size() > max_stretches()) {
                    auto lru = t->nstretch_used.begin();
                    for (auto it = t->nstretch_used.begin(); it != t->nstretch_used.end(); ++it)
                        if (it->second < lru->second) lru = it;
                    auto victim = t->nstretch.find(lru->first);
                    if (victim != t->nstretch.end()) {
                        if (victim->second) t->ndead.push_back(Tape::NDead{victim->second, t->nslot, t->nserial});
                        t->nstretch.erase(victim);
                    }
                    t->nstretch_used.erase(lru);
                    g_nstretch_evicted.fetch_add(1, std::memory_order_relaxed);
                }
                n_reap_stretches(*t);
            }
            if (t->plays_told < 6) {
                ++t->plays_told;
                RAD_INFO("l0 tape %llu: play [%d, %d) of %zu segment(s), %zu native list(s)%s", (unsigned long long)t->id,
                         from, to, t->segs.size(), lists.size(), compose ? " (composed)" : "");
            }
            std::vector<L0Event*> evs;
            for (int i = from; i < to; ++i) evs.insert(evs.end(), t->segs[(size_t)i].evs.begin(), t->segs[(size_t)i].evs.end());
            /* an event-only range after played segments follows the tape's latest play (owner note 42) */
            if (!lists.empty() || (from > 0 && t->nslot >= 0)) n_rerecord(*t, evs); else rerecord(*t, evs);
        });
    }
    const int mode = compose_mode();
    if (mode == 0 || (mode == 3 && to - from < kComposeMin)) {
        return guarded("tape_play", [&] {
            run_on_home_direct(*t, from, to);
            g_plays_direct.fetch_add(1, std::memory_order_relaxed);
            if (t->plays_told < 6) {
                ++t->plays_told;
                RAD_INFO("l0 tape %llu: play [%d, %d) of %zu segment(s), direct", (unsigned long long)t->id, from, to, t->segs.size());
            }
            std::vector<L0Event*> evs;
            for (int i = from; i < to; ++i) evs.insert(evs.end(), t->segs[(size_t)i].evs.begin(), t->segs[(size_t)i].evs.end());
            rerecord(*t, evs);
        });
    }
    return guarded("tape_play", [&] {
        std::shared_ptr<Exec>& ex = t->stretch[{from, to}];
        /* the submission before this play: when it has completed, every earlier command buffer of
         * the tape (the recording's segment runs included) is done */
        const bool had = t->submitted;
        const sycl::event before = t->last;
        if (!ex) {
            const Dev& D = devs()[t->dev];
            Graph g(D.ctx, D.d);
            /* a released segment executable is re-finalized from its graph for the composition only */
            std::vector<std::shared_ptr<Exec>> temps;
            auto seg_exec = [&](int i) -> Exec* {
                Seg& s = t->segs[(size_t)i];
                if (s.ex) return s.ex.get();
                if (!s.g) return nullptr;
                temps.push_back(std::make_shared<Exec>(s.g->finalize()));
                g_segs_refinalized.fetch_add(1, std::memory_order_relaxed);
                return temps.back().get();
            };
            /* RADIANCE_L0_TAPE_COMPOSE=record (step 5 experiment): the stretch recorded from a private
             * in-order queue, each segment a sub-graph submission, so the runtime may build an
             * in-order command list; default: explicit nodes chained by depends_on. */
            static const bool rec = [] { const char* v = std::getenv("RADIANCE_L0_TAPE_COMPOSE");
                                         return v && std::strcmp(v, "record") == 0; }();
            bool any = false;
            if (rec) {
                sycl::queue rq(D.ctx, D.d, sycl::property_list{sycl::property::queue::in_order()});
                g.begin_recording(rq);
                for (int i = from; i < to; ++i)
                    if (Exec* sub = seg_exec(i)) { rq.ext_oneapi_graph(*sub); any = true; }
                g.end_recording(rq);
            } else {
                std::optional<sx::node> prev;
                for (int i = from; i < to; ++i) {
                    Exec* sub = seg_exec(i);
                    if (!sub) continue;
                    auto add = [&](sycl::handler& h) { h.ext_oneapi_graph(*sub); };
                    prev = prev ? g.add(add, sycl::property_list{sx::property::node::depends_on(*prev)}) : g.add(add);
                }
                any = (bool)prev;
            }
            if (any) { ex = std::make_shared<Exec>(g.finalize()); g_stretch_composed.fetch_add(1, std::memory_order_relaxed); }
        }
        if (ex) run_on_home(*t, *ex);
        if (t->plays_told < 6) {
            ++t->plays_told;
            RAD_INFO("l0 tape %llu: play [%d, %d) of %zu segment(s)%s", (unsigned long long)t->id, from, to,
                     t->segs.size(), ex ? "" : " (nothing to run)");
        }
        if (mode == 3) {
            /* the bounded cache: this stretch is the most recently played; evict the least recent past the
             * bound, into the graveyard until its last submission (<= t->last, in order) completes */
            t->stretch_used[{from, to}] = ++t->stretch_clock;
            while (t->stretch.size() > max_stretches()) {
                auto lru = t->stretch_used.begin();
                for (auto it = t->stretch_used.begin(); it != t->stretch_used.end(); ++it)
                    if (it->second < lru->second) lru = it;
                auto victim = t->stretch.find(lru->first);
                if (victim != t->stretch.end()) {
                    if (victim->second) t->dead_stretch.emplace_back(std::move(victim->second), t->last);
                    t->stretch.erase(victim);
                }
                t->stretch_used.erase(lru);
                g_stretch_evicted.fetch_add(1, std::memory_order_relaxed);
            }
            reap_stretches(*t);
        }
        /* note 27, "stretch" mode only: once composed stretches hold every segment (G1, run 12: a step
         * plays the pass as more than one stretch, so the full range alone never came) and nothing of the
         * recording still runs -> release the segment executables (command lists and scratch surfaces) */
        if (mode != 1) return;
        if (t->covered.size() != t->segs.size()) t->covered.assign(t->segs.size(), 0);
        for (int i = from; i < to; ++i) t->covered[(size_t)i] = 1;
        bool all = true;
        for (char c : t->covered) if (!c) { all = false; break; }
        if (!t->segs_released && !keep_segs() && all) {
            bool done = !had;
            if (!done) {
                try { done = before.get_info<sycl::info::event::command_execution_status>() ==
                             sycl::info::event_command_status::complete; }
                catch (const sycl::exception&) { done = false; }
            }
            if (done) {
                int64_t n = 0;
                for (Seg& s : t->segs) if (s.ex) { s.ex.reset(); ++n; }
                t->segs_released = true;
                g_segs_released.fetch_add(n, std::memory_order_relaxed);
                static std::atomic<int> told{0};
                if (told.fetch_add(1) < 4)
                    RAD_INFO("l0 tape: %lld segment graph(s) released after the full-pass graph (%zu segments, "
                             "owner note 27); their recorded graphs stay for re-composition", (long long)n, t->segs.size());
            }
        }
        std::vector<L0Event*> evs;
        for (int i = from; i < to; ++i) evs.insert(evs.end(), t->segs[(size_t)i].evs.begin(), t->segs[(size_t)i].evs.end());
        rerecord(*t, evs);
    });
}
/* Segment by segment: node and event-record counts (SYCL keeps a graph's arguments to itself, so
 * this is weaker than the AQL tape's packet comparison; the pass key fixes the buffers). */
/* Native tapes: the segments' command descriptors byte for byte. 0 when they agree, else the
 * segment's index with the first difference in `why`. */
int n_tape_diff(const Tape& a, const Tape& b, size_t i, char* why, int cap) {
    const std::vector<NCmd>& x = a.segs[i].cmds;
    const std::vector<NCmd>& y = b.segs[i].cmds;
    if (x.size() != y.size()) {
        if (why && cap > 0) std::snprintf(why, (size_t)cap, "segment %zu: %zu commands against %zu", i, x.size(), y.size());
        return 1;
    }
    for (size_t j = 0; j < x.size(); ++j) {
        const NCmd& c = x[j];
        const NCmd& d = y[j];
        const char* nm = c.name ? c.name->c_str() : (c.kind == 1 ? "copy" : "2D copy");
        if (c.kind != d.kind || c.k != d.k) {
            if (why && cap > 0) std::snprintf(why, (size_t)cap, "segment %zu, command %zu: %s against %s", i, j, nm,
                                              d.name ? d.name->c_str() : (d.kind == 1 ? "copy" : "2D copy"));
            return 1;
        }
        if (c.kind == 0) {
            if (std::memcmp(c.gc, d.gc, sizeof c.gc) || std::memcmp(c.bs, d.bs, sizeof c.bs) || c.lds != d.lds) {
                if (why && cap > 0)
                    std::snprintf(why, (size_t)cap, "segment %zu, command %zu (%s): the launch shape differs (%ux%ux%u blocks of %ux%ux%u, %zu B local"
                                  " against %ux%ux%u of %ux%ux%u, %zu B)", i, j, nm, c.gc[0], c.gc[1], c.gc[2], c.bs[0], c.bs[1], c.bs[2], c.lds,
                                  d.gc[0], d.gc[1], d.gc[2], d.bs[0], d.bs[1], d.bs[2], d.lds);
                return 1;
            }
            if (c.arg0.size() != d.arg0.size()) {
                if (why && cap > 0) std::snprintf(why, (size_t)cap, "segment %zu, command %zu (%s): %zu argument bytes against %zu", i, j, nm, c.arg0.size(), d.arg0.size());
                return 1;
            }
            for (size_t o = 0; o < c.arg0.size(); ++o)
                if (c.arg0[o] != d.arg0[o]) {
                    const size_t w = o & ~(size_t)7;
                    uint64_t u = 0, v = 0;
                    std::memcpy(&u, c.arg0.data() + w, std::min<size_t>(8, c.arg0.size() - w));
                    std::memcpy(&v, d.arg0.data() + w, std::min<size_t>(8, d.arg0.size() - w));
                    if (why && cap > 0)
                        std::snprintf(why, (size_t)cap, "segment %zu, command %zu (%s): the by-value argument differs at byte %zu of %zu "
                                      "(the 8 bytes at %zu: 0x%016llx recorded, 0x%016llx now)", i, j, nm, o, c.arg0.size(), w,
                                      (unsigned long long)u, (unsigned long long)v);
                    return 1;
                }
        } else if (c.dst != d.dst || c.src != d.src || c.bytes != d.bytes || c.dp != d.dp || c.sp != d.sp || c.w != d.w || c.h != d.h) {
            if (why && cap > 0)
                std::snprintf(why, (size_t)cap, "segment %zu, command %zu (%s): %p <- %p %zu B (%zu/%zu pitch, %zux%zu) against %p <- %p %zu B (%zu/%zu, %zux%zu)",
                              i, j, nm, c.dst, c.src, c.bytes, c.dp, c.sp, c.w, c.h, d.dst, d.src, d.bytes, d.dp, d.sp, d.w, d.h);
            return 1;
        }
    }
    return 0;
}
int z_tape_diff(RadTape pa, RadTape pb, char* why, int cap) {
    auto* a = static_cast<Tape*>(pa);
    auto* b = static_cast<Tape*>(pb);
    if (!a || !b) return RAD_E_INVAL;
    const size_t n = std::min(a->segs.size(), b->segs.size());
    if (a->native && b->native)
        for (size_t i = 0; i < n; ++i)
            if (n_tape_diff(*a, *b, i, why, cap)) return (int)i;
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
void z_tape_destroy(RadTape p) {
    auto* t = static_cast<Tape*>(p);
    if (!t) return;
    bool held;
    {
        std::lock_guard<std::mutex> lk(held_mutex(t->dev));
        held = g_held[t->dev].t == t;
    }
    if (held) {   /* its device's held play, whichever thread frees it */
        const int saved = t_device;
        t_device = t->dev;
        (void)flush_held();
        t_device = saved;
    }
    release_tape(t);   /* freed once its last play has completed */
}
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
        /* RADIANCE_L0_QUEUE_PROFILING=1 (Intel port, milestone 3, run 57): the card's queues with profiling
         * too, for libxe's launch profiler (RADIANCE_XE_LAUNCH_PROFILE: the live launches' device times, which
         * the tape profiler does not see -- a prefill). Off by default, as the tagged devices were. */
        static const bool prof_all = [] { const char* e = std::getenv("RADIANCE_L0_QUEUE_PROFILING"); return e && *e == '1'; }();
        sycl::property_list pl =
            (D.tag && !prof_all)
                ? (high_priority ? sycl::property_list{in_order(), priority_high()} : sycl::property_list{in_order()})
                : (high_priority ? sycl::property_list{in_order(), priority_high(), enable_profiling()}
                                 : sycl::property_list{in_order(), enable_profiling()});
        auto* st = new L0Stream(sycl::queue(D.ctx, D.d, async_errors, pl));
        st->id = ++g_stream_ids;
        st->dev = dev;
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
    (void)flush_held();
    settle(S(s));                    /* a native play the stream's work follows: waited for here */
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

/* The sleeping wait (owner note 25, run 11). queue::wait and event::wait busy-poll in the driver: with
 * two rank threads each waiting out a 27 ms step, the container sat at 100% of its CPU cap and intel-host's
 * CPU at 70-77 C while the step was all card time. Here the host polls the event's status -- a short
 * spin for the sub-100 us completions, then sleeps that grow to kWaitSleepMaxUs -- and only then calls
 * the runtime's wait, which returns at once and surfaces the asynchronous errors. RADIANCE_L0_WAIT=spin
 * restores the plain runtime wait (the A/B). */
std::atomic<int64_t> g_wait_polls{0}, g_wait_sleeps{0};
bool sleepy_wait_on() {
    static const bool on = [] { const char* v = std::getenv("RADIANCE_L0_WAIT"); return !(v && std::strcmp(v, "spin") == 0); }();
    return on;
}
bool event_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}
void sleepy_wait(const sycl::event& e) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int64_t sleep_us = kWaitSleepMinUs;
    bool told = false;
    for (;;) {
        if (event_done(e)) return;
        g_wait_polls.fetch_add(1, std::memory_order_relaxed);
        const int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
        if (us < kWaitSpinUs) { std::this_thread::yield(); continue; }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        g_wait_sleeps.fetch_add(1, std::memory_order_relaxed);
        sleep_us = std::min<int64_t>(sleep_us * 2, kWaitSleepMaxUs);
        if (!told && us > 10000000) {   /* M1: which host wait holds a rank (a stream or event sync) */
            told = true;
            RAD_WARN("l0: a thread on device %d has waited 10 s in a sleeping wait for the runtime's event (a stream or event sync; plays so far %lld)",
                     t_device, (long long)g_nplays.load());
        }
    }
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
     * passes on card1 (run b2). The sleeping wait (above) polls the queue's last event first, on a
     * GPU device and a stream that has never been gated; the runtime's wait still follows. */
    RAD_TRY(enter(S(s)));
    if (recording()) return seg_cut("rad_stream_sync", [&] { Q(s).wait_and_throw(); });
    const int rc = guarded("rad_stream_sync", [&] {
        if (sleepy_wait_on() && !devs()[S(s).dev].cpu && !S(s).gated.load()) {
            std::optional<sycl::event> last = Q(s).ext_oneapi_get_last_event();
            if (last) sleepy_wait(*last);
        }
        Q(s).wait_and_throw();
    });
    if (rc == RAD_OK) S(s).tags.store(0);   /* the queue is drained: its timestamp events are released */
    reap_tapes();
    return rc;
}

/* ------------------------------------------------------------------ gates */
int z_stream_gate(RadStream s, void** out) {
    RAD_TRY(check_stream(s, "rad_stream_gate"));
    if (!out) return RAD_E_INVAL;
    auto g = std::make_shared<Gate>();
    auto* h = new (std::nothrow) GateHandle{g, &S(s)};
    if (!h) return RAD_E_NOMEM;
    RAD_TRY(enter(S(s)));
    S(s).gated.store(true);   /* before the host task goes on the queue: real_record reads it */
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
    RAD_TRY(enter(S(s)));
    if (recording()) {
        if (!recorded_stream(S(s))) { refuse("an event record on a stream outside the recording"); }
        else if (t_rec.native) {
            /* one in-order list: no device command; recorded for real after the segment runs */
            std::lock_guard<std::mutex> lk(e->m);
            e->gtape = t_rec.t->id;
            e->gseg = (int)t_rec.t->segs.size();
            t_rec.pend.push_back(e);
            return RAD_OK;
        }
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
    return guarded("rad_event_record", [&] { real_record(e, S(s)); });
}

int z_event_wait(RadStream s, RadEvent ev) {
    if (!ev || !s) {
        device_set_error("rad_event_wait: null %s", ev ? "stream" : "event");
        return RAD_E_INVAL;
    }
    auto* e = static_cast<L0Event*>(ev);
    RAD_TRY(enter(S(s)));
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
        if (t_rec.native && (here || earlier)) t_rec.t->rec_waits.emplace_back(e, (int)t_rec.t->segs.size());   /* checked at tape_end (owner note 42) */
        if (here && t_rec.native) return RAD_OK;   /* one in-order list */
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
    int zdev = -1, zslot = -1;
    uint64_t zserial = 0;
    {
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_OK;   /* a never-recorded event is no wait, as hipStreamWaitEvent */
        b = e->ev;
        zdev = e->zdev; zslot = e->zslot; zserial = e->zserial;
    }
    if (zdev >= 0) {
        /* recorded after a native play: a device-side wait on this device's own list (T4: a host wait
         * here deadlocked the two ranks at 1 CPU -- a stream wait is asynchronous by contract), then a
         * fresh slot's signal that the stream's next submission settles on and the next play follows */
        L0Stream& st = S(s);
        ZeDev& Z = zedev(st.dev);
        if (!Z.ok || !Z.imm) { n_slot_wait(zdev, zslot, zserial, "rad_event_wait without a native list"); return RAD_OK; }
        int slot;
        uint64_t serial;
        {
            std::lock_guard<std::mutex> lk(Z.m);
            ZeDev::Ring& r = ring_take(Z);
            slot = (int)(&r - Z.ring.get());
            serial = r.serial.load(std::memory_order_relaxed);
        }
        ze_event_handle_t ev = nullptr;
        {   /* the source slot, under its own lock alone: still pending -> this slot is its consumer */
            ZeDev& Y = zedev(zdev);
            std::lock_guard<std::mutex> lk(Y.m);
            ev = n_slot_event(zdev, zslot, zserial);
            if (ev) Y.ring[(size_t)zslot].consumers.push_back(ZeDev::Consumer{st.dev, slot, serial});
        }
        if (!ev) return RAD_OK;   /* the play completed meanwhile: no wait (the taken slot stays unused) */
        std::lock_guard<std::mutex> lk(Z.m);
        ZeDev::Ring& r = Z.ring[(size_t)slot];
        ze_result_t rc = zeCommandListAppendWaitOnEvents(Z.imm, 1, &ev);
        if (rc == ZE_RESULT_SUCCESS) rc = zeCommandListAppendSignalEvent(Z.imm, r.ev);
        if (rc != ZE_RESULT_SUCCESS) { device_set_error("rad_event_wait: the native wait failed (0x%x)", (unsigned)rc); return RAD_E_DEVICE; }
        r.used = true;
        r.kind = 1; r.tape = 0; r.from = r.to = -1; r.nlists = 0; r.nwaits = 1;
        r.dep_dev = zdev; r.dep_slot = zslot; r.dep_serial = zserial;
        Z.prev_slot = slot;
        Z.prev_serial = serial;
        st.npend_dev = st.dev;
        st.npend_slot = slot;
        st.npend_serial = serial;
        return RAD_OK;
    }
    return guarded("rad_event_wait", [&] { Q(s).ext_oneapi_submit_barrier({b}); });
}

int z_event_query(RadEvent ev) {
    if (!ev) return RAD_E_INVAL;
    RAD_TRY(flush_held());
    auto* e = static_cast<L0Event*>(ev);
    if (pending_here(e)) RAD_TRY(seg_cut("rad_event_query", [] {}));
    std::lock_guard<std::mutex> lk(e->m);
    if (!e->recorded) return 1;
    if (e->zdev >= 0) return n_slot_done(e->zdev, e->zslot, e->zserial) ? 1 : 0;
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
    RAD_TRY(flush_held());
    auto* e = static_cast<L0Event*>(ev);
    if (!(e->flags & RAD_EVENT_HOST_WAIT)) {
        device_set_error("rad_event_sync: the event was made without RAD_EVENT_HOST_WAIT");
        return RAD_E_INVAL;
    }
    if (pending_here(e)) RAD_TRY(seg_cut("rad_event_sync", [] {}));
    sycl::event b;
    int zdev = -1, zslot = -1;
    uint64_t zserial = 0;
    {
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_OK;
        b = e->ev;
        zdev = e->zdev; zslot = e->zslot; zserial = e->zserial;
    }
    if (zdev >= 0) { n_slot_wait(zdev, zslot, zserial, "rad_event_sync"); return RAD_OK; }
    return guarded("rad_event_sync", [&] {
        if (sleepy_wait_on() && !devs()[t_device].cpu) sleepy_wait(b);
        b.wait_and_throw();
    });
}

/* From the records' end timestamps (profiling tags, or the CPU device's profiling queues). */
int z_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    if (!a || !b || !out) return RAD_E_INVAL;
    auto stamp = [](RadEvent ev, uint64_t* t) {
        auto* e = static_cast<L0Event*>(ev);
        std::lock_guard<std::mutex> lk(e->m);
        if (!e->recorded) return RAD_E_STATE;
        if (!e->stamped) {
            /* Records carry timestamps from the first time one is asked for (real_record). */
            if (devs()[0].tag && !g_timing.exchange(true))
                RAD_INFO("l0 events: timing asked for -- records take profiling tags from now on, at most "
                         "%lld a stream between its host syncs", (long long)kTagsPerSync);
            static std::atomic<bool> said{false};
            if (g_rec_untimed.load() > 0 && !said.exchange(true))
                RAD_WARN("l0 events: a stream passed %lld timed records without a host sync; later records "
                         "on it are untimed (bounded, owner note 7)", (long long)kTagsPerSync);
            return RAD_E_STATE;
        }
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
void z_host_wrote(void) { (void)flush_held(); }   /* a held play was issued before the write */

/* ------------------------------------------------------------------ copies and fills */
int z_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_async"));
    RAD_TRY(enter(S(s)));
    if (bytes < 0 || (bytes > 0 && (!dst || !src))) {
        device_set_error("rad_memcpy_async: bad arguments (%p, %p, %lld)", dst, src, (long long)bytes);
        return RAD_E_INVAL;
    }
    if (bytes == 0) return RAD_OK;
    return copy_on(s, dst, src, (size_t)bytes);
}

int z_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch, int64_t width,
                      int64_t height, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_2d_async"));
    RAD_TRY(enter(S(s)));
    if (width < 0 || height < 0 || dpitch < width || spitch < width || (width && height && (!dst || !src))) {
        device_set_error("rad_memcpy_2d_async: bad arguments");
        return RAD_E_INVAL;
    }
    if (!width || !height) return RAD_OK;
    if (native_on(S(s)) && n_hook_copy2d(s, dst, (size_t)dpitch, src, (size_t)spitch, (size_t)width, (size_t)height) == 0)
        return RAD_OK;
    return guarded("rad_memcpy_2d_async", [&] {
        /* sycl_ext_oneapi_graph does not record ext_oneapi_memcpy2d (it throws): on a tape the rows
         * are plain copies, one a row (one copy when both sides are dense). */
        if (Q(s).ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) {
            if (dpitch == width && spitch == width) {
                Q(s).memcpy(dst, src, (size_t)(width * height));
                return;
            }
            for (int64_t r = 0; r < height; ++r)
                Q(s).memcpy((uint8_t*)dst + r * dpitch, (const uint8_t*)src + r * spitch, (size_t)width);
            return;
        }
        Q(s).ext_oneapi_memcpy2d(dst, (size_t)dpitch, src, (size_t)spitch, (size_t)width, (size_t)height);
    });
}

int z_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memset_async"));
    RAD_TRY(enter(S(s)));
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
     * card1. Device memory keeps the runtime's fill; a native recording takes the kernel for both. */
    if (native_on(S(s)) || sycl::get_pointer_type(dst, Q(s).get_context()) != sycl::usm::alloc::device)
        return rad_l0k_fill(s, dst, value & 0xff, bytes) == 0 ? RAD_OK : RAD_E_DEVICE;
    return guarded("rad_memset_async", [&] { Q(s).memset(dst, value & 0xff, (size_t)bytes); });
}

/* The ranges are read here, at the call: the contract keeps the list unchanged until the stream
 * passes the copy, so reading it early is equivalent. One copy a range. */
int z_memcpy_ranges_async(void* dst, const void* src, int64_t prefix, const RadCopyRange* ranges, int n,
                          RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_ranges_async"));
    RAD_TRY(enter(S(s)));
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
    if (prefix) RAD_TRY(copy_on(s, dst, src, (size_t)prefix));
    for (int i = 0; i < n; ++i)
        if (ranges[i].bytes > 0)
            RAD_TRY(copy_on(s, (char*)dst + ranges[i].dst, (const char*)src + ranges[i].src, (size_t)ranges[i].bytes));
    return RAD_OK;
}

int z_memcpy_list_async(const RadCopySeg* segs, int n, unsigned, RadStream s) {
    RAD_TRY(check_stream(s, "rad_memcpy_list_async"));
    RAD_TRY(enter(S(s)));
    if (n < 0 || (n > 0 && !segs)) { device_set_error("rad_memcpy_list_async: bad arguments"); return RAD_E_INVAL; }
    for (int i = 0; i < n; ++i)
        if (segs[i].bytes > 0) RAD_TRY(copy_on(s, segs[i].dst, segs[i].src, (size_t)segs[i].bytes));
    return RAD_OK;
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
    b.host_prebind = z_host_prebind;
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
    rad_l0k_set_flush(flush_held_hook);
    rad_l0k_set_rec(&g_rec_hooks);
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
    /* Milestone 2 (two cards): every GPU of the platform shares ONE context. Level Zero binds a
     * USM allocation to its context, so a peer store into another card's memory, the pinned host
     * pool read zero-copy by both cards and the collectives' mailboxes all need the ranks' devices
     * in one context; per-device contexts (milestone 1) would make every cross-card pointer
     * invalid. Peer access itself is granted per ordered pair (z_enable_peer). */
    if (!cpu && devs().size() > 1) {
        try {
            std::vector<sycl::device> all;
            for (const Dev& D : devs()) all.push_back(D.d);
            sycl::context shared(all);
            for (Dev& D : devs()) D.ctx = shared;
        } catch (const sycl::exception& e) {
            RAD_WARN("l0: no shared context over the %zu cards (%s): per-card contexts, no peer memory",
                     devs().size(), e.what());
        }
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
        if (ok) {
            g = make_backend();
            /* kernel libraries that allocate at init (libxe's all-reduce rows) find rank r's
             * device and the shared context through rad_l0k (l0_kernels.sycl) */
            rad_l0k_set_devices((int)devs().size(),
                                [](int i) -> const sycl::device* { return &devs()[i].d; },
                                [](int i) -> const sycl::context* { return &devs()[i].ctx; });
        }
    });
    return ok ? &g : nullptr;
}

}  /* namespace rad */

/* For tests and the report (owner note 7): out[0] RadEvents alive, out[1] records that reused the
 * queue's last event, out[2] barrier records, out[3] profiling-tag records, out[4] untimed records
 * past the bound, out[5] the most tags any stream holds since its last host sync. */
extern "C" void rad_l0_event_stats(int64_t out[6]) {
    using namespace rad;
    {
        std::lock_guard<std::mutex> lk(event_mutex());
        out[0] = (int64_t)events_alive().size();
    }
    out[1] = g_rec_reused.load();
    out[2] = g_rec_barriers.load();
    out[3] = g_rec_tags.load();
    out[4] = g_rec_untimed.load();
    int64_t mx = 0;
    {
        std::lock_guard<std::mutex> lk(stream_mutex());
        for (L0Stream* s : streams()) mx = std::max(mx, s->tags.load());
    }
    out[5] = mx;
}

/* For tests and the report (owner note 28): out[0] native command lists created, out[1] destroyed,
 * out[2] plays (append calls), out[3] lists appended over every play, out[4] kernel objects created,
 * out[5] 1 when device 0 records natively (2 with in-order lists), 0 otherwise; out[6], out[7] N4's
 * early-completion counters. */
extern "C" void rad_l0_ntape_stats(int64_t out[8]) {
    using namespace rad;
    out[0] = g_nlists_created.load();
    out[1] = g_nlists_destroyed.load();
    out[2] = g_nplays.load();
    out[3] = g_nlists_played.load();
    out[4] = g_nkernels.load();
    const ZeDev& Z = zedev(0);
    out[5] = Z.ok ? (Z.inorder ? 2 : 1) : 0;
    out[6] = g_sig_early.load();   /* plays whose ring event was already signalled right after the append */
    out[7] = g_same_tape_waits.load();   /* plays that waited for the same tape's previous play (its lists still in flight) */
}

/* For tests (owner note 10): out[0] tapes deferred at release because a play was still running,
 * out[1] graveyard tapes waited for (the bound), out[2] tapes freed, out[3] in the graveyard now. */
extern "C" void rad_l0_tape_stats(int64_t out[4]) {
    using namespace rad;
    out[0] = g_tapes_deferred.load();
    out[1] = g_tapes_waited.load();
    out[2] = g_tapes_freed.load();
    std::lock_guard<std::mutex> lk(dead_mutex());
    out[3] = (int64_t)dead_tapes().size();
}

#endif /* RAD_HAVE_L0 */
