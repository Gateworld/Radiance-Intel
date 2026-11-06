// SPDX-License-Identifier: Apache-2.0
// Part of the Intel port of radiance (milestone 3, run 175, owner note 176 item 2).
/* engine_warmup.cpp -- the startup warm-up pass: one synthetic request through every kernel the service
 * uses before the server listens -- a full prefill chunk, a short tail and a few decode steps with the
 * drafts on -- so the first user prompt does not pay the first launches, the tapes' first recordings and
 * the backend's first-use binds (the "cold" first token measured 7.0-7.9 s against 5.8 s warm, run 174).
 * Nothing of it is kept: the request is reaped and its tokens dropped. It cannot change an output: the
 * pass runs the same kernels on other data, and what it leaves behind (KV blocks in the prefix cache,
 * the heat engine's residency, recorded tapes) carries no arithmetic. --startup-warmup off skips it;
 * the time it took is logged as the startup cost. The driver is the KL mode's (engine_kld.cpp): a
 * Request submitted through the scheduler, its sink waited on. */
#include "engine_priv.h"
#include "server/sink.h"
#include "text/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rad {

int Engine::warmup() {
    const auto t0 = std::chrono::steady_clock::now();
    /* A full chunk (--max-num-batched-tokens rows) and a 36-row tail: the shapes of the 7K prompt's
     * last two steps (5,604 = 4,096 + 1,472 + 36; the tail is the remainder below the GDN chunk). */
    const int64_t chunk = cfg_.max_tok;
    const int64_t tail  = 36;
    const int64_t want  = chunk + tail;
    const int32_t gen   = 8;    /* decode steps, with the drafts the configuration gives every request */
    /* Prose repeated to the length, tokenised as the server tokenises a rendered template: the routing
     * of ordinary text, so what the heat engine learns from it is what users send; no control tokens. */
    static const char* kText =
        "The lighthouse keeper logged the weather at dawn: a steady wind from the south-west, a long swell "
        "from the night's storm, and the first fishing boats leaving the harbour behind the breakwater. ";
    Tokenizer tok;
    RAD_TRY(tok.init(vocab_));
    std::vector<int32_t> unit;
    RAD_TRY(tok.encode(kText, unit, false, false));
    if (unit.empty()) return RAD_E_STATE;
    std::vector<int32_t> ids;
    ids.reserve((size_t)want);
    for (size_t i = 0; (int64_t)ids.size() < want; ++i) ids.push_back(unit[i % unit.size()]);

    auto sink = std::make_unique<server::Sink>(64);
    std::unique_ptr<Request> reqs[1];
    reqs[0] = std::make_unique<Request>();
    Request& r = *reqs[0];
    r.id         = UINT64_MAX - 1;   /* reaped before the server hands out its first id */
    r.prompt     = std::move(ids);
    r.max_tokens = gen;
    r.ignore_eos = true;
    r.sp.temp    = 0.0f;
    r.sp.use_control_vector = cvec_default_on();
    r.sink       = sink.get();
    sink->id     = r.id;
    int rc[1] = { RAD_OK };
    sched_.add_all(reqs, 1, rc);
    if (rc[0] < 0) {
        RAD_WARN("startup warm-up: the scheduler refused the %lld-token request: %s", (long long)want,
                 rad_strerror(rc[0]));
        return rc[0];
    }
    while (!sink->done()) {
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        sink->wait(200);
    }
    const server::Finish f = sink->finish_reason();
    const int st = sink->status();
    sched_.reap(sink->id);
    /* Run 204 (owner note 197 item 1): the request leaves its K / V in the blocks it freed, and a new sequence's
     * blocks are not cleared when handed out; the MTP head's first draft reads 1 + n_spec slots past its position
     * before the verify writes them, so a user given those blocks drafted against the warm-up's text (C14: 96 / 480
     * drafts kept, 45.6 tok/s against 70.1; with six seats drafting, two card faults). The step loop zeroes the
     * cache's unlent blocks while idle; listening waits for it. Even a refused warm-up may have touched blocks. */
    kv_reset_pending_.store(true, std::memory_order_release);
    while (kv_reset_pending_.load(std::memory_order_acquire)) {
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (f == server::Finish::Error || f == server::Finish::Cancelled) {
        RAD_WARN("startup warm-up: the request did not finish: %s", rad_strerror(st));
        return st < 0 ? st : RAD_E_STATE;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    RAD_INFO("startup warm-up: a %lld-token prompt (a %lld-row chunk and a %lld-row tail) and %d decode "
             "tokens in %.2f s before listening (--startup-warmup off skips it)",
             (long long)want, (long long)chunk, (long long)tail, gen, s);
    return RAD_OK;
}

/* On the step loop's thread, idle (Engine::kv_reset_pending_): every rank's paged groups, each layer's stripe from
 * block 0 up to the live mark, set to zero on the compute stream and waited for. The blocks above the mark are lent
 * to the expert slab and hold expert units, so they are left alone; the loan moves only on this thread
 * (loan_tick), after this returns, so a fill can only write over zeros. A group with blocks still held (indexed by
 * the prefix cache) is skipped and said: its held blocks cannot be told from the free ones here. Recurrent state
 * needs nothing: a slot is zeroed when it is handed out (state_clears). */
void Engine::kv_reset_after_warmup() {
    if (ranks_.empty()) return;
    /* The warm-up's prompt may sit in the prefix cache, whose blocks are held: drop it first (no user prompt matches
     * the synthetic text), so its blocks are free and cleared with the rest. */
    const int64_t dropped = sched_.reset_prefix_cache();
    if (dropped > 0) RAD_INFO("startup warm-up: %lld prefix-cache entries dropped", (long long)dropped);
    const KVManager& lead = ranks_[0]->kv;
    int64_t bytes = 0, groups = 0;
    for (int32_t g = 0; g < lead.n_groups(); ++g) {
        const KVGroupPlan* lp = lead.plan(g);
        if (!lp || !lp->paged()) continue;
        if (lead.held_blocks(g) > 0) {
            RAD_WARN("startup warm-up: KV group %d still holds %lld blocks; its free blocks are not cleared", (int)g,
                     (long long)lead.held_blocks(g));
            continue;
        }
        const int64_t live = lead.live_blocks(g);
        for (auto& rk : ranks_) {
            const KVGroupPlan* gp = rk->kv.plan(g);
            if (!gp || gp->zero || !gp->base || live <= 0) continue;
            const int64_t n = live * gp->layer_bytes_per_block();
            for (int64_t l = 0; l < gp->n_layers; ++l) {
                char* p = (char*)gp->base + l * gp->layer_stride();
                if (rad_memset_async(p, 0, n, rk->ctx.stream()) < 0) {
                    RAD_WARN("startup warm-up: clearing KV group %d failed", (int)g);
                    break;
                }
                bytes += n;
            }
        }
        ++groups;
    }
    for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
    RAD_INFO("startup warm-up: %lld paged KV groups' unlent blocks cleared (%lld MiB over the ranks), so no request "
             "inherits the warm-up's K / V", (long long)groups, (long long)(bytes >> 20));
}

}  // namespace rad
