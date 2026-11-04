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

}  // namespace rad
