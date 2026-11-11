/* engine_kld.cpp -- the KL mode on the Engine: the driver that replaces the HTTP server, and the
 * copy of a step's scored rows off the cards. The arithmetic and files are core/kld.h's. */
#include "engine_priv.h"
#include "kld.h"

#include "device/device.h"
#include "server/sink.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {

/* The Intel port (run 179, owner note 181 phase 1): the MTP draft head's training pairs. With RADIANCE_MTP_DUMP=<dir>
 * and an architecture that keeps its head's input rows in a buffer named `*dump_hwide` (qwen4exp_fp8:
 * RADIANCE_DUMP_HWIDE=1), every scored position of the KL mode's teacher-forced corpus is written as
 *   meta.bin    int32 doc (the request id), int32 position, int32 token at that position
 *   hidden.bin  the head's input row (bf16, the buffer's width: 4 x 2560 hc streams for Flash-Next)
 *   top32.bin   the served model's logits row at that position as its 32 largest: int32 id, float log-probability
 *               (log-softmax over the whole vocabulary, the ranks' shards merged), largest first
 * Head index i reads hidden(i) and the token at i + 1 and predicts the token at i + 2, whose teacher is the logits
 * row at position i + 1 (rad_block_mtp_hc.h, THE INDEXING). One record a position, in the corpus's step order. */
namespace {
struct MtpDump {
    bool    on = false;
    rad_buf buf = RAD_NULL_HANDLE;
    int64_t width = 0;
    void*   host = nullptr;
    FILE*   fm = nullptr;
    FILE*   fh = nullptr;
    FILE*   ft = nullptr;
    int64_t rows = 0;
};
MtpDump g_dump;
constexpr int kDumpTop = 32;

/* RADIANCE_MTP_PARITY=1 beside RADIANCE_MTP_DUMP (run 180, owner note 181's parity check): the head's own
 * round-1 output for every drafting row of the history pass -- in the KL mode, each document's last prompt
 * row, whose hidden row the dump holds. The head's logits rows are the shared logits buffer's first rows
 * (rad_block_mtp_hc.h op_logits), so it needs the full-logits head (RADIANCE_MTP_DRAFT_HEAD_Q2=0: the trunk's
 * lm_head, not the 2-bit copy). parity.bin, one record a drafting row:
 *   int32 request id, int32 head index i, int32 the token fed (the trunk's draw at i, i.e. the token at i + 1),
 *   then the head's 32 largest: int32 id x 32, float log-probability x 32 (the ranks' shards merged) */
/* Run 283 (owner note 224 item 4, briefs/2026-11-11-draft-head-v2-data.md: parity2.bin): RADIANCE_MTP_PARITY_ROUNDS=3
 * records the head's rounds 1-3 -- the history pass (round 1), then the round-2 and round-3 draft passes, each fed
 * the previous round's GREEDY draft (the serial drafter's own passes; coupling must be off) -- one record a drafting
 * row: int32 request id, int32 head index i, int32 rounds, int32 x 3 the tokens fed (t[i+1] = the trunk's draw, then
 * the round-1 and round-2 drafts), then per round 32 ids + 32 log-probabilities (792 B). ROUNDS=1 (the default)
 * keeps the 11-04 record (268 B) byte for byte. */
struct MtpParity {
    bool                  on = false;
    int                   rounds = 1;  /* rounds recorded a drafting row (1 or 3) */
    std::vector<float*>   host;       /* [round][rank]: max_seqs rows of the rank's logits columns */
    std::vector<uint64_t> seq;        /* the drafting rows' sequences, in draft order */
    int                   n = 0;      /* rows copied by the last history pass, 0 once written */
    int                   issued = 0; /* rounds copied this step */
    FILE*                 f = nullptr;
    int64_t               rows = 0;
    /* The step thread writes after a step's commit, and run_kld closes once every document's sink is done -- which
     * commit itself can make true -- so the close and a late write are serialised here (run 187: MD5c's last step's
     * rows were lost to that order, and a write could have met a closed file). */
    std::mutex            mu;
    float* at(int round, size_t rank) const { return host[(size_t)(round - 1) * nranks + rank]; }
    size_t                nranks = 0;
};
MtpParity g_par;

/* The 32 largest of a logits row over the merged shards and their log-softmax, largest first: an unsorted
 * 32-slot table with its minimum tracked, then sorted; terms more than 30 below the row's max are below
 * float's resolution of the sum. */
void row_top32(const std::vector<KldShard>& shards, int64_t row, int32_t* ids, float* lps) {
    float mx = -INFINITY;
    float tv[kDumpTop];
    int32_t ti[kDumpTop];
    int nt = 0, mi = 0;
    for (const KldShard& s : shards) {
        const float* lr = s.rows + row * s.width;
        for (int64_t c = 0; c < s.width; ++c) {
            const float v = lr[c];
            if (v > mx) mx = v;
            if (nt < kDumpTop) {
                tv[nt] = v; ti[nt] = (int32_t)(s.vocab0 + c); ++nt;
                if (nt == kDumpTop) { mi = 0; for (int k = 1; k < kDumpTop; ++k) if (tv[k] < tv[mi]) mi = k; }
            } else if (v > tv[mi]) {
                tv[mi] = v; ti[mi] = (int32_t)(s.vocab0 + c);
                mi = 0; for (int k = 1; k < kDumpTop; ++k) if (tv[k] < tv[mi]) mi = k;
            }
        }
    }
    float se = 0.0f;
    const float lo = mx - 30.0f;
    for (const KldShard& s : shards) {
        const float* lr = s.rows + row * s.width;
        for (int64_t c = 0; c < s.width; ++c) if (lr[c] > lo) se += std::exp(lr[c] - mx);
    }
    const float lse = mx + std::log(se);
    int ord[kDumpTop];
    for (int k = 0; k < kDumpTop; ++k) ord[k] = k;
    std::sort(ord, ord + nt, [&](int x, int y) { return tv[x] > tv[y]; });
    for (int k = 0; k < kDumpTop; ++k) {
        ids[k] = k < nt ? ti[ord[k]] : -1;
        lps[k] = k < nt ? tv[ord[k]] - lse : -INFINITY;
    }
}
}  // namespace

int Engine::run_kld() {
    /* Each rank's logits buffer must hold a step's tokens after the sampler's rows
     * (RadBuildCtx::max_out_rows); a plugin that ignored the field is refused here rather than
     * overrun. */
    const int64_t need = cfg_.max_tok + cfg_.max_seqs;
    const int64_t nv = meta_.n_vocab;
    kld_vocab0_.clear();
    kld_width_.clear();
    for (const auto& rk : ranks_) {
        const Program& p = rk->program;
        if (p.logits_buf == RAD_NULL_HANDLE || (size_t)p.logits_buf >= p.buffers.size()) {
            RAD_ERR("kld: rank %d declared no logits buffer", rk->index);
            return RAD_E_UNSUPPORTED;
        }
        const RadBufDecl& d = p.buffers[p.logits_buf].decl;
        if (d.dtype != RAD_F32 || d.rank != 2 || d.shape[0] < need) {
            RAD_ERR("kld: the '%s' plugin's logits buffer is %lld rows of %s; the KL mode needs "
                    "f32 rows for a whole step, %lld (it reads RadBuildCtx::max_out_rows)",
                    plugins_.arch_plugin.c_str(), (long long)d.shape[0],
                    d.dtype == RAD_F32 ? "f32" : "another dtype", (long long)need);
            return RAD_E_UNSUPPORTED;
        }
        /* The columns a rank owns: the sampler's split, which declare_sampler checked, or the
         * whole vocabulary on every rank (then read from rank 0 only). */
        const int64_t w = d.shape[1];
        if (w >= nv) {
            if (!kld_width_.empty()) continue;
            kld_vocab0_.push_back(0);
            kld_width_.push_back(nv);
            continue;
        }
        kld_vocab0_.push_back(rk->sampler.config().vocab_off);
        kld_width_.push_back(w);
    }
    int64_t cover = 0;
    for (int64_t w : kld_width_) cover += w;
    if (cover != nv) {
        RAD_ERR("kld: the ranks' logits columns add up to %lld of a %lld-token vocabulary",
                (long long)cover, (long long)nv);
        return RAD_E_STATE;
    }
    kld_host_.assign(kld_width_.size(), nullptr);
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        kld_host_[r] = static_cast<float*>(
            rad_dev_alloc(cfg_.max_tok * kld_width_[r] * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        if (!kld_host_[r]) {
            RAD_ERR("kld: no pinned host memory for a step's logits (%lld MiB a rank); lower "
                    "--max-num-batched-tokens", (long long)(cfg_.max_tok * kld_width_[r] * 4 >> 20));
            return RAD_E_NOMEM;
        }
    }

    /* RADIANCE_MTP_DUMP (see MtpDump above) */
    if (const char* dd = std::getenv("RADIANCE_MTP_DUMP"); dd && *dd) {
        const Program& p0 = ranks_[0]->program;
        for (size_t i = 0; i < p0.buffers.size(); ++i) {
            const std::string& bn = p0.buffers[i].name;
            if (bn.size() >= 10 && bn.compare(bn.size() - 10, 10, "dump_hwide") == 0) {
                g_dump.buf = (rad_buf)i;
                break;
            }
        }
        if (g_dump.buf == RAD_NULL_HANDLE) {
            RAD_ERR("kld: RADIANCE_MTP_DUMP is set and the program has no 'dump_hwide' buffer (the architecture "
                    "keeps one under RADIANCE_DUMP_HWIDE=1)");
            return RAD_E_UNSUPPORTED;
        }
        const RadBufDecl& d = p0.buffers[(size_t)g_dump.buf].decl;
        if (d.dtype != RAD_BF16 || d.rank != 2 || d.shape[0] < cfg_.max_tok) {
            RAD_ERR("kld: the 'dump_hwide' buffer is not [max_tok, n] bf16");
            return RAD_E_UNSUPPORTED;
        }
        g_dump.width = d.shape[1];
        g_dump.host = rad_dev_alloc(cfg_.max_tok * g_dump.width * 2, RAD_MEM_HOST_PINNED);
        const std::string dir = dd;
        g_dump.fm = std::fopen((dir + "/meta.bin").c_str(), "wb");
        g_dump.fh = std::fopen((dir + "/hidden.bin").c_str(), "wb");
        g_dump.ft = std::fopen((dir + "/top32.bin").c_str(), "wb");
        if (!g_dump.host || !g_dump.fm || !g_dump.fh || !g_dump.ft) {
            RAD_ERR("kld: RADIANCE_MTP_DUMP: cannot open %s's files or the pinned rows", dir.c_str());
            return RAD_E_IO;
        }
        g_dump.on = true;
        RAD_INFO("kld: the MTP head's pairs to %s (hidden rows of %lld bf16, the logits' top-%d)", dir.c_str(),
                 (long long)g_dump.width, kDumpTop);

        /* RADIANCE_MTP_PARITY (see MtpParity above) */
        if (const char* pv = std::getenv("RADIANCE_MTP_PARITY"); pv && pv[0] == '1') {
            const char* q2 = std::getenv("RADIANCE_MTP_DRAFT_HEAD_Q2");
            if (!q2 || q2[0] != '0') {
                RAD_ERR("kld: RADIANCE_MTP_PARITY reads the head's full logits rows, which only the trunk's lm_head "
                        "writes: set RADIANCE_MTP_DRAFT_HEAD_Q2=0 as well");
                return RAD_E_INVAL;
            }
            if (!serial_drafting()) {
                RAD_ERR("kld: RADIANCE_MTP_PARITY needs the serial MTP drafter (--num-speculative-tokens > 0)");
                return RAD_E_UNSUPPORTED;
            }
            /* run 283: RADIANCE_MTP_PARITY_ROUNDS (1 or 3; 3 needs a draft depth of 3 and coupling off, so that
             * rounds 2-3 are the head's own greedy chain) */
            g_par.rounds = 1;
            if (const char* pr = std::getenv("RADIANCE_MTP_PARITY_ROUNDS"); pr && *pr) {
                const int want = std::atoi(pr);
                if (want != 1 && want != 3) {
                    RAD_ERR("kld: RADIANCE_MTP_PARITY_ROUNDS=%s: 1 or 3", pr);
                    return RAD_E_INVAL;
                }
                if (want == 3 && (int)drafter_.depth < 3) {
                    RAD_ERR("kld: RADIANCE_MTP_PARITY_ROUNDS=3 needs --num-speculative-tokens >= 3 (the depth is %d)",
                            (int)drafter_.depth);
                    return RAD_E_INVAL;
                }
                if (want == 3 && mtp_couple_) {
                    RAD_ERR("kld: RADIANCE_MTP_PARITY_ROUNDS=3 needs the coupling off (RADIANCE_MTP_COUPLED=0): rounds "
                            "2-3 must be the head's own greedy chain");
                    return RAD_E_INVAL;
                }
                g_par.rounds = want;
            }
            g_par.nranks = kld_width_.size();
            g_par.host.assign((size_t)g_par.rounds * g_par.nranks, nullptr);
            for (int rd = 1; rd <= g_par.rounds; ++rd)
                for (size_t r = 0; r < kld_width_.size(); ++r) {
                    float* p = static_cast<float*>(rad_dev_alloc(
                        (int64_t)cfg_.max_seqs * kld_width_[r] * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
                    if (!p) return RAD_E_NOMEM;
                    g_par.host[(size_t)(rd - 1) * g_par.nranks + r] = p;
                }
            g_par.f = std::fopen((dir + (g_par.rounds == 1 ? "/parity.bin" : "/parity2.bin")).c_str(), "wb");
            if (!g_par.f) {
                RAD_ERR("kld: RADIANCE_MTP_PARITY: cannot open %s/parity%s.bin", dir.c_str(), g_par.rounds == 1 ? "" : "2");
                return RAD_E_IO;
            }
            g_par.on = true;
            RAD_INFO("kld: the MTP head's own round-1%s top-%d to %s/parity%s.bin (the full-logits head)",
                     g_par.rounds == 3 ? " to round-3" : "", kDumpTop, dir.c_str(), g_par.rounds == 1 ? "" : "2");
        }
    }

    auto kld = std::make_unique<Kld>();
    RAD_TRY(kld->open(cfg_, vocab_, nv, cfg_.model));
    /* Published before add_all takes the scheduler's lock; the step loop reads it only on steps
     * planned under that lock afterwards. */
    kld_ = std::move(kld);

    const size_t n = kld_->n_docs();
    /* A document the KV pool cannot hold even alone would wait at the head of the queue forever:
     * admission takes only a chunk the pool can hold, and nothing ever frees enough. Refused up
     * front, naming every such document. The pool is the least that a full paged group holds. */
    {
        const KVManager& kv = ranks_[0]->kv;
        int64_t pool = INT64_MAX;
        for (int32_t g = 0; g < kv.n_groups(); ++g) {
            const KVGroupPlan* p = kv.plan(g);
            if (p && p->paged() && p->window_blocks == 0)
                pool = std::min<int64_t>(pool, p->n_blocks * p->block_size);
        }
        int too_long = 0;
        for (size_t d = 0; d < n; ++d) {
            const int64_t len = (int64_t)kld_->tokens(d).size();
            if (kv.can_ever_fit(len)) continue;
            RAD_ERR("kld: document %zu is %lld tokens and the KV pool holds %lld, so it can never "
                    "be scored; give the cache more room or shorten the document", d,
                    (long long)len, (long long)pool);
            ++too_long;
        }
        if (too_long) return RAD_E_FULL;
    }

    /* --kld-lora: every document runs through one runtime LoRA adapter, held for the whole run --
     * how an adapter is scored against a reference recorded from its merged or HF counterpart. */
    int32_t  lora_slot = -1;
    uint64_t lora_uid = 0;
    std::shared_ptr<void> lora_hold;
    if (!cfg_.kld_lora.empty() &&
        lora_acquire(cfg_.kld_lora, &lora_slot, &lora_uid, &lora_hold) != RAD_OK) {
        RAD_ERR("kld: --kld-lora '%s' is not a loaded adapter (--lora NAME=PATH)", cfg_.kld_lora.c_str());
        return RAD_E_NOTFOUND;
    }
    if (lora_slot >= 0)
        RAD_INFO("kld: scoring through the runtime LoRA adapter '%s' (slot %d)", cfg_.kld_lora.c_str(),
                 lora_slot);

    /* Submit every document under one hold of the queue, so batching depends only on the corpus
     * and the flags and a rerun batches identically. */
    std::vector<std::unique_ptr<server::Sink>> sinks(n);
    std::vector<std::unique_ptr<Request>> reqs(n);
    for (size_t d = 0; d < n; ++d) {
        sinks[d] = std::make_unique<server::Sink>(64);
        auto r = std::make_unique<Request>();
        r->id         = d + 1;
        r->prompt     = kld_->tokens(d);
        r->max_tokens = 1;
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->score      = true;
        /* A loaded control vector scores at the deployment's default (--control-vector on|off). */
        r->sp.use_control_vector = cvec_default_on();
        /* --kld-lora-sources: the other documents run as the base, in the same steps. */
        const bool via = cfg_.kld_lora_sources.empty() ||
                         kld_->source(d).rfind(cfg_.kld_lora_sources, 0) == 0;
        r->lora_slot  = via ? lora_slot : -1;
        r->lora_uid   = via ? lora_uid : 0;
        r->sink       = sinks[d].get();
        sinks[d]->id  = r->id;
        reqs[d] = std::move(r);
    }
    std::vector<int> rc(n, RAD_OK);
    sched_.add_all(reqs.data(), n, rc.data());
    for (size_t d = 0; d < n; ++d) {
        if (rc[d] >= 0) continue;
        RAD_ERR("kld: the scheduler refused document %zu (%zu tokens): %s", d,
                kld_->tokens(d).size(), rad_strerror(rc[d]));
        return rc[d];
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    for (size_t first = 0;;) {
        /* Reaped as they complete, as the server reaps (run 187: holding all of them to the end is the one
         * difference from the service path left after MD5c / MD6 / MD7 -- the host's anonymous memory grew ~30 KB a
         * scored position with the dump, the drafter or libref's gather each taken away). A failed document is
         * still reported below from its sink. */
        while (first < n && sinks[first]->done()) {
            const server::Finish f = sinks[first]->finish_reason();
            if (f != server::Finish::Error && f != server::Finish::Cancelled) sched_.reap(sinks[first]->id);
            ++first;
        }
        if (first == n) break;
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        sinks[first]->wait(200);
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(15)) {
            last = now;
            const double s = std::chrono::duration<double>(now - t0).count();
            RAD_INFO("kld: %lld / %lld positions, %zu / %zu documents, %.0f s",
                     (long long)kld_->rows_done(), (long long)kld_->n_rows(), first, n, s);
        }
    }
    int bad = 0;
    for (size_t d = 0; d < n; ++d) {
        const server::Finish f = sinks[d]->finish_reason();
        if (f == server::Finish::Error || f == server::Finish::Cancelled) {
            RAD_ERR("kld: document %zu did not finish: %s", d, rad_strerror(sinks[d]->status()));
            ++bad;
        }
        sched_.reap(sinks[d]->id);
    }
    if (bad) return RAD_E_STATE;
    RAD_INFO("kld: %zu documents in %.0f s, %lld preemption(s)", n,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
             (long long)sched_.metrics().preemptions);
    if (g_dump.on) {
        std::fclose(g_dump.fm); std::fclose(g_dump.fh); std::fclose(g_dump.ft);
        g_dump.on = false;
        RAD_INFO("kld: the MTP head's pairs: %lld positions written", (long long)g_dump.rows);
        std::lock_guard<std::mutex> lk(g_par.mu);
        if (g_par.on) {
            std::fclose(g_par.f);
            g_par.on = false;
            RAD_INFO("kld: the MTP head's parity rows: %lld written", (long long)g_par.rows);
        }
    }
    return kld_->finish();
}

/* Copy a step's scored rows off the cards. They follow the sampler's rows in the logits buffer in
 * the batch builder's entry order (core/sched/batch.cpp), so the walk below repeats that loop and
 * checks the count. Each rank copies on its own stream behind the step; KL mode does not
 * pipeline, so the next step waits for this. */
int Engine::score_rows(const RadBatch* b, const StepOut& so) {
    const int64_t n = b->n_out - so.n_rows;
    if (n <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    std::vector<KldSpan> spans;
    int64_t row = 0;
    for (const StepEntry& e : plan.e) {
        if (!(e.req && e.req->score && e.is_prefill && e.hidden_row < 0)) continue;
        spans.push_back({ e.req->id, (int64_t)e.ctx_len, (int64_t)e.n_tokens, row });
        row += e.n_tokens;
    }
    if (row != n || n > cfg_.max_tok) {
        RAD_ERR("kld: step %d names %lld scored rows and its plan %lld", b->step, (long long)n,
                (long long)row);
        return RAD_E_STATE;
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        Rank& rk = *ranks_[r];
        RAD_TRY(rad_dev_set(rk.device));
        const auto* src = static_cast<const float*>(rk.ctx.buf_ptr(rk.program.logits_buf));
        if (!src) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(kld_host_[r], src + (int64_t)so.n_rows * kld_width_[r],
                                 n * kld_width_[r] * (int64_t)sizeof(float), rk.ctx.stream()));
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r]->device));
        RAD_TRY(rad_stream_sync(ranks_[r]->ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    std::vector<KldShard> shards(kld_width_.size());
    for (size_t r = 0; r < shards.size(); ++r)
        shards[r] = { kld_host_[r], kld_vocab0_[r], kld_width_[r] };
    if (g_dump.on) {
        /* the head's input rows: the step's token rows, rank 0's copy (the wide stream is replicated) */
        Rank& r0 = *ranks_[0];
        const void* hsrc = r0.ctx.buf_ptr(g_dump.buf);
        if (!hsrc) return RAD_E_STATE;
        const int64_t ntok = b->n_tok;
        RAD_TRY(rad_memcpy_async(g_dump.host, hsrc, ntok * g_dump.width * 2, r0.ctx.stream()));
        RAD_TRY(rad_stream_sync(r0.ctx.stream()));
        /* each scored position: its meta, its hidden row (the step's token row) and its logits row */
        struct DumpJob { int32_t meta[3]; int64_t tok, row; };
        std::vector<DumpJob> jobs;
        int64_t tok_off = 0;
        int64_t srow = 0;
        for (const StepEntry& e : plan.e) {
            const bool scored = e.req && e.req->score && e.is_prefill && e.hidden_row < 0;
            if (scored) {
                for (int64_t j = 0; j < (int64_t)e.n_tokens; ++j) {
                    const int64_t pos = (int64_t)e.ctx_len + j;
                    DumpJob dj{};
                    dj.meta[0] = (int32_t)e.req->id;
                    dj.meta[1] = (int32_t)pos;
                    dj.meta[2] = pos < (int64_t)e.req->prompt.size() ? e.req->prompt[(size_t)pos] : -1;
                    dj.tok = tok_off + j;
                    dj.row = srow + j;
                    jobs.push_back(dj);
                }
                srow += e.n_tokens;
            }
            tok_off += e.n_tokens;
        }
        /* the top-32 over the whole vocabulary and the log-sum-exp (row_top32), rows split over threads (run 179:
         * serial in double, ~2 ms a row, it held the ranks at the step barrier) */
        std::vector<int32_t> tids(jobs.size() * kDumpTop);
        std::vector<float>   tlps(jobs.size() * kDumpTop);
        auto one = [&](size_t i) { row_top32(shards, jobs[i].row, &tids[i * kDumpTop], &tlps[i * kDumpTop]); };
        {
            const unsigned hw = std::thread::hardware_concurrency();
            const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 8u));
            std::atomic<size_t> next{0};
            std::vector<std::thread> th;
            for (unsigned t = 0; t < nth; ++t)
                th.emplace_back([&] { for (size_t i; (i = next.fetch_add(1)) < jobs.size();) one(i); });
            for (auto& t : th) t.join();
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            std::fwrite(jobs[i].meta, 4, 3, g_dump.fm);
            std::fwrite((const uint8_t*)g_dump.host + (size_t)(jobs[i].tok * g_dump.width * 2), 2,
                        (size_t)g_dump.width, g_dump.fh);
            std::fwrite(&tids[i * kDumpTop], 4, kDumpTop, g_dump.ft);
            std::fwrite(&tlps[i * kDumpTop], 4, kDumpTop, g_dump.ft);
            ++g_dump.rows;
        }
    }
    return kld_->score(spans, shards);
}

/* RADIANCE_MTP_PARITY: right after the history pass is issued, before round 2's pass rewrites the logits buffer's
 * first rows, queue each rank's `drafted` rows to the host behind it (the ranks are parked at the barrier, and each
 * stream is in order). */
int Engine::parity_issue(int drafted, int round) {
    if (!g_par.on || drafted <= 0 || round < 1 || round > g_par.rounds) return RAD_OK;
    int n = std::min(drafted, (int)std::min<size_t>((size_t)cfg_.max_seqs, serial_row_.size()));
    if (round > 1) {
        if (g_par.issued != round - 1 || g_par.n <= 0) return RAD_OK;   /* no round 1 behind it: nothing to pair */
        n = std::min(n, g_par.n);                                    /* the same rows, in the same order */
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        Rank& rk = *ranks_[r];
        RAD_TRY(rad_dev_set(rk.device));
        const auto* src = static_cast<const float*>(rk.ctx.buf_ptr(rk.program.logits_buf));
        if (!src) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(g_par.at(round, r), src, (int64_t)n * kld_width_[r] * (int64_t)sizeof(float),
                                 rk.ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    if (round == 1) {
        g_par.seq.resize((size_t)n);
        for (int k = 0; k < n; ++k) g_par.seq[(size_t)k] = serial_row_[(size_t)k].seq;
    }
    g_par.n = n;
    g_par.issued = round;
    return RAD_OK;
}

/* After the step's wait: each drafting row's head index is its entry's last committed position (the verified
 * start plus the drafts accepted; a prompt's last row in the KL mode) and the token fed is the token drawn there. */
int Engine::parity_finish(const StepOut& so) {
    std::lock_guard<std::mutex> lk(g_par.mu);
    if (!g_par.on || g_par.n <= 0) return RAD_OK;
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r]->device));
        RAD_TRY(rad_stream_sync(ranks_[r]->ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    const int rounds = g_par.rounds == 1 ? 1 : g_par.issued;   /* ROUNDS=3 writes the rounds this step really ran */
    std::vector<std::vector<KldShard>> shards((size_t)rounds, std::vector<KldShard>(kld_width_.size()));
    for (int rd = 1; rd <= rounds; ++rd)
        for (size_t r = 0; r < kld_width_.size(); ++r) shards[(size_t)rd - 1][r] = { g_par.at(rd, r), kld_vocab0_[r], kld_width_[r] };
    const StepPlan& plan = sched_.plan();
    for (int k = 0; k < g_par.n; ++k) {
        size_t i = 0;
        while (i < plan.e.size() && !(plan.e[i].seq == g_par.seq[(size_t)k] && so.first_row[i] >= 0)) ++i;
        if (i == plan.e.size()) {
            RAD_WARN("kld: parity row %d (sequence %llu) has no entry in the step; skipped", k,
                     (unsigned long long)g_par.seq[(size_t)k]);
            continue;
        }
        const StepEntry& e = plan.e[i];
        int32_t meta[3];
        meta[0] = (int32_t)e.seq;
        meta[1] = (int32_t)((int64_t)e.ctx_len + e.n_tokens - 1 - e.n_spec + so.accepted[i]);
        meta[2] = so.token[i];
        int32_t ids[kDumpTop];
        float   lps[kDumpTop];
        if (g_par.rounds == 1) {   /* the 11-04 record, byte for byte */
            row_top32(shards[0], k, ids, lps);
            std::fwrite(meta, 4, 3, g_par.f);
            std::fwrite(ids, 4, kDumpTop, g_par.f);
            std::fwrite(lps, 4, kDumpTop, g_par.f);
        } else {
            /* parity2: id, i, rounds, the tokens fed at rounds 1-3 (the trunk's draw, then the round-1 and round-2
             * greedy drafts from the step's slots: StepSlot::tok[0] is the bonus token, tok[r] round r's draft), then
             * per round the 32 largest; a round the step did not run (rounds < 3) writes -1 tokens and zero rows */
            int32_t head[6] = { meta[0], meta[1], (int32_t)rounds, meta[2], -1, -1 };
            const StepSlot& S = slots_host_[e.card];
            if (rounds >= 2) head[4] = S.tok[1];
            if (rounds >= 3) head[5] = S.tok[2];
            std::fwrite(head, 4, 6, g_par.f);
            for (int rd = 1; rd <= 3; ++rd) {
                if (rd <= rounds) row_top32(shards[(size_t)rd - 1], k, ids, lps);
                else { std::memset(ids, 0, sizeof ids); std::memset(lps, 0, sizeof lps); }
                std::fwrite(ids, 4, kDumpTop, g_par.f);
                std::fwrite(lps, 4, kDumpTop, g_par.f);
            }
        }
        ++g_par.rows;
    }
    g_par.n = 0;
    g_par.issued = 0;
    return RAD_OK;
}

}  /* namespace rad */
