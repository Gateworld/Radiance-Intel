/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, owner note 47 item 2).
 *
 * mtp_couple_test.cpp -- libxe's MTP coupling rows against the spechub plugin's host reference rows on the same
 * bytes: `mtp_couple_propose` (x ~ Q over the head's top-8 into idx column 0 and the state plane) and
 * `mtp_couple_verify` (the maximal coupling after the pick over the candidate plane). Both run upstream's inline
 * arithmetic (abi/rad_spechub.h), so the planes must agree byte for byte; the test also requires the coupling to
 * have acted (a row's token changed) so the compared path is the live one, and that the rows the coupling must
 * leave alone (mode 0, a card mismatch) are left alone.
 *
 *   mtp_couple_test LIBXE.so SPECHUB.so        RI_DEVICE=cpu for the CPU SYCL device
 */
#include <sycl/sycl.hpp>

#include "rad_abi.h"
#include "rad_sample.h"
#include "rad_spechub.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

typedef int (*CountFn)(void);
typedef const RadKernelInfo* (*AtFn)(int);

const RadKernelInfo* row_of(void* h, const char* op, int domain, const char* lib) {
    CountFn cnt = (CountFn)dlsym(h, "rad_kernel_count");
    AtFn at = (AtFn)dlsym(h, "rad_kernel_at");
    if (!cnt || !at) { std::printf("%s: no rad_kernel_count / rad_kernel_at\n", lib); return nullptr; }
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* r = at(i);
        if (r && r->op && !std::strcmp(r->op, op) && r->domain == domain && r->launch) return r;
    }
    std::printf("%s: no %s row for op %s\n", lib, domain == RAD_DOMAIN_DEVICE ? "device" : "host", op);
    return nullptr;
}

uint32_t h32(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return (uint32_t)x;
}

RadTensor tn(void* p, uint32_t dt, int rank, int64_t s0, int64_t s1 = 1, int64_t s2 = 1) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = rank;
    t.shape[0] = s0; t.shape[1] = s1; t.shape[2] = s2;
    if (rank == 1) { t.stride[0] = 1; }
    if (rank == 2) { t.stride[0] = s1; t.stride[1] = 1; }
    if (rank == 3) { t.stride[0] = s1 * s2; t.stride[1] = s2; t.stride[2] = 1; }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: mtp_couple_test LIBXE.so SPECHUB.so\n"); return 2; }
    void* hx = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void* hs = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!hx || !hs) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    const RadKernelInfo* xp = row_of(hx, "mtp_couple_propose", RAD_DOMAIN_DEVICE, "libxe");
    const RadKernelInfo* xv = row_of(hx, "mtp_couple_verify", RAD_DOMAIN_DEVICE, "libxe");
    const RadKernelInfo* rp = row_of(hs, "mtp_couple_propose", RAD_DOMAIN_HOST, "spechub");
    const RadKernelInfo* rv = row_of(hs, "mtp_couple_verify", RAD_DOMAIN_HOST, "spechub");
    if (!xp || !xv || !rp || !rv) return 2;

    const char* dev = std::getenv("RI_DEVICE");
    const bool cpu = dev && !std::strcmp(dev, "cpu");
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  { sycl::property::queue::in_order() });
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device: %s; fp64 %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                q.get_device().has(sycl::aspect::fp64) ? "yes" : "NO");

    /* RI_MTP_NCAND (run 312, owner note 246): the verify's candidate width, 64 unless set -- the wide forms (129,
     * 1024, 1025 > the cooperative form's LDS plane, its serial fallback) repeat ids past the vocabulary of 1000, so
     * the first-occurrence rule is exercised too; RADIANCE_XE_MTP_COOP picks the device form under test */
    const char* ncs = std::getenv("RI_MTP_NCAND");
    const int64_t n_cand = ncs && std::atoll(ncs) > 8 ? std::atoll(ncs) : 64;
    const int64_t M = 6, vocab = 1000, rounds = 3, round = 1;
    std::printf("verify width %lld, device form %s\n", (long long)n_cand,
                (std::getenv("RADIANCE_XE_MTP_COOP") && *std::getenv("RADIANCE_XE_MTP_COOP") == '1') ? "cooperative" : "serial");
    const int64_t DW = RAD_MTP_COUPLE_DRAW_WORDS, SW = RAD_MTP_COUPLE_STATE_WORDS;
    /* ---- propose: the head's top-8 a row, the draw rows, the state plane; rows 4 (mode 0) and 5 (card
     * mismatch) must stay untouched */
    std::vector<int32_t> idx((size_t)(M * 8)), seq((size_t)M);
    std::vector<float> val((size_t)(M * 8));
    std::vector<uint32_t> draw((size_t)(M * DW)), state((size_t)(rounds * M * SW), 0u);
    for (int64_t r = 0; r < M; ++r) {
        for (int j = 0; j < 8; ++j) {
            idx[(size_t)(r * 8 + j)] = (int32_t)((h32(1, (uint64_t)(r * 8 + j)) % 900u) + (uint32_t)j * 100u) % (int32_t)vocab;
            val[(size_t)(r * 8 + j)] = 3.0f - 0.45f * (float)j + (float)(h32(2, (uint64_t)(r * 8 + j)) % 100u) / 200.0f;
        }
        /* distinct ids within the row */
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < j; ++k)
                if (idx[(size_t)(r * 8 + j)] == idx[(size_t)(r * 8 + k)]) idx[(size_t)(r * 8 + j)] = (idx[(size_t)(r * 8 + j)] + 1) % (int32_t)vocab;
        seq[(size_t)r] = (int32_t)r;
        RadMtpCoupleDraw d{};
        d.seed = 90210ull + (uint64_t)r; d.base = 17ull + (uint64_t)r; d.seq = (uint64_t)r;
        d.card = r == 5 ? 99 : (int32_t)r;
        d.mode = r == 4 ? 0u : 1u;
        std::memcpy(&draw[(size_t)(r * DW)], &d, sizeof d);
    }
    auto shared = [&](size_t words) { return sycl::malloc_shared<uint32_t>(words, q); };
    uint32_t* x_idx = shared((size_t)(M * 8)); uint32_t* x_val = shared((size_t)(M * 8)); uint32_t* x_seq = shared((size_t)M);
    uint32_t* x_draw = shared((size_t)(M * DW)); uint32_t* x_state = shared((size_t)(rounds * M * SW));
    std::vector<int32_t> r_idx = idx; std::vector<uint32_t> r_state = state;
    std::memcpy(x_idx, idx.data(), idx.size() * 4); std::memcpy(x_val, val.data(), val.size() * 4);
    std::memcpy(x_seq, seq.data(), seq.size() * 4); std::memcpy(x_draw, draw.data(), draw.size() * 4);
    std::memcpy(x_state, state.data(), state.size() * 4);
    RadParam pp[4]{};
    const char* pk[] = { "M", "n_vocab", "round", "max_rounds" };
    const long long pv[] = { M, vocab, round, rounds };
    for (int i = 0; i < 4; ++i) { pp[i].key = pk[i]; pp[i].kind = RAD_P_INT; pp[i].ival = pv[i]; }
    bool ok = true, refused = false;
    auto propose = [&](const RadKernelInfo* row, void* pidx, void* pval, void* pseq, void* pdraw, void* pstate, RadStream s) {
        RadTensor t[5] = { tn(pidx, RAD_I32, 2, M, 8), tn(pval, RAD_F32, 2, M, 8), tn(pseq, RAD_I32, 1, M),
                           tn(pdraw, RAD_U32, 2, M, DW), tn(pstate, RAD_U32, 3, rounds, M, SW) };
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = pp; a.n_p = 4; a.world_size = 1;
        const int rc = row->launch(&a, s);
        if (rc != RAD_OK) { std::printf("%s: propose refused rc %d\n", row->name, rc); refused = true; }
    };
    propose(xp, x_idx, x_val, x_seq, x_draw, x_state, (RadStream)&q);
    q.wait_and_throw();
    propose(rp, r_idx.data(), val.data(), seq.data(), draw.data(), r_state.data(), nullptr);
    if (refused) return 1;
    const bool idx_same = std::memcmp(x_idx, r_idx.data(), idx.size() * 4) == 0;
    /* The state rows field by field: the integers and the draw exact, the FP64 masses q[] within 1e-12 (the
     * device's exp() and the host's libm are different implementations; a last-bit difference in a mass is not
     * a different law, and the picks -- idx column 0 -- must still agree, which the exact idx compare holds). */
    bool st_same = true;
    double dq_max = 0.0;
    for (int64_t rr = 0; rr < rounds * M; ++rr) {
        RadMtpCoupleState a{}, b{};
        std::memcpy(&a, x_state + rr * SW, sizeof a);
        std::memcpy(&b, &r_state[(size_t)(rr * SW)], sizeof b);
        if (std::memcmp(&a.draw, &b.draw, sizeof a.draw) || a.pos != b.pos || a.status != b.status || a.round != b.round ||
            a.x != b.x || a.count != b.count || std::memcmp(a.ids, b.ids, sizeof a.ids))
            st_same = false;
        for (int i = 0; i < 8; ++i) {
            const double d = std::fabs(a.q[i] - b.q[i]);
            if (d > dq_max) dq_max = d;
            if (d > 1e-12 * (std::fabs(b.q[i]) > 1e-300 ? std::fabs(b.q[i]) : 1.0)) st_same = false;
        }
    }
    int sampled = 0, untouched = 0;
    for (int64_t r = 0; r < M; ++r) {
        RadMtpCoupleState s{};
        std::memcpy(&s, &r_state[(size_t)((round - 1) * M * SW + r * SW)], sizeof s);
        if (s.status == 1) ++sampled;
        if (r >= 4 && r_idx[(size_t)(r * 8)] == idx[(size_t)(r * 8)] && s.status != 1) ++untouched;
    }
    std::printf("propose: idx planes %s, state planes %s (max |dq| %.3g); %d of %lld rows sampled (status 1), the mode-0 and "
                "mismatched rows untouched %d of 2\n",
                idx_same ? "identical" : "DIFFER", st_same ? "agree" : "DIFFER", dq_max, sampled, (long long)M, untouched);
    ok = ok && idx_same && st_same && sampled == 4 && untouched == 2;

    /* ---- verify: the candidate plane (the state's ids among the candidates, with their own scores), the
     * parameter rows bound to the draws, the state rows of this round, the pick's tokens */
    std::vector<int32_t> cidx((size_t)(M * n_cand)), tok((size_t)M);
    std::vector<float> cval((size_t)(M * n_cand));
    std::vector<uint32_t> prm((size_t)(M * 24), 0u), vst((size_t)(M * SW));
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t j = 0; j < n_cand; ++j) {
            cidx[(size_t)(r * n_cand + j)] = (int32_t)((r * 131 + j * 7) % vocab);
            cval[(size_t)(r * n_cand + j)] = 2.5f - 0.05f * (float)j + (float)(h32(3, (uint64_t)(r * n_cand + j)) % 100u) / 300.0f;
        }
        /* the head's ids sit at the front with the row's own scores, so Q's support is inside P's */
        for (int j = 0; j < 8; ++j) cidx[(size_t)(r * n_cand + j)] = r_idx[(size_t)(r * 8 + j)];
        for (int64_t j = 8; j < n_cand; ++j)
            for (int k = 0; k < 8; ++k)
                if (cidx[(size_t)(r * n_cand + j)] == cidx[(size_t)(r * n_cand + k)]) cidx[(size_t)(r * n_cand + j)] = (cidx[(size_t)(r * n_cand + j)] + 500) % (int32_t)vocab;
        RadSampleParams sp{};
        sp.seed = 90210ull + (uint64_t)r; sp.pos = 0; sp.temp = 1.0f; sp.top_k = (int32_t)n_cand; sp.mask_row = -1;
        static_assert(sizeof(RadSampleParams) <= 24 * 4, "the sampler's parameter row is 24 words");
        std::memcpy(&prm[(size_t)(r * 24)], &sp, sizeof sp);
        std::memcpy(&vst[(size_t)(r * SW)], &r_state[(size_t)((round - 1) * M * SW + r * SW)], (size_t)SW * 4);
        tok[(size_t)r] = cidx[(size_t)(r * n_cand)];
    }
    uint32_t* x_cidx = shared((size_t)(M * n_cand)); uint32_t* x_cval = shared((size_t)(M * n_cand));
    uint32_t* x_prm = shared((size_t)(M * 24)); uint32_t* x_vst = shared((size_t)(M * SW)); uint32_t* x_tok = shared((size_t)M);
    std::memcpy(x_cidx, cidx.data(), cidx.size() * 4); std::memcpy(x_cval, cval.data(), cval.size() * 4);
    std::memcpy(x_prm, prm.data(), prm.size() * 4); std::memcpy(x_vst, vst.data(), vst.size() * 4); std::memcpy(x_tok, tok.data(), tok.size() * 4);
    std::vector<int32_t> r_tok = tok;
    RadParam vp[3]{};
    const char* vk[] = { "M", "n_cand", "n_vocab" };
    const long long vv[] = { M, n_cand, vocab };
    for (int i = 0; i < 3; ++i) { vp[i].key = vk[i]; vp[i].kind = RAD_P_INT; vp[i].ival = vv[i]; }
    auto verify = [&](const RadKernelInfo* row, void* pi, void* pvl, void* ppr, void* pst, void* pt, RadStream s) {
        RadTensor t[5] = { tn(pi, RAD_I32, 2, M, n_cand), tn(pvl, RAD_F32, 2, M, n_cand), tn(ppr, RAD_U32, 2, M, 24),
                           tn(pst, RAD_U32, 2, M, SW), tn(pt, RAD_I32, 1, M) };
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = vp; a.n_p = 3; a.world_size = 1;
        const int rc = row->launch(&a, s);
        if (rc != RAD_OK) { std::printf("%s: verify refused rc %d\n", row->name, rc); refused = true; }
    };
    verify(xv, x_cidx, x_cval, x_prm, x_vst, x_tok, (RadStream)&q);
    q.wait_and_throw();
    verify(rv, cidx.data(), cval.data(), prm.data(), vst.data(), r_tok.data(), nullptr);
    if (refused) return 1;
    int changed = 0, kept = 0;
    const bool tok_same = std::memcmp(x_tok, r_tok.data(), tok.size() * 4) == 0;
    for (int64_t r = 0; r < M; ++r) {
        if (r_tok[(size_t)r] != tok[(size_t)r]) ++changed;
        if (r >= 4 && r_tok[(size_t)r] == tok[(size_t)r]) ++kept;
    }
    std::printf("verify: token planes %s; the coupling changed %d of 4 bound rows' tokens, kept the 2 unbound rows' %d of 2\n",
                tok_same ? "identical" : "DIFFER", changed, kept);
    ok = ok && tok_same && changed >= 1 && kept == 2;
    std::printf("RESULT mtp_couple_test %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
