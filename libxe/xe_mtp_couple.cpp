/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3, owner note 47 item 2).
 *
 * xe_mtp_couple.cpp -- the linear-MTP coupling's two device rows for Xe2 (the B70): `mtp_couple_propose` (x ~ Q over
 * the head's final top-8, one work-item a row) and `mtp_couple_verify` (the maximal coupling after sample_pick over
 * the target's surviving candidates, one work-item a row). The arithmetic is upstream's, inline in
 * abi/rad_spechub.h and plugin/spechub/spechub.h (the same functions the host reference rows run), in strict FP64:
 * the B70 has the fp64 aspect (Q1, run 53: a 1,000-term double sum on the device equals the host's), and the
 * -ffp-contract=off below keeps the ascending FP64 sums the 53-bit uniforms are compared against. The argument
 * readers and the shape hooks are the spechub plugin's own (plugin/spechub/spechub.cpp, compiled into libxe).
 *
 * This is the serial form of spechub.hip's verify (its cooperative work-group walks the candidates 128 wide); a
 * row's verify is O(n_cand) double exps, which at M <= 8 rows x 1,024 candidates a decode step is latency, not
 * throughput: the step time says whether the cooperative form is worth porting.
 */
#pragma clang fp contract(off)

#include "xe_lanes.h"
#include "../plugin/spechub/spechub.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

void mtp_couple_propose_kernel(RadMtpCoupleProposeCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_propose_row(c, row);
}

void mtp_couple_verify_kernel(RadMtpCoupleVerifyCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_verify_row(c, row);
}

constexpr int kRowsPerBlock = 16;

/* RADIANCE_XE_MTP_COOP=1 (owner notes 243 item 2 / 246, run 312): the cooperative verify, spechub.hip's form -- one
 * work-group a row, the candidates' exps 128 wide into LDS, then every sum and both CDF walks on thread 0 in ascending
 * order, so the token equals the serial row's bit for bit (the same exp((double)v - top) a slot, computed once
 * instead of up to three times; the first-occurrence rule for Q ids is the serial `seen` mask). Ineligible rows write
 * nothing, as the serial row. Wider planes than kMaxCand run the serial row on thread 0. ON by default since run 323
 * (owner note 247: the coupling is the Intel images' default); RADIANCE_XE_MTP_COOP=0 keeps the serial row. */
bool coop_on() {
    static const bool v = [] { const char* s = std::getenv("RADIANCE_XE_MTP_COOP"); return !(s && *s == '0'); }();
    return v;
}

constexpr int kCoopThreads = 128;
constexpr int64_t kMaxCand = 1024;

void mtp_couple_verify_coop_kernel(const int32_t* idx, const float* values, const uint32_t* params,
                                   const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                                   int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                                   int64_t ld4) {
    XE_SHARED(double, weight, kMaxCand);
    XE_SHARED(double, mass, kMaxCand);
    XE_SHARED(double, pq, 8);
    XE_SHARED_VAR(double, sum);
    XE_SHARED_VAR(float, top);
    XE_SHARED_VAR(int, eligible);
    XE_SHARED(int, first, 8);
    XE_SHARED_VAR(int, ix);
    XE_SHARED_VAR(int, dominated);
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    if (n_cand > kMaxCand) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row(c, row);
        return;
    }
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float*   val = c.val + row * c.ld[1];
    if (threadIdx.x == 0) {
        eligible = 0;
        if (s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
            s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab)) {
            float maximum = -INFINITY;
            bool  valid   = true;
            for (int64_t j = 0; j < c.n_cand; ++j) {
                if (ids[j] < 0) continue;
                if (ids[j] >= c.vocab || val[j] != val[j] || val[j] == INFINITY) { valid = false; break; }
                if (val[j] > maximum) maximum = val[j];
            }
            if (valid && maximum != -INFINITY) { top = maximum; eligible = 1; }
        }
    }
    __syncthreads();
    if (!eligible) return;
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x)
        if (ids[j] >= 0 && val[j] != -INFINITY) weight[j] = ::exp((double)val[j] - top);
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] < 0 || val[j] == -INFINITY) continue;
            const double w = weight[j];
            sum += w;
            for (int k = 0; k < s.count; ++k)
                if (ids[j] == s.ids[k]) {
                    if (first[k] < 0) first[k] = (int)j;
                    pq[k] += w;
                    break;
                }
        }
        if (!(sum > 0) || !rad_spechub_finite(sum)) eligible = 0;
        if (eligible) {
            ix = -1;
            dominated = true;
            for (int k = 0; k < s.count; ++k) {
                pq[k] /= sum;
                dominated = dominated && pq[k] >= s.q[k];
                if (s.ids[k] == s.x) ix = k;
            }
            if (ix < 0) eligible = 0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x) {
        double m = 0;
        if (ids[j] >= 0 && val[j] != -INFINITY) {
            bool in_q = false;
            for (int k = 0; k < s.count; ++k)
                if (s.ids[k] == ids[j]) {
                    in_q = true;
                    if (first[k] == (int)j) m = rad_spechub_positive(pq[k] - s.q[k]);
                    break;
                }
            if (!in_q) m = weight[j] / sum;
        }
        mass[j] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    double resid = 0;
    for (int64_t j = 0; j < c.n_cand; ++j) resid += mass[j];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int64_t j = 0; j < c.n_cand; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = weight[j];
                if (w > 0) { out = ids[j]; cdf -= w; if (cdf < 0) break; }
            }
    } else if (pq[ix] >= s.q[ix] ||
               rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT) * s.q[ix] < pq[ix]) {
        out = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] == s.x) continue;
            const double m = mass[j];
            if (m > 0) { out = ids[j]; cdf -= m; if (cdf < 0) break; }
        }
    }
    c.token[row * c.ld[4]] = out;
}

}  // namespace

extern "C" int xe_rad_mtp_couple_propose(const RadArgs* a, RadStream s) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    try {
        sycl::queue* qq = (sycl::queue*)s;
        XE_LAUNCH(qq, mtp_couple_propose_kernel, dim3((unsigned)((c.M + kRowsPerBlock - 1) / kRowsPerBlock)),
                  dim3(kRowsPerBlock), 0, c);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

extern "C" int xe_rad_mtp_couple_verify(const RadArgs* a, RadStream s) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    try {
        sycl::queue* qq = (sycl::queue*)s;
        if (coop_on())
            XE_LAUNCH(qq, mtp_couple_verify_coop_kernel, dim3((unsigned)c.M), dim3(kCoopThreads), 0,
                      c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                      c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
        else
            XE_LAUNCH(qq, mtp_couple_verify_kernel, dim3((unsigned)((c.M + kRowsPerBlock - 1) / kRowsPerBlock)),
                      dim3(kRowsPerBlock), 0, c);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
