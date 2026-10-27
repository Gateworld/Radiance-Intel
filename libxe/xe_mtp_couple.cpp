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

#include <cstdint>

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
        XE_LAUNCH(qq, mtp_couple_verify_kernel, dim3((unsigned)((c.M + kRowsPerBlock - 1) / kRowsPerBlock)),
                  dim3(kRowsPerBlock), 0, c);
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
