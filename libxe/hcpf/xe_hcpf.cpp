// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Deadcode and the radiance contributors
// Part of the Intel port of radiance (milestone 3, run 261; owner note 209, the supervisor's fix approved 21:20 11-07).
//
// xe_hcpf.cpp -- hc_read's prefill products on the matrix units (hc_pf_dpas_kernel<0> / <1>, RADIANCE_XE_HC_PF_DPAS)
// as their own module, kernels/xe/libxe_hcpf.so, so the image carries this one unit built by IGC 2.28.4 while the
// rest of libxe takes 2.42.6: 2.42.6's build of these two kernels hangs the first prefill chunk (B158 / B160 / B167,
// pinned by B172), 2.28.4's does not (B169). The two kernels are r4d_hc_bf16.cpp's text, MOVED unchanged (diffed
// line for line, run 261); the constants and device helpers they share with that unit are hcpf/xe_hcpf_dev.h.
// r4d_hc_bf16.cpp loads this module beside libxe.so at the first >= 64-row chunk and launches through xe_hcpf_run.
// The launch is libxe's own (XE_LAUNCH_SG -> xe::launch_as): this module links librad_l0k.so as libxe.so does, so
// the engine's recorder hooks resolve to the same loaded library and a recorded or glued queue takes these kernels
// as it takes libxe's.
#include <hip/hip_runtime.h>

#include <cstdint>
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"   /* f2bf / bf2f, as r4d_hc_bf16.cpp has them (libr4d's copy, as for that unit) */
#include <sycl/ext/oneapi/matrix/matrix.hpp>

/* A named namespace: kernels in an anonymous namespace would carry the same mangled names as any copy in another
 * library (_GLOBAL__N_1), and the SYCL runtime would hold two images under one kernel name. */
namespace xe_hcpf_mod {

#include "xe_hcpf_dev.h"

/* ------------------------------------------------------------ moved from r4d_hc_bf16.cpp, unchanged */
template <int kMode>
void hc_pf_dpas_kernel(const unsigned short* __restrict__ A, long K,                 /* [rows, K] bf16, pitch K */
                       const unsigned char* __restrict__ W, long wld, long grp,      /* E4M3 rows, scales after K */
                       long nw, const unsigned short* __restrict__ injw,             /* kMode 0: hc bf16 rows of K */
                       int hc, long rows, long n,
                       unsigned short* __restrict__ dout, unsigned short* __restrict__ inj, long irow,
                       const unsigned short* __restrict__ nbuf, unsigned short* __restrict__ x, long xrow) {
    using namespace hcpf;
    XE_SHARED(uint16_t, sA, kRT * kAP);       /* 64 rows x 64 k bf16, pitch 72 */
    XE_SHARED(uint32_t, sB, 4 * 4 * 128);     /* [k step 4][n tile 4][packed 8 x 16 words] */
    XE_SHARED(float, sC, kWv * 8 * 16);       /* one 8 x 16 f32 tile a sub-group */
    XE_SHARED(uint16_t, sU, kMode == 1 ? kRT * kNT : 1);   /* kMode 1: bf16(sigmoid(u)), [64 rows][64 columns] */
    const int  tid = (int)threadIdx.x, wave = tid / kSG, lane = tid % kSG;
    const long m0 = (long)blockIdx.y * kRT;
    const int  ipb = kNT / hc;                                /* kMode 1: channels a work-group owns */
    const long i0  = (long)blockIdx.x * ipb;
    const long n0  = (long)blockIdx.x * kNT;
    /* the weight row behind a column of the tile: kMode 0 column n0 + c (E4M3 below nw, inject rows to nw + hc,
     * else none), kMode 1 hc_up_mix's (stream c / ipb, channel i0 + c % ipb) */
    auto wrow = [&](int c) -> long {
        if constexpr (kMode == 1) return (long)(c / ipb) * n + i0 + c % ipb;
        else return n0 + c;
    };
    sycl::sub_group sg = xe::sg();
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::a, 8, 16, jm::layout::row_major> Am;
    jm::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> Bm;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> Cm[4];
    auto lptr = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };
#pragma unroll
    for (int t = 0; t < 4; ++t) jm::joint_matrix_fill(sg, Cm[t], 0.0f);
    for (long k0 = 0; k0 < K; k0 += 64) {
        xe::syncthreads();                                    /* the previous group's stages are consumed */
        /* the rows: 64 x 64 k, eight bf16 a thread a load (zeros past `rows`) */
        for (int idx = tid; idx < kRT * 8; idx += kThr) {
            const int  r = idx / 8, kc = (idx % 8) * 8;
            const long m = m0 + r;
            uint4 v = {0u, 0u, 0u, 0u};
            if (m < rows) v = *(const uint4*)(A + m * K + k0 + kc);
            *(uint4*)(sA + r * kAP + kc) = v;
        }
        /* the weight rows: 64 columns x 64 k, eight codes a thread a load, packed */
        for (int idx = tid; idx < kNT * 8; idx += kThr) {
            const int  c = idx / 8, kc = (idx % 8) * 8;
            const long j = wrow(c), kk = k0 + kc;
            uint32_t wv[4] = {0u, 0u, 0u, 0u};
            if (j < nw) {
                const unsigned char* wr = W + j * wld;
                const uint2 cd = *(const uint2*)(wr + kk);
                const float sc = ((const float*)(wr + K))[kk / grp] * kS8Scale;
                float f[8];
                e4m3x4_s8(cd.x, f);
                e4m3x4_s8(cd.y, f + 4);
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    wv[p] = (uint32_t)f2bf(f[2 * p] * sc) | ((uint32_t)f2bf(f[2 * p + 1] * sc) << 16);
            } else if (kMode == 0 && injw && j < nw + hc) {
                const uint4 v = *(const uint4*)(injw + (j - nw) * K + kk);
                wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int k2 = kc / 2 + p, step = k2 / 8, kw = k2 % 8;
                sB[((step * 4 + c / 16) * 8 + kw) * 16 + (c % 16)] = wv[p];
            }
        }
        xe::syncthreads();
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            jm::joint_matrix_load(sg, Am, lptr((sycl::ext::oneapi::bfloat16*)(sA + wave * 8 * kAP + step * 16)), (size_t)kAP);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                jm::joint_matrix_load(sg, Bm, lptr((sycl::ext::oneapi::bfloat16*)(sB + (step * 4 + t) * 128)), 32);
                jm::joint_matrix_mad(sg, Cm[t], Am, Bm, Cm[t]);
            }
        }
    }
    /* the tiles out: lane = column, 8 rows a lane */
    const float inv_hc = 1.0f / (float)hc;
    float* sCw = sC + wave * 128;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        jm::joint_matrix_store(sg, Cm[t], lptr(sCw), 16, jm::layout::row_major);
        xe::wave_barrier();
        const int c = t * 16 + lane;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int  rl = wave * 8 + e;
            const long m  = m0 + rl;
            const float v = sCw[e * 16 + lane];
            if constexpr (kMode == 1) {
                sU[rl * kNT + c] = f2bf(sigmoidf_(v));
            } else if (m < rows) {
                const long j = n0 + c;
                if (j < nw) dout[m * nw + j] = f2bf(siluf_(v * inv_hc));
                else if (inj && j < nw + hc) inj[m * irow + (j - nw)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
        }
        xe::wave_barrier();
    }
    if constexpr (kMode == 1) {
        xe::syncthreads();
        for (int idx = tid; idx < kRT * ipb; idx += kThr) {
            const int  rl = idx / ipb, ii = idx - rl * ipb;
            const long m  = m0 + rl;
            if (m >= rows) continue;
            const long i = i0 + ii;
            float acc = 0.0f;
            for (int c = 0; c < hc; ++c)
                acc += bf2f(sU[rl * kNT + c * ipb + ii]) * bf2f(nbuf[m * (long)hc * n + (long)c * n + i]);
            acc *= inv_hc;
            x[m * xrow + i] = f2bf(acc);
        }
    }
}
/* ------------------------------------------------------------ end of r4d_hc_bf16.cpp's text */

}  // namespace xe_hcpf_mod

/* The two launches r4d_hc_bf16.cpp made in place (its lines 5372-5384), the arguments in the kernel's order. 0 on
 * success, -1 on a bad mode or a launch error. `stream` is the caller's hipStream_t (a sycl::queue*). */
extern "C" int xe_hcpf_run(int mode, void* stream, unsigned gx, unsigned gy, const unsigned short* A, long K,
                           const unsigned char* W, long wld, long grp, long nw, const unsigned short* injw, int hc,
                           long rows, long n, unsigned short* dout, unsigned short* inj, long irow,
                           const unsigned short* nbuf, unsigned short* x, long xrow) {
    hipStream_t s = (hipStream_t)stream;
    if (mode == 0)
        (void)XE_LAUNCH_SG(xe_hcpf_mod::hcpf::kSG, s, (xe_hcpf_mod::hc_pf_dpas_kernel<0>), dim3(gx, gy), dim3(xe_hcpf_mod::hcpf::kThr), 0, A, K, W, wld, grp,
                           nw, injw, hc, rows, n, dout, inj, irow, nbuf, x, xrow);
    else if (mode == 1)
        (void)XE_LAUNCH_SG(xe_hcpf_mod::hcpf::kSG, s, (xe_hcpf_mod::hc_pf_dpas_kernel<1>), dim3(gx, gy), dim3(xe_hcpf_mod::hcpf::kThr), 0, A, K, W, wld, grp,
                           nw, injw, hc, rows, n, dout, inj, irow, nbuf, x, xrow);
    else
        return -1;
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
