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
//
// Run 280 (owner note 223, "repair the hc 2.42.6 bug -- a source workaround"): HC1 reproduced the hang without the
// engine (the hc row test at 4,096 rows: a GPU page fault at a wild address at a 64-block grid; clean at one block,
// equal bytes to 2.28.4's build there), and the two compilers' ISA of the kernel differ by +64 `mul` -- the 64-bit
// integer multiply emulation (mul / mach / macl; Xe2 has no native 64-bit integer multiply) of the `long` index
// products. The repair makes the kernel avoid the code IGC 2.42.6 gets wrong: the kernel is instantiated with a
// 32-bit index type (k32), every offset computed in uint32_t and widened ONCE when it is added to the pointer
// (`A + (size_t)(m * K + k0 + kc)` with m, K, k0, kc uint32_t), and xe_hcpf_run takes that instance when every
// product fits 2^31 (rows x K, nw x wld, hc x K, rows x nw, rows x irow, rows x hc x n, rows x xrow -- a 4,096-row
// chunk at K 8,192 is 33.5 M), today's `long` form otherwise (or under RADIANCE_XE_HC_PF_IDX64=1, for comparisons).
// The same addresses, so the same bytes: the data arithmetic is untouched. Unsigned on purpose: a signed product's
// overflow would be undefined, which lets the compiler widen it back into the 64-bit multiply.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"   /* f2bf / bf2f, as r4d_hc_bf16.cpp has them (libr4d's copy, as for that unit) */
#include <sycl/ext/oneapi/matrix/matrix.hpp>

/* A named namespace: kernels in an anonymous namespace would carry the same mangled names as any copy in another
 * library (_GLOBAL__N_1), and the SYCL runtime would hold two images under one kernel name. */
namespace xe_hcpf_mod {

#include "xe_hcpf_dev.h"

/* ------------------------------------------------------------ moved from r4d_hc_bf16.cpp (run 261); the index type
 * made a template parameter in run 280 (owner note 223): k32 = every offset in uint32_t, widened once at the pointer;
 * !k32 = the original `long` arithmetic, instruction for instruction the moved text's semantics */
template <int kMode, bool k32>
void hc_pf_dpas_kernel(const unsigned short* __restrict__ A, long K,                 /* [rows, K] bf16, pitch K */
                       const unsigned char* __restrict__ W, long wld, long grp,      /* E4M3 rows, scales after K */
                       long nw, const unsigned short* __restrict__ injw,             /* kMode 0: hc bf16 rows of K */
                       int hc, long rows, long n,
                       unsigned short* __restrict__ dout, unsigned short* __restrict__ inj, long irow,
                       const unsigned short* __restrict__ nbuf, unsigned short* __restrict__ x, long xrow) {
    using namespace hcpf;
    using idx_t = std::conditional_t<k32, uint32_t, long>;   /* the index arithmetic's type (note 223) */
    XE_SHARED(uint16_t, sA, kRT * kAP);       /* 64 rows x 64 k bf16, pitch 72 */
    XE_SHARED(uint32_t, sB, 4 * 4 * 128);     /* [k step 4][n tile 4][packed 8 x 16 words] */
    XE_SHARED(float, sC, kWv * 8 * 16);       /* one 8 x 16 f32 tile a sub-group */
    XE_SHARED(uint16_t, sU, kMode == 1 ? kRT * kNT : 1);   /* kMode 1: bf16(sigmoid(u)), [64 rows][64 columns] */
    const int  tid = (int)threadIdx.x, wave = tid / kSG, lane = tid % kSG;
    /* the pitches and bounds in the index type (xe_hcpf_run has checked every product fits when k32) */
    const idx_t K_ = (idx_t)K, wld_ = (idx_t)wld, grp_ = (idx_t)grp, nw_ = (idx_t)nw, n_ = (idx_t)n,
                irow_ = (idx_t)irow, xrow_ = (idx_t)xrow, hc_ = (idx_t)hc;
    const idx_t m0 = (idx_t)blockIdx.y * (idx_t)kRT;
    const int  ipb = kNT / hc;                                /* kMode 1: channels a work-group owns */
    const idx_t i0 = (idx_t)blockIdx.x * (idx_t)ipb;
    const idx_t n0 = (idx_t)blockIdx.x * (idx_t)kNT;
    /* the weight row behind a column of the tile: kMode 0 column n0 + c (E4M3 below nw, inject rows to nw + hc,
     * else none), kMode 1 hc_up_mix's (stream c / ipb, channel i0 + c % ipb) */
    auto wrow = [&](int c) -> idx_t {
        if constexpr (kMode == 1) return (idx_t)(c / ipb) * n_ + i0 + (idx_t)(c % ipb);
        else return n0 + (idx_t)c;
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
    for (idx_t k0 = 0; k0 < K_; k0 += 64) {
        xe::syncthreads();                                    /* the previous group's stages are consumed */
        /* the rows: 64 x 64 k, eight bf16 a thread a load (zeros past `rows`) */
        for (int idx = tid; idx < kRT * 8; idx += kThr) {
            const int  r = idx / 8, kc = (idx % 8) * 8;
            const idx_t m = m0 + (idx_t)r;
            uint4 v = {0u, 0u, 0u, 0u};
            if ((long)m < rows) v = *(const uint4*)(A + (size_t)(m * K_ + k0 + (idx_t)kc));
            *(uint4*)(sA + r * kAP + kc) = v;
        }
        /* the weight rows: 64 columns x 64 k, eight codes a thread a load, packed */
        for (int idx = tid; idx < kNT * 8; idx += kThr) {
            const int  c = idx / 8, kc = (idx % 8) * 8;
            const idx_t j = wrow(c), kk = k0 + (idx_t)kc;
            uint32_t wv[4] = {0u, 0u, 0u, 0u};
            if ((long)j < nw) {
                const unsigned char* wr = W + (size_t)(j * wld_);
                const uint2 cd = *(const uint2*)(wr + kk);
                const float sc = ((const float*)(wr + K_))[kk / grp_] * kS8Scale;
                float f[8];
                e4m3x4_s8(cd.x, f);
                e4m3x4_s8(cd.y, f + 4);
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    wv[p] = (uint32_t)f2bf(f[2 * p] * sc) | ((uint32_t)f2bf(f[2 * p + 1] * sc) << 16);
            } else if (kMode == 0 && injw && (long)j < nw + hc) {
                const uint4 v = *(const uint4*)(injw + (size_t)((j - nw_) * K_ + kk));
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
            const idx_t m = m0 + (idx_t)rl;
            const float v = sCw[e * 16 + lane];
            if constexpr (kMode == 1) {
                sU[rl * kNT + c] = f2bf(sigmoidf_(v));
            } else if ((long)m < rows) {
                const idx_t j = n0 + (idx_t)c;
                if ((long)j < nw) dout[(size_t)(m * nw_ + j)] = f2bf(siluf_(v * inv_hc));
                else if (inj && (long)j < nw + hc) inj[(size_t)(m * irow_ + (j - nw_))] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
        }
        xe::wave_barrier();
    }
    if constexpr (kMode == 1) {
        xe::syncthreads();
        for (int idx = tid; idx < kRT * ipb; idx += kThr) {
            const int  rl = idx / ipb, ii = idx - rl * ipb;
            const idx_t m = m0 + (idx_t)rl;
            if ((long)m >= rows) continue;
            const idx_t i = i0 + (idx_t)ii;
            float acc = 0.0f;
            for (int c = 0; c < hc; ++c)
                acc += bf2f(sU[rl * kNT + c * ipb + ii]) * bf2f(nbuf[(size_t)((m * hc_ + (idx_t)c) * n_ + i)]);
            acc *= inv_hc;
            x[(size_t)(m * xrow_ + i)] = f2bf(acc);
        }
    }
}
/* ------------------------------------------------------------ end of r4d_hc_bf16.cpp's text */

/* Run 280 (owner note 223): does every offset the launch computes fit the 32-bit form? Each access is
 * `base + row * pitch + column` with row < count and column < pitch, so (count + 1) * pitch < 2^31 bounds it
 * with a pitch of margin; the per-block products (m0 up to count + 63, the tile's columns) stay inside the same
 * bound because every access is guarded by its row and column checks before the pointer is formed. */
inline bool fits31(long count, long pitch) {
    if (count < 0 || pitch < 0) return false;
    if (pitch == 0) return true;
    return count + 1 <= (0x7fffffffL - 64) / pitch;
}
bool hcpf_fits32(int mode, long K, long wld, long grp, long nw, const void* injw, int hc, long rows, long n,
                 long irow, long xrow) {
    if (K <= 0 || grp <= 0 || hc <= 0 || rows <= 0 || K > 0x7fffffffL || wld > 0x7fffffffL) return false;
    bool ok = fits31(rows, K) && fits31(nw, wld) && fits31(nw + hc, wld);
    if (mode == 0) {
        ok = ok && fits31(rows, nw) && fits31(rows, irow);
        if (injw) ok = ok && fits31(hc, K);
    } else {
        if (n <= 0 || n > 0x7fffffffL / hc) return false;
        ok = ok && fits31(rows, hc * n) && fits31(rows, xrow) && fits31(hc, n);
    }
    return ok;
}
/* RADIANCE_XE_HC_PF_IDX64=1: today's `long` index form regardless (a comparison switch; the default takes the
 * 32-bit form whenever it fits) */
bool hcpf_force64() {
    static const bool v = [] { const char* e = std::getenv("RADIANCE_XE_HC_PF_IDX64"); return e && e[0] == '1'; }();
    return v;
}

}  // namespace xe_hcpf_mod

/* The two launches r4d_hc_bf16.cpp made in place (its lines 5372-5384), the arguments in the kernel's order. 0 on
 * success, -1 on a bad mode or a launch error. `stream` is the caller's hipStream_t (a sycl::queue*). */
extern "C" int xe_hcpf_run(int mode, void* stream, unsigned gx, unsigned gy, const unsigned short* A, long K,
                           const unsigned char* W, long wld, long grp, long nw, const unsigned short* injw, int hc,
                           long rows, long n, unsigned short* dout, unsigned short* inj, long irow,
                           const unsigned short* nbuf, unsigned short* x, long xrow) {
    hipStream_t s = (hipStream_t)stream;
    if (mode != 0 && mode != 1) return -1;
    /* run 280 (owner note 223): the 32-bit index form when every offset fits, else the original `long` form */
    const bool k32 = !xe_hcpf_mod::hcpf_force64()
                     && xe_hcpf_mod::hcpf_fits32(mode, K, wld, grp, nw, injw, hc, rows, n, irow, xrow);
#define XE_HCPF_LAUNCH(MODE, K32)                                                                                   \
    (void)XE_LAUNCH_SG(xe_hcpf_mod::hcpf::kSG, s, (xe_hcpf_mod::hc_pf_dpas_kernel<MODE, K32>), dim3(gx, gy),       \
                       dim3(xe_hcpf_mod::hcpf::kThr), 0, A, K, W, wld, grp, nw, injw, hc, rows, n, dout, inj, irow, \
                       nbuf, x, xrow)
    if (mode == 0) { if (k32) XE_HCPF_LAUNCH(0, true); else XE_HCPF_LAUNCH(0, false); }
    else           { if (k32) XE_HCPF_LAUNCH(1, true); else XE_HCPF_LAUNCH(1, false); }
#undef XE_HCPF_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
/* which form a launch would take (for tests and the ISA record): 1 = the 32-bit index form, 0 = the `long` form */
extern "C" int xe_hcpf_idx32(int mode, long K, long wld, long grp, long nw, int has_injw, int hc, long rows, long n,
                             long irow, long xrow) {
    return (!xe_hcpf_mod::hcpf_force64()
            && xe_hcpf_mod::hcpf_fits32(mode, K, wld, grp, nw, has_injw ? (const void*)1 : nullptr, hc, rows, n,
                                        irow, xrow)) ? 1 : 0;
}
