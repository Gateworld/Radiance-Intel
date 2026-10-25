/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_support.cpp -- libr4d's cross-unit helpers the ported units call, for one card.
 */
#include <sycl/sycl.hpp>

#include "r4d.h"
#include "r4d_args.h"

/* r4d_rows.cpp's tuned-key reads for the ported units: the key if the call carries it, else the
 * default of libr4d's tunable table (tTopk: nch 0, pt1 8), as tune_i does. */
extern "C" void r4d_cfg_topk(const RadArgs* a, R4dTopkCfg* out) {
    out->rc  = 0;
    out->nch = (int)r4d_geti_or(a, "nch", 0);
    out->pt1 = (int)r4d_geti_or(a, "pt1", 8);
}
/* The skinny-GEMM tuning read (r4d_rows.cpp). Named only by the WMMA expert GEMM's adapter in the moe
 * port, which libxe registers no row for (step 4 rewrites it); zeros = libr4d's own rules. */
extern "C" void r4d_cfg_gemm(const RadArgs*, R4dGemmCfg* out) { *out = R4dGemmCfg{}; }
/* The 2-rank all-reduce's block cap (R4D_AR_MAX_BLOCKS); named by the fused quantiser's tp paths,
 * which one card never takes. */
int r4d_ar_max_blocks(void) { return 512; }

/* The recorded form of the 2D copy: sycl_ext_oneapi_graph does not record ext_oneapi_memcpy2d (it
 * throws), so while the queue records a tape the rows are copied by a kernel, 4 bytes an item when
 * every address, pitch and the width allow it. Byte for byte the same result. */
template <typename T>
static void copy2d_kernel(sycl::queue& q, uint8_t* d, const uint8_t* s, size_t dp, size_t sp, size_t w, size_t h) {
    const size_t n = w / sizeof(T);
    q.parallel_for(sycl::range<2>(h, n), [=](sycl::item<2> it) {
        const size_t r = it[0], c = it[1];
        reinterpret_cast<T*>(d + r * dp)[c] = reinterpret_cast<const T*>(s + r * sp)[c];
    });
}

/* libr4d's strided copy through the copy engines (r4d_ar_oneshot_2rank_exact.hip); here the
 * queue's 2D copy (a kernel while recording). Same-card only in milestone 1 (no peer access).
 * Returns 0 or -100 - error. */
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height, long stream) {
    if (!dst || !src || width < 0 || height < 0 || dpitch < width || spitch < width || !stream) return -101;
    if (!width || !height) return 0;
    try {
        sycl::queue& q = *reinterpret_cast<sycl::queue*>(stream);
        if (q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) {
            uint8_t* d = (uint8_t*)dst;
            const uint8_t* s = (const uint8_t*)src;
            if (((uintptr_t)d | (uintptr_t)s | (uintptr_t)dpitch | (uintptr_t)spitch | (uintptr_t)width) % 4 == 0)
                copy2d_kernel<uint32_t>(q, d, s, (size_t)dpitch, (size_t)spitch, (size_t)width, (size_t)height);
            else
                copy2d_kernel<uint8_t>(q, d, s, (size_t)dpitch, (size_t)spitch, (size_t)width, (size_t)height);
            return 0;
        }
        q.ext_oneapi_memcpy2d((void*)dst, (size_t)dpitch, (const void*)src, (size_t)spitch, (size_t)width,
                              (size_t)height);
    } catch (const sycl::exception&) {
        return -102;
    }
    return 0;
}
