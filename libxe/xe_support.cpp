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
/* r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip's geometry (KD, VD, BT): head 128 x 128, chunk 64 */
extern "C" void r4d_gdn_dims(int* head_k, int* head_v, int* chunk) {
    if (head_k) *head_k = 128;
    if (head_v) *head_v = 128;
    if (chunk)  *chunk  = 64;
}
/* The 2-rank all-reduce's block cap (R4D_AR_MAX_BLOCKS); named by the fused quantiser's tp paths,
 * which one card never takes. */
int r4d_ar_max_blocks(void) { return 512; }

/* libr4d's strided copy through the copy engines (r4d_ar_oneshot_2rank_exact.hip); here the
 * queue's 2D copy. Same-card only in milestone 1 (no peer access). Returns 0 or -100 - error. */
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height, long stream) {
    if (!dst || !src || width < 0 || height < 0 || dpitch < width || spitch < width || !stream) return -101;
    if (!width || !height) return 0;
    try {
        reinterpret_cast<sycl::queue*>(stream)->ext_oneapi_memcpy2d((void*)dst, (size_t)dpitch, (const void*)src,
                                                                    (size_t)spitch, (size_t)width, (size_t)height);
    } catch (const sycl::exception&) {
        return -102;
    }
    return 0;
}
