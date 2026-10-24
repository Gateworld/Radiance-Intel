/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_support.cpp -- libr4d's cross-unit helpers the ported units call, for one card.
 */
#include <sycl/sycl.hpp>

#include "r4d.h"

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
