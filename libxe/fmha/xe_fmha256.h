/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 167, owner note 168 item 1; run 169, owner note 173).
 *
 * The flash-attention forward of sycl-tla (Intel's CUTLASS, BSD-3-Clause, third_party/sycl-tla) at this model's
 * head size with the per-query page selection (QSA) applied as a mask: one sequence, bf16 Q / K / V / O, causal
 * (the diagonal at the end of the context), no paging -- the context's K and V dense in memory
 * (xe_attn_qsa.cpp gathers them from the cache). Compiled only when the sycl-tla tree is present (RAD_XE_FMHA,
 * libxe/CMakeLists.txt); GPU only.
 *
 * Run 169: the context length comes from DEVICE memory (`ctx_dev`; `ctx` is then its upper bound, the dense
 * buffers' extent), and the launch goes through the engine's launch path when the caller hands over the recorder
 * hooks (libxe's resolved rad_l0k_rec_device / rad_l0k_rec_launch: a module loaded with dlopen sees no weak symbol
 * of libxe's), so the kernel is ordered behind the engine's native command lists like every libxe kernel. */
#pragma once
#include <stddef.h>
#include <stdint.h>

struct XeFmhaArgs {
    const uint16_t* q;  long q_rs, q_hs;       /* [T][heads][256] bf16: the row and head strides in elements */
    const uint16_t* k;  long k_hs;             /* [kv_heads][ctx][256] bf16: the head stride in elements (rows 256 apart) */
    const uint16_t* v;  long v_hs;
    uint16_t* out;      long o_rs, o_hs;       /* [T][heads][256] bf16 */
    const uint32_t* sel_bits; int sel_words;   /* [T padded to 256][sel_words]: the page selection by block index */
    int T, ctx, heads, kv_heads;               /* ctx: the context length, or its upper bound when ctx_dev is set */
    float scale;
    const int32_t* ctx_dev;                    /* device memory: the context length the kernel reads (null: `ctx`) */
};

/* The engine's recorder hooks as libxe resolved them (abi/rad_l0_rec.h); either null means the plain SYCL submit. */
struct RadL0RecKernel;
struct XeFmhaHooks {
    int (*rec_device)(const void* queue);
    int (*rec_launch)(const void* queue, const RadL0RecKernel* kernel, const uint32_t grid[3], const uint32_t block[3],
                      const void* arg0, size_t arg0_size, size_t lds);
};

/* 0 on success; -1 bad arguments, -2 the kernel refuses the shape, -3 the launch threw. `queue` is a sycl::queue*.
 * Exported by libxe_fmha.so (its own module, the B70 target alone); libxe loads it beside itself with dlopen. */
extern "C" int xe_fmha256_run(void* queue, const XeFmhaArgs* a, const XeFmhaHooks* hooks);
typedef int (*xe_fmha256_run_fn)(void* queue, const XeFmhaArgs* a, const XeFmhaHooks* hooks);
