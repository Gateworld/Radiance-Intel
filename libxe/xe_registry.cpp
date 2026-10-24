/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_registry.cpp -- libxe's kernel rows and plugin exports.
 *
 * libxe declares no schemas of its own: every op it implements is in libref's conventional
 * vocabulary, so libref's schema (which loads either way) is the one its rows bind to. The rows of a
 * ported libr4d unit are libr4d's own (libxe/tools/rows_from_r4d.py -> ports/<unit>.rows.inc), so
 * names, constraints and priorities match the card libr4d serves; the adapters are the port's.
 */
#include "rad_abi.h"
#include "r4d_plugin.h"   /* R4D_PRIO_*, as libr4d's rows use them */

#include <cstdint>

/* libr4d/r4d_rows.cpp's row helpers, verbatim */
#define ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }
#define NOTUNE nullptr, 0, nullptr

/* the ports' adapters and rows (libxe/tools/gen_registry.py) */
#define XE_ADAPTER(n) extern "C" int n(const RadArgs* a, RadStream s);
#include "ports/adapters.inc"

#define XE_ROWS_CONSTRAINTS
#include "ports/all_rows.inc"
#undef XE_ROWS_CONSTRAINTS

/* libxe's own kernels (step 4: matrix-unit ops libr4d does on WMMA, written for Xe2) */
extern "C" int xe_rad_gemm_i8a8_gemv(const RadArgs* a, RadStream s);
static const RadConstraint cXeGemvI8[] = {
    C_DIV("K", 128), C_EQ("group", 128), C_IN("dtype", "i8a8 i8a8h"),
};
extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* a, RadStream s);
extern "C" int xe_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream s);
static const RadConstraint cXeGemvBf16[] = { C_DIV("K", 8), C_IN("dtype", "bf16") };
static const RadConstraint cXeRouterGemm[] = {
    C_GE("M", 1), C_DIV("K", 8), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
    C_LE("top_k", 32), C_IN("dtype", "bf16"),
};

static const RadKernelInfo kKernels[] = {
#include "ports/all_rows.inc"
{ "xe_gemm_i8a8_gemv", "gemm_nt_q", "gemm",
  "the trunk's W8A8 product as a sub-group GEMV: exact int32 group sums, scaled per (row, 128 K)",
  "any M (4 activation rows a weight pass), any N; K a multiple of 128; the stored layout",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 (or f32) scale per (row, 128 K), bf16 C; "
  "i8a8h: the weight Hadamard-rotated at convert against had_quant_act_i8g's activation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvI8), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_i8a8_gemv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_gemm_nt_bf16_gemv", "gemm_nt", "gemm",
  "a bf16 weight's product as a sub-group GEMV, f32 accumulate, one rounding to y",
  "any M (4 rows a weight pass), any N; K a multiple of 8; the stored layout",
  "bf16 or f32 a, bf16 b, bf16 or f32 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvBf16), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_nt_bf16, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_router_gemm_topk_scatter_bf16", "router_gemm_topk_scatter", "moe",
  "the router GEMV (and the shared expert's gate column), then libr4d's router_topk_scatter",
  "any M; n_expert <= 512, top_k <= 32; K a multiple of 8",
  "bf16 a and router weight, the logits as the operand declares them, i32 ids, f32 weights",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeRouterGemm), NOTUNE,
  nullptr, nullptr, xe_rad_router_gemm_topk_scatter, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
};
static const int kNumKernels = (int)(sizeof(kKernels) / sizeof(kKernels[0]));

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libxe",
    "0.1.0",
    "SYCL kernels for Intel Xe2 (Arc Pro B70) on the Level Zero backend: the ops qwen4exp decode "
    "issues on one card",
    "xe2-bmg-g31 sycl-cpu"
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return kNumKernels; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < kNumKernels) ? &kKernels[i] : nullptr;
}
