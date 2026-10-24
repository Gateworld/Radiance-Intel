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
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }
#define NOTUNE nullptr, 0, nullptr

/* the ports' adapters */
#define XE_ADAPTER(n) extern "C" int n(const RadArgs* a, RadStream s);
XE_ADAPTER(r4d_rad_embed_lookup) XE_ADAPTER(r4d_rad_embed_lookup_q) XE_ADAPTER(r4d_rad_gather_rows)
XE_ADAPTER(r4d_rad_gather_rows_u8) XE_ADAPTER(r4d_rad_scatter_rows_u8) XE_ADAPTER(r4d_rad_scatter_rows_bf16)
XE_ADAPTER(r4d_rad_cast) XE_ADAPTER(r4d_rad_rmsnorm) XE_ADAPTER(r4d_rad_add_rmsnorm) XE_ADAPTER(r4d_rad_rmsnorm_p)
XE_ADAPTER(r4d_rad_add) XE_ADAPTER(r4d_rad_mul) XE_ADAPTER(r4d_rad_mul_weight) XE_ADAPTER(r4d_rad_sigmoid)
XE_ADAPTER(r4d_rad_fill) XE_ADAPTER(r4d_rad_logit_softcap) XE_ADAPTER(r4d_rad_scale) XE_ADAPTER(r4d_rad_silu_mul)
XE_ADAPTER(r4d_rad_gelu_mul) XE_ADAPTER(r4d_rad_rope) XE_ADAPTER(r4d_rad_kv_store) XE_ADAPTER(r4d_rad_kv_store_fp8)
XE_ADAPTER(r4d_rad_kv_store_fp8v4) XE_ADAPTER(r4d_rad_kv_store_k6v4)

#define XE_ROWS_CONSTRAINTS
#include "ports/r4d_model_bf16.rows.inc"
#undef XE_ROWS_CONSTRAINTS

static const RadKernelInfo kKernels[] = {
#include "ports/r4d_model_bf16.rows.inc"
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
