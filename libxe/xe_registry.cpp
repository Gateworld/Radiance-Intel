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
#include "../plugin/spechub/spechub.h"   /* the MTP coupling's shape hooks (owner note 47 item 2) */

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
extern "C" int xe_rad_gemm_w2a8_gemv(const RadArgs* a, RadStream s);
extern "C" int xe_w2a8_layout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                              RadLayout*);
extern "C" int xe_w2a8_relayout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                                void*, int64_t);
extern "C" int xe_w2a8_unrelayout(const RadParam*, int, int, const RadEncoding*, const int*, const void*, int64_t,
                                  const RadTensor*, int);
static const RadConstraint cXeGemvW2A8[] = { C_DIV("K", 128), C_EQ("group", 128), C_IN("dtype", "w2a8") };
extern "C" int xe_rad_gemm_nt_bf16(const RadArgs* a, RadStream s);
extern "C" int xe_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream s);
extern "C" int xe_rad_logits_gemm_bf16(const RadArgs* a, RadStream s);
static const RadConstraint cXeLogitsBf16[] = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") };
static const RadConstraint cXeGemvBf16[] = { C_DIV("K", 8), C_IN("dtype", "bf16") };
static const RadConstraint cXeRouterGemm[] = {
    C_GE("M", 1), C_DIV("K", 8), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
    C_LE("top_k", 32), C_IN("dtype", "bf16"),
};
extern "C" int xe_rad_logits_gemm_i8(const RadArgs* a, RadStream s);
static const RadConstraint cXeLogitsI8[] = { C_DIV("n_embd", 128), C_IN("dtype", "bf16") };
extern "C" int xe_rad_moe_gemm_w4p(const RadArgs* a, RadStream s);
extern "C" int xe_moe_planes_layout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                                    int, RadLayout*);
extern "C" int xe_moe_planes_relayout(const RadParam*, int, int, const RadEncoding*, const int*,
                                      const RadTensor*, int, void*, int64_t);
extern "C" int xe_moe_planes_unrelayout(const RadParam*, int, int, const RadEncoding*, const int*, const void*,
                                        int64_t, const RadTensor*, int);
static const RadConstraint cXeMoePlanes[] = {
    C_GE("M", 1), C_DIV("K", 128), C_GE("n_expert", 1), C_LE("n_expert", 65536), C_GE("top_k", 1),
    C_EQ("group", 128), C_IN("dtype", "w4nl64a8h"),
};
/* the MTP coupling (owner note 47 item 2): the rows' launches in xe_mtp_couple.cpp, the shape hooks the spechub
 * plugin's (plugin/spechub/spechub.h, compiled into libxe) */
extern "C" int xe_rad_mtp_couple_propose(const RadArgs* a, RadStream s);
extern "C" int xe_rad_mtp_couple_verify(const RadArgs* a, RadStream s);
static const RadConstraint cXeCouplePropose[] = {
    C_GE("M", 1), C_GE("n_vocab", 1), C_GE("round", 1), C_LE("max_rounds", 8),
};
static const RadConstraint cXeCoupleVerify[] = {
    C_GE("M", 1), C_GE("n_cand", 1), C_GE("n_vocab", 1),
};
extern "C" int xe_rad_attn_paged(const RadArgs* a, RadStream s);
extern "C" int64_t xe_rad_attn_paged_scratch(const RadArgs* a);
/* block_size is derived on attn_paged: no EQ here, so the page size is the model's declared one */
static const RadConstraint cXeAttnPaged[] = {
    C_DIV("head_dim", 128), C_LE("head_dim", 256), C_GE("gqa", 1), C_LE("gqa", 32),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3 bf16"),
};

/* milestone 2, stage 2: all_reduce for two ranks (xe_allreduce.cpp): peer stores, and the host-mailbox
 * twin for a pair without peer access (the engine selects by the device.p2p fact) */
extern "C" int xe_ar2_init(const RadParam*, int, int, int, void**);
extern "C" int xe_ar2_host_init(const RadParam*, int, int, int, void**);
extern "C" void xe_ar2_fini(void*);
extern "C" int xe_rad_ar_oneshot_2rank_exact(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_oneshot_2rank_exact_host(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_gather_2rank(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_gather_2rank_host(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_hc_write_2rank(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_hc_write_2rank_host(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_gather_hc_write_2rank(const RadArgs* a, RadStream s);
extern "C" int xe_rad_ar_gather_hc_write_2rank_host(const RadArgs* a, RadStream s);
extern "C" int64_t xe_ar_gather_hc_write_scratch(const RadArgs* a);
static const RadConstraint cXeArGHcW2[] = { C_EQ("world_size", 2), C_EQ("wire", 0), C_IN("dtype", "bf16"), C_LE("top_k", 32), { RAD_FACT_P2P, RAD_C_EQ, 1, nullptr } };
static const RadConstraint cXeArGHcW2Host[] = { C_EQ("world_size", 2), C_EQ("wire", 0), C_IN("dtype", "bf16"), C_LE("top_k", 32) };
static const RadConstraint cXeGather2[] = { C_EQ("world_size", 2), C_IN("dtype", "f32 fp32 i32 bf16 fp16"), { RAD_FACT_P2P, RAD_C_EQ, 1, nullptr } };
static const RadConstraint cXeGather2Host[] = { C_EQ("world_size", 2), C_IN("dtype", "f32 fp32 i32 bf16 fp16") };
static const RadConstraint cXeArHcW2[] = { C_EQ("world_size", 2), C_EQ("wire", 0), C_IN("dtype", "bf16"), { RAD_FACT_P2P, RAD_C_EQ, 1, nullptr } };
static const RadConstraint cXeArHcW2Host[] = { C_EQ("world_size", 2), C_EQ("wire", 0), C_IN("dtype", "bf16") };
#define C_P2P { RAD_FACT_P2P, RAD_C_EQ, 1, nullptr }
static const RadConstraint cXeAr2[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_P2P,
};
static const RadConstraint cXeAr2Host[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
};

static const RadKernelInfo kKernels[] = {
{ "xe_ar_oneshot_2rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot: each rank stores its payload into the peer's device scratch, a flag a "
  "block, f32 accumulate in ascending rank order (bit-identical across ranks)",
  "exactly 2 ranks, peer-accessible (one Level Zero context over both cards); in place or into y; "
  "32-bit words (an even bf16 count); graph-safe (a device sequence counter)",
  "bf16 / fp16 / fp32 payload, exact wire",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cXeAr2), NOTUNE,
  xe_ar2_init, xe_ar2_fini, xe_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_oneshot_2rank_exact_host", "all_reduce", "ar",
  "the same exchange through a pinned host-memory mailbox (no peer access needed)",
  "exactly 2 ranks; in place or into y; 32-bit words; graph-safe",
  "bf16 / fp16 / fp32 payload, exact wire",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT - 1,
  ROW(cXeAr2Host), NOTUNE,
  xe_ar2_host_init, xe_ar2_fini, xe_rad_ar_oneshot_2rank_exact_host, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_gather_2rank", "all_gather", "ar",
  "all-gather for two ranks over the all-reduce's peer-store mailbox: plain concatenation, or row-interleaved "
  "(`row`) as the vocab-parallel candidate merge reads",
  "exactly 2 ranks, peer-accessible; 32-bit words (message and row bytes multiples of 4); graph-safe",
  "f32 / i32 / bf16 / fp16 payload, a byte move",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cXeGather2), NOTUNE,
  xe_ar2_init, xe_ar2_fini, xe_rad_ar_gather_2rank, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_gather_2rank_host", "all_gather", "ar",
  "the same through the pinned host mailbox",
  "exactly 2 ranks; 32-bit words; graph-safe",
  "f32 / i32 / bf16 / fp16 payload, a byte move",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT - 1,
  ROW(cXeGather2Host), NOTUNE,
  xe_ar2_host_init, xe_ar2_fini, xe_rad_ar_gather_2rank_host, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_hc_write_2rank", "ar_hc_write", "norm",
  "the all-reduce of y (peer stores, exact) followed by libxe's hc_write on the reduced y: two launches, the "
  "bytes of the unfused pair",
  "exactly 2 ranks, peer-accessible, the exact wire (wire 0); a dense y",
  "bf16 y, inj and h",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cXeArHcW2), NOTUNE,
  xe_ar2_init, xe_ar2_fini, xe_rad_ar_hc_write_2rank, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_hc_write_2rank_host", "ar_hc_write", "norm",
  "the same through the pinned host mailbox",
  "exactly 2 ranks, the exact wire; a dense y",
  "bf16 y, inj and h",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT - 1,
  ROW(cXeArHcW2Host), NOTUNE,
  xe_ar2_host_init, xe_ar2_fini, xe_rad_ar_hc_write_2rank_host, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_gather_hc_write_2rank", "ar_gather_hc_write", "norm",
  "libxe's moe_gather into y, then the all-reduce of y (peer stores, exact) and hc_write: three launches, "
  "the unfused chain's bytes",
  "exactly 2 ranks, peer-accessible, the exact wire (wire 0); a dense y; scratch as moe_gather's",
  "bf16 expert rows, f32 expert weights, i32 sorted tokens, bf16 y, inj and h",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cXeArGHcW2), NOTUNE,
  xe_ar2_init, xe_ar2_fini, xe_rad_ar_gather_hc_write_2rank, xe_ar_gather_hc_write_scratch, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_ar_gather_hc_write_2rank_host", "ar_gather_hc_write", "norm",
  "the same through the pinned host mailbox",
  "exactly 2 ranks, the exact wire; a dense y; scratch as moe_gather's",
  "bf16 expert rows, f32 expert weights, i32 sorted tokens, bf16 y, inj and h",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT - 1,
  ROW(cXeArGHcW2Host), NOTUNE,
  xe_ar2_host_init, xe_ar2_fini, xe_rad_ar_gather_hc_write_2rank_host, xe_ar_gather_hc_write_scratch, nullptr, nullptr, nullptr, nullptr, nullptr },
#include "ports/all_rows.inc"
{ "xe_gemm_i8a8_gemv", "gemm_nt_q", "gemm",
  "the trunk's W8A8 product as a sub-group GEMV: exact int32 group sums, scaled per (row, 128 K)",
  "any M (4 activation rows a weight pass), any N; K a multiple of 128; the stored layout",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 (or f32) scale per (row, 128 K), bf16 C; "
  "i8a8h: the weight Hadamard-rotated at convert against had_quant_act_i8g's activation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvI8), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_i8a8_gemv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_gemm_w2a8_gemv", "gemm_nt_q", "gemm",
  "the MTP draft head's w2a8 product as a sub-group GEMV over the canonical planes: exact int32 group "
  "sums, the zero removed through quant_act_i8's group sums, scaled per (row, 128 K)",
  "any M (4 activation rows a weight pass), any N; K a multiple of 128, group 128; codes stored, "
  "scale and zero joined (abi/rad_planes.h)",
  "int8 A in the step order with an f32 scale a row and f32 group sums, u2 W with an f16 scale and a "
  "u8 or u4 zero per (row, 128 K), bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvW2A8), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_w2a8_gemv, nullptr,
  xe_w2a8_layout, xe_w2a8_relayout, nullptr, xe_w2a8_unrelayout, nullptr },
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
{ "xe_logits_gemm_bf16", "logits_gemm", "gemm",
  "the bf16 lm_head's product as the sub-group GEMV",
  "any M; n_embd a multiple of 8; the stored layout",
  "bf16 x and lm_head, bf16 or f32 logits",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeLogitsBf16), NOTUNE,
  nullptr, nullptr, xe_rad_logits_gemm_bf16, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_logits_gemm_i8", "logits_gemm", "vocab",
  "the int8 lm_head's product as a sub-group GEMV over libr4d's stored row (codes, the row's bf16 "
  "scales in its tail), f32 accumulate",
  "any M (4 rows a weight pass), any vocabulary; n_embd a multiple of 128",
  "bf16 x, i8 head with a bf16 scale per row per 128 columns, F32 LOGITS OUT",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LOGITS_FP8,
  ROW(cXeLogitsI8), NOTUNE,
  nullptr, nullptr, xe_rad_logits_gemm_i8, nullptr,
  r4d_rad_layout_i8_logits, r4d_rad_relayout_i8_logits, nullptr, r4d_rad_unrelayout_i8_logits, nullptr },
{ "xe_mtp_couple_propose", "mtp_couple_propose", "draft",
  "the linear-MTP coupling's draw: x ~ Q over the head's final top-8 (FP64 T1 softmax), one work-item a row; "
  "upstream's arithmetic (abi/rad_spechub.h), the spechub plugin's readers and shape hooks (owner note 47 item 2)",
  "M rows, n_vocab, round 1..max_rounds <= 8", "f32 scores, i32 ids, u32 wire rows",
  RAD_DOMAIN_DEVICE, 100,
  ROW(cXeCouplePropose), NOTUNE,
  nullptr, nullptr, xe_rad_mtp_couple_propose, nullptr, nullptr, nullptr, rad_mtp_couple_propose_shape, nullptr,
  nullptr },
{ "xe_mtp_couple_verify", "mtp_couple_verify", "draft",
  "the linear-MTP coupling's verify after sample_pick: the maximal coupling of the bound state over the surviving "
  "candidates in FP64 (accept with min(1, P/Q), else a residual draw), one work-item a row -- the serial form of "
  "spechub.hip's cooperative verify",
  "M rows, n_cand candidates, n_vocab", "f32 scores, i32 ids, u32 wire rows",
  RAD_DOMAIN_DEVICE, 100,
  ROW(cXeCoupleVerify), NOTUNE,
  nullptr, nullptr, xe_rad_mtp_couple_verify, nullptr, nullptr, nullptr, rad_mtp_couple_verify_shape, nullptr,
  nullptr },
{ "xe_moe_gemm_w4nl64_planes", "moe_gemm_q", "moe",
  "the w4nl64a8h experts read from their canonical planes, joined (abi/rad_planes.h), as a sub-group "
  "GEMV: the load copies the experts instead of rearranging them (owner note 4)",
  "any M; K a multiple of 128, group 128; two classes by parity",
  "E4M3 activation with an f32 scale per 128 of K; u4 codes into 16 f32 levels, an E4M3 scale per "
  "row per 64 of K times one f32; bf16 out",
  RAD_DOMAIN_DEVICE, 60,
  ROW(cXeMoePlanes), NOTUNE,
  nullptr, nullptr, xe_rad_moe_gemm_w4p, nullptr,
  xe_moe_planes_layout, xe_moe_planes_relayout, nullptr, xe_moe_planes_unrelayout, nullptr },
{ "xe_attn_paged_split", "attn_paged", "attn",
  "paged attention, split-KV: a work-group per (split, kv head, query row), a sub-group per query "
  "head, f32 online softmax, a merge launch over the splits",
  "head_dim 128 or 256, gqa <= 32, any block size, q_len, causal and window; made for decode and "
  "speculative verify (a long prefill re-reads the cache per query row)",
  "bf16 query, fp8-e4m3 kv cache with optional per-(seq, head) descales, or bf16; bf16 or f32 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cXeAttnPaged), NOTUNE,
  nullptr, nullptr, xe_rad_attn_paged, xe_rad_attn_paged_scratch, nullptr, nullptr, nullptr, nullptr, nullptr },
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
/* Schemas libxe declares WITHOUT a row (milestone 2, stage 2): libr4d's fused collectives, which the
 * hyperconnection block asks for under tensor parallel (arch/common/rad_block_hc.h). The builder
 * needs the op's schema to exist to hand back an unresolved handle (0), on which the architecture
 * emits the plain all_reduce + hc_write pair instead; with no schema at all declare stops ("op has
 * no declared schema", the TP2 declare of run 7). Field for field libr4d's (r4d_rows.cpp). */
#define P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
static const RadParamSpec pXeArHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("world_size"), P_STR("dtype"), P_INTO("wire"),
};
static const RadOperandSpec oXeArHcWrite[] = {
    { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pXeArGatherHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("top_k"), P_INT("world_size"), P_STR("dtype"),
    P_STRO("act"), P_INTO("wire"),
};
static const RadOperandSpec oXeArGatherHcWrite[] = {
    { "y_expert", RAD_OPD_IN, 0 }, { "expert_w", RAD_OPD_IN, 0 }, { "sorted_tok", RAD_OPD_IN, 0 },
    { "shared", RAD_OPD_IN, 1 },   { "shared_gate", RAD_OPD_IN, 1 },
    { "y", RAD_OPD_INOUT, 0 },     { "inj", RAD_OPD_IN, 0 },       { "h", RAD_OPD_INOUT, 0 },
};
static const RadOpSchema kSchemas[] = {
    { "ar_hc_write", ROW(pXeArHcWrite), ROW(oXeArHcWrite),
      "libr4d's fused all-reduce + gated residual write; declared here so the op is known and "
      "unresolved on the Intel build, where the architecture then emits all_reduce + hc_write" },
    { "ar_gather_hc_write", ROW(pXeArGatherHcWrite), ROW(oXeArGatherHcWrite),
      "libr4d's fused moe_gather + all-reduce + hc_write; declared, not served (see ar_hc_write)" },
};
static const int kNumSchemas = (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
extern "C" int rad_kernel_schema_count(void) { return kNumSchemas; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < kNumSchemas) ? &kSchemas[i] : nullptr;
}
extern "C" int rad_kernel_count(void) { return kNumKernels; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < kNumKernels) ? &kKernels[i] : nullptr;
}
