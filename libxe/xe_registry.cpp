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
#include <cstdlib>   /* std::getenv: RADIANCE_XE_GATE_QUANT */

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

extern "C" int xe_rad_hc_write_read(const RadArgs* a, RadStream s);   /* ports/r4d_hc_bf16.cpp, run 206 */
extern "C" int r4d_rad_gate_quant_fp8(const RadArgs* a, RadStream s);  /* ports/r4d_fused_quant_fp8.cpp, run 206 */
static const RadConstraint cXeGateQuant[] = { C_DIV("n", 4), C_EQ("group", 128), C_IN("dtype", "bf16") };
extern "C" int r4d_rad_gated_had_quant_i8g(const RadArgs* a, RadStream s);  /* ports/r4d_fused_quant_fp8.cpp */
static const RadConstraint cXeGatedHad[] = { C_DIV("n", 128), C_EQ("group", 128), C_IN("dtype", "bf16") };
extern "C" int r4d_rad_gated_quant_pair(const RadArgs* a, RadStream s);  /* ports/r4d_fused_quant_fp8.cpp, run 273 */
static const RadConstraint cXeGatedQuantPair[] = { C_DIV("n", 128), C_DIV("n2", 4), C_EQ("group", 128),
                                                   C_IN("dtype", "bf16") };
/* run 273 (pair 3 of the decode step's adjacent launches): libr4d's gated-attention prologue, ported in milestone 1
 * (ports/r4d_qk_norm_rope_gate.cpp) and left unlisted only because libref declares no schema for it; its constraints
 * are libr4d's cQkNormRopeGate field for field */
extern "C" int r4d_rad_qk_norm_rope_gate(const RadArgs* a, RadStream s);  /* ports/r4d_qk_norm_rope_gate.cpp */
static const RadConstraint cXeQkPrologue[] = { C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32) };
/* run 273 (pair 2): router_topk_scatter's constraints (cRouterTopK) at the forms instantiated -- M <= 16 rows, more
 * than 256 experts -- plus the quantiser's width and group */
extern "C" int r4d_rad_router_topk_scatter_quant(const RadArgs* a, RadStream s);  /* ports/r4d_moe.cpp */
static const RadConstraint cXeRouterQuant[] = { C_GE("M", 1), C_LE("M", 16), C_GE("n_expert", 257),
                                                C_LE("n_expert", 512), C_GE("top_k", 1), C_LE("top_k", 32),
                                                C_IN("dtype", "bf16"), C_DIV("n", 128), C_EQ("group", 128) };
/* the same pairing behind the fused router GEMM op the decode step takes (xe_gemv_bf16.cpp): its constraints plus the
 * paired launch's */
extern "C" int xe_rad_router_gemm_topk_scatter_quant(const RadArgs* a, RadStream s);  /* xe_gemv_bf16.cpp */
static const RadConstraint cXeRouterGemmQuant[] = { C_DIV("K", 8), C_GE("M", 1), C_LE("M", 16), C_GE("n_expert", 257),
                                                    C_LE("n_expert", 512), C_GE("top_k", 1), C_LE("top_k", 32),
                                                    C_IN("dtype", "bf16"), C_DIV("n", 128), C_EQ("group", 128) };
/* run 285 (group A of owner note 232's audit): the fused router op above with the shared expert's gate_up as an int8 set
 * in its GEMV launch (xe_gemv_i8.cpp); the router op's constraints plus the paired GEMV's K */
extern "C" int xe_rad_router_gemm_topk_scatter_quant_sh(const RadArgs* a, RadStream s);  /* xe_gemv_i8.cpp */
static const RadConstraint cXeRouterGemmQuantSh[] = { C_DIV("K", 128), C_GE("M", 1), C_LE("M", 16), C_GE("n_expert", 257),
                                                      C_LE("n_expert", 512), C_GE("top_k", 1), C_LE("top_k", 32),
                                                      C_IN("dtype", "bf16"), C_DIV("n", 128), C_EQ("group", 128),
                                                      C_GE("N_sh", 1) };
/* run 274 (pairs 4 / 5 of the decode step's adjacent launches): several projections of one activation in one GEMV
 * launch (xe_gemv_i8.cpp); decode rows only (the GEMV path; the matrix-unit forms serve the prefill chunks) */
extern "C" int xe_rad_gemm_nt_q_x3(const RadArgs* a, RadStream s);          /* xe_gemv_i8.cpp */
extern "C" int xe_rad_gemm_nt_q_bf16_pair(const RadArgs* a, RadStream s);   /* xe_gemv_i8.cpp */
static const RadConstraint cXeGemvMulti[] = { C_GE("M", 1), C_LE("M", 32), C_DIV("K", 128), C_EQ("group", 128),
                                              C_IN("dtype", "i8a8") };
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
/* format B (owner notes 131 / 134 item 3, run 114): xe_moe_i4i8.cpp */
extern "C" int xe_rad_moe_gemm_i4i8(const RadArgs* a, RadStream s);
static const RadConstraint cXeMoeI4I8[] = {
    C_GE("M", 1), C_DIV("K", 128), C_GE("n_expert", 1), C_LE("n_expert", 65536), C_GE("top_k", 1),
    C_EQ("group", 128), C_IN("dtype", "w4g64i8h"),
};
/* format C (owner note 147, run 116): xe_moe_i4i8.cpp, over the w4nl64a8h planes' layout; its instance is the tail
 * skip's (owner note 162 item 1, run 160: RADIANCE_XE_TAIL_SKIP) */
extern "C" int xe_rad_moe_gemm_w4nl_i8(const RadArgs* a, RadStream s);
extern "C" int64_t xe_rad_moe_gemm_w4nl_i8_scratch(const RadArgs* a);   /* run 175: RADIANCE_XE_I4I8_DPAS_PRE's words */
extern "C" int xe_tskip_init(const RadParam*, int, int, int, void**);
extern "C" void xe_tskip_fini(void*);
static const RadConstraint cXeMoeW4nlI8[] = {
    C_GE("M", 1), C_DIV("K", 128), C_LE("K", 4096), C_GE("n_expert", 1), C_LE("n_expert", 65536), C_GE("top_k", 1),
    C_EQ("group", 128), C_IN("dtype", "w4nl64i8h"),
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
/* Run 111 (owner note 135 item 3): GPTQ's calibration Gram on the card -- libr4d's gram_accum unit ported
 * (ports/r4d_gram_accum_fp8.cpp, hipport.py), its row written here by hand with libr4d's constraints */
extern "C" int r4d_rad_gram_accum(const RadArgs* a, RadStream s);
static const RadConstraint cXeGramAccum[] = { C_DIV("n", 4), C_EQ("group", 128), C_IN("dtype", "fp8a8") };
extern "C" int xe_rad_attn_paged(const RadArgs* a, RadStream s);
extern "C" int64_t xe_rad_attn_paged_scratch(const RadArgs* a);
/* run 167 (owner note 168 item 1): a sparse prefill pass's attention as one launch on the matrix units */
extern "C" int xe_rad_attn_qsa_fmha(const RadArgs* a, RadStream s);
extern "C" int64_t xe_rad_attn_qsa_fmha_scratch(const RadArgs* a);
static const RadConstraint cXeAttnQsa[] = { C_EQ("head_dim", 256), C_GE("gqa", 1), C_LE("gqa", 32) };
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
/* run 206 (owner notes 200 item 3 / 201 item 4, the fusion map's top pair): one connection's hc_write folded into
 * the next connection's hc_read -- the architecture declares it under RADIANCE_HC_WRITE_FUSE=1; hc_read_e4m3's
 * constraints, scratch and weight layout (the read's own weights), its schema below */
{ "xe_hc_write_read_e4m3", "hc_write_read", "norm",
  "hc_write (h[s] += inj[s] * y) and the next hc_read in one op: the write rides the read's norm launch "
  "(hc_v2_write_norm_kernel) on the plain V2 form, and runs as hc_write first on any other form",
  "hc_read_e4m3's shapes; y and inj the writing connection's [M, n] and [M, hc]",
  "as hc_read_e4m3, plus bf16 y and inj in",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcReadE4m3), NOTUNE,
  nullptr, nullptr, xe_rad_hc_write_read, r4d_rad_hc_read_scratch,
  xe_hc_layout_e4m3, xe_hc_relayout_e4m3,
  nullptr, xe_hc_unrelayout_e4m3 },
{ "xe_gram_accum", "gram_accum", "quant",
  "h += x^T x over an E4M3 activation, in f32 -- the Gram matrix a GPTQ packer needs (libr4d's unit, ported)",
  "n a multiple of 4 and group 128 (a 64-wide output tile lies inside one scale group); any M; the accumulator "
  "is READ as well as written",
  "E4M3 plus an f32 scale per (row, 128 columns) in; an f32 [n][n] accumulator in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGramAccum), NOTUNE,
  nullptr, nullptr, r4d_rad_gram_accum, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
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
{ "xe_moe_gemm_i4i8", "moe_gemm_q", "moe",
  "format B's experts (w4g64i8h) on the int8 path: exact int32 sums of int4 codes x int8 activations per 64 of K "
  "by DP4A, scaled once a group, read from the canonical planes (no relayout)",
  "any M; K a multiple of 128, group 128; two classes by parity",
  "i8 activation with an f32 scale per 128 of K (had_quant_act_i8g's); i4 codes with a bf16 scale per row per 64 "
  "of K, rotated at convert; bf16 out",
  RAD_DOMAIN_DEVICE, 60,
  ROW(cXeMoeI4I8), NOTUNE,
  nullptr, nullptr, xe_rad_moe_gemm_i4i8, nullptr,
  nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_moe_gemm_w4nl_i8", "moe_gemm_q", "moe",
  "format C: the w4nl64a8h experts' joined planes (codes,table; scale,scale.1) against int8 activations -- each "
  "entry's 16 levels rounded to int8 at its amax, amax / 127 into the scale, exact int32 sums per 64 of K by DP4A",
  "any M; K a multiple of 128 up to 4096, group 128; two classes by parity",
  "i8 activation with an f32 scale per 128 of K (had_quant_act_i8g's); u4 codes into 16 f32 levels, an E4M3 scale "
  "per row per 64 of K times one f32; bf16 out",
  RAD_DOMAIN_DEVICE, 60,
  ROW(cXeMoeW4nlI8), NOTUNE,
  xe_tskip_init, xe_tskip_fini, xe_rad_moe_gemm_w4nl_i8, xe_rad_moe_gemm_w4nl_i8_scratch,
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
{ "xe_attn_qsa_fmha", "attn_qsa_fmha", "attn",
  "a sparse (QSA) prefill pass's attention in one launch on the matrix units: the sequence's page table inverted, "
  "each row's page list a bitmap, the context's K / V gathered dense from the cache, then sycl-tla's "
  "flash-attention forward with the bitmap as a mask (xe_attn_qsa.cpp, fmha/)",
  "head_dim 256, one sequence of >= 1 rows, causal, no window; GPU only (2D block loads); a tolerance form "
  "(position-order sums, bf16 P) judged by the KL replay",
  "bf16 query and out, fp8-e4m3 or bf16 kv cache without descales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cXeAttnQsa), NOTUNE,
  nullptr, nullptr, xe_rad_attn_qsa_fmha, xe_rad_attn_qsa_fmha_scratch, nullptr, nullptr, nullptr, nullptr, nullptr },
};
static const int kNumKernels = (int)(sizeof(kKernels) / sizeof(kKernels[0]));

/* The fusion map's pairs that are libr4d's own fused ops (runs 206-207): each row, and its schema below, is listed only
 * under its flag -- the architecture asks rad_op_declared first, so without the flag it keeps today's ops. Same order
 * here and in kFlaggedSchemas. */
/* the default since run 207 (owner, note 202: the fusion map's exact group); =0 opts out */
static bool env_on(const char* k) { const char* e = std::getenv(k); return !(e && e[0] == '0'); }
static bool xe_gate_quant_on() { static const bool v = env_on("RADIANCE_XE_GATE_QUANT"); return v; }
static bool xe_gated_had_on() { static const bool v = env_on("RADIANCE_XE_GATED_HAD"); return v; }
/* run 273 (owner note 214 item 3, the decode step's adjacent-launch pair 1): opt-in, =1 lists it */
/* the default since run 279 (owner note 215: same image 301, M388 87.14 vs M387 85.55 = +1.9 % single user, the rotation
 * H152 vs H153 +1.6-1.8 % with every prompt length up, byte-identical on the B70 in QK1 / RQ1, M389's kernel table);
 * =0 opts out, as the run-207 set's */
static bool xe_gated_quant_pair_on() { static const bool v = env_on("RADIANCE_XE_GATED_QUANT_PAIR"); return v; }
/* run 273 (pair 3): the fused q / k norm + rope prologue; the default since run 279 (note 215), =0 opts out */
static bool xe_qk_prologue_on() { static const bool v = env_on("RADIANCE_XE_QK_PROLOGUE"); return v; }
/* run 273 (pair 2): the router's selection + sort and the block input's quantiser as one launch; the default since
 * run 279 (note 215), =0 opts out */
static bool xe_router_quant_pair_on() { static const bool v = env_on("RADIANCE_XE_ROUTER_QUANT_PAIR"); return v; }
/* run 274 (pairs 4 / 5): the paired GEMV launches; the default since run 279 (owner note 217: same image 302, M392
 * 88.50 vs M393 87.18 = +1.5 % single user, the rotation H154 vs H152 +1.0-1.2 % with every prompt length up,
 * byte-identical on the B70 in GM1, M391's kernel table, the flag-off kernels unchanged in M394); =0 opts out */
static bool xe_gemv_multi_on() { static const bool v = env_on("RADIANCE_XE_GEMV_MULTI"); return v; }
/* run 285 (owner note 232's audit, groups A and B): the fused router op with the shared expert's gate_up in its GEMV
 * launch (A, j 8) and the gated attention's x3 with the QSA indexer's qk projection in one launch (B, j 9). THE DEFAULT
 * since run 294 (owner note 234: GSH1 byte-identical at M 1 / 4 / 8 / 12 for both groups; M404 / M405 91.06 / 91.01 vs
 * M402 88.99 single-user, +2.3 %; the rotation H157 vs H158 +2.0 % on median / mean / aggregate, texts 11 / 11):
 * RADIANCE_XE_GEMV_MULTI2 unset or =1 lists both, =0 opts out (today's separate launches), =a / =b one group alone */
static int xe_gemv_multi2_mode() {
    static const int m = [] {
        const char* e = std::getenv("RADIANCE_XE_GEMV_MULTI2");
        if (!e) return 3;
        return e[0] == '0' ? 0 : e[0] == 'a' ? 1 : e[0] == 'b' ? 2 : 3;
    }();
    return m;
}
static bool xe_gemv_multi2_on() { return (xe_gemv_multi2_mode() & 1) != 0; }
static bool xe_gemv_multi2b_on() { return (xe_gemv_multi2_mode() & 2) != 0; }
extern "C" int xe_rad_gemm_nt_q_x3_bf16(const RadArgs* a, RadStream s);   /* xe_gemv_i8.cpp (group B) */
static const RadKernelInfo kFlagged[] = {
/* pair 4 (RADIANCE_XE_GATE_QUANT=1): libr4d's gate_quant_fp8, its port's adapter -- the gated attention's sigmoid, mul
 * and quantiser as one pass, the sigmoid rounded to bf16 before the multiply as the unfused chain's buffer does */
{ "gate_quant_fp8", "gate_quant_fp8", "quant",
  "an elementwise gated product and the activation quantiser in one pass, with the gate in its own operand",
  "n a multiple of 4; gate and x both [M][n], their own pitches; group 128",
  "bf16 in, E4M3 or int8 codes with an f32 scale per (row, 128 columns), an optional bf16 out; the codes over the "
  "bf16-rounded product",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGateQuant), NOTUNE,
  nullptr, nullptr, r4d_rad_gate_quant_fp8, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* pair 2 (RADIANCE_XE_GATED_HAD=1, run 207): libr4d's gated_had_quant_i8g, its port's adapter -- format B / C's routed
 * experts' silu x up (bf16-rounded), the 128-wide Hadamard and the int8 quantiser in one pass, for the
 * silu_mul + had_quant_act_i8g pair the MoE block emits today (rad_block_moe_fp8.h, op_egq8) */
{ "gated_had_quant_i8g", "gated_had_quant_i8g", "quant",
  "an elementwise gated product, a 128-wide Hadamard along n and the int8 activation quantiser in one pass",
  "n a multiple of 128; the gated operand [M][2n], gate then up; group 128",
  "bf16 in, int8 codes with an f32 scale per (row, 128 columns), an optional bf16 out (unrotated); the codes over the "
  "rotated bf16-rounded product -- had_quant_act_i8g's",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGatedHad), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_had_quant_i8g, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* the decode step's pair 1 (RADIANCE_XE_GATED_QUANT_PAIR=1, run 273, owner note 214 item 3): the routed experts'
 * gated_had_quant_i8g and the shared expert's gated_quant_fp8 (int8 codes), the two launches a MoE layer issues back
 * to back over different buffers, as one launch whose work-groups are split by row set; each row's arithmetic is its
 * own kernel's (r4d_fused_quant_fp8.cpp, gated_quant_row), so the bytes are the two launches' */
{ "gated_quant_pair", "gated_quant_pair", "quant",
  "two independent gated products quantised in one launch: the first operand set's rows rotated to int8 codes "
  "(gated_had_quant_i8g's), the second's to plain int8 codes (gated_quant_fp8's at int8)",
  "n a multiple of 128 (the rotated set), n2 of 4; each gated operand [M_i][2 n_i], gate then up; group 128; silu",
  "bf16 in, int8 codes with an f32 scale per (row, 128 columns) and an optional bf16 out, per set",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGatedQuantPair), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_quant_pair, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* the decode step's pair 3 (RADIANCE_XE_QK_PROLOGUE=1, run 273): libr4d's qk_norm_rope_gate row, text for text
 * (libr4d/r4d_rows.cpp), over the port's adapter -- the gated attention asks for it at <= 64 rows
 * (rad_block_attn_gated_fp8.h) and issues q norm, k norm, rope q and rope k when it is not declared */
{ "qk_norm_rope_gate", "qk_norm_rope_gate", "attn",
  "the gated-attention QKV preamble in one pass: split [q|gate] per head, QK RMS norm over the "
  "head, partial NeoX RoPE and the gate copy",
  "head_dim 64, 128 or 256; rotary_dim even and <= head_dim, with rotary_dim/2 a power-of-two "
  "multiple of head_dim/32 so the RoPE pair is one shuffle; NeoX style; bf16; n_head may be 0, "
  "which is the K-only form and serves an input with no gate interleaved",
  "bf16 in and out; the norm output is rounded to bf16 before the rotation, as the unfused "
  "reference does by storing it",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeQkPrologue), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* the decode step's pair 2 (RADIANCE_XE_ROUTER_QUANT_PAIR=1, run 273): router_topk_scatter and the block input's
 * had_quant_act_i8g, the two launches a MoE layer issues back to back over the same input, as one grid -- work-group 0
 * the router's body unchanged, the rest the quantiser's rows (r4d_moe.cpp, router_topk_scatter_quant_kernel) */
{ "router_topk_scatter_quant", "router_topk_scatter_quant", "moe",
  "the router's top-k and the stable counting sort that follows it, with the block input's rotated int8 quantiser "
  "(had_quant_act_i8g) over the same rows in the same launch",
  "n_expert in 257 .. 512, top_k <= 32, M <= 16 rows (M*top_k <= 1024, the single-workgroup sort); n a multiple of "
  "128, group 128; bf16 logits and weights, i32 everything else of the router's",
  "bf16 in, bf16 weights + i32 ids, sorted slots, offsets and counts out; bf16 input, int8 codes with an f32 scale "
  "per (row, 128 columns) out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeRouterQuant), NOTUNE,
  nullptr, nullptr, r4d_rad_router_topk_scatter_quant, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* the same under the fused router GEMM op (router_gemm_topk_scatter) the decode step takes: the GEMV pair, then the
 * paired launch above (xe_gemv_bf16.cpp, xe_rad_router_gemm_topk_scatter_quant); the same flag */
{ "xe_router_gemm_topk_scatter_quant", "router_gemm_topk_scatter_quant", "moe",
  "the router GEMV (and the shared expert's gate column), then the router's top-k, the stable sort and the block "
  "input's rotated int8 quantiser (had_quant_act_i8g) in one launch",
  "M <= 16 rows; n_expert in 257 .. 512, top_k <= 32; K a multiple of 8; n a multiple of 128, group 128",
  "bf16 a and router weight, the logits as the operand declares them, i32 ids, f32 weights; bf16 input, int8 codes "
  "with an f32 scale per (row, 128 columns) out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeRouterGemmQuant), NOTUNE,
  nullptr, nullptr, xe_rad_router_gemm_topk_scatter_quant, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* the decode step's pairs 5 and 4 (RADIANCE_XE_GEMV_MULTI=1, run 274): three int8 projections of one activation (the
 * gated attention's q|gate, k, v), and an int8 projection with a bf16 one over the two planes of one activation (the
 * GDN block's in-projection and its a|b), each as one GEMV launch whose rows are the standalone kernels' rows
 * (xe_gemv_i8.cpp); a form the paired kernels are not built for runs as the separate launches */
{ "xe_gemm_nt_q_x3", "gemm_nt_q_x3", "gemm",
  "three int8 GEMVs over one int8 activation in one launch: y_i = a @ b_i^T with the per-(row, 128) scales of each",
  "M <= 32 rows (the decode GEMV path); K a multiple of 128; bf16 weight scales; the NX cross-lane form",
  "int8 a with f32 scales, int8 weights with bf16 scales, bf16 y; each row the standalone GEMV's bits",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvMulti), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_nt_q_x3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
{ "xe_gemm_nt_q_bf16_pair", "gemm_nt_q_bf16_pair", "gemm",
  "an int8 GEMV over an activation's int8 codes and a bf16 GEMV over its bf16 plane in one launch",
  "M <= 32 rows; K a multiple of 128 (and of 8 for the bf16 set); bf16 weight scales; the NX form",
  "int8 a with f32 scales and int8 weights with bf16 scales for the first set; bf16 x, weight and y for the second",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvMulti), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_nt_q_bf16_pair, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* group A of owner note 232's audit (RADIANCE_XE_GEMV_MULTI2=1, run 285): the fused router op with the shared expert's
 * gate_up in its GEMV launch -- the router rows, the gate row and the gate_up rows as one grid (xe_gemv_i8.cpp,
 * gemv_i8a8_bf16x2_kernel), each row its standalone kernel's; then the selection + sort and the quantiser as the fused
 * op's launch. The MoE block issues it at the router's slot and skips the shared gate_up's own launch
 * (rad_block_moe_fp8.h, op_rtq3) */
{ "xe_router_gemm_topk_scatter_quant_sh", "router_gemm_topk_scatter_quant_sh", "moe",
  "the router GEMV, the shared expert's gate column and the shared expert's int8 gate_up in one launch, then the "
  "router's top-k, the stable sort and the block input's rotated int8 quantiser in one launch",
  "M <= 16 rows; n_expert in 257 .. 512, top_k <= 32; K a multiple of 128; n a multiple of 128, group 128; bf16 weight "
  "scales on the gate_up (the NX cross-lane form; any other form runs the fused op and the gate_up as two launches)",
  "bf16 a, router and gate weights, bf16 logits and gate; int8 codes with f32 scales and int8 weights with bf16 scales "
  "for the gate_up, bf16 y; i32 ids, bf16 weights, int8 codes with an f32 scale per (row, 128 columns) out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeRouterGemmQuantSh), NOTUNE,
  nullptr, nullptr, xe_rad_router_gemm_topk_scatter_quant_sh, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
/* group B of owner note 232's audit (RADIANCE_XE_GEMV_MULTI2=1 or b, run 285): the gated attention's q|gate, k, v (the
 * x3) and the QSA indexer's qk projection over the same connection's two planes as one GEMV launch (xe_gemv_i8.cpp,
 * gemv_i8a8_x3_bf16_kernel), each row its standalone kernel's; the plugin declares and issues it (qwen4exp_fp8.cpp) and
 * both blocks skip their own launches */
{ "xe_gemm_nt_q_x3_bf16", "gemm_nt_q_x3_bf16", "gemm",
  "three int8 GEMVs over one int8 activation and a bf16 GEMV over its bf16 plane in one launch",
  "M <= 32 rows (the decode GEMV path); K a multiple of 128 (and of 8 for the bf16 set); bf16 weight scales; the NX form",
  "int8 a with f32 scales, int8 weights with bf16 scales, bf16 y for the three sets; bf16 x, weight and y for the fourth",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cXeGemvMulti), NOTUNE,
  nullptr, nullptr, xe_rad_gemm_nt_q_x3_bf16, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
};
static bool flagged_on(int j) {
    return j == 0 ? xe_gate_quant_on() : j == 1 ? xe_gated_had_on() : j == 2 ? xe_gated_quant_pair_on()
                                        : j == 3 ? xe_qk_prologue_on() : j <= 5 ? xe_router_quant_pair_on()
                                        : j <= 7 ? xe_gemv_multi_on()   /* j 6 and 7 */
                                        : j == 8 ? xe_gemv_multi2_on()  /* j 8: group A */
                                                 : xe_gemv_multi2b_on(); /* j 9: group B */
}
static constexpr int kNumFlagged = (int)(sizeof(kFlagged) / sizeof(kFlagged[0]));
/* the j-th listed flagged entry (counting only those on), or -1 */
static int flagged_index(int j) {
    for (int k = 0; k < kNumFlagged; ++k)
        if (flagged_on(k) && j-- == 0) return k;
    return -1;
}
static int flagged_count() {
    int c = 0;
    for (int k = 0; k < kNumFlagged; ++k) c += flagged_on(k) ? 1 : 0;
    return c;
}

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
/* hc_write_read (run 206): hc_read's params and operands (libr4d's pHcRead / oHcRead, field for field), `h` read and
 * written, then the writing connection's y and inj */
#define P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }
static const RadParamSpec pXeHcWriteRead[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"), P_F64("eps"), P_STR("dtype"),
    P_F64O("wadd"), P_INTO("inject"), P_INTO("group"), P_INTO("rotate"), P_STRO("mix"),
};
static const RadOperandSpec oXeHcWriteRead[] = {
    { "h",        RAD_OPD_INOUT,  0 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "mix_down", RAD_OPD_WEIGHT, 0 },
    { "mix_up",   RAD_OPD_WEIGHT, 0 },
    { "inject_w", RAD_OPD_WEIGHT, 1 },
    { "x",        RAD_OPD_OUT,    0 },
    { "inj",      RAD_OPD_OUT,    1 },
    { "q",        RAD_OPD_OUT,    1 },
    { "scale",    RAD_OPD_OUT,    1 },
    { "rq",       RAD_OPD_OUT,    1 },
    { "rscale",   RAD_OPD_OUT,    1 },
    { "y_write",  RAD_OPD_IN,     0 },
    { "inj_write", RAD_OPD_IN,    0 },
};
/* gate_quant_fp8 (run 206, pair 4): libr4d's pGateQuantFp8 / oGateQuantFp8, field for field */
static const RadParamSpec pXeGateQuant[] = {
    P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"), P_STRO("act"),
};
static const RadOperandSpec oXeGateQuant[] = {
    { "gate", RAD_OPD_IN, 0 }, { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 }, { "scale", RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};
static const RadOpSchema kSchemas[] = {
    { "hc_write_read", ROW(pXeHcWriteRead), ROW(oXeHcWriteRead),
      "hc_write (h[s] += inj_write[s] * y_write, into the unnormed stream) and then hc_read on the same stream, as "
      "one op: the bytes of the two ops in that order" },
    { "ar_hc_write", ROW(pXeArHcWrite), ROW(oXeArHcWrite),
      "libr4d's fused all-reduce + gated residual write; declared here so the op is known and "
      "unresolved on the Intel build, where the architecture then emits all_reduce + hc_write" },
    { "ar_gather_hc_write", ROW(pXeArGatherHcWrite), ROW(oXeArGatherHcWrite),
      "libr4d's fused moe_gather + all-reduce + hc_write; declared, not served (see ar_hc_write)" },
};
static const int kNumSchemas = (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
/* gated_had_quant_i8g (run 207, pair 2): libr4d's pGatedQuantFp8 / oGatedQuantFp8, field for field */
static const RadParamSpec pXeGatedHad[] = {
    P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"), P_STRO("act"),
};
static const RadOperandSpec oXeGatedHad[] = {
    { "gate_up", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 }, { "scale", RAD_OPD_OUT, 0 }, { "out_bf16", RAD_OPD_OUT, 1 },
};
/* gated_quant_pair (run 273): two gated_quant operand sets, the second's names and width suffixed 2 */
static const RadParamSpec pXeGatedQuantPair[] = {
    P_INT("M"), P_INT("n"), P_INT("n2"), P_INT("group"), P_STR("dtype"), P_STRO("act"), P_INTO("rot"), P_INTO("rot2"),
};
static const RadOperandSpec oXeGatedQuantPair[] = {
    { "gate_up", RAD_OPD_IN, 0 },  { "q", RAD_OPD_OUT, 0 },  { "scale", RAD_OPD_OUT, 0 },  { "out_bf16", RAD_OPD_OUT, 1 },
    { "gate_up2", RAD_OPD_IN, 0 }, { "q2", RAD_OPD_OUT, 0 }, { "scale2", RAD_OPD_OUT, 0 }, { "out_bf16_2", RAD_OPD_OUT, 1 },
};
/* qk_norm_rope_gate (run 273): libr4d's pQkNormRopeGate / oQkNormRopeGate, field for field */
static const RadParamSpec pXeQkPrologue[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("n_head_kv"), P_INT("rot"), P_STR("q_dtype"),
    P_F64O("eps"), P_F64O("wadd"), P_INTO("cs_f32"), P_INTO("pos_i64"),
};
static const RadOperandSpec oXeQkPrologue[] = {
    { "q_gate", RAD_OPD_IN, 1 },     { "k", RAD_OPD_IN, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
    { "positions", RAD_OPD_IN, 0 },  { "q_w", RAD_OPD_WEIGHT, 1 }, { "k_w", RAD_OPD_WEIGHT, 0 },
    { "q_out", RAD_OPD_OUT, 1 },     { "k_out", RAD_OPD_OUT, 0 },  { "gate_out", RAD_OPD_OUT, 1 },
};
/* router_topk_scatter_quant (run 273): libr4d's pRouterTopkScatter / oRouterTopkScatter, then the quantiser's */
static const RadParamSpec pXeRouterQuant[] = {
    P_INT("M"), P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
    P_INTO("expert_base"), P_STRO("protect"), P_INT("n"), P_INT("group"),
};
static const RadOperandSpec oXeRouterQuant[] = {
    { "logits", RAD_OPD_IN, 0 },        { "expert_ids", RAD_OPD_OUT, 0 }, { "expert_w", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 }, { "expert_count", RAD_OPD_OUT, 0 },
    { "x", RAD_OPD_IN, 0 },             { "q", RAD_OPD_OUT, 0 },          { "scale", RAD_OPD_OUT, 0 },
};
/* router_gemm_topk_scatter_quant (run 273): libr4d's pRouterGemmTopkScatter / oRouterGemmTopkScatter, then the
 * quantiser's */
static const RadParamSpec pXeRouterGemmQuant[] = {
    P_INT("M"), P_INT("K"), P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
    P_INTO("expert_base"), P_STRO("protect"), P_INT("n"), P_INT("group"),
};
static const RadOperandSpec oXeRouterGemmQuant[] = {
    { "a", RAD_OPD_IN, 0 },             { "b", RAD_OPD_WEIGHT, 0 },       { "gate_b", RAD_OPD_WEIGHT, 1 },
    { "logits", RAD_OPD_OUT, 0 },       { "expert_ids", RAD_OPD_OUT, 0 }, { "expert_w", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 }, { "expert_count", RAD_OPD_OUT, 0 },
    { "gate", RAD_OPD_OUT, 1 },
    { "x", RAD_OPD_IN, 0 },             { "q", RAD_OPD_OUT, 0 },          { "scale", RAD_OPD_OUT, 0 },
};
/* gemm_nt_q_x3 / gemm_nt_q_bf16_pair (run 274): gemm_nt_q's a, a_scale, then each set's weight(s) and y */
static const RadParamSpec pXeGemvX3[] = {
    P_INT("M"), P_INT("K"), P_INT("N1"), P_INT("N2"), P_INT("N3"), P_INT("group"), P_STR("dtype"),
};
static const RadOperandSpec oXeGemvX3[] = {
    { "a", RAD_OPD_IN, 0 },          { "a_scale", RAD_OPD_IN, 0 },
    { "b1", RAD_OPD_WEIGHT, 0 },     { "b1_scale", RAD_OPD_WEIGHT, 0 }, { "y1", RAD_OPD_OUT, 0 },
    { "b2", RAD_OPD_WEIGHT, 0 },     { "b2_scale", RAD_OPD_WEIGHT, 0 }, { "y2", RAD_OPD_OUT, 0 },
    { "b3", RAD_OPD_WEIGHT, 0 },     { "b3_scale", RAD_OPD_WEIGHT, 0 }, { "y3", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pXeGemvPair[] = {
    P_INT("M"), P_INT("K"), P_INT("N1"), P_INT("N2"), P_INT("group"), P_STR("dtype"),
};
static const RadOperandSpec oXeGemvPair[] = {
    { "a", RAD_OPD_IN, 0 },          { "a_scale", RAD_OPD_IN, 0 },
    { "b1", RAD_OPD_WEIGHT, 0 },     { "b1_scale", RAD_OPD_WEIGHT, 0 }, { "y1", RAD_OPD_OUT, 0 },
    { "x", RAD_OPD_IN, 0 },          { "b2", RAD_OPD_WEIGHT, 0 },       { "y2", RAD_OPD_OUT, 0 },
};
/* router_gemm_topk_scatter_quant_sh (run 285): the fused router op's params and operands, then the shared expert's
 * gate_up as an int8 set over the block input's codes (a_q, a_scale: the hc read's), its weight pair and its output */
static const RadParamSpec pXeRouterGemmQuantSh[] = {
    P_INT("M"), P_INT("K"), P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
    P_INTO("expert_base"), P_STRO("protect"), P_INT("n"), P_INT("group"), P_INT("N_sh"),
};
static const RadOperandSpec oXeRouterGemmQuantSh[] = {
    { "a", RAD_OPD_IN, 0 },             { "b", RAD_OPD_WEIGHT, 0 },       { "gate_b", RAD_OPD_WEIGHT, 1 },
    { "logits", RAD_OPD_OUT, 0 },       { "expert_ids", RAD_OPD_OUT, 0 }, { "expert_w", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 }, { "expert_count", RAD_OPD_OUT, 0 },
    { "gate", RAD_OPD_OUT, 1 },
    { "x", RAD_OPD_IN, 0 },             { "q", RAD_OPD_OUT, 0 },          { "scale", RAD_OPD_OUT, 0 },
    { "a_q", RAD_OPD_IN, 0 },           { "a_scale", RAD_OPD_IN, 0 },
    { "b_sh", RAD_OPD_WEIGHT, 0 },      { "b_sh_scale", RAD_OPD_WEIGHT, 0 }, { "y_sh", RAD_OPD_OUT, 0 },
};
/* gemm_nt_q_x3_bf16 (run 285, group B): gemm_nt_q_x3's params and operands, then the bf16 set over the activation's
 * bf16 plane `x` */
static const RadParamSpec pXeGemvX3Bf[] = {
    P_INT("M"), P_INT("K"), P_INT("N1"), P_INT("N2"), P_INT("N3"), P_INT("N4"), P_INT("group"), P_STR("dtype"),
};
static const RadOperandSpec oXeGemvX3Bf[] = {
    { "a", RAD_OPD_IN, 0 },          { "a_scale", RAD_OPD_IN, 0 },
    { "b1", RAD_OPD_WEIGHT, 0 },     { "b1_scale", RAD_OPD_WEIGHT, 0 }, { "y1", RAD_OPD_OUT, 0 },
    { "b2", RAD_OPD_WEIGHT, 0 },     { "b2_scale", RAD_OPD_WEIGHT, 0 }, { "y2", RAD_OPD_OUT, 0 },
    { "b3", RAD_OPD_WEIGHT, 0 },     { "b3_scale", RAD_OPD_WEIGHT, 0 }, { "y3", RAD_OPD_OUT, 0 },
    { "x", RAD_OPD_IN, 0 },          { "b4", RAD_OPD_WEIGHT, 0 },       { "y4", RAD_OPD_OUT, 0 },
};
/* kFlagged's schemas, in its order; listed under the same flags */
static const RadOpSchema kFlaggedSchemas[] = {
    { "gate_quant_fp8", ROW(pXeGateQuant), ROW(oXeGateQuant),
      "the same, with the gate in its own operand rather than packed beside x." },
    { "gated_had_quant_i8g", ROW(pXeGatedHad), ROW(oXeGatedHad),
      "gated_had_quant_fp8 at int8 codes: silu(a)*b or sigmoid(a)*b, a `group`-wide Hadamard, then int8 codes at "
      "amax / 127 (the bf16 out unrotated)." },
    { "gated_quant_pair", ROW(pXeGatedQuantPair), ROW(oXeGatedQuantPair),
      "gated_had_quant_i8g over the first operand set (`rot` 1) and gated_quant_fp8 at int8 codes over the second "
      "(`rot2` 0, width `n2`), silu, in one launch; `M` is the two sets' rows together. The two ops' bytes." },
    { "qk_norm_rope_gate", ROW(pXeQkPrologue), ROW(oXeQkPrologue),
      "the gated attention's prologue in one launch: per head, the q half of [q|gate] and k RMS-normed (the output "
      "rounded to bf16), partial NeoX RoPE from the cos/sin table at each token's position, the gate copied when "
      "asked. The four ops' bytes (rmsnorm q, rmsnorm k, rope q, rope k)." },
    { "router_topk_scatter_quant", ROW(pXeRouterQuant), ROW(oXeRouterQuant),
      "router_topk_scatter over `logits` and had_quant_act_i8g over `x` (the same rows) in one launch: the two ops' "
      "bytes." },
    { "router_gemm_topk_scatter_quant", ROW(pXeRouterGemmQuant), ROW(oXeRouterGemmQuant),
      "router_gemm_topk_scatter with had_quant_act_i8g over `x` (the same rows) in its router launch: the two ops' "
      "bytes." },
    { "gemm_nt_q_x3", ROW(pXeGemvX3), ROW(oXeGemvX3),
      "three gemm_nt_q (dtype i8a8) over one activation in one launch, widths N1 / N2 / N3: the three ops' bytes." },
    { "gemm_nt_q_bf16_pair", ROW(pXeGemvPair), ROW(oXeGemvPair),
      "gemm_nt_q (dtype i8a8, width N1) over an activation's codes and gemm_nt (bf16, width N2) over its bf16 plane "
      "`x` in one launch: the two ops' bytes." },
    { "router_gemm_topk_scatter_quant_sh", ROW(pXeRouterGemmQuantSh), ROW(oXeRouterGemmQuantSh),
      "router_gemm_topk_scatter_quant with the shared expert's gate_up (gemm_nt_q at dtype i8a8, width N_sh, over the "
      "block input's codes a_q / a_scale) in its GEMV launch: the two ops' bytes." },
    { "gemm_nt_q_x3_bf16", ROW(pXeGemvX3Bf), ROW(oXeGemvX3Bf),
      "gemm_nt_q_x3 (dtype i8a8, widths N1 / N2 / N3) over an activation's codes and gemm_nt (bf16, width N4) over its "
      "bf16 plane `x` in one launch: the two ops' bytes." },
};
static_assert(sizeof(kFlaggedSchemas) / sizeof(kFlaggedSchemas[0]) == kNumFlagged, "a schema for each flagged row");
static int own_schemas() { return kNumSchemas + flagged_count(); }
static const RadOpSchema* own_schema_at(int i) {
    if (i < kNumSchemas) return &kSchemas[i];
    const int k = flagged_index(i - kNumSchemas);
    return k >= 0 ? &kFlaggedSchemas[k] : nullptr;
}
/* libxe's own schemas, then the MTP coupling's two (plugin/spechub/spechub.cpp, compiled into libxe): the loader
 * refuses a row whose op no loaded plugin declares a schema for, and the service's kernel list is libxe:libref
 * without the spechub plugin; two plugins declaring the same schema is accepted when they agree (loader.cpp,
 * schema_same), which these do, being the same definition. */
extern "C" int rad_kernel_schema_count(void) { return own_schemas() + rad_spechub_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0) return nullptr;
    if (i < own_schemas()) return own_schema_at(i);
    return rad_spechub_schema_at(i - own_schemas());
}
extern "C" int rad_kernel_count(void) { return kNumKernels + flagged_count(); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    if (i < kNumKernels) return &kKernels[i];
    const int k = flagged_index(i - kNumKernels);
    return k >= 0 ? &kFlagged[k] : nullptr;
}
