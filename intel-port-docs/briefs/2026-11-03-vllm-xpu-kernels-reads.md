# vllm-xpu-kernels, read against ours (owner note 169 item 2; run 163, 2026-11-03)

Read-only assessments of third_party/vllm-xpu-kernels (commit 29ba15c, Apache-2.0) against the engine's kernels, a
paragraph each: does it fit, and would it be faster. Written from the sources (treated as data), file:line pointers
below each paragraph. The prefill attention (note 169 item 1's start) is the first section.

## chunk_prefill (csrc/xpu/attn/xe_2/) -- prefill attention

Intel's CuTe / sycl-tla flash-attention-2 forward (XeFMHAFwdKernel + FMHAFwdMainloop): a causal / varlen / paged
FMHA with flash-style online softmax that runs both GEMMs on DPAS (XMX) through XE_DPAS_TT atoms, streams Q/K/V
with Xe 2D-block loads, keeps P in registers, and accumulates S and O in fp32. Q/O are bf16 (or f16); K/V are the
same type or FP8 E4M3/E5M2, cast to bf16 in registers with per-tensor scales folded into the softmax scale. For
head 256 it uses 256-query x 32-key tiles, 32 sub-groups of 16 lanes (512 work-items, 256 GRF), 8 query rows a
sub-group (DPAS M = 8), and the P.V product split into eight 32-wide V slices. Its only dependency is header-only
CUTLASS (intel/sycl-tla at a pinned revision, not vendored in our tree) plus the DPC++ SPIR-V extensions for DPAS,
2D block I/O and split barriers. Our heads fit exactly (24 q / 2 kv heads, head_dim 256, GQA by head division;
"256,paged,causal" is a policy Intel ships for Qwen3-Next), and it would read our FP8 cache rows as-is (the K|V
interleaving is a stride plus a V pointer at +256 B; scales = pointers to 1.0). **The blocker is the page: it needs
block_size a multiple of 16 (each 16/32-key tile is one 2D block load inside one page), while this model's cache
uses 4-token QSA pages, and the QSA sparse path (per-query block lists) is outside its model entirely.** The
practical port runs it non-paged on the chunk's dense bf16 K/V (plus a gather of prior context), replacing only the
dense attn_split launches. **Expect ~0.4-1.5 ms a launch against 7.25 ms (5-15x), taking attention from ~20 % to
~2-4 % of prefill device time**; the slow end applies to 256-row launches (24 work-groups on 32 cores).

- Kernel loop: per 32-key tile, Q and K 2D-block loads + reorder (fp8 -> bf16) + DPAS S = Q.Kᵀ over d-slices of 32;
  causal mask; remainder mask; online softmax in the exp2 domain; P -> bf16; then per V slice: V load + reorder,
  rescale O, DPAS O += P.V; the next page's K prefetch; one WG split-barrier a tile: attn/xe_2/collective/
  chunk_prefill_mainloop.hpp:572-701; the softmax :713-753; Q reloaded every K tile :608.
- DPAS atom XE_DPAS_TT<gcd(SGTileQ,8), float, ElementQ> (fp32 accumulate) attn/xe_2/chunk_prefill.hpp:280-285;
  VTiles :311-312; WG = SGPerWG x 16 lanes attn/xe_2/kernel/chunk_prefill_kernel.hpp:215; the grid (V tiles, Q
  tiles, batch x q_heads) attn/xe_2/collective/chunk_prefill_scheduler.hpp:65-68.
- Head-256 policy: ShapeQK <256,32,32>, ShapePV <256,32,32>, ShapeOut <256,256>, 32 sub-groups attn/fmha_utils.hpp:
  116-121; the _b16 variant for pages that are multiples of 16 only :223-228; heads 64..512 attn/xe_2/fmha_xe2.cpp:
  241-245, the dispatch :349-372.
- Dtypes: Q bf16/f16 with K/V the same or e4m3/e5m2; O = Q's type chunk_prefill.hpp:386-491; the fp8 scales are
  per-tensor floats dereferenced unconditionally mainloop.hpp:510-514, V's applied once after the loop :703-709.
- GQA: head = head_q / (num_heads_q / num_heads_kv) kernel.hpp:247,270. Causal with full_tile_offset = seq_len_kv −
  seq_len_qo (the diagonal at the end of the context: chunked prefill OK) :276-299. Varlen by int32 cu_seqlens_q; in
  paged mode cu_seqlens_k[b] is the per-sequence context length :219-237.
- Paged KV: the cache [num_blocks, block_size, kv_heads, head_size], strides explicit, V not transposed
  fmha_xe2.cpp:16,181-188, chunk_prefill.hpp:80-87,271-274; get_paged_idx mainloop.hpp:254-267; non-contiguous page
  strides attn/paged_kv_utils.h:6-13; block_size a positive multiple of 16 fmha_xe2.cpp:256-273; 2D surfaces need
  64-B aligned bases attn/xe_2/collective/block_2d_align.hpp:19-45 and < 2^24 rows mainloop.hpp:321-418. Extras:
  sliding window, sink, the softmax LSE output, a per-batch is_prefill mask kernel.hpp:267,441-496.
- Build: header-only CUTLASS by FetchContent from intel/sycl-tla @ 7b56b936b87f799f80dde60c2fd8d11c1128cebb
  (CUTLASS_ENABLE_HEADERS_ONLY) CMakeLists.txt:333-371, the include dirs :393-401 -- no checkout under third_party/.
  Headers: cutlass/*, cute/tensor.hpp, cute/algorithm/subgroup_algorithms.hpp, flash_attention_v2/collective/
  fmha_fusion.hpp, copy_block_slm.hpp (only the ReduceK > 1 epilogue) chunk_prefill.hpp:3-16, mainloop.hpp:35-44,
  epilogue.hpp:35-47; the launch by syclcompat with sub_group_size<16> and grf_size<256> chunk_prefill.hpp:226-251.
  Flags: -fsycl-targets=spir64_gen, -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,
  +SPV_INTEL_subgroup_matrix_multiply_accumulate, AOT -device pvc,bmg,... -cl-intel-256-GRF-per-thread
  CMakeLists.txt:280-312, cmake/utils.cmake:583-610; the documented toolchain oneAPI 2026.1 DPC++ (README.md:47).
  Ours: intel/llvm 7.2.0, -fsycl-targets=intel_gpu_bmg_g31,spir64 (src/cmake/RadL0.cmake:44-48); we already launch
  SIMD16 / 256-GRF kernels (src/libxe/xe_lanes.h:27). The host wrapper is torch-bound (fmha_xe2.cpp,
  chunk_prefill_utils.hpp); the kernel and chunk_prefill_args_t are not (chunk_prefill.hpp:44-95).
- Ours: attn_split_kernel = one WG per (split, kv head, query row), one sub-group of 32 a q head, scalar FMA + a
  5-step xor-shuffle dot, per-element E4M3 decode, fp32 online softmax (src/libxe/xe_attn.cpp:5-12,44-51,89-176;
  the launch grid :229-247); ~2-3 TFLOPS. Heads 24 q / 2 kv, head_dim 256, gqa 12 (src/arch/qwen4exp_fp8/
  qwen4exp_fp8.cpp:19-24); the cache one tensor [block][kv_head][pos][K(256)|V(256)], E4M3 or bf16, an int32 block
  table, no stored scales (xe_attn.cpp:128,132,156,275-281,294-296; src/arch/common/rad_block_attn_gated_fp8.h:
  596-604) -- maps onto the kernel's strided paged model without a format change (k_stride_seq 512 B,
  k_stride_heads block x 512 B, V = K + 256 B, page_stride_elements kv_heads x block_size, cu_seqlens_k = ctx).
- The blocker: this model's attention page is 4 tokens (the QSA compress block, qwen4exp_fp8.cpp:1092-1096) against
  block_size % 16 == 0 (fmha_xe2.cpp:258-263, mainloop.hpp:255); the default RAD_KV_BLOCK is 16 (rad_arch.h:634);
  the QSA sparse path issues per-query block lists (rad_block_attn_gated_fp8.h:67-72,583-594). Adapting: (a) the
  non-paged varlen form on the chunk's dense bf16 w.k / w.v (rad_block_attn_gated_fp8.h:34-36) plus a gather of the
  prior context from the 4-token pages (~1 KB a token); or (b) a K/V copy that assembles a 16-row tile from four
  4-row pages (CuTe copy-atom surgery, risky); either way only the dense path (ctx <= qsa_exact_to) is replaced;
  the numerics change (P in bf16, the exp2 domain), as note 168 item 1 expects.
- Speed: ~21 GFLOP a launch in 7.25-9.75 ms today (~2-3 TFLOPS scalar); the B70's XMX bf16 peak is roughly 150-180
  TFLOPS (from memory, not the tree); FMHA at head 256 with 8 V slices, the fp8 reorders and the Q reloads plausibly
  reaches 15-50 TFLOPS: 0.4-1.5 ms a launch. The low end for 256-row launches (24 WGs of 512 WIs: one WG fills a
  core at 256 GRF, <= 24 of 32 cores busy, no latency hiding); whole-chunk launches (2048 rows -> 192 WGs) sit
  nearer the high end. Net: attention 20 % -> ~2-4 % of prefill device time (~16-18 % of the prefill); the
  uncertainty is mostly occupancy and the untested 7.2.0-against-sycl-tla toolchain match.

## gdn_attn (csrc/xpu/gdn_attn/) -- the gated delta rule against our chunk scan and recurrent update

Same math, different packaging; not worth replacing. Their Xe2 prefill path implements exactly our recurrence: g =
−exp(A_log)·softplus(a+dt_bias) with an in-chunk inclusive cumsum, beta = sigmoid(b), q/k L2-normalised (eps 1e-6,
q·dk^-½), chunk 64, A = (I + tril(β·KKᵀ·e^{g_i−g_j}))⁻¹, W = A(βe^g⊙K), U = A(β⊙V), O = e^g·QS +
tril(QKᵀe^{Δg})(U−WS), S ← e^{g_last}S + (U−WS)ᵀ(e^{g_last−g}K). The state is [slot][v_head][V][K], V-major --
identical to our h0/ht [N,H,V,K] -- in fp32/fp16/bf16. Differences: five kernels (prepare, compute_A, inverse,
compute_wu, fwd_o) with A/w/u bf16 intermediates round-tripping HBM, versus our kkt_solve + cs_jm keeping W/V′/S
on-chip; generic head dims (64-wide dv/dk loops) where ours is fixed 128/128; tokens must sit on a chunk-padded
"virtual" axis with b/a as fp32 [H, T_virtual]. Mainloops are sycl-tla CuTe: XE_DPAS_TT<8,float,bf16>, sub-group 16,
GRF 256, 64x64x32 work-group tiles (2x2 / 2x1 / 4x2 sub-group layouts; 16x16x16 for the inverse), 2D-block loads
with 3-deep prefetch, no SLM staging. The KKᵀ gram is recomputed per v head (ours: once per k head). fwd_o is
(batch × v_heads) work-groups walking chunks serially -- the same serial structure as cs_jm. Their decode kernel
keeps state in registers per (batch, head, 32-column v bucket) with tree reductions; equivalent to our recurrent
update but less tuned. Speed: the serial part roughly parity to 1.5x faster (register-resident block-2D loads, fused
W×S / Q×S against our SLM scratch round-trips), minus their A/w/u HBM traffic and five launches: 0.7-1.3x of our
4.15 ms; the ceiling is ~2 % of prefill. Borrow gemm_TTS_fused_2A and the block-2D staging if cs_jm is ever
revisited; nothing for decode.

- Gating / norm: gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp:82-124, sigmoid on b
  xe_2/chunk_causal_conv1d_xe2.hpp:517, l2norm xe_2/l2norm_kernel.hpp:53-54; ours ports/r4d_gdn_recurrent_update_*.cpp:10,
  ports/r4d_gdn_kkt_solve_k128_c64_bf16.cpp:9. Chunk 64: gdn_attn_utils.h:10; ours
  ports/r4d_gdn_chunk_scan_k128_v128_c64_bf16.cpp:36.
- A / W / U / O / S: kernels_xe2.hpp:248-269, 827-875, 1070-1085, 1108-1168, 1183-1211; ours ports/r4d_gdn_chunk_scan_jm.inc:15-19.
  State layout: xe_2/chunk_gated_delta_rule_xe2.cpp:18, kernels:1019-1022; ours chunk_scan.cpp:517, jm.inc:81.
- Tiles / DPAS / SG16: kernels:15-42, 1278, 1287-1289; the mainloop xe_2/gemm.hpp:106-136; the fused dual-A sharing S
  loads 391-489. HBM intermediates: kernels:1560-1568; the virtual padding gdn_attn_interface.cpp:445-466.
- Dependencies: sycl-tla (cutlass-sycl) commit 7b56b93 by FetchContent, not vendored (CMakeLists.txt:339-370);
  SPIR-V 2d_block_io / subgroup_matrix_multiply_accumulate (:307); torch; vllm::xpu::is_bmg (kernels:1351).

## grouped_gemm (csrc/xpu/grouped_gemm/) -- the expert GEMMs against format C at prefill

Reuse the tiling and scheduling; the datapath is bf16 DPAS, not int8. It computes D[M_e,N] = A[M_e,K]·B_eᵀ (+
bias[E,N]) over rows pre-sorted by expert (no tile padding): the Python remap_hidden_states yields rows_per_expert +
a token→row map; moe_gather applies the top-k weights. Xe2 takes bf16/fp16 activations only; B may be bf16, int4
([E,N,K/2] + [E,N,K/gs] scales, gs 32-256), MXFP4, per-tensor FP8, MXFP8 (E8M0/32) or block-FP8 ([E,K/128,N/128]).
Every path dequantises B into the activation dtype in registers (reorder + a per-column scale applied each k-tile by
a vISA mul) and runs XE_DPAS_TT<8,float,bf16>; block-FP8 instead rescales the fp32 accumulator once per 128-K block.
Scheduling is a persistent grid (sm_count·512/WG-threads) walking the expert list with per-expert tile counts
ceil(M_e/tile_m), per-expert pointers A+pre_rows·K, B+e·N·K, and an atomic tile counter broadcast through SLM.
Policies: 256x256x32 (SG 8x4), w4a16 128x256x32 (4x8), w8a16 128x128x16 (4x2), and small-M 8/16/32 x64x32 (1x4),
picked by the average rows an expert (<= 4 / <= 8 / <= 128). Sub-group 16, GRF 256, 6-deep k-tile prefetch to hide
the packed-weight decode. moe_shared_fused_decode swaps the atomic scheduler for a one-WG tile map (<= 512 routed
rows, FP8 per-tensor only). Fit: not directly usable (bf16 datapath; a stacked-B stride, not our per-expert pointer
tables and parity classes). Reuse the shape: at ~80 rows an expert their choice is 32x64x32 (SG 32x16) -- 32-row
weight reuse against our 4. For int8 DPAS (8x16x32, int32 accumulate) the K group must be 64, not 128: our weight
scale is per 64-K, so close the int32 partial every two k-steps and fold ws[n]·as[m] into the fp32 accumulator (their
block-FP8 trick with two vectors). The sycl-tla int8 DPAS atom must be confirmed at the fetched revision (the tree
is not vendored). **Speed: gate_up today reads each expert's weights ~20x (80 rows / 4 a run) ≈ 10 GB a launch --
bandwidth-bound at 17 ms. A 32-row int8-DPAS tile cuts the weight traffic ~8x and the compute is ~160 GOP at a
realistic ~100 TOPS: 4-6 ms gate_up, ~2 ms down, i.e. ~3x on these launches, 49 % -> ~17 % of prefill (~1.5x
overall).** The cheaper fallback -- activations dequantised to bf16 and the codes through a 16-entry table inside
their int4 path -- would land near 8-10 ms.

- Dtypes / dispatch: grouped_gemm/xe_2/gemm_xe2.hpp:55-66; xe_2/grouped_gemm_xe2_interface.hpp:191-207, 284-305,
  357-369, 553; A bf16 only grouped_gemm_interface.cpp:47-49.
- The expert list / pointers / atomic: xe_2/grouped_gemm_xe2.hpp:112-150, 173-176, 219-224; the launcher :114-132.
- Group-scale handling: gemm_xe2.hpp:346-347 (prefetch 6), 409-455, 485-502; block-FP8 accumulator rescale 412-425.
- Policies: xe_2/gemm_xe2_policy.hpp:9-105; the row sorting vllm_xpu_kernels/fused_moe_interface.py:519-545, 664-666.
- The fused decode map: xe_2/moe_shared_fused_decode_xe2.cpp:87-230, 278-361.
- Ours: src/libxe/xe_moe_i4i8.cpp:175 (kMT = 4 rows a run; the 16 is lanes a column, :551), 512-thread WG :175,
  format C levels / scales :177-180, 306-318, 340-348, the DP4A loop :400-433, the sorted / off tables :196-224, the
  dispatch :545-597.

## mhc (csrc/xpu/mhc/) -- DeepSeek V4's Manifold-Constrained Hyper-Connections against our hc

"mhc" is DeepSeek V4's Manifold-Constrained Hyper-Connections, not Qwen-style hc. Its pre-step projects the
flattened residual [N, 4·H] bf16 onto 24 fp32 vectors (4 pre-gates, 4 post-gates, one 4×4 "comb"), scales those dot
products by one RMS over all 4·H elements, applies sigmoid gates and a Sinkhorn-normalised 4×4 mix, and emits
layer_input = Σ_m pre[m]·residual[m]. The post-step is out[o] = Σ_i comb[i,o]·residual[i] + post[o]·x. Ours is a
different function: per-stream grouped RMSNorm with gain, then two low-rank E4M3 products (W_down 320×10240 → silu →
W_up 10240×320) giving a sigmoid gate per (stream, channel), the stream mean, and inject gains; our write is rank-1
(h[c] += inj[c]·y) with no stream-to-stream matrix. So no mhc kernel computes any of ours; only HC = 4 (hard-coded)
matches, there is no low-rank bottleneck, and their weights are fp32 on TF32 DPAS where ours are group-scaled E4M3
decoded to bf16. What transfers is structure, not code: (a) split-K with a fused reduce/epilogue for our down product
(K = 10240, only 324 columns, so hc_pf_dpas<0> has 6 column tiles × ⌈rows/64⌉ work-groups); (b) cute block-2D loads
with 3-deep prefetch straight into DPAS fragments, against our LDS staging, scalar E4M3 decode and two barriers per
64-k step, repeated per 64-row tile. Porting as-is would not be faster (TF32 is half the bf16 DPAS rate; it needs
cutlass/sycl-tla plus ATen); re-tiling hc_pf_dpas along (a)/(b), with W decoded to bf16 once per chunk, plausibly
would.

- Name and math: mhc/mhc_interface.h:1,11-13,23; mhc/xe_2/mhc_pre.cpp:24-29 (fn is [24, K=4·H] fp32);
  mhc/xe_2/mhc_post.cpp:3-5; mhc/xe_2/hc_head_fused.cpp:1 ("DeepSeek V4"). HC pinned to 4: mhc_pre.cpp:932.
- Vector (small-M) path: hc_head_fused.cpp:32-36 (2 tokens/WG, VEC 8 = 16 B bf16 and 32 B fp32 loads, sub-group 16,
  WG 256), :81-110 (dot + Σx² in one pass), :117-135 (sub-group reduce_over_group, SLM across sub-groups);
  mhc_pre.cpp:284-289 (stage 1: 12 columns, VEC 4), :546-602 (sigmoid + Sinkhorn on sub-group 0 via
  permute_group_by_xor), :615-636 (weighted reduction, VEC 8). No joint_matrix anywhere.
- Split-K DPAS path (M >= 128): mhc_pre.cpp:60-64 (WG tile 16×32×32, 2 sub-groups = 32 threads, prefetch 3), :76-92
  (n_splits 32/4/8 from the Xe-core count), :158-160 (make_block_2d_copy_A/B/D), :208-242 (prefetch warm-up,
  barrier_arrive/wait K loop, no SLM), :232-237 (Σx² from the A fragment inside the K loop), :840 (XE_DPAS_TT<8,
  float, tfloat32_t> -- TF32, fn fp32), :884-885 (sub-group 16, GRF 256), :971-974 + csrc/utils.h:140-149
  (VLLM_MHC_USE_TF32; off = the vector path for every M). Fused reduce + stage 2: mhc_pre.cpp:452-639 (1 WG/token,
  256 threads, fp32 workspace [n_splits·M_padded, 24] at :1006-1007).
- Post: mhc_post.cpp:32-35 (1 token/WG, VEC 8, sub-group 16), :60-99 (4×4 fp32 comb in registers, 4 streams in/out
  per 16 B).
- Dependencies: mhc_pre.cpp:31-50 (ATen/torch, cutlass device_kernel, cute compat launch, cute tensor/subgroup
  algorithms, KernelHardwareInfo); csrc/CMakeLists.txt:360-364 (intel/sycl-tla fetched); mhc/xe_2/CMakeLists.txt:3
  (Xe2 AOT library).
- Ours: src/libxe/ports/r4d_hc_bf16.cpp:17-22 (math), :27-30 (n 2560, hc 4, LR 320, 6.55 MB a matrix), :468-492
  (hc_write rank-1), :1783-1811 (hc_pf_dpas: sub-group 16, 8 sub-groups, 64×64 tile, E4M3 decoded once a tile),
  :1845-1890 (K loop: two syncthreads, scalar decode into packed LDS, joint_matrix 8×16×16 bf16), :4806-4843 (four
  launches: hc_v2_norm, <0> grid (⌈324/64⌉, ⌈rows/64⌉), <1> grid (n/16, ⌈rows/64⌉), hc_quant_rows); STATE.md (P43:
  <1> 1.41 ms, <0> 0.93 ms a launch).

## layernorm.cpp / layernorm_quant.cpp -- fused add + RMSNorm, and the quantising forms

layernorm.cpp is vLLM's row-wise family: rms_norm (y = x·rsqrt(mean x² + ε)·(w + bias), bias = 1 for the Gemma
forms), fused_add_rms_norm (residual ← bf16(input + residual) in place, then input ← norm(residual)·w), and
three-pass LayerNorm / fused-add LayerNorm. layernorm_quant.cpp fuses the same norm (optional residual add) with a
quantiser: per-token fp8/int8 (scalar, three passes), per-block fp8/int8 with group scales (row staged in SLM,
optional UE8M0), MXFP4, and static-scale fp8. All run one work-group per token, sub-group 32, 16-byte
vec_n_t<bf16,8> loads, fp32 Σx², reduce_over_group, rsqrt via SLM. The fused-add math is bit-for-bit our
add_rmsnorm_body (bf16-rounded sum, f32 Σx², the same rsqrt, gain + wadd); the differences are cosmetic (in-place
output, bf16-only gain where ours takes f32, no mult or pitches). Our quant fusions already go further
(rmsnorm_had_quant_i8 keeps the row in registers and adds the Hadamard; fused_quant_fp8 is bit-identical to the
unfused pair), so nothing there is new. For note 165 item 2 these files contain no all-reduce at all: the hard part
-- a work-group owning whole rows of the reduced message so a row's Σx² is local, in today's exact rank-0-then-rank-1
order -- is absent; and in the hc model the op after the all-reduce is hc_write, already fused behind
RADIANCE_XE_AR_HC_FUSE, not an RMSNorm. Speed: theirs loads 16 B a lane where ours loads 2 B; that helps only at
prefill-sized M, and add_rmsnorm is not on our profile's top list, so expect a small gain from copying the load
pattern, none from porting.

- rms_norm: csrc/layernorm.cpp:45 (reqd_sub_group_size 32), :79-104 (vector Σx², reduce_over_group, SLM rsqrt),
  :114-134 (second pass, weight + weight_bias), :462-465 (block 1024/256 by token count, vec 8 for 2-byte types),
  :492-532 (16 rows/WG only when hidden/8 <= 32, i.e. head-dim norms).
- fused_add_rms_norm: :609-673 (pass 1 bf16 add written to residual + Σx²; group_barrier; pass 2 re-reads residual,
  writes input), :633 (add in scalar_t), :777-795 (vec 8, grid = tokens, block = hidden/8 capped); Gemma =
  weight_bias 1.0 at :1343-1359. LayerNorm three-pass: :867-934, :1071-1143.
- Quant: csrc/layernorm_quant.cpp:64-130 (dynamic per-token fp8 e4m3 / int8, scalar, 3 passes, scale_ub), :199-313
  (per-block: 16 B VEC, row in SLM, xor-shuffle group max, UE8M0 :280-284), :333-483 (MXFP4 e2m1, power-of-two
  scales), :500-572 / :585-666 (static fp8, VEC = gcd(8, hidden) :823-827).
- Dependencies: layernorm.cpp:1-8 and layernorm_quant.cpp:4-17 (ATen DeviceGuard, dispatch macros, utils.h,
  quantization/utils.h vec_n_t at :10-12, fp8 quant_utils, mxfp4); plain parallel_for + local_accessor, no cutlass.
- Ours: src/libxe/ports/r4d_model_bf16.cpp:226-240 (math, wadd, f32 gain, two passes), :233-234 (plugins declare
  gains RAD_F32), :280-310 (add_rmsnorm_body: bf16 sum written, Σx², no barrier between passes, scalar 2-B loads);
  ports/r4d_rmsnorm_had_quant_i8.cpp:7-40; ports/r4d_fused_quant_fp8.cpp:7-30. The all-reduce today:
  src/libxe/xe_allreduce.cpp:14-17 (block b owns a contiguous chunk of the flattened message, the sum in rank order),
  :293-343 (hc_write fused into emit), :600-603 (RADIANCE_XE_AR_HC_FUSE); r4d_hc_bf16.cpp:10-11 (the hc model: no
  pre-norm, no residual add).
