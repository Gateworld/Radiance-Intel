# Radiance on Intel Arc Pro B70: source audit

Date: 2026-10-24. This was a read-only audit of `~/code/radiance-intel/src`, a fresh clone of
https://codeberg.org/StillDeadcode/radiance (main). Nothing was built, run on a GPU, ported, or
changed in production. Paths below are relative to `src/` unless noted.

How it was done: `grep`/`sed` over the tree; a small Python parser read the production `.rad`
header (the first bytes and the directory, read-only); `podman inspect radiance` for the labels
and command. The op list in section 4 comes from the source, not from a `--debug-graph` run, so it
is a superset (prefill forms and alternative rows are included, then marked).

---

## 1. Versions

| | value |
|---|---|
| main HEAD | `9eaf10da6eea6c1fd0053936adf20ad5a101b7a3`, 2026-10-23 11:57 +0200, "radiance 1.11.0" (`git describe` = `v1.11.0`) |
| v1.9.0 tag | `685ec5b`; main is 103 commits past it |
| production image | `localhost/radiance:1.9.0-keyenv`; label `org.opencontainers.image.version = 1.9.0+api-key-env`, `revision = 73caca3` (not in upstream history: it is the local keyenv build), `radiance.gpu-targets = gfx1201`, base `docker.io/stilldeadcode/radiance@sha256:36d044d5…`, local patch `0001-config-api-key-falls-back-to-RADIANCE_API_KEY` |
| production model | `/srv/models/llm/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad` (121,969,901,568 B). Header: `arch_id = qwen4exp`, created by `rad-convert Oct 2 2026 from Qwen/Qwen3.8-Flash-Next`, recipe text = `data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` |
| production flags | `--tp 2 --tp-wire exact --placement expert_tiered --host-pool-mib 12288 --kv-cache-dtype fp8 --ngram-placement ram --num-speculative-tokens 3 --max-num-seqs 8 …` (note: the upstream `deploy/compose/flashnext.yaml` says `--tp-wire wht6`; production runs `exact`) |

**About the names.** There is no `qwen3.8-next` plugin and no `iq4r` format in the code. "iq4r"
appears only in the file name, README and docs. The arch plugin is `arch/qwen4exp_fp8/`
(`arch_id` `qwen4exp`, docs/ARCHITECTURES.md:16, :315). The served recipe is
`qwen4exp-w4nl64-i8-hc8m`. Its experts are 4-bit codebook weights, kernel dtype `w4nl64a8h`; the
trunk is **int8**, not FP8. What the production container holds, by bytes (from its directory):

| entries | GiB | encoding |
|---:|---:|---|
| 50,156 | 59.21 | routed experts, `gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32(2^-13) fwht128` |
| 1 | 47.68 | n-gram table (`ple_ngram`), E4M3 at one fixed bf16 scale |
| 223 | 3.41 | trunk block linears + lm_head, int8, bf16 scale per (row, 128) |
| 883 | 2.27 | checkpoint bf16 (norms, router, embed, conv, A_log, …) |
| 202 | 0.65 | hc mixing matrices + MTP fc, E4M3 rows, f32 scale per 128 / 80 |
| 1 | 0.16 | MTP 2-bit draft head (u2, u8 zero, f16 scale per 128) |
| 20 | 0.09 | the 10 protected experts, bf16 |

---

## 2. How tightly the core is tied to HIP

**Short answer:** the core reaches the GPU only through a backend vtable
(`core/device/device.h:150-336`, `struct DeviceBackend`). The ABI above it is vendor-neutral
(`abi/rad_device.h:19-24`: "Nothing above this layer names a vendor, product or instruction
set"). However, the only GPU backend that exists is AMD-only in practice. It is HIP plus raw
HSA/AQL plus KFD ioctls, and it intercepts HIP's own launch symbols.

### HIP runtime calls outside libr4d (engine code; tests/tools left out)

85 call sites, all inside 6 files: `core/device/{aql.cpp,hip.cpp,aql_blit.hip}`,
`core/runtime/{xprefetch.hip,wtab.hip}`, `core/sched/advance.hip`, plus one type in
`abi/rad_types.h`. Tests/tools add 38 more (the emulator header and `tools/vmemprobe.cpp`).

| group | n | where / examples |
|---|---:|---|
| device queries | 27 | `hipGetDeviceCount` core/device/aql.cpp:384, `hipDeviceGetPCIBusId` aql.cpp:390, `hipSetDevice` hip.cpp:70, `hipDeviceGetAttribute` hip.cpp:150, `hipGetDeviceProperties` hip.cpp:168, `hipRuntimeGetVersion` aql.cpp:3096 |
| memory | 12 | `hipMalloc` hip.cpp:288, `hipMemGetInfo` hip.cpp:194, `hipPointerGetAttributes` hip.cpp:330, `hipHostGetDevicePointer` hip.cpp:341, `hipExtMallocWithFlags(…Uncached)` aql.cpp:2735. aql.cpp:3705-3745 are the engine's **own definitions** of `hipMemcpy2DAsync/hipMemcpyAsync/hipMemsetAsync/hipMemset`, not calls |
| launches | 16 | `hipLaunchKernelGGL` in core/device/aql_blit.hip:191-243 (copy/fill kernels), core/sched/advance.hip:70-73, core/runtime/wtab.hip:17, core/runtime/xprefetch.hip:291-302; aql.cpp:3688 **defines** `hipLaunchKernel` |
| streams/events | 3 | all are interposed definitions: aql.cpp:3768 `hipStreamSynchronize`, :3776 `hipEventRecord` and :3782 `hipStreamWaitEvent` (both refused on engine streams) |
| peer access / IPC | 4 | `hipDeviceCanAccessPeer` hip.cpp:220, `hipDeviceEnablePeerAccess` hip.cpp:268. No IPC in the core |
| graphs | 0 | none in the engine (only libr4d/isa/graphrate.hip, a bench tool) |
| errors | 23 | `hipGetLastError`, `hipGetErrorString` … |

Pinned/host-registered memory does **not** go through HIP at all. aql.cpp:3223-3245 pins host
memory with `hsa_amd_memory_lock_to_pool`, explicitly "HIP is never asked".

### What does the real work: HSA and KFD (all in core/device/aql.cpp, 3,808 lines)

119 `hsa_*` call sites: `hsa_queue_create` (aql.cpp:1486, one HSA queue per stream),
`hsa_signal_*` (events), `hsa_amd_memory_pool_allocate` (kernarg rings, aql.cpp:1497),
`hsa_amd_vmem_*` (elastic memory), `hsa_executable_load_agent_code_object` (aql.cpp:476, which loads
plugin code objects itself), `hsa_amd_profiling_*`. KFD ioctls: `AMDKFD_IOC_CREATE_QUEUE` for a
user-mode **SDMA** queue (aql.cpp:2181, :2397), `ALLOC/MAP_MEMORY_OF_GPU` (:2312, :2323),
`WAIT_EVENTS` (:1314).

The design is stated at aql.cpp:1-29: *"every stream is an HSA queue the engine owns, and every
kernel, copy, fill and fence on it is an AQL packet written here … What stays with HIP:
allocation, device properties, peer access, IPC, elastic memory, and compiling and registering
kernels."* The engine defines `hipLaunchKernel`, `__hipRegisterFatBinary` and
`__hipRegisterFunction` (aql.cpp:3638, :3666, :3688), so every plugin's launch binds to the
engine ahead of the HIP runtime.

### Who owns memory, streams and launches

- **Core owns** memory (`rad_dev_alloc`), streams, events, copies, fills, pinned host pools,
  elastic memory, recorded passes ("tapes"), and the expert-prefetch DMA. All of it goes through the
  vtable.
- **Plugin owns** the launch itself. A kernel library's `launch(const RadArgs*, RadStream)`
  (abi/rad_abi.h:96) calls `hipLaunchKernelGGL` on the `RadStream` cast to `hipStream_t`. On AMD,
  the engine's interposed `hipLaunchKernel` turns that into an AQL packet. libr4d also allocates
  some memory itself: `hipMalloc` for split-tail workspaces and all-reduce scratch, outside the
  ledger (core/device/dispatch.cpp:120-124), and `hipHostMalloc` for the host-mailbox all-reduce.
- `RadStream` is `typedef void*` (abi/rad_types.h:331). What it points to is an unwritten
  contract between the backend and the kernel library.

### What a new backend would face

- In a HIP build there is **no plain-HIP fallback**. `hip.cpp`'s table leaves streams, events,
  copies, fills, tapes, vmem and code-object checks null (core/device/hip.cpp:345-370: "filled in
  by the AQL backend"). `select_backend` takes AQL or the host backend
  (core/device/dispatch.cpp:87-104). AQL aborts if its interposition is not live
  (aql.cpp:3531-3546). So a chipStar build needs a streams/events/copies half written for
  `hip.cpp`, or a new backend.
- The host backend (core/device/host.cpp, 825 lines) is a complete implementation of the same
  vtable. It gives a size and a template for a new backend.
- Optional vtable parts may stay null: tapes, vmem, xprefetch, gates, chains, DMA lists, code
  checks (device.h: "Null on a backend without them … that is a supported configuration").
- The core has 653 lines of its own HIP device code that a new backend must replace: copies and
  fills (aql_blit.hip, 247), the step-advance kernel (sched/advance.hip, 76), weight-table patches
  (runtime/wtab.hip, 19), and expert-prefetch logits/plan (runtime/xprefetch.hip, 311; optional).
  The core's `RAD_HAVE_HIP` gates are confined to core/device/, sched/advance.cpp,
  runtime/wtab.cpp, runtime/xprefetch.cpp and engine_diag.cpp.

---

## 3. The plugin interface (docs/PLUGIN.md, abi/)

What a kernel library must export (docs/PLUGIN.md:22-41): `rad_plugin_abi_version()`
(= `RAD_ABI_VERSION`, 26, abi/rad_abi.h:20), `rad_plugin_info()` (kind, name, version,
`build_target`), `rad_kernel_count/at()`, and optionally `rad_kernel_schema_count/at()` for new
ops. Each `RadKernelInfo` row carries an op name, constraints, domain (`RAD_DOMAIN_DEVICE` |
`RAD_DOMAIN_HOST`, rad_abi.h:46), a priority and `launch(const RadArgs*, RadStream)`. Optional
hooks: init/fini, scratch, layout/relayout (the kernel's own weight order), tunables, opd_shape,
replaces. Optional exports: `rad_plugin_open`, `rad_kernel_concurrent`, `rad_kernel_collective`,
`rad_plugin_reference`.

Key lines:

- PLUGIN.md:306 — *"The core assumes nothing about a card's wave size, LDS, fp8 hardware or links."*
- PLUGIN.md:308-311 — *"The vocabulary is what you implement. A model runs on your library when every op its architecture issues has a row there … libref's host code is the specification and `rad-kbench` the check."*
- PLUGIN.md:314-315 — *"fp8 bytes are OCP E4M3 (`e4m3fn`, bias 7, max 448) wherever the engine reads them."*
- PLUGIN.md:319-320 — *"Tensor parallelism is ops you supply. `all_reduce`, `all_gather` and the fused collectives are kernel-library rows like any other; the engine inserts none and assumes no link."*
- PLUGIN.md:162-164 — *"On a card the `RadStream` a kernel receives is a queue the engine writes packets into itself (core/device/aql.cpp), not a `hipStream_t`."*
- PLUGIN.md:172-177 — only `hipLaunchKernel` (plus the copy/memset/sync listed there) takes an engine stream; `hipModuleLaunchKernel`, cooperative launches, rocBLAS/hipBLASLt/MIOpen are refused. Code-object limits (PLUGIN.md:179-187): no hostcall/printf, no dynamic stack, kernargs ≤ 768 B, LDS ≤ 64 KiB per dispatch, uncompressed fat binary.

**Can a non-HIP device be added as a plugin alone? No.** A plugin supplies kernels; it cannot
supply a device. Devices, memory, streams and events come from the `DeviceBackend` vtable, which
is internal to the core (`core/device/device.h`, compiled into the engine), and only `host` and
`aql` exist. The interface is designed for a new backend: one vtable, a frozen
`abi/rad_device.h`, and no vendor names above it. So adding Intel means **a new core device
backend plus a kernel library**. The arch plugin, scheduler, planner, server and format stay as
they are. One caveat: §5 of PLUGIN.md and the loader's code-object checks (`code_mark/code_runs`,
AQL-owned fat-binary registration) assume HIP fat binaries. A SPIR-V or SYCL library would leave
those null, and then "every plugin runs (1)" (device.h:393-395).

---

## 4. Flash-Next's GPU path (arch `qwen4exp_fp8`, recipe w4nl64-i8-hc8m)

Model shape (docs/ARCHITECTURES.md:315-330): 48 layers in 12 × (3 GatedDeltaNet → MoE, 1 QSA
attention → MoE); hidden 2560, hc_count 4 (residual stream 10240); 512 experts × 640 at top-10
plus a shared expert; 24 q / 2 KV heads, head_dim 256; vocab 248,320; QSA indexer 4 heads × 128,
budget 2048; MTP 1 layer.

The arch (`arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, 1,693 lines, plus `arch/common/rad_block_*`) can
issue 78 libr4d ops. Below they are grouped by what a decode step does, with the libr4d row and
file each resolves to under the production recipe. "AMD constructs" covers the file and the shared
headers it includes. LOC is the whole file; most files also hold prefill forms.

Shared headers that carry most of the AMD-specific code (nearly every kernel includes the first
one):

| header | LOC | constructs |
|---|---:|---|
| r4d_common.h | 311 | `update_dpp` ×11 (row reductions, :128), `permlanex16` (:130), `readlane`, `perm`, `s_barrier`, `sched_barrier`, `global_load_tr_b128_v8i16` (gfx12 transpose load), `exp2f`; `R4D_WAVES` = `amdgpu_waves_per_eu` (:39) |
| r4d_fp8_wmma.h | 220 | `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` (:56), `wmma_i32_16x16x16_iu8_w32_gfx12` (:67), `cvt_pk_f32_fp8` / `cvt_f32_fp8` (hardware E4M3 converts) |
| r4d_gdn_wmma.h | 111 | `wmma_f32_16x16x16_bf16_w32_gfx12` (:99), `asm("" : "+v"(x))` VGPR barrier (:76) |
| r4d_dt16.h | 107 | bf16/f16 WMMA (:30, :51), `ballot_w32`, `cvt_pk_f32_fp8`, `cvt_pkrtz` |
| r4d_q8.h | 39 | `cvt_pk_fp8_f32` (hardware f32→E4M3) |
| r4d_chain.h | 85 | `kernarg_segment_ptr` (:34), `s_sleep` (:65): chained launches, **AQL-only** |
| r4d_moe_sort.h | 198 | `update_dpp` ×4 wave scans, `readlane` |

Across the ~28k lines of decode-path files and headers, the AMD builtins are:
`cvt_pk_fp8_f32` 20, `update_dpp` 14, `readfirstlane` 10, `perm` 10, `fence` 9, `readlane` 8,
`cvt_pk_f32_fp8` 8, bf16 WMMA 5, iu8 WMMA 2, fp8 WMMA 1, f16 WMMA 1, and one or two each of
`sched_barrier`, `s_sleep`, `s_barrier`, `exp2f`, `cvt_f32_fp8`, `cvt_pkrtz`, `rcpf`, `permlanex16`,
`kernarg_segment_ptr`, `global_load_tr_b128`, `fdot2`, `ds_bpermute`, `ballot_w32`. The only real
inline asm is `s_wait_storecnt 0x0` (all-reduce paths: r4d_ar_oneshot_2rank_exact.hip:98/160/269,
r4d_ar_wide.h:55, r4d_fused_quant_fp8.hip:484). The other `asm` uses are empty, register-constrained
compiler barriers (r4d_gdn_wmma.h:76, r4d_prefetch.hip:32/50).

Verdict key: **plain HIP** = scalar HIP that compiles on another target once a shim header
replaces the lane builtins (DPP/readlane/ballot/perm/fp8 convert) with portable shuffles and
software conversion. **needs rewrite** = matrix-unit (WMMA) code, whose per-lane fragment layout
is baked into the indexing, or AQL/peer-specific code.

| step part | op → row | file (LOC) | AMD-specific | verdict |
|---|---|---|---|---|
| MoE routing | `router_gemm_topk_scatter` / `router_topk_scatter` → `*_bf16`; `moe_scatter_i32`, `moe_gather_bf16`, `moe_gather_inv_bf16` | r4d_moe.hip (3,564) + r4d_moe_sort.h (198) + r4d_moe_gather.h (92) | `kWave=32` lane math (:61, :159, :227, :273-316), DPP scans, readlane/readfirstlane, `__ballot`, router GEMM on bf16 WMMA (`rg_mma` :3241) | topk/scatter/gather plain HIP; router GEMM **needs rewrite** (or a GEMV) |
| expert GEMV/GEMM, 4-bit | `moe_gemm_q` → `moe_gemm_w4a8` at `w4nl64a8h` (`moe_gemm_launch<2>`, :2909-2912) | r4d_moe.hip kernel :664; code expand `w4_e4m3` :441-459 | 3 × `__builtin_amdgcn_perm` per 8 codes **expand to E4M3 bytes** and feed **`wmma_f32_16x16x16_fp8_fp8`** (r4d_fp8_wmma.h:56); LDS staging, `R4D_WAVES`, uniform-load tricks for gfx12 `flat_load` | **needs rewrite** (no FP8 matrix on Xe2) |
| expert GEMM, 10 protected bf16 | `moe_gemm` → `moe_gemm_bf16` | r4d_moe.hip :1794 (`mb_mma` :1777) | bf16 WMMA | **needs rewrite** |
| expert activation quant | `had_quant_act_fp8`, `gated_had_quant_fp8` | r4d_quant_act_fp8.hip (359), r4d_fused_quant_fp8.hip (1,066), r4d_fwht.h (56) | `cvt_pk_fp8_f32`, 128-wide FWHT by DPP/shuffle, `kWave` | plain HIP (software E4M3) |
| trunk W8A8 int8 | `rmsnorm_had_quant_i8g` / `quant_act_i8g` → `gemm_nt_q` = `gemm_i8a8_nt_m16` | r4d_gemm_fp8a8.hip (1,491), r4d_fp8_frag.h (37) | `wmma_i32_16x16x16_iu8` (r4d_fp8_wmma.h:67, used at r4d_gemm_fp8a8.hip:146-147); weights relaid into WMMA fragment order at load (r4d_fp8_frag.h); 36 KiB LDS staging | **needs rewrite** (maps well to INT8 XMX) |
| lm_head | `logits_gemm` → `logits_gemm_i8`; `logit_rerank_i8` | r4d_gemm_fp8a16_nt_m1.hip (906); r4d_rowtopk_bf16.hip (619) | bf16 WMMA in `lm_wmma_body` (:581), "over the 64 KiB" LDS note; rerank has no builtins | GEMM **needs rewrite**; rerank plain HIP |
| FP8 dequant | everywhere E4M3 is read: hc mixing, MTP fc, n-gram rows, KV cache, expert scales | r4d_q8.h, r4d_dt16.h, r4d_fp8_wmma.h | hardware `cvt_pk_f32_fp8` / `cvt_pk_fp8_f32` | plain HIP (software or table convert; E4M3 is exact in f16) |
| IQ4R (w4nl64) dequant | inside `moe_gemm_w4a8` | r4d_moe.hip :425-500 | `v_perm_b32` 16-entry table (kW4nlPerm, r4d_args.h:270), output E4M3 | the dequant itself is plain (a 16-entry LUT); the consumer is the fp8 WMMA |
| Gated DeltaNet (decode) | `gdn_conv_update`, `gdn_recurrent_update` / `gdn_conv_recurrent_update`, `gdn_gated_rmsnorm` (+ `gated_had_quant_i8g`) | r4d_gdn_conv_w4_h128_bf16.hip (616), r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip (1,132), r4d_gdn_gated_rmsnorm_h128_bf16.hip (141), r4d_gdn_gated_norm_had_quant_i8.hip (148) | readfirstlane, `cvt_pk_fp8_f32`, nontemporal ×6, 9 `__shared__`, system-scope atomics ×2; **no WMMA in the decode kernels** | plain HIP |
| Gated DeltaNet (prefill only) | `gdn_chunk_scan`, `gdn_kkt_solve` | r4d_gdn_chunk_scan…hip (608), r4d_gdn_kkt_solve…hip (314) | bf16 WMMA, `sched_group_barrier`, `wave_barrier` | needs rewrite (prefill) |
| QSA indexer | `qsa_block_key`, `qsa_score`, `qsa_select`, `qsa_tail_store`, `qsa_work` | r4d_qsa_block_key_bf16.hip (227), r4d_qsa_score_bf16.hip (469), r4d_qsa_select_bf16.hip (1,208), r4d_qsa_tail_store_bf16.hip (120), r4d_qsa_work_bf16.hip (221) | score: bf16 WMMA in `qsa_score_mx_kernel` (:245, :311); select: `__ballot` ×11 with 32-lane masks, readlane/readfirstlane, 16 `__shared__` | score **needs rewrite**; the other four plain HIP |
| sparse attention (QSA layers) | `attn_paged_gate_quant` → `attn_decode_gq_h256_gqa12_fp8kv` (page of 4 = a QSA block) | r4d_attn_paged_h256_gqa12.hip (23, defines) → r4d_attn_paged_gqa8.hip (429) → r4d_attn_decode_h256_gqa6.hip (1,046) | f16/bf16 WMMA via r4d_dt16.h (`D::wmma` :453, :532), `fdot2`, `perm`, `cvt_pk_fp8_f32`, LDS 6-12 KiB | **needs rewrite** |
| q/k norm + RoPE + gate + KV store | `qk_norm_rope_gate_kv_store_fp8` | r4d_qk_norm_rope_gate.hip (477) | `cvt_pk_fp8_f32`, `__shfl` ×6 | plain HIP |
| hyper-connections | `hc_read_e4m3`, `hc_write`, `hc_project`, `hc_enter`; at tp2 fused into `ar_hc_write` / `ar_gather_hc_write` | r4d_hc_bf16.hip (3,777), r4d_hc_fp8.h (43) | `kWave=32` (:72, 144 wave mentions), 46 `__shared__`, LDS 25-58 KiB, `cvt_pk_*fp8`, one bf16-WMMA kernel `hc_down_wide_kernel` (:1534, mma at :1663) | mostly plain HIP; `hc_down_wide` **needs rewrite**; LDS near 64 KiB |
| n-gram gather / PLE | `ngram_ids_i32`, `embed_lookup_q_bf16` (E4M3 rows, fixed scale), `ple_gate_bf16`, `ple_conv_bf16` | r4d_ngram_ids.hip (220), r4d_model_bf16.hip (2,286), r4d_ple_gate_bf16.hip (181), r4d_ple_conv_bf16.hip (240) | E4M3 convert only; production reads the 47.7 GiB table from pinned host RAM (`--ngram-placement ram`) | plain HIP |
| small ops | rmsnorm, add, mul, silu_mul, sigmoid, scale, fill, cast, gather_rows, rope_table, quant_act_i8 | r4d_model_bf16.hip, r4d_quant_act_i8.hip (188), r4d_encoder_bf16.hip | `cvt_pk_fp8_f32`, `__shfl` | plain HIP |
| sampling | `sample_chain_f32` and the stage rows (`sample_temp/topk/topp/minp/penalties/ngram_ban/pick/argmax/logprobs…`), `pair_argmax_f32` | r4d_sample_stages_f32.hip (1,550), r4d_sample_chain_f32.hip (384), r4d_sample_ngram_f32.hip (214), r4d_pair_argmax_f32.hip (105) | no builtins; wave-32 constants, 19 `__shared__` | plain HIP |
| MTP | `mtp_enter_e4m3` (r4d_hc_bf16.hip); the MTP layer reuses the GDN/attn/MoE kernels; 2-bit draft head → `gemm_w2a8_nt` | r4d_gemm_w2a8_nt.hip (297) | iu8 WMMA (`w2_mmai` :68) | enter plain; draft head **needs rewrite** (INT8 XMX fits) |
| all-reduce (tp2) | `ar_oneshot_2rank_exact`, fused `ar_hc_write`, `ar_gather_hc_write`, `ar_rmsnorm_*` | r4d_ar_entry.hip (1,852), r4d_ar_oneshot_2rank_exact.hip (540), r4d_ar_seq.h (89), r4d_ar_hc.h (111) | system-scope `__hip_atomic`, `__builtin_amdgcn_fence`, `s_barrier`, inline `s_wait_storecnt`, peer pointers, fine-grained memory | **needs rewrite**; not needed on one card |
| expert prefetch hint | `prefetch` | r4d_prefetch.hip (79) | empty `asm volatile("" :: "v")` | plain after a macro |
| chained launches | `r4dchain_*` forms of several kernels | r4d_chain.h | `kernarg_segment_ptr`, `s_sleep`; AQL backend only | compile out (`rad_dev_chain_enabled` = 0) |

Totals: libr4d is 107,784 lines (64,410 of .hip/.h; the rest is rows, layout, shapes and a
28,644-line selftest). The decode-path files and headers above are ~28k lines. About 9 kernels
need a matrix-unit rewrite for decode: expert w4 GEMM, expert bf16 GEMM, router GEMM, trunk i8a8
GEMM, lm_head, QSA score, decode attention, hc_down_wide, and the w2a8 draft head. Everything
else is scalar HIP behind ~25 distinct builtins.

**What `kWave = 32` in r4d_moe.hip:61 means.** It is the wavefront width of RDNA's wave32 mode
(the gfx12 WMMA builtins are the `_w32` variants), hard-coded as a compile-time constant, not
read from the device. Kernels split `threadIdx.x` into `wave = tid / 32` and `lane = tid % 32`
(:276, :316, :360), size blocks as `NW * kWave` (`__launch_bounds__`, :273, :310), and bound
problem sizes by it: `n_expert <= MAXOWN * kWave` (:159), with each lane owning `n_expert/32`
router logits, and `top_k <= kWave`, where round r parks its winner on lane r (:227). The
reductions and scans assume 32 lanes in lockstep (DPP row ops, `permlanex16` across the two
16-lane halves, readlane by lane index, 32-bit ballot masks). On Intel this needs a sub-group of
exactly 32. chipStar's shuffles require that too. Xe2 offers SIMD16 and SIMD32 sub-groups (to
confirm on the card), and SIMD32 on Intel is usually the less efficient width.

**Reference coverage.** All of the conventional ops above have a libref host implementation
(`libref/ref_registry.cpp`, e.g. `moe_gemm_q` :1458 with w4nl in ref_moe.cpp). The only ops with
no reference are the device-only fusions: `ar_*`, `attn_paged_gate_quant`,
`gate_quant_fp8`/`gated_*quant*`, `rmsnorm_had_quant_i8g`, `qk_norm_rope_gate(_kv_store)`,
`swiglu_clamped_quant_fp8`, `moe_*_inv`, `router_sigmoid_topk_scatter`, `prefetch`. So a new
library can be checked op by op with `rad-kbench` against libref, and it can skip the fusions if
the arch falls back to the unfused chain (not verified for every fusion).

There is also a portability asset: **tests/r4d_emu/emu/hip/hip_runtime.h** (242 lines) gives host
semantics for every gfx12 builtin libr4d uses, including all four WMMA variants' fragment layouts,
DPP, `permlanex16`, `ds_bpermute`, `global_load_tr` and the fp8 converts. It already runs 14
libr4d units on the CPU (tests/CMakeLists.txt:386-419). It is the specification for a shim header.

---

## 5. Tensor parallel: the exact all-reduce and wht6

- **Process model:** one process, one thread per rank, one address space. *"No IPC handles … Once
  hipDeviceEnablePeerAccess has run, a device pointer from rank 0's card is valid on rank 1's
  card … the exchange is a pointer written to a shared array plus a barrier"*
  (libr4d/r4d_ar_entry.hip:8-13). `hipIpcGetMemHandle` is still attempted at
  r4d_ar_oneshot_2rank_exact.hip:307, but only for an out-of-process peer, and a failure is
  ignored.
- **Exact, 2 ranks** (r4d_ar_oneshot_2rank_exact.hip:1-20): a one-shot push. Each block writes its
  chunk of the input into the **peer's** scratch over PCIe BAR (P2P), sets the peer's per-block
  flag with a system-scope release (`store_sys_rel`, :29-31), spins on its own flag
  (`load_sys_acq`), and reduces local + received in f32. The scratch is double-buffered by a
  per-launch parity, with device-resident sequence counters, so it is graph- and tape-replay safe.
  It uses fine-grained (uncached) device memory (`hipExtMallocWithFlags(…Finegrained)`, :297) and
  `s_wait_storecnt` drains. **It needs peer access** (`hipDeviceEnablePeerAccess`, :502;
  `device.p2p == 1` row constraint). 3/4/8-rank one-shot and two-shot variants exist
  (r4d_ar_*_Nrank_*.hip).
- **wht6** (r4d_ar_oneshot_2rank_wht6.h:1-35): the same transport and flags, but the payload is
  compressed. Each group of 64 is rotated by an orthonormal Walsh-Hadamard, scaled by its amax and
  packed as 6-bit uniform codes + a bf16 scale (6.25 bits/element, about 2.56× fewer bytes than
  bf16). The receiver unpacks both halves, adds in f32 and inverse-rotates. Both ranks fold the
  same pair, so their results are bit-identical (built with `-ffp-contract=off`). It is used above
  a message size (`--tp-wire-min-kb`). `WAVE 32` and lane-group shuffles are hard-coded (:42-46).
  Production currently runs `exact`.
- **Without P2P:** every collective row has a derived `_host` twin (`device.p2p == 0`) whose
  mailbox lives in pinned, coherent, mapped host memory (libr4d/r4d_rows.cpp:9297-9305;
  `hipHostMalloc(…Coherent|Mapped|Portable)` at r4d_ar_oneshot_2rank_exact.hip:478-485). So **host
  staging is already supported**. It still needs system-scope atomics from both cards on shared
  host memory.
- On a single B70 none of this is needed.

## 6. Weights: FP8 and the "IQ4R" layout

- **FP8 is OCP E4M3** (`e4m3fn`, bias 7, max 448) everywhere the engine reads fp8 (PLUGIN.md:314-318).
  This container uses it for: hc mixing matrices and MTP fc as E4M3 rows with an **f32 scale per
  row group** (128 columns for down/fc, 80 for up, because the low-rank width 320 is 2.5 blocks of
  128); the n-gram table at **one fixed bf16 scale** (1.993e-4); the expert scales; the KV cache
  (`--kv-cache-dtype fp8`); and activations between ops (experts take E4M3 activations from
  `had_quant_act_fp8`). FP8 checkpoints elsewhere use 128×128 tile scales; Flash-Next does not.
  The trunk and lm_head are **int8**, symmetric, with a bf16 scale per (row, 128 of K), served
  W8A8 against per-group int8 activations.
- **"IQ4R" = `w4nl64a8h`** (docs/MOE-W4.md:77-170, libr4d/r4d_layout.cpp:136-193):
  - encoding `affine:codes=u4[1x1],scale=fp8_e4m3[1x64],scale.1=f32[*x*],table=f32{1x16}/fwht128`;
  - weight `[N][K/2]` bytes, row-major, two codes per byte. After libr4d's load-time reorder,
    byte b of a dword holds element b (low nibble) and b+4 (high nibble); the container stores
    canonical order;
  - code → level from the 16-entry **w4nl codebook** ±{1.125, 3.25, 5.5, 8, 11, 14, 18, 22}, each
    exact in E4M3 (and in f16), not an integer grid;
  - scale: one **E4M3 per (row, 64 of K)** times a fixed f32 of **2^-13** (`kW4G64Scale`,
    r4d_args.h:292). Stored scales sit in 2^-15…2^-10.6;
  - weights are rotated along K by a **128-wide Hadamard** at convert. The kernel never
    un-rotates; the activation is rotated the same way before its fp8 quantisation, and the factor
    of 128 is folded into the stored scale;
  - 4.125 bits/weight; 10 "protected" experts are kept as bf16; codes come from GPTQ plus three
    sweeps of coordinate descent.
  - Implication for Xe2: the levels are not integers (×8 would reach 176), so there is no exact
    INT8 path. The exact translation is codes → f16 and E4M3 activations → f16 into an **f16 XMX**
    (DPAS) with f32 accumulate. The products match the FP8 WMMA's exactly; only the summation
    order differs.

---

## 7. Verdict

Facts that set the frame:
- One B70 has 32 GB, the same as one R9700. Upstream ships a supported **one-card** Flash-Next
  config (deploy/compose/flashnext-tp1.yaml): experts that do not fit sit in a 42 GiB pinned host
  pool, the n-gram table is read from disk, and the host needs ~64 GB RAM. Prefill is link-bound
  (2,545 tok/s at 4096-token steps on the R9700). So the target is a single card with expert
  streaming. TP and peer access are not needed.
- Decode is GEMV-shaped: at most 8 seqs × 4 tokens, and top-10 of 512 experts touches each expert
  with about 0.6 rows. So decode is memory-bound, and matrix units matter mostly for prefill.
- FP8 matrix hardware is not needed for this model. The trunk is int8 (XMX has INT8), and the
  experts can run as f16 with identical products.
- The engine's own claim (aql.cpp:4-8) is that HIP's launch path (0.5-1.3 µs host time a
  dispatch, a system-scope fence per kernel) costs milliseconds at ~1,600 dispatches per rank per
  decode step. Any non-AQL backend pays something like that, unless it records the pass once and
  replays it (the core's "tape" hooks are meant for this).

### (a) chipStar + a new kernel library, single GPU

Work:
1. Core: a streams/events/copies/fills half for `core/device/hip.cpp` on plain HIP (host.cpp's
   825 lines show the size; estimate ~800-1,500 lines). Select it when AQL is absent, instead of
   aborting. Do not export the engine's `hipLaunchKernel`/`__hipRegister*` in this build. Leave
   tapes, vmem, xprefetch, DMA lists and chains null. Port the core's 653 lines of HIP kernels
   (aql_blit, advance, wtab; xprefetch can stay off).
2. Kernels: a shim header for the ~25 builtins (start from tests/r4d_emu's semantics), software
   E4M3 converts, macros for `R4D_WAVES`/asm barriers, chain forms compiled out. Then replace the
   ~9 WMMA kernels with scalar FMA/dot GEMVs, because chipStar has no warp-matrix support and so
   no route to XMX. Estimate 3-6k lines new or changed, reusing the plain-HIP rest (~20k lines)
   unchanged.

Risks: chipStar compiling this code at all (clang's SPIR-V target, the `_w32` builtin surface,
64-bit `__ballot`, system-scope `__hip_atomic`, `hipExtMallocWithFlags`, nontemporal builtins);
forced sub-group 32 on Xe2; launch overhead on ~1,600 dispatches a step through chipStar's Level
Zero path; no matrix units, so prefill would be very slow; pinned-host-pool streaming performance
through chipStar's memcpy. Payoff: the shortest route to "it runs and matches libref". It answers
whether the seam holds, but it would not give a competitive engine.

### (b) Native Level Zero / SYCL device backend + SYCL kernel library

Work:
1. Core: a `DeviceBackend` on Level Zero (estimate ~2-4k lines): immediate command lists as
   streams, ze events, USM device/host (pinned) allocation, the copy engine for
   `memcpy_list_async` (the host-pool expert streaming), `zeVirtualMem*` for elastic memory
   (optional), and optionally command-list record/replay as "tapes" to beat launch cost. Port the
   653 lines of core kernels. Define what `RadStream` is (e.g. a `sycl::queue*` that wraps the
   command list).
2. Kernels: a SYCL library for the ~45 ops a single-card decode+prefill step issues. The ~30
   plain ops translate mostly mechanically (SYCLomatic can draft them). The ~9 matrix kernels are
   rewritten on `joint_matrix` or ESIMD DPAS: INT8 for the trunk, lm_head and draft head, f16 for
   experts, attention, router, QSA score and hc_down. Estimate 15-25k lines, with weight relayout
   hooks (`relayout`) to put weights in DPAS-friendly order at load.

Risks: the size (this is most of a second libr4d); sub-group 16 vs 32 decisions throughout;
reaching AQL-class dispatch cost with L0 immediate lists; matching the bit-level contracts that
libref checks; Intel's driver/compiler maturity on Xe2 for joint_matrix. Payoff: the only option
that can use XMX and the copy engine properly, so it is the only one with a real performance case.

### (c) Vulkan compute

The same core work as (b), on Vulkan: buffers via `VK_KHR_buffer_device_address`, which is
mandatory because RadTensor, the weight tables and the MoE pointer tables are raw 64-bit device
pointers; timeline semaphores for events; a transfer queue; sparse binding for vmem. Kernels as
GLSL/Slang compute shaders with `VK_KHR_cooperative_matrix` for INT8/f16. The size is similar to
(b) or larger (~20-30k lines of shader plus a 3-4k-line backend), with weaker tooling: no
SYCLomatic, and pointer arithmetic is awkward in shaders. The upside is that it runs on Mesa/ANV
with no oneAPI stack and the same code would also run on the AMD cards. Risks: the pointer-heavy
design against Vulkan's buffer model, cooperative-matrix shape limits on ANV, and per-dispatch
cost on ~1,600 dispatches (needs pre-recorded command buffers).

### Recommendation

Only **(b)** is worth it as a real port. If the goal is just "does it run on the B70", do **(a)**
first, scoped to a one-card plain-FMA decode, as a bounded probe (core HIP stream half + shim +
scalar GEMVs). Its main job is to prove the device seam and the arch/scheduler work unchanged on
a non-AMD backend. Do not plan (c) unless intel-host cannot run oneAPI/Level Zero. In all three, the
model needs no FP8 matrix hardware, and the int8 trunk suits Xe2's INT8 XMX.

Not checked here: intel-host's RAM (the one-card config wants ~64 GB), the B70's PCIe link width and
speed, how many B70s intel-host has, and whether Xe2 runs SIMD32 sub-groups well. These need a look at
intel-host, which this audit did not touch.
