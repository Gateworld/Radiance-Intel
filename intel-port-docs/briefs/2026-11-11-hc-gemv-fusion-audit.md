# Desk audit: the decode hc chain's producer -> consumer pairs, and the standalone GEMV launches (owner note 232)

Run 285, 2026-11-11 09:3x UTC. Desk only, no build, no card time. Sources: the kernel table of M403 (image 309, the
split default, `svcmtpx-gnt0-hc8-rp-gnx-prof`, 323 steps, 28.2 ms of device time a rank-step), the tape-edge table of
M373 (image 296, `-prof-edges`, 355 steps), `src/libxe/ports/r4d_hc_bf16.cpp` (level 8's decode chain, HEAD 78538f4),
`src/arch/common/rad_block_moe_fp8.h` / `rad_block_attn_gated_fp8.h` / `rad_block_gdn_fp8.h` / `rad_qsa.h` (the blocks'
op order), `src/libxe/xe_gemv_i8.cpp` (the x3 / pair adapters). "A step" = one rank-step of the MTP-on single prompt
(M403's launches / 323). The rule applied is owner note 200's: fuse only phases that share a shape and whose dependency
is satisfied inside one work-group (the N1 lesson: a fence + last-arriver fold cost +10 %); exact bits only.

## 1. The hc connection's decode chain today (97 connections a step, level 8 + the defaults of notes 202 / 215)

    ar_kernel<0,0,5> (105.6 a step, 20.1 us) -> hc_v2_write_norm (93.5, 7.3 us) -> hc_v2_down_rows<4,0,2> (97.5, 25.5 us)
    -> hc_v2_gu<4> (96.5, 28.5 us: the gains + the up mix) -> hc_quant_rows<1> (101.6, 3.5 us) -> the first consumer

Shapes: write_norm hc x rows work-groups of 256 threads (one per (stream, row); ~0 LDS); down_rows (lr / 2) x (rows / 4)
groups of 64 threads (two columns of W_down a group, four rows a wave); gu (hc x rows gain blocks + n / 16 = 160 mix
blocks) x 256 threads, ~54 KB LDS (the up mix's `d` + its 64 staged W_up rows); quant_rows one wave per (row, 128-channel
scale group) = rows x 20 items, 8 waves a block; the first consumers 256-thread GEMV launches.

| pair (producer -> consumer) | a step | what the consumer needs from the producer | fits one work-group? | saving if fused | verdict |
|---|---|---|---|---|---|
| write_norm -> down_rows | 97.5 | a column of `d` needs the WHOLE normalised row (all hc x n = 10,240 values of nbuf, written by 4 norm groups) | no: a cross-group dependency; NR (note 200 1a) ran both in one launch with a sync and lost (74 vs 32 us) | 0.77 us boundary + part of 7.3 us a connection = <= 0.78 ms at stake | measured lost (NR); closed |
| down_rows -> gu (the up mix half) | 96.5 | the mix needs ALL 320 values of `d` for its rows (every column group's output) | no: the hc merge (RADIANCE_XE_HC_AB, upstream 1.14's grid) did it with a ticket wait and lost (179 us vs 67 a connection) | <= 2.5 ms at stake, none recoverable this way | measured lost (AB); closed |
| down_rows -> gu (the gains half) | 96.5 | the gains need only nbuf (not `d`) | yes -- and it is DONE: GU runs the gains and the mix in one launch (default since 11-07) | done | done |
| gu -> hc_quant_rows (note 232's first candidate) | 101.6 | a code's scale is the amax over a 128-channel group of `x`; an up-mix block writes 16 channels (kUpRows / hc) of `x`, so a group spans 8 mix blocks | no at today's tile (a counter + last arriver = the N1 pattern). A re-tile to 128 channels a block = 512 W_up rows a block (164 KB of staging against the 60 KB LDS cap, so 8 serial 64-row passes) and n / 128 = 20 blocks x 1 row group on 32 Xe cores: lost by construction (the NR / AB lesson: fewer, longer blocks) | 3.5 + 0.77 us x 97 = 0.41 ms at stake (1.5 % of the step) | NOT a build candidate on the producer side. The consumer side (each GEMV group quantising `x` itself) is exact but redundant work in every group of every consumer launch, and the ROUTED experts' rotated codes (fwht128 + quant, `rq`) would move into the format C kernel, which is instruction-issue bound: no |
| hc_quant_rows -> the first GEMV launch | 101.6 | the GEMVs need the codes (i8 sets) or `x` (bf16 sets) | a HORIZONTAL fold (quant waves as extra blocks of a bf16 consumer's launch that does not read the codes) is exact and waits on nothing: at FFN connections the router pair launch (bf16, 48 a step), at attention connections the indexer's qk launch (bf16, 13 a step); at GDN connections the first consumer is the i8 + bf16 pair, whose i8 set needs the codes: no fold there (36 a step) | 61 x (3.5 + 0.77) = 0.26 ms (0.9 %) | possible but DOMINATED: groups A / B below put an int8 set that needs the codes into those same launches, which is worth more; the two exclude each other. Not proposed |
| ar_kernel -> write_norm | 93.5 | the write needs the all-reduced `y` row; the norm needs its stream's whole row (sum of squares over n = 2,560) | not at today's blocking: an all-reduce message of 4 rows is split over ~6 blocks, so a row's reduction spans blocks. It WOULD fit if the LL kernel ran one row a block at decode (4 rows -> 4 blocks of 5 KB), with each block then applying the write to its row's 4 streams and finishing the 4 norms in hc_v2_norm_body's order (exact by construction) | 7.3 + 0.77 us x 93.5 = 0.75 ms at stake (2.7 %), minus whatever the re-blocked all-reduce costs or gains (AR2: rows 2's 3-block form was the slow one) | the ONE hc item left with real size; it changes the all-reduce's blocking, so it needs a design note (note 219 B's area) and a bench of the 1-row-a-block LL form first, not a build. Proposed to the coordinator as a design, not started |
| hc_v2_gu -> (the gain's consumer: the next connection's write) | 96.5 | `inj` (4 gains a row) is read by the NEXT connection's write_norm, 2 blocks later | cross-launch by nature | -- | nothing to fuse |

Summary of the hc chain: after GU and WRITE_FUSE (both default), every remaining pair either carries a whole-row
dependency across work-groups (norm -> columns, columns -> mix, mix -> codes) or was measured lost (NR, AB). The quantiser
cannot be folded into its producer at any tile that keeps the card full. The only item with size is the all-reduce +
write + norm as one launch at one row a block (~0.75 ms at stake), which is a design, not a candidate under this note.

## 2. The standalone GEMV launches a step (note 232's addendum; M403: gemv_i8a8<1,4,1,0,5> 46,670 / 323 = 144.5 a step at 13.4 us = 1.94 ms)

What the 145 are (the blocks' issue order; the counts from M403, the pairings from the M373 edge table):

| launches a step | kernel | op / weight | input it reads | where in the step | consumer | a sibling reading the same activation at the same point? |
|---|---|---|---|---|---|---|
| 13 (12 attention layers + the draft head's) | gemv_i8a8 | `self_attn.o_proj` | attn_out codes (gate_quant_fp8's) | after the attention gate | ar_kernel | none: the only reader of attn_out |
| 36 | gemv_i8a8 | `linear_attn.out_proj` | gdn_o codes (the gated norm's) | after the GDN update + norm | ar_kernel | none |
| 48 | gemv_i8a8 | `mlp.shared_expert.{gate,up}_proj` (640 x 2,560 a rank) | the FFN hc read's int8 codes (`w.h` q8 / s8) | issued right behind the routed gate_up (format C) -- the AMD build overlaps them on two queues; our in-order list serialises them | gated_quant_pair | YES: the router pair (gemv_bf16_pair: `mlp.gate` 512 cols + `shared_expert_gate` 1 col, bf16 over the same read's `x` plane), 47.7 a step, issued at the block's start. Nothing between them depends on either |
| 48 | gemv_i8a8 | `mlp.shared_expert.down_proj` (2,560 x 320 a rank) | sff codes (gated_quant_pair's shared half) | right behind the routed down | moe_gather | none: the routed down reads eff (a different input, a different kernel family) |
| 13 | gemv_i8a8_x3 (pair 5, default) | `q_proj|gate, k_proj, v_proj` | the attention hc read's codes | after the QSA indexer's chain | qk_norm_rope_gate | YES: the indexer's `index_qk_proj` GEMV (gemv_bf16<0,0,4,0>, 15 a step at 19.1 us, bf16 over the same read's `x`), issued FIRST after the read; its chain (two rmsnorms, rope, qsa_work / block_key / tail / score / select) only feeds the attention kernel |
| 36 | gemv_i8a8_bf16_pair (pair 4, default) | `in_proj_qkv` + `in_proj_a|b` | the GDN hc read's codes + `x` | first after the read | gdn_conv_recurrent_update | already paired; nothing else reads that activation before the update |
| 3 | mtp_rows_gemv / gemv_w2a8_wide | the draft head | -- | the draft passes | -- | the head's own path, not audited here |

So of the 145 standalone int8 GEMV launches, 48 (the shared gate_up) have a sibling and 97 (the output projections and
the shared down) do not. Two groups come out:

**Group A -- the FFN block's start: the router pair + the shared expert's gate_up as ONE launch (48 a step).** Today two
launches: gemv_bf16_pair (11.8 us: router 512 x 2,560 bf16 = 2.6 MB + the 1-column gate) then, ~5 launches later, the
standalone gemv_i8a8 for the 640 x 2,560 int8 shared gate_up (~1.6 MB: at 13.4 us average it is launch / latency bound,
its byte floor is ~3 us). Both read only the hc read's outputs (`x` and its int8 codes); the shared gate_up moves ahead of
the routed gate_up (exact: it never depended on it; the two were only "interleaved" for the AMD build's second queue).
Form: the gemv_i8a8_bf16_pair machinery with the bf16 set being today's router pair (two bf16 weights) -- one int8 set +
two bf16 sets, each row the standalone kernel's bits (pairs 4 / 5's rule). Expected saving: the smaller launch's fixed cost
plus one boundary, ~8-10 us x 48 = **~0.4-0.5 ms a step (1.4-1.8 %)**. Needs: a merged op declared by MoeFP8 at decode
rows (as op_inab / op_qkv3 are, behind RADIANCE_XE_GEMV_MULTI or its own flag; the fallback is today's two ops), its row
in xe_registry / xe_gemv_i8.cpp, the CPU row test (bytes), the card row check, profile first, M-pair, rotation.

**Group B -- the attention block's start: the indexer's qk projection + the x3 as ONE launch (13 a step).** Today
gemv_bf16 (19.1 us, bf16 over `x`) then, after the indexer's chain, the x3 (35.7 us, int8 over the codes). No dependency
between them (the indexer's output reaches the attention kernel, not the projections). Form: an "x3 + one bf16 set"
instance of the same multi-set kernel. Expected saving ~15 us x 13 = **~0.2 ms a step (0.7 %)**. Same gates.

Together ~0.6-0.7 ms of the 28.2 ms rank-step (~2.2 %), i.e. roughly +1.5-2 tok/s on the 88.99 reference, inside pairs
4 / 5's kind of work (~one run with the two builds and the card checks). Not started: note 232 is an audit; the build is
the coordinator's / owner's call. If built, A first (the larger, 48 a step), B with it (the same kernel template).

## 2b. The designs (run 285, after the coordinator's "build now")

**Group A -- `router_gemm_topk_scatter_quant_sh` (built this run, opt-in `RADIANCE_XE_GEMV_MULTI2=1`).**
- libxe: `gemv_i8a8_bf16x2_kernel<WS_BF16, MB, WCC>` = one int8 set + two bf16 sets over one activation's two planes
  (work-groups [0, g1) the gate_up rows by `gemv_i8a8_set_row`, [g1, g1 + g2) the router rows and the rest the gate row
  by `gemv_bf16_set_row` -- the bodies pairs 4 / 5 already copy from the standalone kernels, so each row's bytes are
  today's). The adapter takes the fused router op's 13 operands + `a_q, a_scale, b_sh, b_sh_scale, y_sh` (+ param
  `N_sh`), launches the three sets as one grid (M in chunks of 4 rows, as the pair adapter), then the selection + sort +
  quantiser launch through the fused op's own tail (`xe_router_topk_quant_tail`, refactored out of
  `xe_rad_router_gemm_topk_scatter_quant`). A form the paired kernel is not built for (f32 scales, an unaligned bf16 set,
  a GEMV env form) runs the fused op and the standalone int8 GEMV: the same bytes, one launch more, said once.
- The MoE block (`rad_block_moe_fp8.h`): `op_rtq3` declared beside `op_rtq2` when the layer has the gate and the shared
  arm's gate_up is the plain int8 form over the block input's fed codes (`sh_gate_up.i8 && !op_q && w.h.q8_fed`), reads
  {`w.h.x`, `w.h.q8`, `w.h.s8`}, writes the fused op's nine + `w.sgu`; issued at the router's slot with `hq_fused` and
  `gu_fused`; `shared_up(.., gu_done)` skips `sh_gate_up.step`. Dependency check (the coordinator's condition): the gate_up
  reads exactly `w.h.q8 / s8` today (`LinearFP8::step`'s int8 path with `op_q` null); in the MoE pass nothing writes
  them (the routed arm's quantiser writes `w.hr.*`; `nq_h` runs only without `ext_in`, and Flash-Next's blocks run with
  it); `w.sgu` is excluded from buffer sharing (`rad_buf_concurrent`, the AMD build's concurrency rule), so writing it
  ahead of its declared op is safe, as `w.sgate`'s is; its only reader is the gated quantiser after both gate_ups.
- Gates: the ctest `xe_router_sh_cpu` (today's two launches vs the one, every output byte for byte at M 1 / 4 / 8 / 12),
  the card row check (`logs/gshstep.sh`), the M-pair `-gvm2-` against the flagless reference M402 (88.99, texts = M285),
  the `-prof` twin (expected: `gemv_i8a8_kernel<1,4,1,0,5>` standalone 145 -> ~97 a step, `gemv_bf16_pair_kernel` 47.7 ->
  0, the new kernel ~48, `router_topk_scatter_quant_kernel` unchanged), one 10-minute rotation `-gvm2` against H156 on
  the new image (both halves on the same image). Expected ~0.4-0.5 ms a step (1.4-1.8 %).

**Group B -- `gemm_nt_q_x3_bf16` (designed; to build after A's gates, the coordinator's review first).**
- Today at an attention layer (13 a step): `l.qsa.step()` issues the indexer's qk projection (`proj.step`: a bf16
  `gemm_nt`, width (heads + 1) x head_dim = 640, over `w.h.x`), then its norms / rope / qsa_work / block_key / tail /
  score / select; then `l.attn.step()` issues the x3 (`op_qkv3`: qg | k | v over `w.h.q8 / s8`) and the rest. Nothing of
  the indexer's chain reads qg / k / v, and the x3 reads nothing the indexer writes: the two projections are independent
  reads of one hc read's two planes, a boundary apart only because they belong to two blocks.
- libxe: `gemv_i8a8_x3_bf16_kernel` = three int8 sets + one bf16 set (work-groups by set, the same two row bodies);
  the adapter `xe_rad_gemm_nt_q_x3_bf16` with operands a, a_scale, (b1, b1_scale, y1) x 3, x, b4, y4 and params M, K,
  N1 .. N4, group, dtype; the fallback = the x3 adapter + the bf16 GEMV. Listed under the same flag (`RADIANCE_XE_GEMV_
  MULTI2`: `1` lists both groups; `a` / `b` one of them, so the M-pair can measure them apart).
- The plugin (`qwen4exp_fp8.cpp`): the indexer's weight is declared by `QsaIndexer::declare`, AFTER the attention block,
  so the merged op is declared by the plugin once both exist (`lay.attn.qg / kp / vp` + `lay.qsa.proj`, the buffers
  `aw.h.q8 / s8 / x`, `b_qg / b_k / b_v`, `m.qsaw.qk`), guarded by `rad_op_declared` (the AMD build declares no such row
  and changes nothing). At step time (the layer loop, "the indexer runs first") the plugin issues the merged op at T <=
  kGemvMultiRows before `l.qsa.step(.., proj_done = true)` and `l.attn.step(.., qkv_done = true)`: two new default-false
  parameters that skip `proj.step` and the `op_qkv3` / three-launch branch. `QsaIndexer::store` (the prefill path) keeps
  its own `proj.step`.
- Gates as A's: a case in `gemv_multi_test.cpp` (the x3 row + the bf16 row vs the new row, bytes), the card row check,
  the M-pair / `-prof` (the indexer's `gemv_bf16_kernel<0,0,4,0>` 15 -> ~2 a step, `gemv_i8a8_x3_kernel` 13 -> 0, the
  new kernel 13), the rotation. Expected ~0.2 ms a step (0.7 %) -- below the 1.5 % line alone; it rides with A.

## 3. What this does NOT change

The three big items of the step are untouched by any pair here: the format C expert kernels (8.4 ms a step, ~3x their
byte floor), the trunk GEMVs' own time (the standalone 145 at 13.4 us are ~1.9 ms; the groups save launch overhead, not
bytes), the hc kernels' E4M3 decode (~5 ms). The fusion map of briefs/2026-11-05-megakernel.md section 8 stands; this
audit adds the quantiser's verdict (no, at any tile that fills the card), the ar + write + norm design candidate, and the
two GEMV groups.
