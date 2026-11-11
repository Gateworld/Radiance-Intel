# The matrix units (DPAS / XMX) for the 4-row decode kernels: what exists, what it could buy, how to judge it

2026-11-01, supervisor (read-only investigation, owner's request). Sources: STATE.md (runs 99-100, 114-122),
briefs/2026-10-28-bottleneck.md, briefs/2026-10-31-kernel-headroom.md, src/libxe (working tree). Not an owner note:
nothing here is queued until the owner approves it.

## 1. Why the matrix units, and where they help

- With MTP on, every decode step pushes the verify's 4 rows through each layer. The bottleneck brief (§2) puts
  13.4-13.6 ms of the 4-row step's extra time in kernels that redo per-column work for each row on the same weights
  (trunk 5.9, hc 3.4, bf16 experts 2.3, all-reduce 1.2, GDN 0.6).
- The trunk GEMV at 4 rows (gemv_i8a8<1,4,0>) costs 9.07 ms a step at 164 GB/s, **1.90x its 1-row time for the
  same bytes**. The headroom brief (§1.1 item 1) reads that as per-row ALU and shuffle issue (4 DP4A + 3 cross-lane
  steps per row per 16-byte chunk), not bytes.
- One int8 DPAS instruction sequence (joint_matrix 8 x 16 x 32) does every row's products for a 16-column tile at
  once, so a 4-row call should cost about what a 1-row call costs. The tile's M = 8 leaves half of it idle at 4
  rows, but the kernel is not compute-bound, so that is free.
- **Estimate:** the trunk at 4 rows from ~9.1 to ~5 ms a step (about the 1-row time): ~4 ms, ~8 % of the 51.5 ms
  MTP-on step as a ceiling. Past kernels have recovered a third to a half of such ceilings end to end, so **~+3-5 %**.
  hc at 4 rows (hc_up_mix's extra ~17.7 us is per-row work; the down group ~2.9 ms of room): another **~+2-3 %**.
- Not worth it for the experts: format C's kernels already run within 1.3-1.5x of their weight-streaming floor.

## 2. What already exists (both in the working tree, uncommitted)

| Form | Flag | Built | Accuracy | Timed? |
|---|---|---|---|---|
| trunk W8A8 at 2-8 rows on int8 joint_matrix 8x16x32 | `RADIANCE_XE_GEMM_DPAS_ROWS=1` (xe_gemv_i8.cpp) | run 99 (10-30), note 105 item 1 | all kbench cases pass; worst rel_l2 1.06e-4 vs the GEMV's 6.5e-5; **not bit-identical** (integer products exact, an f32 step differs); flips a near-tie, so M105 / M108 texts changed | **no**: heat stops and the import-bind stall that day, then the exactness gate stopped it |
| hc at 2-63 rows on the matrix units | `RADIANCE_XE_HC_PF_DPAS=2` (ports/r4d_hc_bf16.cpp) | run 100, note 105 item 3 | BK13: 40/40; at M 2-8 **closer to libref than today's kernel** (8.4e-3-1.10e-2 vs 1.15-1.36e-2 with inject) | **no**: "needs a two-card load" |

Both were shelved by the old rule (byte-identical or the agreement bar). Under note 149's relative bar (judged
against the non-bit-identical controls, as GL, ATTN_MX and format C were), neither has been tried.

## 3. How to judge a non-exact form's speed (a lesson from GL)

- GL (note 141): kernel -41 %, single prompt +7.6 %. But the rotation (H99 / H100) gave **median -4.7 %, mean
  +1.6 %, aggregate +2.6 %**. It was faster on short prompts and slower on tool prompts. Its different texts moved
  the mix (80 vs 60 stops) and MTP acceptance (0.695 vs 0.705).
- So a rotation of a non-exact form measures two things at once: the kernel's speed, and different texts with
  different lengths and acceptance. The median gate can reject a faster kernel, or accept a slower one, by chance of
  text.
- **Suggestion:** for non-exact forms, time **teacher-forced replays**. Feed the control's answers through both
  forms, as the KL check already does (kld-gl.sh), and record the time per verify step and per token on the same
  tokens and drafts. That separates speed from text. Then keep the rotation as a sanity check, and judge by mean /
  aggregate as well as the median.

## 4. A possible note (for the owner to approve, not queued)

After note 151 A-C and note 152 item 1:
1. Time the existing `RADIANCE_XE_GEMM_DPAS_ROWS=1` and `RADIANCE_XE_HC_PF_DPAS=2` (rebased on today's tree, with
   format C on) by teacher-forced replay against the control: verify-step time and tokens/s on the same tokens.
   Their KL by kld-gl.sh, judged by note 149's bar.
2. If either is faster by >= 1.5 % on the replay and passes the bar: one rotation as a sanity check (no worse than
   -1.5 % median, mean / aggregate reported), and keep it opt-in.
3. Revisit GL the same way (its replay timing), since its rotation verdict is confounded by text.
About $80.
