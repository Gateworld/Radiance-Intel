# The gate_up epilogue fold (owner note 254; ships as v1.1-intel, note 257): the design

2026-11-12, run 341 (agent). Part of the Intel port. Desk and build only until the v1.0.1 soak passes; card steps only
in a booked window after it (note 249 item 6).

## 1. What is folded, and why both arms

Today's decode MoE layer issues, back to back (rad_block_moe_fp8.h `pass`):
- `op_gu`: format C gate_up, `moe_i4i8_run_kernel<5,16,1,false,true>` (sub-group 16, 512 threads, 32 columns a
  work-group; one 16-lane team a column), writing `w.egu` [T][2F] bf16, gate then up (`parts` 2, F = `n_ff_exp`);
- `op_gqp`: `gated_quant_pair` (default on since run 279). One launch: the routed rows' silu × up, the width-128
  Hadamard and int8 codes (`gated_had_quant_i8g`), then the shared expert's silu × up, plain int8
  (`gated_quant_fp8` at int8).

When the router launch already ran the shared gate and gate_up (`gu_fused` and `gate_done`, the decode path),
`shared_up` issues nothing between the two. The quant launch goes only if **both** arms leave it: folding the routed arm
alone would leave the pair launch in place with the shared arm's rows. So the fused op takes both arms. The shared
gate_up's output `w.sgu` was written by the router launch, before the routed gate_up.

## 2. The kernel (xe_moe_i4i8.cpp, format C's file only)

- **The body is unchanged.** `moe_i4i8_run_kernel` becomes a thin wrapper over an inlined body template with
  `EP = false`, so the default kernel is the same code. A new `moe_i4i8_run_ep_kernel` calls the body with `EP = true`
  and one more argument struct (`GqEpi`).
- **Routed arm: a counter epilogue.** A work-group's 32 columns lie inside one 128-column group of one half (F % 128 ==
  0, 128 % 32 == 0), so a group of a run needs 8 work-groups: 4 for its gate columns and 4 for its up columns. Each
  work-group writes its bf16 columns as today, then:
  1. every thread does a device-scope release fence, then the work-group barrier;
  2. thread 0 adds 1 to the run's counter for that group, `cnt[i · (F/128) + g]` (i = the run's first sorted row);
  3. the 8th arrival quantises the group for the run's rows. It does an acquire fence, re-reads the 2 × 128 bf16
     values a row, writes the codes and the scale, and stores 0 back into the counter, ready for the next launch or a
     tape replay.
  - Every row is in exactly one run: a padding or out-of-bound row is its own run of 1; a run of up to 4 rows of one
    expert belongs to its first row's work-groups.
  - The early exits (columns past a narrow class's width, a run the tail skip drops whole) write zeros and still take
    part in the count; the quantiser quantises those zeros, as today.
- **Shared arm: extra work-groups.** `grid.y = M_b + T`. Rows `y < M_b` are the shared arm's: work-group `x == 0` runs
  the row, the rest return. They come first in the grid, so they are dispatched first and stay off the tail.
- **The arithmetic, exactly the port's** (`r4d_fused_quant_fp8.cpp` `gated_quant_row` + `quant_group`):
  - `a = g / (1.0f + expf(-g))`, `h = f2bf(a * bf2f(u))` with the port's own `f2bf` (no NaN case), the optional bf16
    out, `v = bf2f(h)`;
  - the rotation: butterflies at strides 1, 2, …, 64 in that order. The port holds 4 values a lane on 32 lanes; here it
    is 8 a lane on 16 lanes, so stride 4 moves into the lane. The element arithmetic is the same: low `a + b`, high
    `a - b` (the port's `(-b) + a`, identical in IEEE);
  - `m` = the max of |v| (order-free), `sc = m > 0 ? m * (1.0f / 127.0f) : 1.0f`, `inv = 1.0f / sc`, codes
    `(int)rintf(fminf(fmaxf(v * inv, -127.0f), 127.0f))`, four to a dword, value 0 low.
  - The shared arm's width only needs to be a multiple of 4: each lane's 8 columns are two halves of 4, each live or
    zero, as the port's guard (`c0 < n`).
  - Both TUs build with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`.
- **Counter memory:** the op's row instance (as the tail skip's `TskipInst`) allocates `T_max × F/128` u32 at init,
  zeroed once.

## 3. The switch and the architecture

- `RADIANCE_XE_GU_EPI=1`, opt-in, lists the op row `moe_gemm_q_gq` in libxe.
- The MoE block declares it only with format C, two classes, `op_gqp` declared and the router-fused shared gate_up
  available. Each pass issues it in place of `op_gu` + `op_gqp` when that pass's shared gate and gate_up were both done
  by the router launch; otherwise today's two ops.
- Reads: gate_up's reads plus `w.sgu`. Writes: `w.egu` and the pair's outputs.

## 4. Gates (note 254)

- CPU row test: the fused op's bytes (egu, the routed codes and scales, the shared codes, scales and bf16 out) against
  gate_up + gated_quant_pair, at decode shapes of 1 / 2 / 4 / 8 tokens, both classes, a narrow class, padding rows,
  and a shared width that is not a multiple of 128.
- Card row check, the -prof twin, the M-pair (texts = M285, ≥ +0.5 %), the sampled + greedy rotation pair.
- Pass: WAITING (v1.1-intel, the owner's decision; owner note 257). Fail: record and stop.

## 5. Expected

At most the pair launch a layer: brief 2026-11-12-moe-megakernel-paper-vs-ours §2, ~0.36 ms a step, plus 48 boundaries.
Less the epilogue: the last work-group of each group quantises at gate_up's tail, and every work-group adds a fence and
an atomic. Net maybe 0.15-0.25 ms of ~25.6 ms (0.6-1 %), against note 254's +0.5 % bar.
