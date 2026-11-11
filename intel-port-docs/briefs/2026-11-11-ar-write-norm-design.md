# Design: the decode all-reduce, the hc write and the next read's norm as ONE launch, one row a block

Run 285, 2026-11-11 (owner note 232's audit, the one hc pair with size; the coordinator's "design only, to me, after
A / B"). Nothing built. Sources: `src/libxe/xe_allreduce.cpp` (ar_kernel, HcW, launch), `src/libxe/ports/r4d_hc_bf16.cpp`
(hc_v2_write_norm_kernel, hc_v2_norm_body), M403 / M406's kernel tables, AR2's phase stamps (run 280).

## 1. Today

At every connection (93.5 a step at the single prompt) the chain is `ar_kernel<0,0,5>` (20.1 us) -> a boundary (0.77 us)
-> `hc_v2_write_norm` (7.3 us) -> a boundary -> `hc_v2_down_rows`. The all-reduce of a 4-row message (n32 = 4 x 2,560 x
2 B / 4 = 5,120 words) runs nblk = ceil(5,120 / (4 x 256)) = 5 blocks of 256 threads, 1,024 words a block, so a block's
words STRADDLE rows (block 1 holds the end of row 0 and most of row 1). The fused write (RADIANCE_XE_AR_HC_FUSE, the HW
form, run 100) handles that per word (r = j0 / n) and is off in the service: the write is folded into the next read's
norm instead (RADIANCE_HC_WRITE_FUSE, hc_v2_write_norm_kernel: work-group (c, r), 256 threads, thread tid owns i = tid +
k x 256 ascending, the squares accumulated as it writes, wave_sum, the 8 partials added in order by tid 0, rsqrt, then
nbuf[j] = f2bf(h[j] * sc * (w[j] + wadd))). The norm needs a whole stream of a row (2,560 values); that is why the pair
cannot fuse at today's blocking.

## 2. The form

Re-block the decode all-reduce ROW-ALIGNED: per = n / 2 words (1,280 for n = 2,560: even, 16-byte aligned, so PR 5's
pairs and PR 6's quads hold), nblk = M (1 block a row: 4 at the verify, 1 at a draft pass; today 5 and 2). Each block
then owns one complete row of y after its emit phase, and can finish, in the same launch, after one work-group barrier:
1. the write of its row into the hc streams: h[r][c][i] = f2bf(bf2f(h) + g_c x bf2f(y[r][i])) for c in 0..3, i over
   the row -- hc_write_kernel's expression, the thread mapping of hc_v2_write_norm (i = tid + k x 256); the gains
   `inj[r][c]` are this connection's (written by the read's gains blocks before the block's compute, as now);
2. the next connection's norm of each of the 4 streams in hc_v2_norm_body's order: the squares accumulated over the
   written bf16 values in the same thread mapping, wave_sum, the 8 partials in order, rsqrt(ss / n + eps),
   nbuf[r][c][j] = f2bf(bf2f(h) * sc * (w[c][j] + wadd)) and scaleout[r][c]; the four streams one after another in one
   block (hc_v2_write_norm does them in four work-groups: the same arithmetic, the same order within a stream).
The y values the write reads are the block's own stores (global, after the barrier) -- the same bits the separate
kernel reads. The next read's columns launch (hc_v2_down_rows) follows as today; nothing after it changes.

Exactness: by construction -- every element's operations and their order are hc_write_kernel's and hc_v2_norm_body's
(the two the CPU row test compares today); the all-reduce's sums are unchanged (the word -> block mapping moves, the
per-word sum does not). A CPU row test runs the three kernels against the one on the same inputs, bytes.

## 3. What it would save, and what it costs

- At stake: 93.5 x (7.3 + 0.77) us = 0.75 ms a rank-step (2.7 % of 27.5 ms), i.e. ~+2 tok/s single-user if free.
- The cost inside the all-reduce: each block does 4 streams' write + norm serially (~100 KB of h, y, w traffic a row;
  hc_v2_write_norm spreads the same work over 16 work-groups in 7.3 us, latency-bound). Expect 3-6 us added to the
  20.1 us launch, so the net is ~0.2-0.4 ms a step (0.7-1.5 %) -- the gate is marginal; a bench decides before any
  engine code.
- Block count: 4 blocks a verify instead of 5, 1 a draft instead of 2. AR2's stamps showed the exchange's own span is
  ~6.5 us and rows 2's 3-block form was the slow case (dispatch stagger); fewer, fatter blocks might help or hurt the
  wait phase -- measure with the B.0 stamps (XE_AR_STAMPS=1) at nblk = M (XE_AR_BLOCKS already overrides the count).

## 4. The protocol surface (why this is a design, not a candidate)

The LL form's tear and sequence protection (forms 0-3's races, note 120) is per word and per block: the row-aligned
per changes only which words a block carries; the sequence number, the parity slots, the flags and the arrival counter
are untouched. But any change to ar_kernel's blocking is a change to the collective, so it needs the coordinator's
gates for that surface: allreduce_test at every size incl. XE_AR_TEST_TEAR, the 1 M-launch soak on both ranks (0
mismatches, 0 stalls), the peer probe, then the usual (profile first, texts, M-pair, rotation).

## 5. The op and where it lives

A new op `ar_hc_write_norm` (TP2 only): the all-reduce's operands (y in place), inj, h, then the NEXT connection's norm
weight w, eps, wadd, and its outputs nbuf, scale. HyperConn declares it where it already couples a write with the next
read (declare_write_read, run 206: hc_ffn after hc_mix, hc_mix after the previous layer's hc_ffn), and the next read then
starts at its columns (the norm skipped, as the fused write_read skips the separate write today). libxe's row: a new
template bool of ar_kernel (the WN form) on PR 5 at M <= kResidentBlocks and n even; any other shape = today's launches.
Where the all-reduce is not issued by the connection (ar_out_take / the pipelined pieces) nothing changes.

## 6. Staged plan

1. Bench (card1 + card0, no engine, ~$15): the all-reduce at nblk = M vs today's nblk at 1 / 2 / 4 rows with the stamps;
   then the WN form's extra time a launch on the same harness (a copy of write_norm's row inside the kernel). Go only if
   the exchange does not slow down and the WN form stays under ~5 us added at 4 rows (the net >= ~0.35 ms a step).
2. The kernel + the CPU row test (bytes vs ar + write_norm), the tear test, the soak.
3. The op in HyperConn behind a flag, the M-pair (texts), a rotation; default by the owner.
Estimate: ~$120 across two runs; the exactness is by construction, the gain 0.7-1.5 % at best -- below A's, so it
should come after the cheaper exact items unless the bench surprises.
