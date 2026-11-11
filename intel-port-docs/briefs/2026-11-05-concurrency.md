# Independent decode kernels side by side: a design note and its first probe (owner note 191 item 2)

2026-11-05, run 193 (agent). Part of the Intel port. **Desk work; no card time until the soak ends.**

## Why

At 4 rows most decode kernels sit at 2-5x their byte floor (P87, the kernel scoreboard): each launches too few
work-groups to fill the B70's 32 Xe cores, or spends its time in latency it cannot hide alone. Two kernels that
do not depend on each other could share the card. Each keeps its own registers, LDS and code, which is what sank
the megakernel's fused forms (brief 2026-11-05-megakernel.md, 6.2-6.3).

## The independent pairs inside a decode layer

| pair | why independent | rough time a layer (P87, 4 rows) |
|---|---|---|
| the shared expert (gate_up + down, int8 GEMVs) beside the routed experts (format C run kernel) | both read the post-norm `x`, and their outputs meet only at the final sum | shared ~2 x 20 us; routed 114 + 67 us |
| the router (gate GEMV + top-k) beside the shared expert | the shared expert does not depend on the routing | router ~10-15 us |
| attention's projections (q, k, v GEMVs on a full-attention layer; GDN's in / z projections) | they read the same input and write separate outputs | ~3 x 15-25 us |
| the MTP head's draft passes against the trunk | **not** independent: each draft reads the trunk's last hidden state | -- |

**The best candidate is the shared expert beside the routed experts.** The routed kernel is the largest decode
item, and at 1-4 rows it leaves cores idle (27 experts x a few column blocks). The shared expert's two GEMVs are
small and need no routing.

## What "side by side" means on this card

- **On Level Zero, two lists are not by themselves two engines.** If the B70 exposes one compute engine (the CCS
  group's queue count, read at startup: to be checked in the probe), kernels from two immediate lists still go
  through one hardware ring, in submission order.
- **What lets two kernels overlap on one engine is the absence of a barrier between them.** An in-order list
  (ours, Z.imm) puts a barrier between every pair of kernels. A list without the in-order flag, with explicit
  events only where the data flows, lets the next kernel's work-groups start while the previous one's last ones
  drain.
- With two or more CCS engines, a second list on another engine (GLUE_ALL=2's machinery, ximm, on a compute group)
  gives true concurrency.

So the probe measures all three forms. The engine change then follows whichever wins.

## The probe (stage 1): one pair, standalone, timed

- **Kernels:** the shared expert's gate_up GEMV (gemv_i8a8 at its decode shape, 4 rows) and the routed experts'
  gate_up (moe_i4i8_run_kernel at the verify mix). Both are called through libxe's own entries, as
  xe_fc_tile_probe calls the expert GEMM, with random inputs.
- **Forms, each timed over 48 pairs x 5 runs (host wall, median):**
  1. today: both on one in-order queue (the sum of the two);
  2. one Level Zero command list without the in-order flag and no barrier between the pair (an event joins them);
  3. two in-order immediate lists, one each, joined by an event; on two CCS engines if the device has them.
- **Exactness:** both outputs compared byte for byte with form 1's. The kernels are untouched and nothing they
  read changes, so they must be equal.
- **Bar:** form 2 or 3 at ≤ 85 % of form 1's time is worth the engine change; otherwise report and stop.
- **Cost:** ~$20-30 of desk work and a canary-sized card step, after the soak.

## If it pays (stage 2, not started)

- The op graph gains a fork / join around the pair, on a second stream: rad_event_record, rad_event_wait.
- The recorded tape keeps both streams. The native glue then needs the non-in-order form, or a second compute
  list, for that segment.
- **Exact:** the same kernels, the same inputs, and a join before the sum, so the texts must be identical.
- **Gates:** the M-pair texts, a profiled step, the rotation (the default at ≥ 1.5 %).
