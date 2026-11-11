# A persistent decode kernel ("megakernel") for Flash-Next on the two B70s: a design brief (owner notes 183.3, 186.3a)

2026-11-05, run 188 (agent). Part of the Intel port. **A design only; nothing is built.** Stage 1 (section 6) is
the only part asked for next, behind a flag, about $150; larger stages wait for the owner.

## 1. What it would replace: today's decode step (the kernel scoreboard, P87, MTP on, ~2.9 tokens a step)

The step is ~51 ms of wall time and ~36 ms of device time a rank. Its kernel rows sum to ~32.5 ms against a
~9.5 ms byte floor (briefs/kernel-scoreboard.md). The step is already a recorded tape, a native Level Zero command
list replayed at ~0.6 us a launch against ~3 us through the runtime. The host's launch cost is therefore mostly
gone (run 174: ~5.7 % device idle). **What is left is inside the device: ~1,500 small kernels a step, each with
its own ramp, drain and cold start.** By group:

| group (P87 rows) | ms a step | floor ms | how a megakernel could help |
|---|---|---|---|
| expert gate_up + down, format C (1, 3) | 9.29 | 2.2 | little: these are long kernels running at 3.9-4.6x their floor *inside* the kernel (note 167's target), not launch-bound |
| trunk int8 GEMVs, 4 rows (2) | 5.46 | 2.4 | the ~5 us per-launch floor x 220 launches; prefetching the next GEMV's weights while one finishes |
| the hc family: up mix, down rows, gains, norm, write, quant rows (4, 6, 9, 13, 15 part) | ~7.5 | ~2.2 | ~6 kernels a connection x ~97 connections; one persistent pass per connection removes ~5 boundaries each |
| all-reduce + hc write (5) | 2.66 | ~0.85 | the LL protocol is already a device-side flag exchange; inside a persistent kernel it overlaps with the next phase's weight loads |
| router, small GEMVs, GDN update, tails (8, 10, 11, 14, 15) | ~4.5 | ~1.3 | boundary removal; the router can feed the expert phase without a host round trip |
| lm_head, MTP draft head (7, 12) | 2.27 | 1.7 | near its floor (lm_head); the draft head is 183.1's kernel |

A fair ceiling from removing boundaries alone is about **8-12 ms a step**: most of the non-expert gap, none of the
expert kernels' own inefficiency. That is **~20-30 % of the device step, ~1.2-1.4x decode** at 1 user, *if* the
persistent form reaches the bandwidth the individual kernels reach between boundaries. The expert kernels need
their own work either way (note 167 / 183.1 / format C).

## 2. What the published decoders gained, and how this case differs

- **Hazy Research, "Look Ma, No Bubbles" (2025-05-27):** Llama-3.2-1B's whole forward pass as one kernel on H100 at
  batch 1, ~78 % of memory bandwidth against roughly half for conventional stacks (as summarised by Cohere's
  write-up; the original post was not retrieved here). One thread block per SM, resident for the step; an
  instruction interpreter on each SM; counters in global memory as the dependency barriers.
- **Mirage Persistent Kernel (MPK, OSDI '26, arXiv 2512.22219):** a compiler that turns a model into one
  megakernel, with an SM-level task graph that pipelines across operators. It reports 1.0-1.7x over SGLang / vLLM
  on A100 / H100 / B200, single and multi-GPU (the Qwen3-8B demo).
- **Cohere (2026-09-08):** a decode megakernel serving engine, 1.25-1.41x over vLLM end-to-end on one BF16 H100.
- **AutoMegaKernel (arXiv 2606.09682):** generated persistent cooperative kernels for Llama-family models. Its best
  result is up to 1.33x on an L4 at batch 1, mixing W8A16 against bf16, which the authors flag.

All of these are NVIDIA (AMD's Fleet paper is a task abstraction for multi-die GPUs), dense or modest MoE, and in
VRAM. **This case adds three things none of them handles:**
1. **Experts that miss the card.** The heat engine stages cold experts from the host pool over PCIe; a miss is
   known only after the router runs, on the device.
2. **Tensor parallel over PCIe**: two cards, an all-reduce per layer (the LL exchange ~8-11 us, PEER2 / PR5).
3. **Intel's stack:** SYCL has no supported grid-wide barrier. The DPC++ Compatibility Tool's `nd_range_barrier`
   is an experimental atomic-counter helper; intel/llvm's forward-progress extension is a draft. So co-residency
   of all work-groups has to be arranged and checked by us.

## 3. How it would work on Level Zero / SYCL

- **One persistent launch a decode pass a rank**, recorded into the tape like today's kernels, so replay, the
  glue and the ring events stay as they are. The kernel is sized to the device's resident capacity
  (work-groups = Xe cores x resident groups per core, queried at launch). A launch too large to be co-resident
  is refused and the pass runs as today. That refusal is the safety net against a deadlock on a missing group.
- **Phases and barriers:** each layer's ops become phases; a phase waits on a device-scope atomic counter (the
  `nd_range_barrier` pattern: one increment a work-group, a spin with a bounded count). Phases that only need
  their own rows wait on per-tile flags instead, MPK's finer-grained form, so the next phase can start on the
  tiles that are ready.
- **The all-reduce inside:** the LL protocol's peer stores and flags are already device-side; a phase writes its
  share and spins on the peer's flag. Both cards' kernels must be running at once. The tape already plays both
  ranks together, but a preempted or delayed peer would hang the other. So every spin is bounded and writes an
  error word the host checks, instead of spinning until the GuC kills the job. (We saw that end on 11-05: GA2's
  host-side wait became a "Timedout job" and a CAT fault.)
- **Expert misses:** the cleanest first form keeps the MoE outside, so the step is several persistent segments,
  "dense chain -> MoE -> dense chain". A later form puts the MoE inside, with a per-expert ready flag that the
  stager's copy sets: the copy engine appends a small flag write after each expert's DMA, and the MoE phase spins
  on that expert's flag. That form needs the stager to run during the kernel, which it already does on the copy
  engine.
- **Weights:** each phase's weights stream through the L2 as today. The gain comes from starting the next phase's
  loads while the current phase drains, which a kernel boundary forbids.

## 4. Risks

- **Co-residency and deadlock:** a grid barrier with a group not resident never completes. Mitigation: size from
  the device query, refuse otherwise, bound every spin.
- **The driver's submit path and timeouts:** a step-long kernel (~36 ms) is far under the xe job timeout (seconds).
  A kernel that waits across steps, for example for the host, would need the long-running mode. The design keeps
  one launch a pass, never across passes.
- **Exactness:** a megakernel must reproduce each op's arithmetic order to keep the texts bit-identical (the gates).
  Ops whose kernels choose their reduction split by M (split-K) keep the same split.
- **Occupancy on the B70:** a persistent kernel's register and SLM budget is the maximum of its phases'. The expert
  and DPAS kernels use grf256 / large SLM, so a single kernel holding everything would lose occupancy. That is
  another reason the first forms keep the expert kernels separate.
- **Debuggability:** one kernel per pass hides which op went wrong. Mitigation: a per-phase error word and phase
  counters read after the pass, and the flag-off path as the A/B for every result.
- **Effort:** the published systems are compilers (MPK, AutoMegaKernel) or months of hand work (Hazy). A
  hand-written persistent pass for this model is weeks; the staged plan below keeps each stage measurable.

## 5. Note 183 item 2 (fewer hc kernels): a step toward this, not thrown away

183.2 fuses the hc family pairwise: gains into down rows, norm with write, the quantiser into its producer. That
is the same boundary removal, done one pair at a time, exact by keeping the arithmetic order. **Its kernels are
the phases a persistent hc segment would run**: the fused bodies move into the segment unchanged. So 183.2 is
not wasted. But stage 1 below *is* the hc chain as one persistent launch, and it subsumes the pairwise fusions.
**Recommendation: do 183.2 as stage 1**, one persistent hc launch a connection, rather than as separate pairwise
fusions first. Skip the separate 183.2 unless stage 1 fails. If stage 1 fails (no gain, or occupancy), the
pairwise fusions are the fallback with the same arithmetic.

## 6. The staged plan

| stage | what | measure | cost | owner |
|---|---|---|---|---|
| **1 (asked for; ~$150)** | **one hc connection's chain (up mix -> down rows -> gains -> norm -> write -> quant rows) as one persistent launch** behind `RADIANCE_XE_HC_PERSIST=1`, phases on a device-scope counter, sized to residency, bounded spins | the chain's us a connection against today's sum (P87: ~77 us a connection over ~6 kernels), then the step time; texts bit-identical (M-pair), KL-exact (the dump's logits byte-identical, as GA4 / GB1) | ~$150 | reports after |
| 2 | the dense chain between MoE layers as one segment: attention + hc + router | step ms; the same gates | ~$300-500 | waits |
| 3 | the all-reduce inside (LL flags with bounded spins, both ranks) | step ms; fault-free hours | ~$300 | waits |
| 4 | the MoE inside, on per-expert ready flags from the stager | step ms, misses | ~$500+ | waits |

Stage 1's first measurable step is a standalone timing of the persistent hc chain against the six kernels it
replaces, on card1, small, before any engine run. That is the canary run UNATTENDED asks for, for a new kernel.

**Expected outcome of stage 1:** the hc family is ~7.5 ms a step over a ~2.2 ms floor. Removing five boundaries a
connection at ~5 us each is ~2.4 ms a step (~6-7 % of the device step) if the persistent form holds the bandwidth.
Below ~1 ms gained, it is not worth going on, and the report will say so.

### 6.1 Stage 1a, measured (run 189): exact, and slower

Stage 1a was the first piece: hc_read's level-8 down half (the norm over the connection, then the down rows and
the gains) as one persistent launch. One work-group per Xe core, two phases split by a device-scope barrier with a
bounded spin (`RADIANCE_XE_HC_PERSIST=1`, image 236, commits dd5ca52 and 687492e).
- **Exact:** the CPU device's row test was bit-identical with it off and on, and so were the M-pair texts on both
  cards (M268 against M262).
- **Slower:** 47.5 and 55.2 tok/s against 57.2 and 70.2 (**-17 % and -21 %**). The kernel log was clean.

The likely reasons, neither profiled yet:
- **Too few groups.** One group per core leaves each core one group of latency hiding. The six kernels it replaced
  each launched hundreds of groups and filled the card.
- **The barrier costs more than the boundaries.** One device-scope barrier (atomics on a global counter, then every
  group spinning on it) costs more than the ~5 us a kernel boundary costs inside a recorded tape. Recorded plays
  already strip most of the launch cost that a megakernel saves elsewhere.

**What a stage 1b would have to change before it is worth another run:**
- several groups per core (occupancy-sized, not one);
- no device-wide barrier where the dependency is local: the norm is per row, so a row's groups can go on when their
  own row's reduction is done (per-row flags in place of one counter).
- First, a unitrace profile of M268's step, to see whether the time goes to the spin or to the phases' own
  bandwidth.

**Until then, 183.2's pairwise fusions (section 5's fallback) are the better next step.** Each removes a boundary
with no device-wide synchronisation.

### 6.2 The profile and stage 1b, measured (runs 189-190, owner note 187): exact, still slower, stopped

- **The profile** (tape profiler, M269 against M270, texts identical): the three kernels take **42.7 us** a
  connection (norm 6.8, columns 25.7, gains 10.2). 1a's launch takes **132.3 us**, 29 % of the step's device time.
- **Stage 1b** (`RADIANCE_XE_HC_PERSIST=2`, commit 16de61f, image 238):
  - work by ticket, per-row ready flags in place of the barrier, and occupancy-sized groups (192 for 648 units);
  - CPU row test bit-identical, M274's texts identical, kernel log clean;
  - **114.2 us a launch: better than 1a, still ~2.7x the separate kernels.**
- **What it says:** removing the device-wide barrier and restoring occupancy recovered only 18 us of 1a's 90 us
  excess. What 1a and 1b still share is the three bodies fused into one kernel. Every phase then runs with the
  largest body's register footprint and the code IGC makes for the combined kernel, and that is the likely cost.
  A persistent form on these bodies does not pay as built.
- **Stopped, as note 187 says. The owner decides:** stage 2 (which would inherit the problem unless the bodies are
  kept as separate kernels with lighter joins), 183.2's pairwise fusions, or parking it.

### 6.3 Stage 2 on paper (owner note 190: parked; a desk estimate only, no build, no card time)

**What stage 2 is.** The dense chain between two MoE layers as one persistent kernel: hc_read (norm, down rows,
gains, up mix, quant rows), the attention (QSA or GDN by layer, 4 rows), the router and the trunk int8 GEMVs. The
all-reduce and hc_write stay outside, as stage 3's and the MoE's.

**The gain, on measured numbers (P87).**

| | ms a step | what stage 2 could take |
|---|---|---|
| the hc family | ~7.5 (floor ~2.2) | the boundaries between ~5 kernels a connection |
| the trunk GEMVs (4 rows) | 5.46 (floor 2.4) | the ~5 us per-launch floor x ~220 launches |
| router, small GEMVs, GDN update, tails | ~4.5 (floor ~1.3) | boundary removal |
| **the ceiling** | | **about 5-8 ms a step** (the brief's 8-12 ms less the parts that stay outside) |

**The cost, measured on the smallest case.** Stages 1a and 1b fused only three of the hc bodies, and each launch
came out **+71 to +90 us a connection** over the separate kernels (114-132 against 42.7). Across ~97 connections
that is **+7-9 ms a step: the whole of stage 2's ceiling, lost on its first and easiest piece.** So stage 2 is
worth building only if that penalty's cause is found and removed first. Nothing measured yet says what it is.

**Does the fused cost grow with more and more varied bodies?** What is known and what is not:
- **Known, structural:**
  - A persistent kernel has one LDS and register budget, the largest of its phases', for every phase. 1a and 1b
    already carried the gains' 2 x HN bytes of LDS (20 KB) through the column phase, which needs none.
  - Stage 2 would add attention (LDS tiles) and the GEMVs (the large register file, grf256), so the worst phase's
    budget would govern ever more of the pass. This grows with the variety of bodies, not just their number.
- **Known, from the build:**
  - IGC printed no spill warnings for 1a's or 1b's kernels, so spills (the usual codegen penalty) are not the cause
    measured.
  - One kernel holding many phases also has a larger instruction footprint. Whether the instruction cache matters
    here is unmeasured.
- **Not known:** which of these, or the per-unit synchronisation itself (a ticket atomic, a flag acquire and two
  group barriers a unit, about 3 units serial a group in 1b), made up 1b's 71 us. A counter run (unitrace: active
  and stall share, SLM and instruction-cache counters) on M274's kernel would say. That is ~$10 and a short card
  step, but **it is not asked for while stage 2 is parked.**

**Would a different design avoid it?** Yes, plausibly, because these designs keep each kernel's own budget:
- **Separate kernels with overlapped loads.** The next kernel's weights are prefetched into L2 by the current
  kernel's tail (an L2 prefetch sweep over the next operand). Each kernel keeps its own occupancy and code. The win
  is the next kernel's cold start: bounded by the device's measured idle between kernels (~5.7 % of the step, run
  174, ~2 ms) plus whatever the prefetch hides of the first loads.
- **Kernels that overlap their ramp and drain.** The tape's lists are in order, so each kernel waits for the last to
  drain. A list without the implicit barrier, with explicit events only where the data flows, would let a kernel's
  prologue run under its predecessor's tail. Same bound, ~2 ms.
- **Upstream 1.14's approach** (note 190 item 2): hc_read's two decode kernels as one ordinary grid, their dots kept
  off LDS round trips: 34.7 → 25.2 us at 4 rows on upstream's card. It is a normal launch, so no persistence and no
  shared budget across phases.

**Recommendation.** Keep stage 2 parked. The better route to the same ~5-8 ms is:
1. upstream's merge (note 190) for the hc family;
2. then the load-overlap form for the trunk GEMVs, with each kernel kept separate.
Both are measurable one kernel at a time, the way stage 1 should have been checked first.

### 6.4 Upstream's merge on the B70, and the cooperative launch (runs 202-203, owner notes 190 / 196)

The first item of that route lost too. Upstream's merge (RADIANCE_XE_HC_AB, hc_v2_ab_kernel<4>) is exact, but on the B70:

| | per connection |
|---|---|
| the three kernels it replaces (M270) | 67.3 us (hc_up_mix 31.5 + down_rows 25.6 + down<2> 10.2) |
| the one-grid kernel (M280) | 179.4 us |

The M-pair runs at 52.6 tok/s against 70.4 (M282, −25 %). It stays opt-in.

Note 196 asked whether a cooperative launch (Level Zero's zeCommandListAppendLaunchCooperativeKernel) would rescue
the kernels whose work-groups wait on each other. RADIANCE_L0_COOP_QUERY=1 measured the limits:

| kernel | groups launched | cooperative cap |
|---|---|---|
| hc merge (hc_v2_ab<4>) | 216 | 128 |
| stage 1b (hc_v2_down_ticket) | 72 | 128 |

The B70's compute queue group advertises cooperative kernels on both cards, but residency is not the problem:
- **1b fits under the cap and still loses 2.7x.** Its groups can all be resident at once already.
- **The hc merge's ticket order prevents starvation.** A block waits only after every producer's ticket has been
  taken by a block that is already running.

The cost is elsewhere:
- **One LDS budget for all phases.** Every block declares the largest phase's LDS (the up mix's, up to 60 KB, ~4
  groups a core), so the column phase runs far below its own kernel's occupancy.
- **The ticket and fence protocol.** Every block pays it.

Note 196 stopped at step 1 (supervisor, 00:29 11-06).

**The rule for this card:** fused phases must not share one LDS or register budget sized by the largest phase.
Keep a dependency at the kernel boundary (or across lists); fuse only phases with the same footprint.

## 7. The counters: why the fused forms lose (owner note 198 item 1; run 205)

### 7.1 The static half: Level Zero's kernel properties (measured)

RADIANCE_L0_COOP_QUERY=1 logs, at each kernel's first recorded launch, its properties and its cooperative residency
cap at the launch's block. Image 248, decode M-pairs: M287 (the hc merge) and M288 (1b). Both kernel logs clean.

| kernel | group | SLM per group | private | spill | residency cap (groups) | launched |
|---|---|---|---|---|---|---|
| **hc merge `hc_v2_ab<4>`** | 256 | 31,264 + 26,624 B = **57.9 KB** | 4 KiB | **576 B** | **128** | 216 |
| **1b `hc_v2_down_ticket`** | 256 | 24,640 + 20,480 B = **45.1 KB** | 0 | **4,864 B** | **128** | 72 |
| separate: up mix `hc_up_mix<4>` | 256 | 27,168 + 26,624 B = 53.8 KB | 4 KiB | 0 | 128 | 160 |
| separate: down rows `hc_v2_down_rows<4>` | **64** | **32 B** | 0 | 0 | **1,024** | 160 |
| separate: gains `hc_v2_down<2>` | 256 | 24,608 + 20,480 B = 45.1 KB | 0 | 0 | 128 | 60 |
| separate: norm `hc_v2_norm` | 256 | 64 B | 0 | 0 | 256 | 4 |

This confirms the LDS-budget explanation with numbers:
- **The column phase loses 8x residency.** On its own, the column phase (down rows) runs as 64-thread groups with
  32 B of SLM, up to 1,024 resident. Inside the hc merge it runs as 256-thread groups with the up mix's 57.9 KB, so at
  most 128 are resident. In 1b, the norm and the column phase both carry the gains phase's 45.1 KB.
- **Both fused kernels spill.** The hc merge spills 576 B and 1b 4,864 B; none of the separate kernels spill. The
  fused body holds every phase's live registers at once.
- **Residency is not the bottleneck.** 1b's 72 groups fit its cap of 128 (section 6.4), so the loss is the budget,
  not starvation.

### 7.2 The dynamic half: stall reasons (run 206)

- **The method:** unitrace `--stall-sampling` on ri-m3ut:4, through logs/mutstep.sh (U2 base, U3 the hc merge, U4 1b;
  all kernel logs clean).
- **What was sampled:** under sampling the engine's startup runs about 10x slower, so none reached serving within the
  step. The startup's recordings were sampled, though, and they include the decode shapes. m3/tools/stall-sum.py sums
  each kernel's events by reason and by source line.
- **The fused grids are thinly sampled:** the merge has 5.8 K events and 1b's ticket kernel 141 K, against millions for
  the busiest kernels. Shares inside a kernel are still meaningful.

| kernel | active | waiting on a dependency (thread 0 spins, the group waits at the broadcast) | barrier | scoreboard waits elsewhere |
|--------|--------|------------------------------------------------------------------------------|---------|----------------------------|
| hc merge (hc_v2_ab_kernel<4>) | 12 % | ~54 % (50 % at the broadcast, ~4 % in xe_persist.h's spin) | 5 % | ~21 % |
| 1b (hc_v2_down_ticket_kernel) | 31 % | 18 % | 12 % | ~21 % |
| separate down rows (hc_v2_down_rows<4>) | 49 % | -- | 0 % | 29 % (send stalls 10 %) |
| separate gains (hc_v2_down<2>) | 23 % | -- | 45 % | 17 % |

- **The split note 198 asked for:** in the merge, more than half the threads' time is waiting on the ticket / flag
  dependency. The cross-phase wait does not hide inside the grid, as the design hoped; it turns into idle threads. 1b
  spends about 30 % on its waits and barriers. The separate down rows have neither and run nearly half active.
- **Together with section 7.1:** a fused body pays twice, once in residency (the larger LDS / GRF budget) and once in
  the dependency waits. A shape-matched fusion with no cross-phase dependency (note 200's GU: the gains and the up mix)
  is the only form that has won, which fits.

### 7.3 What this says for note 198 item 2

- **A smaller up-mix tile (2a)** cuts the hc merge's 57.9 KB, but the column phase still runs in 256-thread groups.
  Its 1,024-cap shape (64 threads, no SLM) can't be had inside a 256-thread launch.
- **Fusing only similar phases (2c)** fits the numbers best. Norm and down rows are both tiny-SLM phases, so one
  kernel of those two keeps a high cap. Up mix and gains are both 45-58 KB phases. Every fusion that crosses the two
  groups pays the larger budget.
- **Trimming the spills** (fewer live registers per phase, or `launch_as_grf256` for the fused body) is a
  precondition for any fused form to win.

## 8. The Intel fusion map (owner notes 200 item 3 / 201 item 4; run 206)

- **The measurement:** RADIANCE_L0_NTAPE_EDGES=1 (commit 7420846), with the tape profiler. Every idle gap inside a
  play is charged to its kernel pair. M302: image 255, the M-pair base, texts identical, klog clean.
- **What a boundary costs:** every boundary inside a recorded list costs the same 0.76-0.78 us, whichever kernels meet
  there. A decode step has about 1,700 of them, about 1.3 ms. Removing one saves its 0.77 us plus most of the smaller
  kernel's fixed cost.
- **The hc connection's chain** (about 87-91 times a step):

  ar_kernel -> hc_write -> hc_v2_norm -> hc_v2_down_rows -> hc_v2_down<2> (gains) -> hc_up_mix -> hc_quant_rows -> the
  trunk GEMVs.

| pair | a step | shape and dependency | the rule's verdict | expected saving a step |
|------|--------|----------------------|--------------------|------------------------|
| hc_write -> hc_v2_norm | ~87 | both 256 threads, ~0 LDS. The norm is per (stream, row), and each thread reads back only the elements it owns, so one work-group can apply `h += inj[c] y` to its stream (the same bf16 rounding), write it and reduce it | **fuse: exact, no dependency** | ~0.4 ms (the 4.0 us write + 0.77) |
| silu_mul -> quant_act_fp8 | ~45 | row-local: quant_act_fp8 is a work-group a row, and silu x up can be computed on its load | **fuse: exact** | ~0.15 ms |
| rmsnorm -> rmsnorm (q / k norms) | ~44 | independent, the same 1,024-thread shape: one launch, two row sets | **fuse (horizontal)** | ~0.13 ms |
| sigmoid -> binary -> quant_act (the attention gate) | ~14 x 2 | element-wise, then row-local | **fuse** | ~0.08 ms |
| rope -> rope -> kv_store_fp8 | ~14 x 2 | head-local | **fuse** | ~0.08 ms |
| gains -> hc_up_mix | ~87 | no dependency between the phases (GU) | **won**; opt-in, +0.5-0.7 % | done |
| hc_v2_norm -> hc_v2_down_rows | ~87 | the columns need the whole norm (NR) | lost: 74 us vs 32 | -- |
| down_rows -> gains (the hc merge) | ~87 | a cross-phase ticket wait (section 7.2) | lost | -- |
| hc_up_mix -> hc_quant_rows | ~91 | a 128-channel scale group spans 8 up-mix work-groups, so fusing needs the arrival counter | no (a dependency) | -- |
| ar_kernel -> hc_write | ~96 | the all-reduce is ticketed and polling; RADIANCE_XE_AR_HC_FUSE exists (run 100) | not on this rule | -- |
| gemv_bf16_pair -> router, router -> gather | ~45 | the router needs the whole logits row; the gather needs all the routing | no | -- |

- **The total of the fusable pairs:** about 0.8 ms a step, about 2 % of a step, or roughly +1.5 tok/s on today's 70.2.
- **Where the work lives:** every pair the rule allows crosses an op boundary. hc_write and hc_read are separate ops,
  as are silu_mul and quant_act, and the two norms. So none can be built inside libxe alone. Each needs a fused op
  emitted by the architecture plugin (as `add_rmsnorm` already is), plus a libxe row for it. That keeps every pair
  exact and behind a flag, since the plugin emits the old two ops without it.
- **The order to build:** hc_write + hc_v2_norm first (the largest, ~1 %), then silu_mul + quant_act, then the
  norms.

## Sources

- Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (2025-05-27), as summarised in
  [Cohere, "Megakernels" (2026-09-08)](https://cohere.com/blog/megakernels)
- [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs (OSDI '26)](https://www.usenix.org/conference/osdi26/presentation/cheng);
  [arXiv 2512.22219](https://arxiv.org/html/2512.22219v1); [code](https://github.com/tiledsl/mirage)
- [AutoMegaKernel (arXiv 2606.09682)](https://arxiv.org/html/2606.09682v1)
- [intel/llvm: a forward-progress extension draft (PR 14828)](https://github.com/intel/llvm/pull/14828);
  [intel/llvm: a grid barrier in the CUDA backend (PR 7598)](https://github.com/intel/llvm/pull/7598)
- [Intel DPC++ Compatibility Tool, DPCT1087 (no cross-group sync in SYCL; nd_range_barrier)](https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2024-2/dpct1087.html)
- This project: briefs/kernel-scoreboard.md (P87, B66), STATE.md (run 174's device idle; PEER2 / PR5 the LL
  exchange; GA2's timed-out job, 11-05)
