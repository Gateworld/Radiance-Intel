# Recorded passes ("tapes") on the B70: findings, and what xe direct was for

2026-10-29. Radiance on 2x Intel Arc Pro B70 (intel-host), Flash-Next, milestone 3. Sources: owner notes 52 and 58-64;
briefs/2026-10-28-bottleneck.md; briefs/2026-10-28-xe-direct-design.md (the phase 0 design, with its probe data);
STATE.md rows H20-H26, M35-M44, V1-V3 and X1-X8. **M** = measured, **C** = computed from measured numbers.

## 1. Summary

- **Tapes are not the main reason the B70 is slower than the AMD pair.**
  - The gap is about 4-6x: amd-host's R9700 pair does ~218 tok/s on this model, and the B70 pair does 37.8 tok/s
    MTP off and 51.8 MTP on.
  - A B70 decode step is 27.9 ms, and **26.8-26.9 ms of it is kernels running** (M, P3/M20). Tapes, launches and
    list boundaries account for about 1 ms.
  - The time goes to kernel form: hc 8.0 ms, trunk 6.7, experts 6.5, GDN + attention 3.4, all-reduce 1.7 (M,
    bottleneck brief). Xe2 has no FP8 math, so every E4M3 weight is decoded by ALU, and the kernels move bytes at
    ~21 % of the card's bandwidth.
- **What tapes cost is VRAM.** The driver keeps **2.74 GiB a card on a single prompt and 3.6-4.8 GiB in the service
  rotation** for the recorded command lists, and never returns it (M). That memory is not available to the expert
  cache. It also forces the 2 GiB recordings floor (note 58) and caps how many tapes fit with 6 users.
- **What freeing it would buy (C):**
  - rotation: +5-8 % by the agent's estimate, up to +10-15 % if the extra free VRAM after load can go to experts
    (section 5);
  - single prompt: +2.5-7 % from fewer list boundaries and lighter barriers;
  - 6 users: probably the largest gain, not yet estimated.
- **xe direct** was the plan to free it: an engine-owned submission path, as the AMD build has. It is **closed**
  (note 64). A safety classifier stopped the work four times, three times in the unattended runner and once in an
  attended session, before proof (i) passed.
- **There is a cheaper route that stays inside Level Zero** (section 6): keep fewer lists alive at the peak. In the
  rotation the peak was 3,636 lists, against 542 needed in steady state, and the memory follows the peak.

## 2. Tapes on AMD and on Intel

| | AMD build (R9700, core/device/aql.cpp) | Intel build (B70, core/device/l0.cpp) |
|---|---|---|
| Launch path | The engine writes the hardware queue's packets itself (~6 ns each); HIP is kept for allocation and code loading | Level Zero immediate command lists (~2 us host time an append, M) |
| A tape is | the finished packets plus their argument blocks, in VRAM the engine owns | driver-owned recorded command lists, one per segment |
| Dropping a tape | frees its memory | returns nothing to the system (M38, X1) |
| VRAM floor needed | no | yes (2 GiB since note 58) |

Intel's driver stack gives a process no user-mapped hardware queue to write into: submission goes through the
runtime (NEO) and the xe kernel driver. NEO already runs its own ring without a system call per submission
(direct submission, active on the B70: 3 exec calls for ~4,000 launches, M X3). So the system-call cost is not
the issue. The issue is the memory behind each list.

## 3. Measurements

### 3.1 What a recorded list costs (X1-X3, probe on card1, NEO 1.14.37020 under xe)

- **1.19 MiB of VRAM a list** (256 empty lists: +304 MiB), about 3.2 driver buffers each (M).
- **Destroying lists returns 0 MiB.** Later lists reuse the same memory: +0.0 MiB for sets of 0-128 launches (M).
  So the cost is the **high-water mark of lists alive at once x ~1.2 MiB**, held for the life of the process.
- GPU time a launch inside a played list: **0.57-1.19 us in an in-order list, 0.04-0.12 us with no dependence**
  (M). The extra ~0.5-1.1 us per dependent launch is the runtime's wait between commands.
- Host cost: 1.97 us an append; 5.5 us for an immediate launch plus host synchronize (M).

### 3.2 How much the engine's recordings hold (note 61 item 0)

| step | setup | result |
|---|---|---|
| V2 | single prompt, tapes on, VRAM trace | VRAM grows with lists alive: ~1.22 MiB a list + ~55 MiB, **2.74 GiB at 2,265 lists**; engine allocations +0; 51.33 tok/s (the trace costs nothing) |
| V3 | single prompt, **tapes off** | **+1.8 GB on the first prompt anyway**, then flat: first-use memory, not tapes |
| H26 | service rotation, VRAM trace | **+3.68 GiB when the floor first fires, +4.92 GiB at the end; peak 3,636 segment lists alive**, when recordings of new shapes overlap before their first plays. Steady state needs only **542 composed lists**, but the driver keeps the peak's memory. 70 requests, median 40.8 |
| V1 | first trace attempt | heat stop at 91 s (two driver-counter reads a segment cost CPU); its 14 lines agree (~1.28 MiB a list) |

A1's earlier "+4.5-4.8 GB" was this tape memory plus V3's first-use 1.8 GB.

### 3.3 What the floor does in service

| step | floor | result |
|---|---|---|
| H20 | 1 GiB | median 41.7 |
| H21 | 1 GiB | **out of device memory on card0 at request 21** (m3 fault 14): ~135 MiB free, the engine hung on a failed append |
| M38 | dropping tapes under the floor | **VRAM did not come back**; every pass ran live, MTP off 29.4 tok/s; dropping made opt-in |
| H23 / H24 / H25 | 2 GiB | clean, 38.9 / 41.8 / 41.5; the floor fires at ~1.55 GiB free, so the cards stay clear of the ceiling |
| M43 / M44 | 2 GiB | single prompt 37.77 / 51.36 tok/s (MTP off / on): the floor does not cost the single prompt |

Since note 58 (da8a4d6), a failed append is retried, and a segment that still cannot be appended fails the step
visibly. A silent wrong answer was caught first (E11) and fixed (E12, E13).

### 3.4 Expert residency against VRAM

- H25 (service, :70): 18,288 of 25,078 expert units resident (72.9 %), **80.7 misses a step**, rotation step
  82.7 ms (M).
- H22 (FP16 hc, 517 fewer units resident): **90.8 misses (+10.1)** (M). The local slope is **~0.0195 misses a step
  per unit**. A unit is 1.27 MB.
- One miss costs an exposed 150-220 us in the rotation (fitted, bottleneck brief), so today's misses are 12-18 ms of
  the 82.7 ms step (C).

## 4. What xe direct was

**Goal.** Do what aql.cpp does on AMD:
- the engine owns submission and the memory of its recorded passes;
- Level Zero keeps compiling and loading kernels, device properties, allocation and the live path (prefill);
- l0.cpp stays the default and the fallback.

**Design** (phase 0 brief, sections 2-3):
- one engine-owned address space and compute queue per card, through the xe render node as an ordinary user, with
  no driver, firmware or root change;
- a recorded pass becomes one buffer of ~0.4-0.6 MB (C), against thousands of ~1.2 MiB lists today;
- freeing the buffer returns its memory at once;
- passes are played through a GPU-polled ring, with barriers only between dependent commands;
- the engine's existing allocations are shared into that address space at the same GPU addresses.

**Gates:**
- the tiny model's tests bit-identical;
- Flash-Next text identical to M37;
- two clean rotations with a VRAM trace;
- target: no recordings floor, or one well under 1 GiB.

**Status at closing:**
- Phase 0 done (~$6). Phase 1 written but not committed (`src/core/device/xe_direct.*`, the proof under
  `src/libxe/test/`).
- Phase 1 proof (i) **runs to its end without a fault on card1**: own address space, queue and batch. **But the
  kernel's stores are not seen.**
  - The single "right" output of 4,096 is a coincidence: element 12's expected value equals the fill value.
  - The one GPU fault (X4, m3 fault 15: a wrong local-id layout) was fixed.
- The agent's leading guess for the missing stores was two unapplied kernel-binary relocations. **That is ruled
  out.** NEO's linker patches that relocation only for kernels that need implicit arguments (stack calls, a
  debugger, or a metadata flag). This kernel needs none, and both sites already hold 0, which is what NEO runs.
  The cause is still open.
- About $30 of the $320 approved for phases 1-2 was spent. The rest went back to the milestone.

## 5. What freeing the tape memory would be worth

**Rotation.** It depends on how much of the freed VRAM the expert cache can take.

| assumption | units to experts | misses a step | median (C) |
|---|---:|---:|---:|
| The agent's revision: headroom 4 -> ~2.3 GiB (keeps the 1.8 GB first use + 0.5 margin) | ~1,440 | ~53 | **43.5-45 (+5-8 %)** |
| The phase 0 brief: +3 GiB | ~2,540 | ~31 | 45.6-47.9 (+10-15 %) |

- The traces show **~25.5 GiB used a card after load, ~7.2 GiB free of 32 GiB**, more than the 4 GiB headroom.
  With tapes off (V3), ~5.3 GiB stays free after first use.
- If that room can go to experts once recordings are MBs, the gain is at or above the brief's figure. Each extra
  unit is colder, so the return falls as more fit.
- Where the extra ~3 GiB is budgeted was not found before closing (note 62 item 3).

**Single prompt.**
- From the list boundaries (0.9-1.1 ms measured gap) and lighter barriers: 0.65-1.7 ms of 26.5 ms, so MTP off
  37.8 -> 38.7-40.4 and MTP on 51.4 -> 53-55 (+2.5-7 %, C).
- The single prompt's real levers are the kernel forms: the hc rewrite and the expert GEMV grids, ~4-5 ms.

**6 users.**
- C9 (run 39, 4 GiB headroom): the 6-user phase was floor-bound, with 22 tapes at ~0.8 GB a trunk tape (M).
- With passes in MBs, every shape's tape fits. Likely the largest gain of the three. Not estimated.

## 6. Routes that stay inside Level Zero

1. **Lower the peak of lists alive.** The memory is the high-water mark, not the steady state.
   - In H26 the peak, 3,636 segment lists, came from recordings of new shapes overlapping before their first plays.
     Steady state needs 542 composed lists (~0.65 GB).
   - Bounding how many recordings are in flight at once, or composing and releasing segments sooner, would hold the
     peak near the steady state.
   - That is ordinary scheduling in l0.cpp, the same API in use today. If the peak stays near ~800 lists, the pool
     is ~1 GB a card instead of 3.6-4.8 GiB. That frees most of what xe direct would have freed, and the floor
     could come down.
   - **This is the recommended next step for the tape problem.**
2. **Count the pool's reusable memory in the floor rule.** Memory from destroyed lists is reused by new ones, so
   the floor should not treat it as lost. That stops false refusals (~1 run) but gives no VRAM to experts.
3. **Find the 1.8 GB of first-use memory** (V3). If it is scratch for the prefill kernels that spill (the GDN chunk
   scan, tiled hc; decode spills 0 B), the GDN chunk-scan rewrite already in the queue may free most of it.
4. **Report to Intel (NEO, open source).** Destroyed command lists' memory is never returned to the system. A
   release call, or a pool cap setting, would fix this for everyone.

## 7. Open questions

- Where the ~3 GiB of free VRAM beyond the 4 GiB headroom is budgeted, and how much of it experts may take.
- What the 1.8 GB of first-use memory is.
- Why proof (i)'s stores are not seen. Not pursued; xe direct is closed.
- The 6-user gain from fitting every tape.
