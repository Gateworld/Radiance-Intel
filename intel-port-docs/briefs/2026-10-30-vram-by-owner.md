# One card's VRAM by owner (owner note 95 item 1)

Part of the Intel port of radiance (milestone 3, run 96, 2026-10-30).

## How it was measured

- **Setting:** H61's, the best measured so far: the two-card fix, NEO's 64 KB command buffer, floor 2048 MiB,
  headroom 3072 MiB, MTP on.
- **Image:** :90 (5a7e0e2 plus the ledger below).
- **Tools:**
  - **The engine's ledger** (RADIANCE_L0_VRAM_TRACE): every device or host-mapped allocation of at least 16 MiB, with
    the process's VRAM and the driver's free VRAM after it.
  - **NEO's own allocation log** (NEOReadDebugKeys=1 LogAllocationType=1 LogAllocationStdout=1
    LogAllocationMemoryPool=1): every graphics allocation with its type, pool, card and size. m3/neo_allocs.py sums
    it.
  - **The kernels' scratch and private sizes** from libxe.so's zebin metadata (m3/zeinfo_scratch.py).
- **Runs:**
  - **H68:** the rotation (72 requests, median 43.8 tok/s, misses 66.3 a step, equal to H61's 44.0 / 66.4).
  - **B33:** the first prompt at 20, 990 and 5,604 tokens in one load.

The card is 32,616 MiB: process VRAM plus the driver's free VRAM equals that throughout. NEO's logged allocations add
up to the driver's per-process figure within 15 MiB.

## Card1 at steady state after the rotation (H68)

| owner | MiB | when | notes |
|---|---|---|---|
| expert slab | ~22,058 | start-up | 18,216 slots; the weights pool is 25,673 MiB with the static weights |
| static weights | 3,615 | start-up | every rank's own shard |
| activation arena | 1,501 | after the load | 1.29 GiB + 2 x 90.63 MiB scratch; the mover lends 1.20 GiB of it to the slab while steps are small; the plan said 1.15 GiB |
| KV cache | 688 | start-up | 50% of 2 x 65,536 tokens |
| engine buffers made at the first requests | 68 | first prompt | 20 + 48 MiB |
| **engine total** | **27,932** | | |
| recorded passes: command buffers | 794 | the recordings | 6,345 x 128 KiB |
| recorded passes: linear streams | 489 | the recordings | 7,793, ~64 KiB |
| recorded passes: heaps | 398 | the recordings | 6,366 x ~64 KiB |
| kernels' private surfaces | 239 | first launches | 36 surfaces, 0.75-10 MiB each |
| scratch | 208 | first prompt | 128 + 64 + 16 MiB, one per queue context; 64 KiB a thread x ~2,048 threads |
| preemption (context save) | 184 | before the pools | 3 x 61.3 MiB, one per hardware context |
| kernel ISA | 32 | first launches | |
| printf surfaces | 16 | first prompt | 4 x 4 MiB |
| other driver buffers | ~10 | | timestamps, rings, fences |
| **NEO total** | **~2,370** | | the driver's figure: 2,354 |
| **free** | **2,330** | | the headroom asked 3,072; the 2,048 floor never fired |

The checkpoint store is not VRAM: 1,578 MiB of pinned host memory, resident on card1 as an import. The host pool's
pieces (8 x 2 GiB + 607 MiB) are host memory too.

## The first-use memory, by prompt size (B33)

| first prompt | NEO added on card1 | what |
|---|---|---|
| 20 tokens | +693 MiB | scratch 208, private 105, ISA 32, printf 16, the first recordings' lists ~330 |
| 990 tokens | +301 MiB | lists, private +41 |
| 5,604 tokens | +25 MiB | lists |

So "V3's 1.8 GB" from run 76 was mostly the arena (1.5 GiB, allocated right after the load) plus this ~0.7 GiB.

## The kernels' per-thread scratch (zebin metadata, libxe.so: 899 kernels, 420 with scratch)

| per thread | kernels |
|---|---|
| 128 KiB | hc_down_tiled, hc_up_tiled (spills; tiled prefill forms) |
| 64 KiB | vit_gemm_wa x10, qsa_score / qsa_score_mx x3, r4d_gdn_recurrent_update x4 |
| 32 KiB | r4d_gdn_recurrent_update x170, qsa_score x13, gdn_chunk_scan x6, mtp_prefill_gemm x4, hc wide / PF2 forms, mtp_enter_tiled, attn_vit |

The 128 MiB scratch surface is a 64 KiB kernel. The 3 surfaces belong to 3 queue contexts, and each context also
holds a 61 MiB preemption buffer.

## Item 3: the experts sized from what is left (run 97)

Three things stood between the experts and the free VRAM.

1. **The budget's card was too small.** It read SYCL's global_mem_size (30.30 GiB), and the runtime caps its free
   reading at that figure, so ~1.3 GiB of the 32,616 MiB card was never split. RADIANCE_VRAM_CAP_FROM_FREE=1
   (f4f2f70) reads the real free memory: a 2 GiB probe brings the reading under the cap, and the budget takes it as
   the card.
2. **The second card's import binds stall when it is fuller.** Rank 1's host memory is owned by card0, so on card1
   every host-pool piece and the checkpoint store are imports. Some of those binds never return (xe VM_BIND EBUSY,
   the runtime retrying without end), and no log line marks it:
   - the 1.54 GiB pinned store at headroom 2048 (H63 / H64 / H67 / M76);
   - the host pool's later pieces at the real-card budget (H70);
   - a fifth piece with MTP off even on an empty card (M81).

   The fix is fewer and smaller imports:
   - a host pool sized to what it holds: 8 GiB instead of 16 (it held 8.82 GiB at the old setting and ~6-7 GiB at
     the new one);
   - a pinned store of 2 snapshots (225 MiB) instead of 14 (1.54 GiB).

   A pageable store (d7b4da2, opt-in) also avoids the bind, but its copies serialise and cost ~8-10 % on the
   rotation (H69 / H71), so it is not used.
3. **The headroom** now covers what comes after the plan, as measured above. At 2560 MiB a card keeps ~0.47 GiB
   free at the rotation's steady state, note 95's 512 MiB margin.

| | H61 (best before) | the candidate |
|---|---|---|
| setting | fix, 64 KB key, headroom 3072, floor 2048, host pool 16 GiB, 14 pinned checkpoint slots | the same with the real-card budget, headroom 2560, floor 256, host pool 8 GiB, 2 pinned slots |
| expert slab slots (rank 0 / 1) | 18,168 / 18,216 | 19,766 / 19,815 (+1,600, ~+2.2 GiB) |
| rotation median tok/s | 44.0 | 47.5 / 47.3 (H74 / H75; H73 at a 9 GiB host pool 47.6) |
| misses a step | 66.4 | 43.2 / 43.1 (-35 %) |
| free VRAM at the steady state | ~2.3 GiB | ~0.47 GiB |
| M-pair run 2, MTP off / on | 37.78 / 51.52 (M65 / M66) | 37.83 / 51.88 (M82 / M80), text identical to M43 / M44 |
| out of memory, retries, lost tapes, floor firings | none | none |

The cost: a long session keeps 2 linear-state snapshots instead of 7, so a resume beyond the newest one prefills
more (B35 measures the 7K / 16K first tokens).

## Where the room is (for note 95 item 2)

1. **The recorded passes' lists, 1,681 MiB.** About 0.26 MiB a list (128 KiB command buffer + ~64 KiB stream + ~64
   KiB heap) over ~6,400 lists. Fewer or shorter lists, or smaller heaps, are the largest single lever.
2. **Free VRAM, 2,330 MiB.** The headroom is a guess, and the planner does not count NEO's 2.35 GiB. Sizing the
   experts last from the measured free VRAM is note 95 item 3.
3. **Contexts, scratch and private surfaces, 631 MiB.** Spill-free forms of the 64 KiB kernels would halve the
   largest scratch. Fewer queue contexts would save 61 MiB of preemption plus a scratch surface each.
4. **The arena, 1,501 MiB.** 1.20 GiB of it is already lent to the slab at small steps; the rest is ~0.3 GiB.
5. **KV, 688 MiB,** sized to 2 x 65,536 tokens at 50%.
