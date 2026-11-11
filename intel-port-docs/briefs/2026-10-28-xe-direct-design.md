# xe direct: an engine-owned submission backend for the B70 (owner note 59, phase 0)

Run 75, 2026-10-28. This is a design brief: no backend code is written yet. The new card work was three read-only
probe steps on card1 (X1-X3) with the probe committed as c6209e0. Each number below is either **M** (measured, with
its file) or **C** (computed here from measured numbers). **Phase 1 waits for the owner's approval of the budget in
section 7.**

## 1. What the driver holds, and how it submits (item 1)

The probe is `xe_list_pool_probe` (src/libxe/test/list_pool_probe.cpp). It runs on its own context with a
one-work-group kernel and no model, and reads its own DRM client from `/proc/self/fdinfo`. It ran on image :70 as
NEO 1.14.37020 under xe, on card1 (`results-m3/card01/X1-out.txt`, `X2-out.txt`, `X3-out.txt`,
`X3/strace.txt`).

**The memory behind a recorded list is the driver's reuse pool, not a leak.**
- The first 256 regular in-order lists, empty and closed, took **+304 MiB of VRAM: 1.19 MiB a list (M, X1)**. In
  strace that is ~3.2 BOs a list: 829 `XE_GEM_CREATE` + `GEM_MMAP_OFFSET` in the first set's 0.14 s (M, X3).
  These are the list's command buffer and heaps.
- Destroying the 256 lists returned **0 MiB** (M).
- Every later set reused the same memory with **+0.0 MiB**. That held for 256 lists of 0, 1, 16 or 128 launches,
  in-order or plain, twice over (M).
- No `GEM_CLOSE` came before process exit (837 at exit) (M).

So what NEO holds is the **high-water mark** of lists alive at once, times their size. The engine's lists are
larger than the probe's: A1 measured +4.5-4.8 GB for ~1,182 segment lists, ~3.8-4 MB a list (M, run 42), because
their command streams are longer.

This explains the whole tape VRAM history:
- Dropping a tape frees nothing to the kernel (M38).
- The floor sees free VRAM fall and never climb back.
- The 4.5 GB a card stays held for the process's life. A recording made after a drop reuses the dropped lists'
  memory, but the floor rule cannot see that.

**Direct submission is active on our immediate lists (NEO's ring, "ULLS").**
- Over the probe's whole run, about 4,000 immediate launches and plays, strace on the host saw **3
  `DRM_IOCTL_XE_EXEC`** (M, X3):
  - one at start;
  - one when the first immediate list began (06.68 s);
  - one after the probe's 2-second idle (08.75 s): the ring is restarted after idling.
- The run also made 5 `EXEC_QUEUE_CREATE` and 93 `XE_WAIT_USER_FENCE`. strace decodes the latter as
  `ETNAVIV_PM_QUERY_DOM`, same number 0x0a IOWR. The user fences mean a long-running (LR) VM.
- The engine's idle-busy share stayed at 0.5 % (M, X1), so the parked ring does not show in fdinfo cycles.
  Forcing `EnableDirectSubmission=1` (echoed by the driver) changed none of the numbers (X2).

Consequences:
- The engine **already pays no system call per submission**. The host cost is NEO's own: **1.97 us an
  append** (M, X1-X3, the same under strace, so it is not a syscall).
- An immediate launch plus host synchronize is a **5.5 us round trip** (M, X1/X2; 22 us under strace's ptrace
  cost on the waits).
- A GPU-polled ring of our own buys its gain on host time and on the GPU-side command stream, not on the
  kernel crossing.

**The GPU time of one command inside a played list (M, X1-X3; the in-order range is three runs):**

| list | 1 launch | 512 launches | a launch |
|---|---|---|---|
| in-order (the recorder's lists) | 10.3-25.2 us | 317-616 us | **0.57-1.19 us** |
| plain, no dependence | 9.3-23.2 us | 43-72 us | **0.04-0.12 us** |

So a dependent launch costs **~0.5-1.1 us of GPU time beyond an independent one**. That is NEO's in-order
post-sync / pipe-control wait plus the walker's drain and launch. Round 2 measured 0.593 us in-order (M,
round2/STATE.md:62). The spread between runs (0.57 under strace, 1.17-1.19 without) is not explained; clock state
is the likely cause. Phase 1 measures it on our own commands.

## 2. The minimum from xe and from the hardware (item 2)

### 2.1 The xe uAPI (intel-host's `/usr/include/drm/xe_drm.h`, the render node we already open)

| need | ioctl | notes |
|---|---|---|
| device facts | `DRM_IOCTL_XE_DEVICE_QUERY` (engines, mem regions, gt list, config) | the CCS engine instance, VRAM region id, VA bits |
| our address space | `DRM_IOCTL_XE_VM_CREATE` with `DRM_XE_VM_CREATE_FLAG_LR_MODE` | LR, because a ring that never ends has no job timeout; LR forbids syncobj out-fences, so all completion is **user fences** -- what NEO does (93 waits in X3) |
| memory | `DRM_IOCTL_XE_GEM_CREATE` (VRAM placement, `cpu_caching` WC for host-written), `DRM_IOCTL_XE_GEM_MMAP_OFFSET` + mmap, `DRM_IOCTL_GEM_CLOSE` | **freeing a recorded pass = unbind + GEM_CLOSE: the kernel gets the pages back at once** |
| mapping | `DRM_IOCTL_XE_VM_BIND` (MAP / UNMAP, `pat_index`, a user-fence sync) | `pat_index` is per platform (Xe2 PAT table; take NEO's values for device-cached and host-WC) |
| shared buffers | `DRM_IOCTL_PRIME_FD_TO_HANDLE` on a dma-buf fd, then `VM_BIND` **at the same GPU address** | how our VM sees the engine's Level Zero allocations (2.3) |
| a queue | `DRM_IOCTL_XE_EXEC_QUEUE_CREATE` (class `DRM_XE_ENGINE_CLASS_COMPUTE`, width 1, one placement) | one per card; destroy at exit |
| start / restart the ring | `DRM_IOCTL_XE_EXEC` (one batch address, a user-fence sync) | once at start and after a park, as NEO's 3 execs |
| block when idle | `DRM_IOCTL_XE_WAIT_USER_FENCE` (addr, value, mask, op GTE, timeout) | the hot path polls memory; the ioctl is for long waits only |

No driver, module, firmware or root change. Everything above is the render node as an ordinary user.

### 2.2 The Xe2 commands (NEO's `shared/source/generated/xe2_hpg_core/hw_cmds_generated_xe2_hpg_core.inl`, read run 75)

| command | DWORDs | role in xe direct |
|---|---|---|
| `PIPELINE_SELECT` (GPGPU) | 1 | once at ring start |
| `STATE_BASE_ADDRESS` | 22 | once at ring start. Instruction base = our ISA heap; general state base = our indirect-data heap; surface state unused (the kernels are stateless, A64 pointers) |
| `STATE_COMPUTE_MODE` | 3 | once (large-GRF mode per kernel class if any kernel needs 256 GRF) |
| `CFE_STATE` | 6 | once: max threads, **scratch space** (only the spilling prefill kernels need it; decode spills 0 B, K1) |
| `STATE_SYSTEM_MEM_FENCE_ADDRESS` | 3 | once (Xe2 needs it for system-scope fences) |
| `COMPUTE_WALKER` | 40 | one per dispatch. Embedded `INTERFACE_DESCRIPTOR_DATA` (8 DW: `KernelStartPointer`, `SharedLocalMemorySize`, `NumberOfThreadsInGpgpuThreadGroup`, `NumberOfBarriers`), `ThreadGroupId{X,Y,Z}Dimension`, `SimdSize`, `GenerateLocalId` / `EmitLocal`, `IndirectDataStartAddress` / `IndirectDataLength`, **`InlineData[8]`** (the first 32 B of the arguments) and an embedded `POSTSYNC_DATA` (5 DW: a write or timestamp when the walker completes) |
| `PIPE_CONTROL` | 6 | the dependence between dependent dispatches (CS stall + the dataport flush the next kernel's reads need), and the end-of-pass fence write |
| `MI_SEMAPHORE_WAIT` | 5 | the ring's idle point: poll a tail word in memory (`WaitMode` polling, `CompareOperation` >=) |
| `MI_BATCH_BUFFER_START` | 3 | the ring jumps into a recorded pass (second-level batch); also loops the ring |
| `MI_BATCH_BUFFER_END` | 1 | a recorded pass's last command (returns to the ring) |
| `MI_STORE_DATA_IMM` | 5 | the per-play fence value the host polls |
| `MI_MEM_FENCE` | 1 | ordering for host-visible writes where PIPE_CONTROL is heavier than needed |
| `MI_ATOMIC` / `MI_LOAD_REGISTER_IMM` | 11 / 3 | not needed in phase 1 |

Field layouts come from that file. Its license is MIT, compatible with the tree's Apache-2.0, but the backend
writes its own packers for the ~10 commands rather than vendoring 325 KB.

**Generation risk.** These are Xe2-HPG (BMG) layouts. Xe3 changes `COMPUTE_WALKER` (heapless) and Xe-HPG differs.
The backend must refuse any device whose IP version is not the one it was written for, by name, and fall back to
l0.cpp.

### 2.3 Kernels and memory outside Level Zero

**Kernel code.** libxe.so carries **46 native zebin images for intel_gpu_bmg_g31** (M: `.ze_info` × 46,
registered through `__sycl_register_lib`). There are two ways to reach them:

- **(a)** Ask Level Zero for the module's native binary (`zeModuleGetNativeBinary` on the module SYCL built).
  Level Zero keeps compiling and loading, as the note wants.
- **(b)** Read the images at registration, as aql.cpp reads HIP bundles at `__hipRegisterFatBinary` /
  `__hipRegisterFunction` (aql_code.cpp's job).

(a) is less code; (b) needs no Level Zero call per module. **Phase 1 uses (a).**

A zebin is an ELF:
- `.text.<kernel>` is the ISA, copied into our instruction heap at a 64-B aligned offset, which becomes
  `KernelStartPointer`.
- The `.ze_info` YAML gives each kernel:
  - `execution_env`: `simd_size`, `grf_count`, `slm_size`, `barrier_count`, `inline_data_payload_size`,
    `has_*` flags;
  - `payload_arguments`: offset, size and type of each argument (`arg_bypointer` stateless = an 8-B A64 pointer,
    `arg_byvalue`, `group_count`, `local_size`, `global_id_offset`, `buffer_offset`, `private_base_stateless`);
  - `per_thread_payload_arguments` (local ids, or none when `GenerateLocalId` makes them).
- Relocations, if any (global constants), are patched at load.

A kernel whose `.ze_info` asks for anything the backend does not implement is **refused by name and stays on
l0.cpp**. aql.cpp does the same with `Hidden::Unsupported`.

**Arguments.** The tape layer already captures each command's kernel, group counts, group size, SLM and the
by-value argument bytes. That is the audit's byte-for-byte record (l0.cpp:1120-1210). xe direct maps those bytes
onto the zebin's payload layout: the cross-thread data goes to an indirect-data slot, and its first 32 B go into
the walker's inline data.

**Memory.** The engine's buffers (weights, KV, workspaces) are Level Zero allocations in NEO's VM. Our VM sees them
by **dma-buf export + PRIME import + `VM_BIND` at the same GPU address**. Level Zero exports with
`ze_external_memory_export_desc_t` (DMA_BUF) at allocation time, then `ze_external_memory_export_fd_t`, and the
engine allocates few, large slabs. Every pointer in a recorded argument then stays valid unchanged.

Phase 2 can instead move allocation into the backend (GEM_CREATE in our VM, imported into Level Zero for the live
path). That is more change, and only worth it if export turns out restricted.

## 3. The design: `xe.cpp` beside `l0.cpp`

The new backend sits beside l0.cpp in the same dispatch table (dispatch.cpp), as aql.cpp sits beside hip.cpp.
**l0.cpp stays the default and the fallback**: `RADIANCE_XE_DIRECT=1` selects the new backend per card. Any refusal
leaves the card on l0.cpp with one warning: device id, a kernel's `.ze_info`, export failure, or a failed probe
dispatch at start.

**What stays with Level Zero / SYCL:**
- compiling and loading kernels (and handing us their zebin);
- device properties;
- allocation (exported);
- the live path: prefill, and any pass that is not recorded;
- copies and peer access.

**What xe direct owns:**
- one LR VM, one compute exec queue and one ring per card;
- the recorded passes' memory;
- every play.

**A recorded pass becomes one BO.** It holds:
- a second-level batch: walkers, with a pipe control only between dependent commands (the tape knows the lanes);
- the pass's indirect-data slots;
- its fence write and `MI_BATCH_BUFFER_END`.

Size: ~1,380 commands a rank-step × (160 B walker + ~128-256 B indirect data + 24 B pipe control) ≈ **0.4-0.6
MB a pass (C)**, against ~3.8-4 MB a *list* today and ~1,182 lists for 4 recordings.

**Freeing a pass** is `VM_BIND` UNMAP + `GEM_CLOSE`, and the VRAM is back with the kernel at once. Then:
- "dropping a tape returns nothing" (M38) is gone;
- the recordings floor can go to ~0;
- the drop policy only has to bound MBs.

**A play:**
1. The host writes `MI_BATCH_BUFFER_START(pass)` + `MI_STORE_DATA_IMM(fence, seq)` into the ring.
2. It bumps the tail word the ring's `MI_SEMAPHORE_WAIT` polls.

No ioctl and no driver code; host cost ~tens of ns (aql.cpp's 6 ns a packet is the analogue). The host waits by
polling the fence word, and `XE_WAIT_USER_FENCE` covers long waits. When idle, the ring stays parked on the
semaphore; it does not need restarting.

**Ordering with the live path (Level Zero).** The engine already synchronizes at step boundaries. A play waits for
earlier Level Zero work through a host wait at the hand-off, then the ring runs. A later phase may make a GPU-side
semaphore on an exported event word. Recorded steps (all of decode) never hand off mid-step.

**Lighter barriers.** NEO puts a full post-sync wait between every in-order command (the 0.57-1.19 us). We emit:
- no barrier between independent commands (the tape's lanes and the shared-expert / routed-expert pairs, note
  54 item 3);
- the minimal pipe control (CS stall + untyped dataport flush, no L3 or state flushes) between dependent ones.

Whether the minimal form is cheaper than NEO's is phase 1's per-dispatch measurement. It is not assumed below.

## 4. Gates (from note 59)

- **Phase 1** (card1 only):
  - (i) own VM, one buffer, one kernel (a libxe kernel loaded from its zebin), one `XE_EXEC`, the result checked
    against the CPU;
  - (ii) the GPU-polled ring: 10,000 plays with no ioctl between them (strace count), the result checked, and the
    time per dispatch for dependent and independent commands against X1's 0.57-1.19 / 0.04-0.12 us;
  - (iii) a forced hang (a semaphore that never completes) recovered by xe's reset with a fault row, and the
    backend reporting it and falling back.
- **Phase 2:**
  - the tiny model's E-tests bit-identical through xe direct;
  - Flash-Next's M-pair text identical to M37 / M43;
  - two clean rotations with a VRAM trace;
  - **target: no recordings floor, or one well under 1 GiB.**
  - card1 first; both cards only after card1 passes.

## 5. Risks

1. **GPU hangs from wrong commands.** A bad walker or pipe control hangs the CCS. xe's reset recovers it, which is
   a fault row (note 59). Two in one run pause the runner.
   - Mitigations:
     - every new command form is first run in a 1-dispatch batch with a short `XE_EXEC` user-fence timeout;
     - the ring is only used after (i) passes;
     - card1 only.
2. **Generation-specific layouts.** Xe2-HPG only; refuse others by IP version (2.2).
3. **The zebin contract.** New `.ze_info` keys from a compiler update. Refuse unknown argument types per kernel
   (they stay on l0.cpp); pin the compiler in the build image (already pinned: ri-m1build:3, Intel LLVM v7.2.0).
4. **Export / import.** If NEO refuses DMA_BUF export for device allocations, or binds them with a PAT our VM cannot
   match, phase 2 moves allocation into the backend (2.3). It is a larger change but the same design.
5. **Scratch.** The spilling prefill kernels (GDN KT=2, hc tiled: never launched in decode) need `CFE_STATE`
   scratch. Phase 2 records decode only, and spilling kernels are refused.
6. **Two submitters on one engine.** NEO's ring and ours share the CCS. The GuC time-slices LR queues, so a
   hand-off costs a context switch (~us). The live path is prefill, where that is noise.
7. **Preemption of LR queues on eviction.** xe evicts an LR VM's memory only with preempt fences. VRAM is not
   overcommitted (the plan fills it below the ceiling), and a preemption is a delay, not a fault.

## 6. Expected tok/s (from measurements)

### 6(a) The rotation with the recovered VRAM

Inputs:
- **H25** (:70, the service): 18,288 of 25,078 expert units resident (72.9 %), **80.7 misses a step**, median
  **41.5** tok/s, rotation step wall median 82.7 ms (M).
- **H22** (the same image, FP16 hc): **17,771 resident (-517) → 90.8 misses (+10.1)**, median 40.6 (M). The local
  slope is **0.0195 misses a step per resident unit** (C).
- A unit is 1,268,480 B. The cost of a miss inside a rotation step is the brief's fitted 150-220 us exposed
  (briefs/2026-10-28-bottleneck.md §2, M-fitted). So 80.7 misses ≈ 12-18 ms of the 82.7 ms step (C).

What xe direct frees on a card is everything the recordings now take beyond their ~MBs. The service reserves **4
GiB of headroom**, and the floor fires at ~1.5 GiB free (H25: 1,555 MiB), so **~2.5 GiB is held by NEO's list
pool** and 1.5 GiB more is idle headroom kept for it. With recordings in MBs, the headroom can drop to ~1 GiB:
- **+3 GiB ≈ 2,540 units** → 20,830 resident (83 %) (C);
- the local slope gives **~31 misses a step (-50)** (C). This agrees with a straight line from H25's 72.9 % to
  m2's 87.5 % / 8.3 misses (C, a weaker anchor: MTP off, other traffic);
- **-50 misses × 150-220 us = 7.5-11 ms off an 82.7 ms step → ~72-75 ms → median ~45.6-47.9 tok/s (+10-15 %)**
  (C).

Two bounds frame this:
- The owner's single-prompt slope, -500 units ≈ +4 ms on the MTP step, would give ~20 ms for 2,540 units. The
  whole miss time in the rotation step is only 12-18 ms, so that slope overstates here.
- With headroom at 0.5 GiB (+3.5 GiB, 2,960 units, ~85 %), misses fall to ~23 and the median to ~47-49.

**Expected: ~46-48 tok/s rotation median.** The floor's live passes (190 of 28,710 in H25) add nothing
measurable.

### 6(b) The single prompt from the gaps between kernels

Inputs:
- The MTP-off warm step is **26.51 ms** (M43, M).
- The device windows sum to 26.78-26.93 ms over the profiled window against a 27.86 ms wall (P3/M20). So
  **0.9-1.1 ms of the step is host/sync/list-boundary gap**: 66 list boundaries at 12-38 us (bottleneck brief row
  7) (M/C).
- ~1,380 commands a rank-step, nearly all in-order dependent. At X1-X3's **0.5-1.1 us dependent-launch excess**
  (in-order minus independent), **0.7-1.5 ms of the step is NEO's dependence cost between commands** (C). Part of
  it is the true drain of a dependent kernel, which no barrier removes.

What xe direct recovers (C):
- **The list boundaries:** a pass is one second-level batch with no boundaries. That is 0.5-0.9 ms (row 7's
  ceiling was ~0.5 ms with fewer lists; the measured gap is 0.9-1.1).
- **Independent commands without a barrier:** the tape's lanes, today serialized by the in-order list. If ~20 % of
  the commands are independent of their predecessor (the attention side chain, the shared expert against the
  routed experts), that is 0.15-0.3 ms.
- **The minimal pipe control instead of NEO's:** 0 to ~0.5 ms. Unknown until phase 1 times it, and counted at 0
  in the low estimate.

**Total: 0.65-1.7 ms of 26.5 ms → MTP off 37.8 → ~38.7-40.4 tok/s (+2.5-7 %).** The MTP step (~89 ms with ~3.5×
the commands) scales the same way: **51.4 → ~53-55** (C).

This is the smaller half. **The single prompt's real levers remain the kernel forms** (the hc rewrite and the
expert GEMV grids, the brief's #3/#4: ~4-5 ms). xe direct mainly earns its cost on the rotation and the VRAM
problem.

## 7. Estimate for phases 1-2

The milestone has averaged ~$7 a run (about $515 over 75 runs). Code-heavy runs with builds and card steps cost
$12-20.

| phase | work | runs | dollars |
|---|---|---|---|
| 1 | xe.cpp skeleton (device query, LR VM, GEM/BIND, exec queue, user fences); zebin reader (`.ze_info` YAML subset, ISA heap); the ~10 command packers; proof (i) and ring (ii); hang test (iii); a ctest for the packers and the zebin reader on the CPU | 4-6 | **$60-100** |
| 2 | dispatch-table backend; export/import of the engine's slabs; tape → pass BO; play through the ring; fences and host waits at hand-offs; refusals and fallback; E-tests, M-pair, two rotations with the VRAM trace; then card0 | 8-12 | **$130-220** |
| | **total** | **12-18** | **$190-320** |

The milestone has ~$485 left. With this, the remaining items fit only in part: the CPU-expert split, the occupancy
items, the hc rewrite (worth ~4-5 ms on the single prompt) and GDN. **The owner decides the budget before phase 1
(note 59).**

## 8. Recommendation

Approve phase 1 at **$100** (proof + ring + hang test). Decide phase 2 on its per-dispatch numbers.

There is also a cheaper, partial fix inside l0.cpp: count destroyed lists' memory as room for new recordings. It
would stop the floor's false refusals, but it cannot give the 2.5-3.5 GiB back to the experts, which is where
6(a)'s +10-15 % comes from. I would not take it instead of xe direct, but it is ~1 run if the owner wants the floor
rule right before phase 2 lands.

## Addendum (run 76, owner note 61 item 0): the recordings' VRAM measured in the engine

`RADIANCE_L0_VRAM_TRACE` (609653b) reads the engine's own per-card VRAM from fdinfo against lists alive:
- **V2, the single prompt:** +2,810 MiB a card over the recordings. It follows the peak of segment lists alive
  (2,265 at ~1.22 MiB each). Engine allocations +0; composed stretches +0, since they reuse the segments' memory.
- **H26, the rotation:** +3,680 MiB when the floor first fires, and +4,921 MiB at the peak (3,636 segment lists
  alive at once).
- **V3, tapes off:** the first prompt adds +1,858 MiB too. A1's +4.5-4.8 GB was this first-use memory plus the
  recordings.

So the memory xe direct would own and free is 2.74 GiB (single prompt) to 3.6-4.8 GiB (rotation) a card, which
passes note 61's 2.5 GiB bar.

**Section 6(a) is corrected.** It credited the experts with +3 GiB of headroom. But the headroom still has to hold
the 1.8 GB of first-use memory, so it can fall from 4 GiB to ~2.3 GiB:
- +1.7 GiB ≈ 1,440 units;
- ≈ -28 misses a step at the measured slope;
- -4.2 to -6.2 ms of the 82.7 ms step;
- **rotation median ~43.5-45 tok/s (+5-8 %)**, not 46-48.

The rest returns only if the first-use 1.8 GB is attributed and made smaller. That is outside xe direct.
