<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Part of the Intel port of radiance (milestone 3, run 108, owner note 109 item 8). -->
# One process per card: a design brief (owner note 109 item 8)

Note 109 item 8: "'one process per card' (no imports; a cross-process all-reduce): design and measure last, as a brief
first." This is that brief. Nothing is built. The measurement it proposes is phase 0 below.

## Today (facts, with file:line under src/)

- **One process, one thread per rank.** engine.cpp:1659 spawns `rank_loop(i)` per rank; engine_bringup.cpp:3461
  binds `devs[i]`. The step's batch is shared by pointer (rank_barrier.h:3-6: an IPC broadcast "costs tens of
  microseconds per step"; engine.h:112-114: "no IPC or serialisation").
- **One SYCL / Level Zero context over both cards** (core/device/l0.cpp:2871-2881), so that a USM pointer from either
  card is valid on the other. If that fails it falls back to per-card contexts (l0.cpp:2883).
- **What crosses cards.**
  - Each rank's host pool: pinned host memory in the shared context (l0.cpp:344, mem/pools.cpp:269-289, "the imports
    a second card binds"). It is cut into pieces of RADIANCE_HOST_POOL_PIECE_MIB, because xe will not bind a host
    buffer over 4 GiB that a second card imports (pools.h:83-85).
  - The checkpoint store: one pinned HostArena per engine holding every rank's shard (engine.h:485-488,
    hostarena.h:4-30), bound by card1 as an import.
  - The VRAM pools stay on their own card. With RADIANCE_L0_EXPLICIT_RESIDENCY=1, an allocation is resident on its
    own card, and on every card if it is <= 1 GiB (l0.cpp:273-312). Without it, a kernel makes every allocation in
    the context resident on its card (l0.cpp:274-277).
- **The all-reduce** (libxe/xe_allreduce.cpp; two ranks, peer = rank ^ 1):
  - Each rank writes its message into the peer's scratch slot (a remote write over PCIe, or into a pinned host
    mailbox in the `_host` form) and polls its own slot.
  - It sums f32(rank 0) + f32(rank 1) in a fixed order.
  - The peer's pointers come through a process-static table (xe_allreduce.cpp:106-178).
  - The LL form (the default since 6ec1fb3): 8-byte uncached stores of {seq, data}, polled per word; no flag, no
    fence.
  - Device time a decode exchange (AB4, graph-timed): 14.0-15.7 us at 1 row, 20.9-22.4 us at 4 rows. About 96
    exchanges a step.
  - No IPC exists anywhere in src (no zeMemGetIpcHandle, dma-buf, shm or SCM_RIGHTS).

## What the imports have cost (milestone 3)

- **Runs 84-93: every two-card load hung at its first step.** card1 spun in an xe VM_BIND that failed EBUSY while
  binding host memory the shared context had imported from card0.
  - m3 faults 21-24 included a kernel NULL dereference at 8 GiB on card1, and a GT reset.
  - The cure: the host pool in 2 GiB pieces, explicit residency, and NEO's DisableIndirectAccess (4fb114b).
- **Standing constraints.** The 4 GiB import limit (hence the pieces). Card1 binds the checkpoint store
  (RADIANCE_CKPT_PAGEABLE is the opt-out). A shared context makes every allocation a candidate for residency on both
  cards.
- **Since the fix, the two-card loads have been stable** (every run from 93 to 108). So the case for one process per
  card is robustness and simplicity, not speed. A speed gain is not expected (see the estimate).

## Design

- **Processes.** P0 (card0) is today's engine: HTTP server, scheduler, sampler, rank 0. P1 (card1) is a worker:
  rank 1's weights, KV, heat engine and mover, on a context of card1 alone. Each process imports nothing from the
  other card. Its host pool and its checkpoint shard are its own, bound only by its own card, so the 4 GiB import
  limit and the piece rule no longer apply.
- **Control plane (P0 -> P1, each step).** The step plan (rows, positions, block tables, draft and verify layout)
  goes over a shared-memory ring (memfd, mapped in both), with a futex doorbell; P1's acknowledgement comes the
  same way.
  - The plan is a few KB. The cost is the "tens of microseconds" rank_barrier.h names: against today's ~47 ms
    MTP-on step (56.9 tok/s at ~2.7 tokens a verify) that is ~0.1 %.
  - The tapes need the plan before the step's launches, as today.
- **Data plane: the all-reduce across processes.** Two transports, both keeping the LL protocol unchanged on the
  device:
  - **A, peer device memory over IPC.** Each process exports its scratch slots with zeMemGetIpcHandle (a dma-buf fd
    on Linux), sends the fd over a unix socket (SCM_RIGHTS), and the peer opens it with zeMemOpenIpcHandle.
    - The device path is today's: remote 8-byte stores into the peer card's VRAM over PCIe, so expect today's
      14-22 us an exchange.
    - It is still an import, but of ~40 MB of slots, not 16 GiB of host pool, and of device memory, not userptr.
  - **B, a host mailbox.** A memfd mapped in both processes, each importing only its own mapping into its own card
    (the `_host` form's path).
    - No card touches the other card's memory. Each exchange crosses PCIe twice (write to host, peer reads), so
      expect ~1.5-2 x A's latency; to be measured.
- **Sampling, MTP and the heat engine.** These stay in P0 as now. The final all-reduce already gives both ranks the
  full hidden state; P1 sends nothing back but the step's done signal (and its /stats share, over the ring).
- **Startup and failure.** P0 spawns P1 (or a unit starts both) and rendezvouses over the socket. If either exits,
  the other exits (the fail-fast exit's rule, note 112). The init rendezvous's 120 s timeout stays.

## Estimate

- **Speed.** About zero with transport A. With B, -1 to -2 % MTP-on if the exchange doubles (96 x ~15 us more =
  ~1.4 ms on ~47 ms).
- **Robustness.** It removes the import class of hangs (m3 faults 21-24) and the 4 GiB piece rule. It does not
  remove the driver's own P2P path under A.
- **Effort.**
  - rank_loop becomes a worker main: the shared-pointer batch becomes a serialised plan, and the static World
    table becomes a socket rendezvous.
  - Bring-up runs per process: weights, pools, tapes, heat engine.
  - /stats and the dashboard merge P1's share.
  - Roughly 8-14 runs. Phase 0 is 1-2.

## Phase 0: measure first (~$15-25, 1-2 runs)

1. **allreduce_test gains AR_XPROC=A|B.** The test forks: the child takes card1 (ZE_AFFINITY_MASK=1) and the parent
   card0, each on its own context. They exchange slots (A: IPC fds over a socketpair; B: one memfd), then run the
   existing correctness cases and AR_BENCH's graph-timed exchange.
   - Gate: every case PASS; time an exchange against AB4 (14.0-15.7 / 20.9-22.4 us).
2. **A control-plane ping (no card).** A 4 KB plan through a memfd ring with a futex doorbell, P0 -> P1 -> P0, timed
   on intel-host. Gate: < 50 us round trip. intel-host's CPU heat rule applies: no busy-poll, futex only.
3. **Decision.**
   - If A matches today's exchange and the ping is < 50 us: the full form is worth its 8-14 runs only if the owner
     wants the import class gone for the service.
   - Otherwise it stays a brief.

## Recommendation

Run phase 0 only if robustness against the import class matters for the service. The present arrangement has been
stable since run 93, and the brief expects no speed gain. The m3 numbers do not depend on it.
