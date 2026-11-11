# Radiance on Intel: system review

2026-10-27 ~14:00 UTC, after run 39. Branch `intel-l0` at 6ebd5d8 plus two uncommitted files; image :39 not built (note 41). `./intel-host-check.sh pre` and `git -C src log` were denied here; intel-host's state is the 13:41:58 UTC precheck (both cards free, 117 GiB available).

## 1. Executive summary

1. Works: two B70s serve Flash-Next (TP2) from recorded passes on direct Level Zero command lists; the tiny model replays token-identical (E1), the op oracles pass on the card (E2-E4, O1-O4: 0 mismatches), tool calls match production 3 of 3.
2. Works: MTP is a net gain alone, 42.0 tok/s against 35.7 off (M26/M25, run 33); the rotation median went 22.2 to 37.5 (H5, run 29, image :35); the pool fell from ~10 GB to 2.5 GB a card.
3. Does not work: first token, 53 s at 5.6K tokens against production's 0.5 s; `--ngram-placement ram` is refused (M23).
4. Does not work: concurrency. MTP loses from 3 users unless gated (C3b 41.3 / 49.4 against C2's 59.4 / 71.1 off); the gate stalled at 4 GiB headroom (C6) and is uncommitted; aggregate at 6 is 72.4 of 120.
5. Does not work: stability under new pass mixes. Seven engine stalls in two days (M4, M9, M11, M12, T4, T6, C6); the longest clean multi-user serve is about two minutes (C5, C7), the longest clean replay H1's 20 minutes at one user.
6. Open correctness: prompt 10's first token differs from production by 6 nats and is the port's numerics (note 40).
7. Heat sets the schedule: 13 guard stops, every 2-CPU replay reaches 75 C in 3-8 minutes; kills produced five of the eight m3 fault entries plus fault 6's reset; the real faults were two device OOMs (1, 7). A 24-hour soak cannot run on this heatsink.
8. Spend: $234 of $1,000 over 39 ticks in ~30 hours; heat and the 30-minute precheck window constrain, not budget.
9. The rotation and aggregate figures are stale: image :38 at 2 GiB gave 26.2 (H6, heat-stopped at 201 s); at 4 GiB (H7) it heat-stopped at 293 s, no figure.
10. Where this lands: 36-39 / 44-50 tok/s single prompt (off / on), 34-42 rotation, 75-95 aggregate, first token 5-22 s; the 60 / 120 / 5 s targets are out of reach.

## 2. Code and backend: five risks for a 24-hour multi-user soak

**R1. Rank-asymmetric tape decisions.** `core/runtime/ctx.cpp:1357` reads its own card's free VRAM for the floor, `:1385-1391` evicts its own LRU tape, `:1307` audits every 1,024 plays of its own counter, so after one refusal one rank issues a pass live while the peer plays it. Scenario: C6 (run 35): at the first multi-row history recording device 0's play waited for good while rank 1 finished the step. Fix: a one-byte vote at the step barrier (both go live if either would); audit on the step generation.

**R2. A collective that gives up corrupts silently.** `libxe/xe_allreduce.cpp:57` spins about two minutes, `:251-253` then skips the sum so `y` keeps stale bytes, `:273-277` still arrives so the sequence advances, the report (`:80-81`) is a device printf (unreadable, M10), and the host-readable counter at `:514` is read by nobody. Scenario: in hour 6 a recording under memory pressure delays rank 1 past the budget (H3 shows 10-second barrier waits already); rank 0's blocks time out, one step's activations are stale, nothing is logged. Fix, one build: memcpy `seq_ctr[3]` on both ranks every step and fail the step when it moves.

**R3. Pass-shape explosion against kTapes = 24.** The key (`ctx.cpp:961-970`) mixes n_tok, n_seq, n_tok_decode, n_draft_out, n_checkpoints and six pointers, so every admission, finish or drafting flip at 3-6 users is a new 1,036-segment recording (a cold pass at 4-9 tok/s, ~1,400 host launches) and an LRU eviction (`core/runtime/ctx.h:490`); the audit (`ctx.h:492`) re-issues a live pass every 1,024 plays; H3 recorded 52 shapes in 6 minutes with one user. Scenario: a 3-user soak records hundreds of shapes an hour, the steps go host-bound (C4: 35 CPU-seconds per 19 s), the guard kills at 75 C in the first hour. Fix: pre-record the shapes with an untimed warm rotation at 1-3 users; `RADIANCE_TAPE_MAX=64`; audit once in the soak.

**R4. VRAM the plan cannot see.** The room check (`core/device/l0.cpp:289`) and the floor (`ctx.cpp:1353`, 1 GiB) read the driver's free memory, but `zeCommandListCreate` (`l0.cpp:729`) allocates outside `g_used` (`:315`) and outside `--gpu-headroom-mib`, and a failed bind is not an error: the play "never completes". Scenario: fault 7 (C3, run 33): two minutes into 3 users rank 1's memcpy fails, rank 0 waits forever, the runner pauses. Fix: 4096 MiB headroom as the port's default; an OOM inside a recording drops the tape and issues that pass live (an append failure already throws at `:999`).

**R5. The speculation gate flips per step.** `core/sched/scheduler.cpp:2163-2171` recomputes `may_draft` from the running count every step, so at 2-3 users a finish or admission toggles drafting; each toggle is a new key (n_draft_out) and, for the no-draft history pass, a live issue (`ctx.cpp:953`). Fix: two consecutive steps over or under before flipping, never in an admission step, plus a dangling-wait check at `tape_end` (`l0.cpp:1350`) before that pass is recorded again.

## 3. Measurement and method

Sound: the op oracle on the card against libref at 2e-2 (E2-E4, O1-O4; the DP4A GEMV bit-exact); the tiny model's 256-token identity across native, graph and plain (E1); the two-measurement rule; the tape profiler (5f71cfa), whose device-time sum equals the step (P3: 27.8 ms of 27.8) and which corrected the serialized m2 profile. `--profile-ops` is unusable here (P1/P2 stall on profiling tags), so live passes and host ops are never profiled, only inferred from container CPU seconds.

Three gaps, cheapest closure each:

1. **The rotation median at the service configuration is unmeasured.** 37.5 is image :35 at 2 GiB (H5); :38 at 2 GiB gives 26.2 (H6); 4 GiB (H7) has no figure. Closure: an untimed warm rotation before the 5-minute window so the recordings fall outside it, `m2-heat.sh mtp4g` on :39.
2. **Concurrency is N copies of one 256-token prompt fired at once** (`m3/intel-host/m3-conc.sh:81-95`): identical lengths, prefix-cache hits, no first-token or tail latency. Closure: pick `bodies[k % 11]` from the rotation (one line) and report TTFT and p50/p90 token latency from the per-chunk timestamps already in the .sse files.
3. **Agreement is qualitative** (1 of 11 byte-identical, 10 "coherent", tool calls 3 of 3) and prompt 10 is open. Closure: the 11 requests through prod-api.sh with top_logprobs (0 of 20 used) against the card's logprobs: first-divergence token and top-4 gap per request. No build, one serve.

## 4. Runner and supervision

`run.sh`: a 10-minute timer behind a gate; a precheck older than 30 minutes ends the run (`UNATTENDED.md:192`), which cut about half the 39 runs after one to three card steps. The fault judge (`run.sh:249-513`) keys incidents by dump id and PCI address, judges recovery at 10 minutes and, since note 35, ignores lines within 20 s of a heat stop (`:402-416`). It paused three times: runs 22 and 28 on kill artifacts and run 33 on fault 7, whose vm-worker:-12 signature matched a fix-claimed configuration line (`STATE.md:289`); that pause cost 3.5 hours with the owner's watcher down.

The heat guard stops at 75 C held 2 s and kills with `podman stop -t 2` (`intel-host-engine-guard.sh:91`): 13 stops, five kernel-log artifacts, one GT reset whose stuck blitter queue hung the next load 7 minutes (M21). Owner notes: 43 headings in 72 hours, two numbered 41, several revising each other; the agent re-reads 857 lines every run. Permission friction: runs 36-37 lost a build and a pause to `bash m3/build-l0.sh` without `timeout`; `bash -n` and `grep /proc/meminfo` were denied too; this session lost `git -C src log` and `./intel-host-check.sh pre`.

Change:

- Graceful stop: SIGTERM with 20 s grace before `podman stop`, and an engine handler that stops admitting and waits for `tape_idle`.
- fix-claimed lines only for commits; vm-worker:-12 has had three causes and should notify, never pause.
- The runner refreshes `intel-host-precheck.txt` on its own timer; a permission smoke test at run start, logged, never a pause.
- One current-rules sheet replacing the note chain for CPUs, heat, pauses and images.

## 5. Platform limits on intel-host

| limit | binds today | measured cost | what a change buys |
|---|---|---|---|
| failing CPU heatsink | yes, hardest | 13 stops; replays die at 3-8 min, oracles at 99 s (O5: 84 C); engines capped at 2 CPUs | the Nov 1 heatsink: a soak, 4-8 CPU loads, no kill artifacts |
| PCIe Gen4 x8, 14.4 GB/s measured (`round2/STATE.md:105`) | not for decode | 8.3 misses a step x ~1.2 MiB = ~0.7 ms of 27 ms on the rotation, 0.2 ms on one prompt | x16: ~0.4 ms a step; prefill staging gains proportionally |
| DRAM-less NVMe (Micron 2450) holding the 47.7 GiB n-gram table | yes, for first token | up to ~2 ms a step (unmeasured as an A/B); ~12 of 53 s and ~34 of 94 s first token | RAM or a drive with DRAM; the second drive (Crucial P310) is also DRAM-less, so `disk=PATH` buys nothing |
| RAM 121 GiB | yes | one 51.2 GB pinned allocation fails (M23); prefill transients of 27-53 GB (H3/H4) leave 12 GiB to the guard floor | 256 GiB: the table in RAM (chunked pinning) plus room for the transients |
| no FP8 matrix on Xe2 | yes, structurally | the hc rows on E4M3 are 7.8 of 27.8 ms (28 %); 4-column forms cost 2x | only other silicon (section 7) |
| 2 x 32 GB VRAM for 122 GB of weights | yes | 87.5 % resident at 512 MiB headroom, ~82 % at 4 GiB; readmits 95 % on varied prompts | nothing in the B70 line; the mover policy is the lever |

## 6. Plan

The remaining order (A kernels, B prefill, C mover, D service and soak) is wrong in two places.

Reorder: the stall work first (note 42 items 1-3), because every later measurement and the soak depend on it and C6 is unexplained; then C (one flag-guarded run; it may return the 2.5 tok/s the rotation lacks); then B's cheap half (the n-gram table) and one XMX prefill run; then the two remaining A kernels; then D.

Cut: the persistent-kernel exchange (ar_kernel is 2.0 ms of the step, the gain under 1.5 ms); further GDN work (M20); MTP-under-concurrency work beyond the gate.

Add: a 1-hour soak at 3 users before any 24-hour one; the 24-hour soak only after the heatsink; prompt 10's per-layer divergence (one run) before any promotion.

Realistic end state on this hardware:

| measure | now | realistic | reasoning |
|---|---:|---|---|
| single prompt, MTP off | 35.7 | 36-39 | device-bound 27.8 ms; the largest kernels are 4-5 ms each and the trunk GEMV at 437 GB/s |
| single prompt, MTP on | 42.0 | 44-50 | 53.6 ms at 2.25 tokens a step; gemv<4> on XMX -3 to -4 ms, hc_read_a -2 to -3; acceptance varies 39-49 % |
| rotation median (on, gate, 4 GiB) | 37.5 stale | 34-42 | the 4 GiB headroom costs ~5 % residency; the mover change may return 1-3 |
| aggregate at 6 users | 72.4 | 75-95 | the 6-row step is ~82 ms; multi-column XMX GEMVs could take 15-20 ms off; 120 needs a 50 ms step |
| first token at 7K | 53 s | 16-22 s on disk, 5-10 s with the table in RAM | XMX GEMMs at 700-1,500 tok/s give 4-8 s; the disk table adds ~12 s; 5 s needs both and a path near the silicon's 2,500 tok/s |

## 7. Strategy for the owner

**Option 1, finish the port as planned.** 10-20 more runs, roughly $150-300 of the remaining $766, 4-8 runner days plus the heatsink date for the soak. Buys a Flash-Next endpoint at the ranges above, with an open numerics item and a backend that has never served several users for an hour. Costs the owner's attention; it stays about 5x slower than production's pair in decode and 10-40x in first token.

**Option 2, stop at a documented port and serve a B70-shaped model.** The 27B dense at 58-60 tok/s single user, 190 aggregate at 4, first token 0.3-7.5 s on one B70 (note 33): days of work on a supported upstream stack, one card. Costs: not Flash-Next (no tool-call parity). The port's 4,211 lines, 17 tests and 27 targets become a documented artifact and a credible upstream contribution.

**Option 3, buy AMD.** Two R9700 32 GB (gfx1201, production's platform; upstream's one-card Flash-Next config, `audit.md:310`). Buys the production software path, 218 tok/s on the pair, 0.5 s first token, no port maintenance. Costs roughly $1,300 a card at list (unverified), intel-host's power and case heat, and surplus B70s.

**Recommendation:** option 2 now on card1, with a bounded close-out of the port on card0 (the stall items, one rotation at the service config, one XMX prefill run), then stop and document. Option 3 only if a Flash-Next replica on intel-host is required. The measurements that decide:

1. A 1-hour 3-user soak at the service config after the stall fixes: a stall or heat stop means the port is not a service.
2. First token at 7K after one XMX run: above 15 s, users will not accept it.
3. The rotation median at that config against 40, with the calibration in hand: 59 tok/s for a 14 GB model on one card against 42 for this one on two.

## 8. Top ten recommendations

1. Agent: read the all-reduce stall counter on both ranks every step and fail the step on a change (R2; one build).
2. Agent: make record, skip and drop rank-symmetric through the step barrier (R1); name the tape and event in the 10-second play warning (`l0.cpp:865`).
3. Supervisor: graceful engine stop in the guard (SIGTERM, 20 s) before `podman stop`.
4. Agent: hysteresis on `RADIANCE_SPEC_MAX_SEQS` (two steps, never at admission), then commit the gate.
5. Owner: choose between options 1 and 2 on section 7's measurements within two runs.
6. Agent: the rotation at the service config with the warm rotation outside the window (gap 1); rotation-mixed concurrency with TTFT and p90 (gap 2).
7. Supervisor: fix-claimed lines only for commits; the OOM signature notifies, never pauses; the runner refreshes the precheck.
8. Agent: the logprob agreement table on the 11 requests (gap 3) and prompt 10's per-layer first divergence, one run, before any promotion.
9. Owner: the heatsink before any soak; then 256 GiB of RAM or a DRAM NVMe for the n-gram table, not the P310.
10. Agent: cut the persistent-kernel exchange and further GDN work; the next kernel run goes to the XMX prefill GEMM, the user-facing gap.
