# The GEN0 stall (owner note 236): reproducer, diagnosis, design

Run 294, 2026-11-11 (radiance-intel-2e). For the coordinator's review before anything is built (note 236 step 3).

## 1. The reproducer (step 1) -- it stalls in 1 to 2 tries

`logs/stallstep.sh STEP IMAGE [TRIES] [FORM] ...` (host side, genstep.sh's step rules) + `m3/intel-host/stall-serve.sh`
(inside the container, image ri-m3gen:2 = today's runtime 313 + python3): each try starts a fresh engine with the
service's card args (M285's: tapes on, headroom 2048), waits for /health, sends GEN0's three documents at once (dolly 1,
glaive 721 = the tool-call conversation, gsm8k 1301; concurrency 3) through the generation client, and watches
serve.log; the rank barrier's 10 s warning = STALL: the lines around the first failed append on each device, the barrier
and the stalled threads' stacks are printed, serve.log is kept as serve-stall-N.log, the engine gets SIGTERM then SIGKILL
(it dies on SIGKILL in 2 s -- GEN0's engine only ignored SIGTERM), and the host step kills the container if anything is
left. A finished client = PASS.

| step | form | result |
|---|---|---|
| STALL1 (14:27) | svc (headroom 2048) | try 1 PASS, **try 2 STALL** 20 s after the requests started |
| STALL2 (14:35) | fake4 (the test hook RADIANCE_L0_FAKE_APPEND_FAIL=4) | the first append and its 3 retries all failed -> "after 3 retries -- this rank's step fails": the step loop stopped the engine with a device error at warm-up. No hang: that path ends. |
| STALL3 (14:47) | fake (one fake failure) | the fake at startup drained and retried FINE (the engine came up); then **the real STALL at try 1**, 22 s in, at the same point |
| STALL4 (15:0x) | cmdbuf256 (OverrideCmdListCmdBufferSizeInKb=256) | the prevention probe, running (10 tries) |

Both real stalls happen at the same moment: the tool-call request's FIRST turn finishes (finish=tool_calls, 68 tokens),
the client sends the conversation's second turn (msgs=4, tools=1, prompt 552 tokens), the other two documents are done
-- a new batch shape (one sequence, a 552-token prefill with the grammar) at the VRAM ceiling.

## 2. The diagnosis (step 2)

**Why the ranks leave step.** Evidence from the stalled runs' serve logs (results-m3/card01/STALL1/serve-stall-2.log,
STALL3's, GEN0's full log on intel-host):

1. The new shape is not recorded: "rank 0: under the recordings' floor every one of the 3 tapes was played in the last
   2048 passes, so none is dropped: the new shape runs live (RADIANCE_TAPE_KEEP_PASSES)" -- free VRAM is already under
   the recordings' floor (1024 MiB by default; the service's run.sh sets 2 GiB) when the second turn arrives.
2. The live pass's launches are glued onto the device's immediate list (l0.cpp l.1727 `fn(Z.imm, r.ev, ...)`); at the
   ceiling that append fails: "D dispatch.cpp:41 device: a library launch: the glued append failed (0x70000003)" -- a
   RAD_E_DEVICE for that op. In STALL3 the op was `moe_gemm_q`'s "operand 2: the weight table could not be patched" ->
   "E rank 0 step 72" -> rank 0's step FAILED.
3. On the other device a recorded segment's append fails the same way: "l0 tape 55/56: appending 1 list(s) failed out
   of device memory (0x70000003); draining device N's immediate list and retrying (attempt 1 of 3; owner note 58)" ->
   l0.cpp l.1542-1553: `zeCommandListHostSynchronize(Z.imm, UINT64_MAX)` under Z.m -- an UNBOUNDED host wait for every
   command in flight on that device.
4. The in-flight work on that device includes the step's collectives, which wait on the peer (xe_allreduce.cpp: the
   peer's flag; bounded at kSpin = 400M polls, minutes), and device-side event waits on the peer's plays; the peer has
   just failed its step (2) and issues nothing more. So the drain never returns: the barrier shows "2 of 3 arrived"
   (the scheduler and the failed rank) and one rank thread "state R, running" inside libze_intel_gpu
   (zeCommandListHostSynchronize spinning). In GEN0 both devices failed their appends ("1 of 3 arrived").
5. The failed step is fatal by design (engine.cpp l.717-721: "a device fault is not recoverable in-process (spec 17)"
   -> fail_loop -> stop): the engine logs "stopping: draining in-flight requests" and then waits for the rank thread that
   is stuck in (3)/(4) -- it neither serves nor exits (GEN0's "did not exit on the kill"; SIGKILL works).

So the ranks do not "leave step" through the retry logic itself (STALL3's startup fake drained and retried cleanly):
one rank's step dies on the OOM wave while the other rank's retry waits forever for device work that needed the dead
rank. The unbounded drain is the hang; the OOM is its trigger.

**Why the tool-call turn needs the VRAM -- what fills the card.** The engine's own log: "l0 tape: the driver takes
1.13 MiB a command list on device N (NEO OverrideCmdListCmdBufferSizeInKb not set)" and "lists created 31427, destroyed
28938 (alive 2489)": a decode tape is 762 segments = 762 Level Zero command lists, three tapes are kept (2048 passes
each, note 58's rule), so ~2.5K live lists x 1.13 MiB = **~2.8 GiB of VRAM in command buffers**, none of it in the
headroom's accounting (the headroom measures "already held" at start-up: 1.90 GiB). The second turn then needs, at
once, the live pass's glued lists and the prefill's expert staging ("a prefill step streams every routed layer's pooled
experts onto the card") -- at headroom 2048 that is the ceiling; at 4096 (GEN1, the hr4g form) it is not. Nothing is
specific to the grammar: its masks are small; the trigger is "a new shape after the recordings filled the card".

## 3. The design (step 3) -- for the coordinator

Three tiers; I propose building 0 and 1 together (small, exact), and ask about 2.

**Tier 0 -- prevent the OOM (the owner's preference).**
- (a) The command-list memory: NEO's command buffer per list is 1.13 MiB where the tapes' lists hold a few commands
  each (1215 commands over 762 lists). NEO's debug key `OverrideCmdListCmdBufferSizeInKb` (read only with
  `NEOReadDebugKeys=1`; l0.cpp l.1131-1154 measures the effect at start-up and says "in effect" under 0.6 MiB a list;
  run 80's X9 measured 0.25 MiB a list with 64 against 1.19 by default) cuts the lists' VRAM ~4.5x (~2.8 -> ~0.6 GiB).
  STALL5 (`cmdbuf64`) measures it at headroom 2048: the per-list MiB in the log, then 10 tries. (STALL4 set the key
  without NEOReadDebugKeys: no effect -- a control, in effect.) If it holds, the engine sets both variables in its own
  environment at start-up before the first zeInit (l0.cpp), unless the user set them, with the measured line as the
  check; if not, (b)/(c).
- (b) Count the lists in the budget: the l0 layer knows its live list count and the measured bytes a list; the
  recordings' floor check (ctx.cpp l.1388-1400, "runs live" at l.~1515) compares free VRAM with the floor already --
  add the lists' bytes to the headroom reservation when a tape is kept, so a new shape under the floor DROPS the least
  recently played tape's lists (tape_drop) before it runs live, instead of running live on a full card. Run 73 found a
  dropped tape "returns no VRAM" (M38) -- that was the stretch cache keeping its executables; the drain path shows the
  lists do return when destroyed ("its completed work returns the driver's command buffers and the graveyard's tapes can
  go"), so a drop must destroy the lists, not only unlink the tape.
- (c) The blunt one: --gpu-headroom-mib 4096 as the service default (GEN1 ran clean on it); costs 2 GiB of pool
  (fewer cached tokens). Keep as the fallback if (a) fails and (b) is not wanted now.

**Tier 1 -- never hang, never wait indefinitely (lock-step on failure).**
- The drain bounded: `zeCommandListHostSynchronize(Z.imm, 1 s)` in a loop up to RADIANCE_L0_DRAIN_S (default 20 s),
  logging the slot waited for every 5 s (the n_slot_wait form); on expiry the append is not retried: RAD_E_DEVICE for
  this rank's step (the "after 3 retries" path, which STALL2 showed ends cleanly) with a clear line ("the drain found
  the peer gone").
- A shared step-poison flag: the first rank whose step fails (any RAD_E_DEVICE, incl. the glued append and the table
  patch) sets it and aborts the barrier (engine.cpp l.430's `barrier_->abort()` -- today only on the stop path); a rank
  entering the OOM retry checks it first and fails its step at once instead of draining; the ranks' threads then both
  return, fail_loop runs, the engine stops within the drain bound, SIGTERM is honoured, systemd restarts the service and
  the in-flight requests end with an error (the owner's "the request fails cleanly").
- Gates (note 236 item 4): the reproducer at headroom 2048 >= 10 tries with no hang (with Tier 0 it should not even
  fail; with Tier 0 off -- `svc` on the fixed build -- it must END within the bound, not hang); texts = M285 on the
  M-pair; one rotation with a tool mix; klog clean.

**Tier 2 -- degrade instead of stop (a bigger change; the owner's call).** Treat the OOM-class step failure (not a GPU
fault) as recoverable in-process: both ranks abandon the step at a resync barrier, the scheduler fails the step's
requests with 503 and keeps the others, the devices are waited idle (bounded), the failed shape's lists are destroyed,
serving continues without a restart. Needs spec 17's rule relaxed for this error class and the ranks' resync path; I
would not start it without a note.

**What I would build now:** Tier 0 (a) if STALL4 says the knob holds (one line + a log line), else (b); plus Tier 1
(bounded drain + the poison flag + barrier abort on a failed step). Both exact (no numerics). One build, one image, the
gates above.
