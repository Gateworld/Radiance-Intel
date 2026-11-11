# OWNER-NOTES.md — the owner's standing notes for milestone 1

Read this file **first, at the start of every run**, before UNATTENDED.md and PROMPT.md. It overrides
both where they differ. The owner adds notes here between runs; check for new ones every time.
Record in STATE.md, under "Owner notes", which notes you have acted on and how.

## 1. Decode replays a recorded graph (2026-10-24)

Decode must **replay a recorded SYCL graph** (`sycl_ext_oneapi_graph`) **or a regular Level Zero command
list**, not launch kernel by kernel on the in-order queue. Round 2's M3 measured 0.59 µs a launch for
replay against 2.47 µs on the in-order queue: at ~1,600 launches a token that is about 3 ms a token.

- The in-order queue is fine as the stream (`RadStream`). Recording, replay and re-recording when the
  step's shape changes belong to the backend's recorded-pass support (`DeviceBackend` tape hooks:
  `tape_begin`, `tape_end`, `tape_play`, …; see `core/device/device.h` and how `Ctx::run_step` uses them),
  so the engine's existing logic decides when to record and when to replay.
- **Step 5 must deliver the graph path and measure it.** Per launch and per decode step, with replay and
  with plain queue launches. Use the same kernels, show the outputs are identical, and record both in
  STATE.md.
- Steps 6 and 7 run decode with replay on. Step 7 reports tok/s with replay (MTP on and off), and once
  with plain queue launches for comparison.
- If a graph cannot be used for a step (an unsupported pattern, a host callback, dynamic shapes), say
  which op or pattern, fall back for that step only, and count how often it happens.

## 2. The clang grammar-test failure must be understood before step 7 (2026-10-24)

`gbnf_test_shard3` (`json_schema_walks_write_only_valid_instances`, seed 69, writes the invalid UTF-8
`"bion\xf0\x85\xaa\x8d"`) passes with g++ and fails with the Intel clang++ that compiles the L0 build's
host code. The grammar code constrains **tool-call output**, so this is not only a question for the
author.

- **Before step 7, find the cause.** Rebuild the grammar code and its test with clang under UBSan (and
  ASan if useful) in the build container, with `-fno-strict-aliasing`, and with `-funsigned-char` /
  `-fsigned-char`, to separate undefined behaviour, char signedness and optimiser differences. Compare
  the failing walk with g++'s step by step.
- Then either **fix it** or **show that it is confined to the test** (for example the test's own
  generator, not `core/sample/gbnf*.cpp`). A fix:
  - is the smallest change that makes the code correct under both compilers;
  - keeps g++ results unchanged (the CPU and HIP builds' grammar tests still pass);
  - goes in its own commit, with a message saying what was undefined or compiler-dependent.
  Record the evidence in STATE.md and the report either way.
- **Step 7's agreement check includes tool-call prompts:**
  - at least 3 of the 10 prompts are tool calls;
  - they use the `tools` field through `./prod-api.sh` (allowed) and the same tool schema on your engine;
  - with the grammar constraint on in both;
  - compare the tool-call JSON as well as the tokens.
  If the cause is not understood by step 7, step 7 still runs, but the report must say the tool-call
  path on the Intel build is not cleared.

## 3. Heat on intel-host: no host reference work there, and a hard stop at 80 °C (2026-10-24)

Run 8's card1 step `b9` (rad-kbench + tests, 21:56–22:01 UTC) computed libref's host references on
intel-host's CPU: about one core at 100% for 4.5 min. With the failing heatsink Tctl went 41 → 81 °C in a
minute and peaked at 88.8 °C; intel-host's own thermal guard (stop at ≥ 88 °C, or ≥ 85 °C for 10 s) then
stopped chatterbox-vllm at 22:01:02. Your sampler logged 88.8 °C, yet b10 and b11 ran after it: the
85 °C rule was not followed. From now on:

- **No host-side reference work on intel-host.** Compute libref references, rad-kbench's host side and the
  engine's op oracle (`RADIANCE_OP_ORACLE`) on **amd-host** (CPU device or host build), save the expected
  outputs and inputs as files, copy them to the intel-host folder, and on intel-host run only the device side and
  compare to those files. A card1 step does device work; its host CPU use must be small (setup,
  copies, comparison).
- **Every card1 container** is named `ri-m1-<step>` (not `ri-<step>`), runs with **`--cpus 1`**, and runs
  under the guard, started before the container:

      timeout 3h ./intel-host-engine-guard.sh ri-m1-<step> 600 > logs/guard-<step>.txt 2>&1 &

  The guard now also **stops the container at Tctl ≥ 80 °C for 2 s** (and below 8 GiB MemAvailable, as
  before). A heat stop (exit 11) counts like a failed correctness check (twice → `PAUSED`); find what
  used the CPU and move it to amd-host before retrying.
- **Before every card1 step:** `./intel-host-engine-guard.sh --cool 1800` must return 0 (Tctl < 50 °C). If it
  does not, end the run; the timer retries.
- **Read your sampler during a step**, not only after. If Tctl reaches 80 °C and the guard has not
  stopped the container yet, stop it yourself (`./intel-host-engine-guard.sh --stop ri-m1-<step>`).
  Never start another card1 step in a run that saw ≥ 85 °C; record it and end the run.
- Step 7's `.rad` copy and model load are the exception that cannot be moved: run them with the guard
  and the 80 °C stop too, with `--cpus 1` for the copy's receiving side where you control it, and rate-
  limit the copy further (`--bwlimit=30000`) if Tctl climbs past 70 °C.
- Record every step's peak Tctl in STATE.md. The report gets a table: step, CPU seconds, peak Tctl.

## 4. Weight relayout at load, and intel-host's heat (2026-10-25)

One core at 100% takes intel-host past the 80 °C guard within a minute (b8, b9), so the real-model load's
relayout (each kernel's `relayout` hook rearranging the container's canonical planes into its own
order, done by the loader's relayout workers on the host) will probably be stopped by the guard.
`spec.md` §4.2 says the `.rad` is "canonical, not kernel-shaped" by design, so check carefully.

- **Before copying the `.rad`**, find out how Radiance lays out weights at load: read the loader
  (`core/engine_bringup.cpp`, `core/place/`, `core/format/`, the `relayout` hooks of the rows the
  production model resolves to in libxe) and the conversion and export tools (`core/build/`, `tools/`,
  `rad-convert`'s options such as `--reuse`). Measure on amd-host how much host CPU the relayout of the
  production model's weights costs (time and bytes, per weight kind), e.g. with the loader on the CPU
  device or a small harness that calls the same hooks.
- **Can the relayout be done once, ahead of time, and saved so intel-host only reads?** For example a
  pre-laid-out container, a cache file the loader can map, or a layout chosen at rad-convert time.
  - **If yes:** do that relayout on amd-host's CPU (`nice -n 19`, ≤ 8 threads, `MemAvailable` > 40 GiB),
    and copy the result to intel-host instead of, or alongside, the original `.rad`. Record the tool, the
    flags, the output's size and sha256 in STATE.md.
  - **If no:** do everything in step 7 that does not load the model (the copy, the hash, the config, the
    memlock check), then create `PAUSED` with the first line **"load waits for intel-host's heatsink
    (Nov 1)"**, and the exact command to resume the load.
- Ways that are allowed if they keep intel-host under the guard, before you pause: relayout on **card1**
  (upload the canonical planes and rearrange them with a kernel), or a load that is slower but cooler
  (**fewer** CPUs than 1, e.g. `--cpus 0.3`, one relayout worker, pauses between weights), measured with
  the guard running. Record what you tried and its peak Tctl.
- **Never loosen the 80 °C guard or the `--cpus 1` cap on intel-host** to force a load through. Tighter is
  fine; looser is not.
- **Memlock is fixed** (owner, 2026-10-25 01:04 UTC): `ssh intel-host 'ulimit -Hl; ulimit -Sl'` prints
  `unlimited` twice, and a rootless container with `--ulimit memlock=-1:-1` gets `unlimited`. Step 7 no
  longer pauses for it; run engine containers with `--ulimit memlock=-1:-1`.
- **The milestone budget is now $400** (budget.txt; the daily cap is still $60 per ET day).

## 5. Step 6 accepted; one decisive-greedy check before step 7's copy (2026-10-25)

The owner accepts step 6's verdict on the teacher-forced criterion (all ops within tolerance under the
op oracle, replay == plain token for token, top-1 95.6% vs the host reference, misses at near-ties).
**Before step 7's `.rad` copy**, add one check:

- **Rebuild the tiny qwen4exp model with its lm_head scaled ×10** so greedy decoding is decisive (large
  top-1 margins instead of near-ties). Scale the lm_head weight itself before conversion (or, equivalently,
  its stored per-group scales in the converted container), nothing else; keep the same seed and the
  production recipe. Record the change and the new container's sha256.
- **Rerun the free-running greedy comparison:** 256 tokens, the same prompt, `ignore_eos`, MTP off,
  graph replay on, on card1 against the host reference (`--kernels libref`, host backend). The host
  reference runs **on amd-host** (owner note 3); only the card1 run goes to intel-host, under the guard, at
  `--cpus 1`, after `--cool`.
- **Expect a one-for-one match over all 256 tokens.** Record in STATE.md, and in the report's step-6
  section: tokens matched, the first divergence if any, and the reference's top-1 margin there.
- **If they don't match one for one, investigate before moving on:** the step and position where they
  part, the reference's margin there, and the op oracle (`RADIANCE_OP_ORACLE`) on that step to find the
  first op whose output differs beyond tolerance. Fix it, or show it's a genuine tie, which a ×10 head
  should make very unlikely. A mismatch you can't explain counts like a failed correctness check
  (twice → `PAUSED`).
- **Then proceed with step 7, starting with owner note 4's relayout question** (before any copy).

## 6. Run 11's copy stays; the order for run 12 (2026-10-25)

Notes 4 and 5 were written at 01:10 UTC, after run 11 had already started the `.rad` copy to intel-host
(01:07:54 UTC). The owner keeps that copy: **don't delete it, redo it or copy it again.** "Before the
copy" in notes 4 and 5 now means "before the rest of step 7". Run 12 goes in this order:

1. **Note 5's check first:** the tiny model with its lm_head ×10, 256 free-running greedy tokens on
   card1 against the host reference on amd-host.
2. **Then note 4's relayout question:** read the loader, measure the relayout on amd-host, and follow
   the yes or no path. If the answer is yes, copy the pre-laid-out output **alongside** the
   `.rad` that is already on intel-host.
3. **Then the rest of step 7.** The copy may have moved out of your runs: if the user unit
   `ri-copy-resume` exists (`systemctl --user status ri-copy-resume`), the owner is finishing the copy
   there (`dryrun/m1/copy-resume.sh`, same guard and `ri-m1-copy` container). **Don't touch that unit,
   its container or the file while it runs.** Start the sha256 check only after `logs/run.log` has a
   `copy-resume: DONE` line. If it logs `copy-resume: FAILED`, or the unit doesn't exist and the copy
   isn't complete, finish it yourself with `m1/copy_rad.py` (it resumes with `--append`) under the
   same guard and limits as before.
   - What happened at 01:46 UTC: the owner stopped run 11's rsync (rc 20 at 01:46:00) and started
     `ri-copy-resume` at 01:46:09. Run 11's own resume at 01:46:24 failed at once (the `ri-m1-copy` name
     was taken; exit 12), but both wrote `logs/copy-rad-2.txt` and `logs/guard-copy2.txt`. So STATE.md's
     line "Resumed 01:46:24 (copy_rad.py with --append ...)" is wrong: the copy that is running belongs to
     `ri-copy-resume`. Correct that line. The owner then stopped run 11 (SIGTERM) so run 12 could start;
     that counts as one failed run in `.fails`, not your error.

STATE.md "Next (run 12)" 0 ends with "PAUSED for memlock". That is out of date: memlock is fixed (note
4), so don't pause for it.

## 7. After the second fault: fix the event leak first, then S1 with per-token measurements (2026-10-25)

The owner resumed the milestone after reading PAUSED. Card1 checked at 12:38 UTC: no kernel log lines
after 09:28:17, and the card still enumerates (renderD129). A third card1 fault pauses again at once,
as run.sh already does.

**Before the S1 rerun, in this order:**
1. **Fix the event leak in `src/core/device/l0.cpp`** with a pooled, recycled event set: a fixed pool of
   L0 events that `rad_event_record` reuses once an event has completed, instead of a new runtime
   event or profiling tag for every record. Bound the pool. When it runs out, wait on the oldest
   event, or fail with a clear error. Never grow without limit.
2. **Verify it on amd-host's CPU device:** a stress test of `rad_event_record` / `rad_event_wait` with
   no stream syncs, far beyond 74,000 records (say 1,000,000). Show that the number of live events
   stays flat and that every event still orders work correctly. Commit with the test.
3. **Then the small mover stress on card1** (canaries, klog, under the guard), and only after that S1.

**In S1's tok/s runs (MTP on and off), record per token:**
- host CPU use inside the container, against its `--cpus` cap (from the container's cgroup `cpu.stat`)
- expert residency, misses, mover promotions and PCIe MB (from the engine's `/stats`)
- that graph replay was used, confirmed for each step
- the time split between the engine's device step and host work

Save the per-token rows as a file in `results-m1/card1/` and summarise them in STATE.md.

**If host CPU sits at the cap** (say ≥ 90% of `--cpus` through most of a run), **repeat that run once
at `--cpus 2`.** This is the only exception to the `--cpus 1` cap in notes 3 and 4. For that run, start
the guard as `./intel-host-engine-guard.sh --hot 75 ri-m1-<step> 600` so it stops at 75 °C, not 80, and read
Tctl during the run. If the guard stops it, don't retry; record the peak and the token reached.

**In the report, explain the gap between S1's tok/s (7.6 with MTP on) and round 2's one-card estimate
of 40–51 tok/s,** using these measurements: how much is host CPU, expert misses or PCIe, launch or
replay overhead, and device kernel time. Say which assumptions in round 2's estimate were wrong.

The heat report in `results2/heat/` is being corrected by the owner (the central skew now comes from
the point estimate, not the fit). Use the corrected `report.md` when it has a line saying so.

## 8. Aim: finish step 7 today (2026-10-25)

The owner raised the daily cap for 2026-10-25 (ET) to $160 (`daily-cap-2026-10-25.txt`) so the
milestone can **finish step 7 today, Eastern time.** The report (step 8) is not part of today's aim:
start it only once step 7 is done. The total budget is still $400. **No safety rule changes:** notes
3, 4 and 7 (heat, `--cpus`, the guard, the fault pause) apply exactly as written.

Order: the card1 mover stress, then S1 with note 7's per-token measurements (MTP on and off, the
`--cpus 2` repeat only if CPU sits at the cap), then the production comparison, then the 20-minute
heat replay. Keep runs moving. When a precheck expires mid-run, end the run promptly so the next one
brings a fresh precheck, instead of doing other work while card1 waits. If something blocks a step,
record it and go on with the steps that don't depend on it.

## 9. After the third fault: finish step 7 with static placement, then the report, then the mover (2026-10-25)

The owner read PAUSED (run 27) and chose the agent's option 2 now and option 1 after the report.
Card1 checked at 16:37 UTC: one kernel log line after 15:10:30 (the coredump's deletion at 16:13; the
faulting address is lost), the card still enumerates (renderD129), no ri- containers left.

**1. Finish step 7 now, with the heat engine's moves off (static placement):**
- Rerun H, the 20-minute heat replay, with the same traffic, image and limits (`--cpus 0.3`, guard,
  cool check), but with the mover's promotions and demotions off: experts placed once at load and
  never moved. Record the exact flag or config and show in `/stats` that promotions and demotions stay
  at 0. Record residency, misses and link MB per step as before. Label the result "static placement".
- Run the staged prompt-10 check (the card's exact rendered prompt sent raw to production as well), to
  tell whether the split comes from the engine version (1.9.0 vs 1.11.0) or from the card's numerics.
  Stay within the 20-request limit (14 used so far).
- **Any card1 fault in these steps pauses at once. Don't retry.** Static placement still uses the copy
  engine for prefill staging, so a fault there is possible.
- Then mark step 7 done in STATE.md.

**2. Then the report (step 8).** Report the mover fault (faults 2 and 3) as an open problem: what ran,
the counters (65,005 promotions, 78% readmits, 128,309 quarantines, ~1 TB in ~10 min), the kernel
lines, and the suspected cause, marked as unverified. Use both H results: the 10 minutes of real
heat-engine counters before the fault, and the static-placement replay.

**3. After the report, the first job is option 1: the mover's copy path.** On amd-host, read
`core/place/mover.cpp` (slot reuse, quarantine) and the copy functions in `l0.cpp`. Reproduce the
promotion and demotion churn on the CPU device with a small model and many moves, then fix it and
commit it with a test. Then a mover-only stress run on card1 (canaries, klog, guard) before any replay
with moves on. Note the thrash itself (78% readmits on one card) as a heat-engine policy finding, apart
from the fault.

## 10. Write the report now; then milestone 1b in the same budget (2026-10-25)

The owner read PAUSED (run 28). **Fault 4's devcoredump is saved:**
`intel-host:/mnt/fast-storage/models/test/radiance-intel/devcoredump-fault4.txt` (495,024 B, taken as
root at 17:08 UTC). Read it there or copy it to amd-host. From now on a udev rule on intel-host saves every
card1 devcoredump to `/mnt/fast-storage/models/test/radiance-intel/coredumps/` as it appears.

**1. The report (step 8), now.** Copy its numbers from the result files and STATE.md.
- Step 7's 20-minute heat replay goes in as **not done**: both attempts (H with the heat engine on,
  H2 with static placement) ended in the copy-engine fault. Report the counters from before each fault.
- **Open problem: the copy-engine fault** (faults 2, 3 and 4, with fault 4's coredump). Give the
  hypothesis as unverified: stale buffers in 2D copies recorded inside tapes since 5b909d9 (stored as
  per-row plain copies, replayed against a per-request buffer that has moved). Include what H2 ruled
  out (the mover's moves, long prompts).
- **Open problem: prompt 10.** Production gives "The" at −0.05 on the raw prompt, the card gives "We"
  at −0.32, raw and chat. That is a forward-pass difference, either the card's numerics or a change
  between 1.9.0 and 1.11.0.
- Mark everything else that wasn't done as undone, and say why.

**2. After the report: milestone 1b, inside the same $400 budget.** Record each part's result in
STATE.md under "Milestone 1b".

(1) **The copy fault.**
- Review 5b909d9 on amd-host. Fix how recorded copies reference per-request buffers: either the stored
  copy resolves the buffer at replay time, or the tape is re-recorded per request when a buffer it
  copies from or to changes. Use fault 4's coredump (the faulting address) to check the diagnosis.
- Verify the fix on the CPU device with a test that changes the per-request buffers between replays
  and checks every copied byte. Commit the fix with the test.
- Then on card1, **two replays, in this order:** the same replay as H2 with `RADIANCE_L0_TAPE=0`
  (plain launches, old image), then the same replay with the fix (tapes on).
- **Any card1 fault pauses at once. Don't retry.** Keep notes 3, 4 and 7's limits (`--cpus 0.3`,
  guard, cool check).

(2) **Prompt 10 on the CPU backend.**
- Run prompt 10's exact rendered prompt (462 tokens) through the CPU backend at the commit the card's
  image was built from, on amd-host. Compare its first token and log-probabilities with production's
  "The" (−0.05) and the card's "We" (−0.32).
- **This needs the production weights on amd-host again.** That is an owner exception to PROMPT.md's
  "the `.rad` is read once": a second read, for this one comparison, on these terms:
  - read-only and mmap'd, never copied
  - idle I/O class, `nice -n 19`, at most 8 threads
  - amd-host's `MemAvailable` stays above 40 GiB the whole time
  - production is never touched
- Check first, from the loader, whether the CPU backend can run the model that way: lazily mapped
  weights, with no full load into RAM. If it can't, or if `MemAvailable` falls toward 40 GiB, stop,
  record why, and leave (2) undone.

## 11. New fault and pause rules, and decide instead of asking (2026-10-25)

The owner changed the pause rules; UNATTENDED.md ("GPU faults and canaries", "Decide instead of
asking") and run.sh now carry them. **They replace** "Any card1 fault pauses at once. Don't retry" in
notes 7, 9 and 10, PROMPT.md's GPU fault rule (pause on the second fault), and the runner's old count.

- **A card1 fault doesn't pause.** Save the dump reference, diagnose from the dump, fix and verify on
  amd-host's CPU device, then retest on card1 at a small size first. List each fault under "## Reviewed
  GPU faults" in STATE.md (`reviewed-fault:`) and each fix (`fix-claimed:`), in the formats
  UNATTENDED.md gives. Faults 1–4 are already listed there, and `12fc16f` is recorded as the fix for
  fault 2's signature (`vm-worker:-12`).
- **Pause only** when card0 is affected or chatterbox stops; card1 hasn't recovered 10 minutes after a
  fault; the same signature recurs after a claimed fix; or two faults happen in one run.
- **Decide instead of asking:** take the cheapest path that still produces a report or a verified fix,
  write `decision: … | alternative: … | why` in STATE.md, and continue. Pause to ask only for what
  can't be undone: production or card0, root, deleting data, spending past the budget.
- **When the owner removes `PAUSED`,** don't create it again for anything that happened before that.
- **Unchanged:** every heat rule (notes 3, 4 and 7: `--cpus`, the guard, `--cool`, 85 °C, and two heat
  stops → `PAUSED`), card1 only, and production untouched.

For milestone 1b this means: the copy-engine fault work (note 10, part 1) goes on through faults
without pausing, within those four conditions. One more card1 fault in a run where one already
happened still pauses.

## 12. Finish milestone 1 now (2026-10-25)

The owner lifted the plan's weekly limit and the daily cap (`daily-cap-*.txt` = $400 for 2026-10-25 to
2026-10-31 ET), so only the $400 milestone budget limits spending. Finish milestone 1 in one go:
1. Complete the report: the `.json` beside the `.md`, and check its numbers.
2. Milestone 1b (note 10): the copy-engine fault, then prompt 10 on the CPU backend. Use the fault rules
   in note 11: a fault doesn't pause unless one of their conditions applies.
3. Update the report with 1b's results (what was found, fixed and verified, and what stays open), then
   write `status: complete` in STATE.md.

If the budget won't stretch to all of 1b, finish what fits, record the rest as open in the report and
STATE.md, and complete the milestone without passing the budget.

## 13. Milestone 2 (2026-10-25): what carries over, and what is new

Notes 1–12 stand where they still apply (heat: 3, 4, 7; fault rules: 11). Milestone 1 is archived in
`m1/` (its STATE.md, PROMPT.md, results, runs, logs, report). This milestone's PROMPT.md is the owner's
brief, job C Part 2; the whole brief is `briefs/2026-10-25-prod-update-intel-host-retire-m2.md`.

- **Budget $500, daily cap $80, $20 a run.** Report `<date>-radiance-intel-m2.{md,json}`.
- **Containers are `ri-m2-<step>`.** The guard stops `ri-m1-*` and `ri-m2-*`.
- **Production will move to Radiance 1.11.0 during this milestone (job A, in a night window).**
  Production runs 1.9.0 until then; your build is 1.11.0. Until OWNER-NOTES says the move is done,
  mark any divergence from production as *possibly version-related*; after it, agreement checks
  compare like with like. Milestone 1's prompt 10 ("The" −0.05 on production vs "We" −0.32 on the
  card) will be re-asked of production 1.11.0 in that window; the answer goes here.
- **Chatterbox and Qwen-Image are being retired from intel-host (job B, owner present).** Until then the
  card0 service is `active` and card0 has clients: stage 2's two-card steps wait on the card0 gate
  (UNATTENDED.md). After B3, `intel-host-precheck.txt` shows the service `inactive`, `card0_clients: 0`
  and `dri_containers: none`: that is the gate opening, not a problem. intel-host's RAM is then all yours
  within the guard's 8 GiB floor and the "> 16 GiB free" rule for pinned memory.
- **Production API: 30 requests this milestone** through `./prod-api.sh` (count reset to 0).
- **Heat:** intel-host's heatsink is still the failing one. Every heat rule stays: `--cpus 0.3` for engine
  steps, the guard at 80 °C (`--hot 75` where a note allows `--cpus 1` or `2`), `--cool` before each
  step, no host reference work on intel-host. Stage 1's profiling and tok/s runs at 2 CPUs are the note-7
  kind of exception: one run each, guard at `--hot 75`, stop and record if it trips.
- **`m2b-run.sh`** (stage 2 step 1) is round 2's M2b script: `probe2/intel-host/m2b-run.sh` here, already
  staged on intel-host as `/mnt/fast-storage/models/test/radiance-intel/m2b-run.sh` (it uses image
  ri-tune:6 there; see `round2/STATE.md`). It needs card0, so it waits on the gate too.

## 14. When stage 1 is done, write one marker line (2026-10-25)

The owner retires chatterbox-vllm (the card0 service) only after stage 1 finishes, and the owner's own
watcher does the stopping, never you. So that it can tell: when stage 1 ends (the step ≤ 25 ms, or 10
kernels tuned, whichever comes first), put this line on its own in STATE.md, exactly in this form:

    stage1: done | step <N> ms | <K> kernels tuned

and send the stage-1-done notification as PROMPT.md says. After that, `intel-host-precheck.txt` will at
some point show the card0 service `inactive`, `card0_clients: 0` and `dri_containers: none`: the
card0 gate has opened (note 13).

## 15. Get it done sooner (2026-10-25, 21:25 UTC)

The owner lifted the daily cap for the rest of the milestone (`daily-cap-*.txt` = $500 through 2026-11-01
ET): only the $500 budget limits spending now. Keep runs moving: end a run promptly when card1 is waiting
on a precheck or a cool-down and nothing else is useful, so the next cycle brings a fresh precheck; don't
idle inside a run. Everything else stands: the heat rules, the card0 gate, the fault rules, the budget
judgement by $350 (UNATTENDED.md). If stage 1 reaches 10 kernels with stage 2 still ahead and the budget
looks short for the two-card serving, prefer finishing stage 2's exchange and two-card serving over more
stage-1 kernels, and say so in STATE.md.

## 16. Faster (2026-10-25, 21:35 UTC): card0 opens early, CPUs by heat, stage 1 capped at 6, $80 a run

Four owner decisions. They override PROMPT.md and notes 3, 4, 7, 13 and 14 where they conflict.

1. **card0 opens as soon as job B's archive is verified**, not after stage 1. The owner's watcher stops
   chatterbox-vllm then (tonight, most likely). From that moment `intel-host-precheck.txt` shows the card0 service
   `inactive`, `card0_clients: 0`, `dri_containers: none`, and stage 2's two-card work (M2b, the exchange,
   the collectives, the TP2 plan) may start and **interleave with stage 1**: do it while card1 cools or a
   load runs. Note 14's marker line is still wanted when stage 1 ends, for the record.
2. **Engine loads and serving steps may use as many CPUs as the heat allows.** The 80 °C guard stays as it
   is, and the guard runs at `--hot 75` on every step. Rule: start a step at `--cpus 2`; after a step whose
   peak Tctl stayed ≤ 68 °C, the next may double (4, then 8); after a peak above 72 °C, halve. A guard stop
   at 75 °C (exit 11) is **not** a strike under this note: record the peak and the CPUs in the heat table,
   halve, and go on. Never go above 8. Profiling runs keep their 2 CPUs. Nothing about host reference work
   on intel-host changes (none).
3. **Stage 1 stops at 6 tuned kernels** (or a step ≤ 25 ms, whichever first). Then the MTP check, the
   20-minute heat replay with moves on, the marker line, and on to stage 2.
4. **Per-run budget is $80** (`--max-budget-usd 80`, from the next run). The 3-hour run cap stays.

## 17. amd-host builds at 16 threads (2026-10-25, 22:00 UTC)

amd-host has 48 threads and production uses about one of them. From the next run the service's CPU quota is
1600% and UNATTENDED.md says 16 threads: every local build and CPU-device run may use `--cpus 16
--memory 28g`, `-j16`, still `nice -n 19` and still with amd-host's `MemAvailable` above 40 GiB (check it
before a build; two builds at once would break that rule, so run one at a time). `m2/build-l0.sh` and the
oracle scripts still say 8: change them to 16 when you next touch them. intel-host's limits are unchanged
(note 16).

## 18. The card0 gate is open (2026-10-25, 23:28 UTC)

chatterbox-vllm was stopped and disabled on intel-host at 23:28 UTC (job B done: archive verified on both
hosts). `intel-host-precheck.txt` now shows the card0 service as `failed` or `inactive` (podman's stop exit
code makes systemd call it `failed`; either word means retired), `card0_clients: 0` and
`dri_containers: none` whenever only your own containers run. That is the gate of notes 13 and 16: card0
(`/dev/dri/card0`, `/dev/dri/renderD128`, PCI `0000:0b:00.0`) is yours for stage 2's two-card steps,
interleaved with stage 1 as note 16 says. Name two-card containers `ri-m2-<step>` as before; the guard
and the fault rules cover both cards. intel-host's RAM is all yours within the 8 GiB floor and the 16 GiB
rule for pinned memory.

## 19. Either card for single-card steps (2026-10-26, 00:40 UTC)

Single-card steps (loads, tok/s A/Bs, profiles, benches, the tiny model) may run on **card0 or card1**,
whichever is free and did not just fault: card0 = `0000:0b:00.0`, nodes `/dev/dri/card0` and
`/dev/dri/renderD128`; card1 = `0000:0f:00.0`, `/dev/dri/card1` and `/dev/dri/renderD129`. Check the
card by PCI address inside the container as you do for card1. Say which card each step used in the heat
table. **Never two engine or bench steps at once** (one card each): intel-host's heat is the CPU's, so two
steps double it; the guard and the cool check are per step as before. The precheck reports both cards'
clients. When card1 is within 10 minutes of a fault, use card0 and let card1's recovery be judged.

## 20. card0 follows card1's fault rules (2026-10-26, 00:50 UTC)

Now that both cards are the port's, a fault on card0 is handled like one on card1: no pause; save the
dump reference (the udev rule saves every dump of either card to `coredumps/`), diagnose, list it under
"Reviewed GPU faults", fix, verify on the CPU, retest small. The runner pauses only for a wedged card, a
card not recovered 10 minutes after its fault, a signature recurring after a `fix-claimed:` line, or two
faults in one run on either card. The runner's count is now distinct faults on both cards.

## 21. Cache the CPU-device JIT (2026-10-26, 02:05 UTC)

Every oracle, all-reduce test and CPU-device run spends 10-20 minutes JIT-compiling the kernels for the
OpenCL CPU device before it does anything. Make that compile happen once: run those containers with a
persistent SYCL kernel cache, bind-mounted from the host, e.g. `-v $HOME/code/radiance-intel/sycl-cache:/sycl-cache`
and `-e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl-cache` (the OpenCL CPU runtime honours the same
variables through SYCL's program cache; if a run shows it still compiling from scratch, check
`SYCL_CACHE_DIR` is writable in the container and that the image hash did not change the program key).
Add the same two variables to the card containers on intel-host with a cache dir under
`/mnt/fast-storage/models/test/radiance-intel/sycl-cache/` (AOT kernels need no JIT, but the runtime still
caches the SYCL graph and the few JIT'd parts). Say in STATE.md how long the first and second
CPU-device run take after this; if the second is not faster, say so and drop it.

## 22. Run cap 3 h -> 12 h (2026-10-26, 03:30 UTC)

The owner lifted the per-run time cap: from the next run, a run may last up to 12 hours (`run.sh` CAP_S
43,000 s; the unit's TimeoutStartSec 12.5 h). The $80 per-run budget still ends a run, and so does a
clean waiting point when card work is blocked. Everything else is unchanged. A long run should still
write STATE.md as it goes (every step), since a run that dies on the cap keeps nothing it has not
written.

Addendum to note 22 (03:33 UTC): the milestone budget is now **$1,000** (`budget.txt`), the daily caps
match it, and the budget judgement in UNATTENDED.md moves to $700. Spend only on work that moves the
milestone; the cap is a ceiling, not a target.

## 23. Keep amd-host's disk above the gate yourself (2026-10-26, 04:05 UTC)

Twice tonight the runner skipped a cycle because `/` on amd-host fell under 50 GB after your image builds
(the owner pruned both times). After every `podman build` of a runtime image: `podman image prune -f`
(dangling layers only), and remove `ri-m2rt` tags older than the two newest unless STATE.md still
names one for a pending card step. Check `df -B1 ~/code` before each build; if a build would leave
under 55 GB, prune first. Record what you removed in STATE.md.

## 24. Graph memory: measure it, then try host placement (2026-10-26, 05:00 UTC)

Owner priority: right after the two-rank tape fix has had its card run (pass or fail), and before any
further kernel tuning, spend one bounded run on the recorded graphs' VRAM:
1. **Measure what the graph pool holds.** With graphs on, record one decode pass on both cards and
   report the device memory it took, split by kind: command lists/batch buffers, per-node argument and
   event storage, and kernel scratch (register spills). Use the backend's tape stats plus the runtime's
   allocation counters or sysman's VRAM-used before/after; name the kernels that take scratch and how
   much. Say how many graphs are alive at once (recordings plus the graveyard) and the total.
2. **Try host placement for the command lists.** Find the driver's control for placing command
   lists / batch buffers in system memory instead of VRAM (the Intel compute runtime has placement
   debug keys; Level Zero itself has no flag): set it for the engine containers, confirm from the
   allocation report that the command lists moved, then one two-card tok/s run with graphs on,
   compared with graphs on in VRAM and with graphs off. Report the VRAM freed, the tok/s delta and the
   per-launch cost. If the driver offers no such control, say so with the evidence and stop.
3. **Decide and record** (`decision:` line): graphs on with host-placed command lists, graphs on in
   VRAM with a sized headroom, or graphs off, for the rest of stage 2's measurements. Scratch stays
   in VRAM whatever happens.
Everything else in notes 16-23 stands. Keep the run cheap: one build at most, three card runs at most.

## 25. Heat stops under note 16 are not strikes; the spinning wait comes first (2026-10-26, 05:55 UTC)

- **Rule clarified:** a stop by the guard at `--hot 75` (exit 11) is never a strike and never a pause,
  as note 16 says; record the peak and CPUs, halve, cool, go on. The "two heat stops -> PAUSED" rule
  of notes 3/7 now applies only to a stop at the 80 C guard or by intel-host's own 85/88 C guard. Run 10's
  pause was lifted by the owner. Your wrapper's lesson stands: never chain two model loads in one
  container; one load per step, a cool check between.
- **Priority 1, before the remaining note-24 card run:** the sleeping wait. Your finding that both rank
  threads spin at 100% in the device wait while the step is all card time explains the 1-core ceiling
  at every CPU cap and most of intel-host's heat per step. Replace the spin in the backend's stream sync
  with a sleeping/yielding wait (event wait with a bounded poll, then futex/condvar), measure: tok/s
  unchanged or better, container CPU-seconds per step, peak Tctl per step. If it holds, every later
  card step gets cooler and the CPU cap stops mattering.
- **Then** the one remaining note-24 card run (narrower placement mask if the full one aborts the
  driver; the kernel-properties tool with 56bc5fb), the decision line, then stage 2's measurements
  (MTP on with replay, concurrency 1/3/6, coherence, tools, agreement), then the persistent-kernel
  exchange as the first tuning item.

## 26. Decision: graphs on, with the graph pool in system RAM (2026-10-26, 06:20 UTC)

This replaces the provisional "graphs off" and the remaining choices in note 24. Keep recorded graphs
on for the rest of stage 2 and put their memory in system RAM, by whichever of these the driver allows:
1. **Explicit placement, narrow mask:** the compute runtime's placement key for command buffers only
   (not heaps or linear streams, which aborted the driver with the wide mask). One card run; if the
   driver aborts again, record the exact key and abort line and stop trying keys.
2. **The runtime's own spill** (what T2 showed: ~7 GB of graph memory in GPU-mapped host memory at no
   measurable speed cost): find the smallest `--gpu-headroom-mib` at which the first recordings succeed
   (try 512, then 1024), and give the rest of the headroom back to the experts.
Measure for the chosen route: tok/s MTP off and on, VRAM used and GPU-mapped host memory per card
(`drm-total-vram0`, `drm-total-gtt`), expert units resident, and the per-step host CPU. Write the
`decision:` line with the numbers. This comes right after the sleeping wait (note 25) and before the
stage 2 measurements, which then run with graphs on in this configuration.

## 27. Shrink the graph pool (2026-10-26, 06:30 UTC)

After S2's measurements and note 26's headroom run, one bounded run on the pool's size itself:
1. **Measure the split:** how much of the recorded pass's VRAM is scratch surfaces (375 spilling kernels;
   the surface is the largest spill per thread times the hardware thread count, per command list)
   versus command buffers, heaps and node storage. Name the ten kernels with the largest spill per
   thread (`xe_kernel_props`).
2. **Cut the multiplication, in this order:** (a) fewer graphs per pass: record a decode pass as a few
   large graphs rather than ~1,036 segment graphs, so there are a few scratch surfaces instead of a
   thousand; (b) the audit's doubling and the graveyard: keep only what a replay needs alive; (c) the
   worst spillers: reduce register pressure in the top few (the DeltaNet update first), which shrinks
   every surface and speeds those kernels.
3. **Report** the pool before and after (VRAM and GPU-mapped host memory per card), the expert units
   returned to VRAM, and tok/s. Stop at one run; what is left goes in the report as a tuning item.
This comes after note 26 and before the persistent-kernel exchange unless the exchange is already
written, in which case measure the exchange first.

## 28. Fix the backend: tapes on direct Level Zero command lists (2026-10-26, 06:45 UTC)

Owner decision. The recorded-pass ("tape") implementation on SYCL graphs costs ~2.5 MiB per graph and
~10 GB per card per pass; the owner wants it nearly free, as HIP graphs are on the AMD path. Replace it:
- **Design:** record a pass into Level Zero regular command lists directly (`zeCommandListCreate` /
  `zeCommandListAppendLaunchKernel` with the kernel arguments set once, events for the cross-stream
  and cross-rank waits, `zeCommandListClose`, replay by `zeCommandQueueExecuteCommandLists` or an
  immediate list's `AppendCommandLists`), bypassing the SYCL graph layer; one command list per
  recorded stretch between host decision points, so a pass is a few lists, not ~1,036. Keep the public
  tape API (`tape_begin/mark/end/play/destroy`) unchanged so the engine does not notice. Keep the
  per-device state and the deferred free from ba174f6 / c81bc4e. Interoperate with SYCL through
  `sycl::get_native<backend::ext_oneapi_level_zero>` for queues, contexts, kernels and events.
- **Order:** after run 12's measurements (MTP on, the heat replay, prefill first-token times) and note
  27's cheap reductions (which may already take the pool under 1 GB; do them first, measure, then
  this). Before the persistent-kernel exchange.
- **Acceptance:** the tape, device and all-reduce tests pass on the CPU device and on the cards; the
  tiny model's recorded and plain runs give identical tokens; two-card T1 tok/s with replay is no worse
  than before; the recorded pass's VRAM and GPU-mapped host memory per card measured before and after
  (`drm-total-vram0`, `drm-total-gtt`); kernel logs clean. Write the numbers in STATE.md.
- **Budget:** at most two runs. If it is not passing after two, leave the SYCL-graph tapes in place
  (with note 27's reductions), record where it stopped, and go on to the exchange; it goes in the
  report as the milestone-3 item.

## 29. Report sooner (2026-10-26, 07:25 UTC)

The owner wants the milestone 2 report earlier than note 28's two-run budget allows. This note narrows
note 28 and fixes the order of what is left; everything else in notes 26–28 stands.

- **Note 28 gets this run (13) and at most one more (14).** Whatever state the direct Level Zero
  tapes are in at the end of run 14, stop working on them and write the report. If they serve, record
  the before/after numbers from note 28's acceptance list; if they do not, record where it stopped,
  leave the SYCL-graph tapes with the hybrid cache in place, and put the remainder in the milestone 3
  plan.
- **Do not start the persistent-kernel exchange or any further stage-1 kernel before the report.**
  Both go into the milestone 3 plan with a size: the exchange with M2b's 24 µs and the ~10 ms of 27 ms
  a step it addresses; the kernels with the current per-kernel table.
- **MTP on:** record S3's result (9.4 tok/s warm, step 402 → 175 → 71 ms as the stretch cache warms)
  with your diagnosis (four draft passes thrash the 512-entry stretch cache?) as a finding, not a fix
  to make now. One cheap experiment is allowed if it fits inside run 13 or 14 without delaying note
  28: a larger bound (for example 2048 entries) for one MTP-on tps run, to show whether the thrash
  explanation holds. Otherwise it is an open item.
- **Report at the end of run 14 at the latest** (`/srv/models/test/reports/<date>-radiance-intel-m2.md`
  + `.json`, then `status: complete`), with every PROMPT measurement already in hand: the per-kernel
  table before/after, tok/s one card and two cards, MTP on/off, by concurrency (1/3/6), the heat
  replay's readmit rate and policy finding (S4), the prefill first-token times (S5), the exchange's
  measured cost, the estimate comparison (uniform 81–116, skew-corrected 150–220, amd-host's 218 against
  the measured ~36–37), agreement results, and the milestone 3 plan. Reference-doc changes as before.
- If run 13 ends before S4 or S5 has run, run 14 does them first, then the rest of this note.

## 30. Milestone 3 starts (2026-10-26, 08:10 UTC)

Owner: "continue build it, the result needs improvement." Milestone 2 is archived in `m2/` (STATE,
PROMPT, DONE, notes, UNATTENDED, runs, logs). PROMPT.md is now milestone 3, built from the report's §8
plan with targets; UNATTENDED.md says milestone 3 and `ri-m3-*` (the guard also still accepts `ri-m2-*`).
- Every note 1–29 still applies, except: **note 29's cut-off is lifted** — note 28's two-run budget is
  back and is stage A item 1; and the production-request count restarts at **20** for this milestone.
- **The owner's production window** (job A, Radiance 1.11.0 on amd-host) runs 2026-10-27 06:00–07:30 UTC
  from `a2-window.timer`. It is not yours. No amd-host builds or other amd-host CPU load 05:45–07:45 UTC that
  day; intel-host-side steps may continue.
- **Measure both ways, always** (PROMPT's targets): the single warm prompt and the 11-request rotation's
  median. A speed claim with only one of them is incomplete.
- Stage D prepares the service and soaks it; **nothing is switched to it**. The owner promotes.
- Milestone 2's report path is now fixed in the runner (`*-radiance-intel-m3`); `DONE` is created by the
  runner only, from `status: complete` plus both report files, as before.

## 31. Richer Telegram run summaries (2026-10-27, 02:05 UTC)

The owner wants more detail in the per-run Telegram message. run.sh now sends up to 1,500 characters of
the run's final summary (newlines kept, `**` stripped) instead of a 300-character single line; the run
log is unchanged. This is a runner change made by the supervisor between runs; nothing for you to do,
and run.sh stays off-limits as before. When you write the end-of-run summary, remember its first
1,500 characters are what the owner reads on the phone: lead with the numbers.

## 32. Prior art for the multi-column XMX path (owner, 2026-10-27, 02:30 UTC)

llama.cpp PR 29864 (ggml-org, merged 2026-10-22, commit d81d0d8, SYCL backend, MIT license) did on an
Arc Pro B70 what stage A item 4 and stage B need: multi-column `mul_mat_vec_q` kernels on the XMX units,
int8 DPAS through ESIMD, for 2–8 column verification batches and 9–80 column short prompts, plus a fused
gate/up ("up * act(gate)", up to 16 columns) and a GRF sweep. Their stated reason the old path was slow is
our profile's too: per-column repetition of weight unpacking and scale arithmetic. Author-measured on a
B70: speculative decode end-to-end 28.6 -> 53.6 tok/s (Q3_K_S); pp16 50.3 -> 308 t/s (Q4_K_S).
- **Use it as prior art, not as code to copy blindly:** read the PR and its `esimd.hpp` / kernel diffs
  (fetch from github.com/ggml-org/llama.cpp, MIT; cite the PR in a comment where a technique is taken).
  Apply the ideas to (a) the trunk INT8 GEMV at 2–8 columns — the MTP verify pass, which is item 2's
  remaining speed problem; (b) the gate/up fusion for the expert path; (c) stage B's short-batch GEMMs.
- **Their caveats are ours to check:** MoE (MUL_MAT_ID) was out of scope and issue 31068 reports MoE
  prompt processing ~35% slower on a B70 under that path, so the 4-bit expert GEMM needs its own
  verdict from our oracle and bench, not theirs; some decode cells regressed 2–10%; gains shrank with
  their flash attention on. Measure ours as always (oracle, bench, M18/M19 pair, both measurements).
- This does not change the order of work: finish the GDN question cheaply, then this is the preferred
  technique for the remaining item 4 kernels and stage B.

## 33. Calibration: Qwen3.8-27B serves at 58-60 tok/s on ONE B70 (owner link, 2026-10-27 02:50 UTC)

github.com/Danmoreng/intel-b70-qwen38-vllm (MIT): vLLM 0.31.0+xpu, EXL3 4.00 bpw Qwen3.8-27B on one
32 GB B70 at 180 W. Decode 58.4-59.7 tok/s single user (C4 aggregate 190.8); prefill 1,712-2,493 tok/s,
TTFT 0.30 s @512 / 0.82 s @2K / 7.53 s @16K; FP8 KV; MTP 3 draft tokens adopted (their MTP4 and DFlash
lost); decode-only graphs, capture 1-64. Use ONLY as calibration, not scope: a 14 GB dense model resident
in VRAM is not our 122 GB streamed MoE. What it tells us: (a) the silicon and driver sustain ~59 tok/s
decode and ~2,500 tok/s prefill, so our remaining gaps are software; (b) stage B's first-token targets
are conservative; (c) MTP-3 on this card is the right speculative config when passes are cheap. Put the
comparison (one line) in the report's estimate section. No change to PROMPT or order of work.

## 34. The runner moves to Opus 5.5 (owner, 2026-10-27 04:15 UTC)

Run 27 ended at 04:07 on "You've reached your Fable limit" after 94 turns (the cards were idle and clean;
P3/P4's profiles and image `:34` survived). Two runner changes by the supervisor, between runs:
- `run.sh` now passes `--model claude-opus-5-5` (override per run with `RI_MODEL=...`). The owner chose
  Opus 5.5 for the rest of milestone 3 over waiting for the Fable window.
- A model/plan limit message is classified as a limit (retry next cycle, no notification spam) instead of
  a failed run, so it can no longer reach the three-failure pause.
Nothing for you to do; your session resumes with `--resume` as always. Expect per-run cost figures to
differ from the earlier runs: the budget numbers are CLI estimates at list prices, so the $1,000 milestone
budget now counts Opus-rate estimates. If the pace of the budget worries you, say so in STATE.md instead
of changing anything.

## 35. A CAT error in the guard's own kill is not a recurrence (owner, 2026-10-27 05:00 UTC)

Twice now (m3 fault 3 at 23:18 on 10-26, m3 fault 5 at 04:42 today) the guard's heat stop killed the
engine with device commands in flight, the kernel log recorded `CAT:bcs` / fault responses in that same
second, and the runner paused because `CAT:bcs` carries milestone 1's fix-claimed line. The cards were
fine both times and the owner resumed within minutes. The judge in `run.sh` now reads `logs/memguard.log`
and treats a kernel-log incident within 20 s of a `STOPPED (heat)` line as a kill artifact: it logs and
notifies, does not count the fault, and does not pause. `wedged` still pauses, whatever else is true.
- You keep writing `reviewed-fault:` lines for these when they happen (the record matters), but you no
  longer have to wait for a resume.
- The heat stops themselves are the real problem to engineer around, not the log lines: oracle and
  profile steps must end before the host's load builds (your ORACLE_MAX decision line) and engine loads
  stay at 2 CPUs (H2b).

## 36. Publish the progress numbers after every run (owner, 2026-10-27 05:30 UTC)

The owner watches a live view of the milestone's numbers. It reads a file, and that file has been
maintained by hand, so it goes stale between runs. From now on **you** keep it current, because you are
the one who measures.

At the end of every run, write `progress.tsv` in the project directory: one row per target, tab-separated,
with a `#` comment header. Columns: `name`, `baseline`, `now`, `target`, `note`. Keep the rows and their
order exactly as below, update `now` whenever a run measures it (and only then), and say in `note` which
step and run produced the figure. Rewrite the whole file each time; it is a snapshot, not a log.

```
# name	baseline (m2)	now	target	note
decode 1-prompt	36.5	36.0	60	MTP off, warm (M20, run 26)
rotation median	22.2	29.8	40	11 prompts, MTP on (H3, run 24)
aggregate @6	75.6	75.6	120	not re-measured since m2
MTP 1-prompt	9.4	40.3	36.6	M22, run 29: beats MTP off
first token 7K	53	53	5	stage B, not started
tape pool GB/card	10	2.5	1	VRAM+host over plain (T5, run 23)
replay misses/step	8.3	8.3	4	stage C, not started
```

Rules: the numbers must be ones you measured in this milestone (or m2's baseline), never estimates or
hopes; if a figure is stale because its last measurement is now superseded by a code change, keep the
old number and say so in the note ("stale since 8bb3f99"); if a target is met, keep the row. The report's
targets table is this file plus prose, so keeping it current costs you nothing later. `targets.tsv` is
the owner's fallback copy -- leave it alone.

## 37. Upstream issue 130 explains the n-gram table's cost (owner link, 2026-10-27 05:55 UTC)

codeberg.org/StillDeadcode/radiance issue 130 (open, 2026-10-19) is the same ground as stage B item 2,
with other people's measurements on R9700s:
- With `--ngram-placement disk`, **decode issues about 15 x 4 KiB reads per generated token** and prefill
  at 16K tokens runs at 1.5-3.7k tok/s, bounded by the drive, not the GPU (GPU busy % falls with no
  throttle bit; step time correlates r ~ +0.9 with the drive's await). On a DRAM-less consumer NVMe the
  host memory buffer (40 MiB, mapping ~40 GiB) cannot cover the 47.7 GiB table, so reads settle at
  17-20k/s. Drives with on-board DRAM, or Optane, do not show it.
- Our own numbers fit: m2's table charged `embed_lookup_q` **2,128 us for one host launch a step**, and
  intel-host's model lives on `/dev/nvme1n1` (Micron 2450, a DRAM-less OEM drive) -- the exact case in the issue.
- The workarounds are `--ngram-placement ram`, a model drive with DRAM, or the reporter's LD_PRELOAD shim.
  The maintainer welcomes a PR for `--ngram-placement disk=PATH`; `bakon` proposes it. Not ours to write
  now, but worth knowing the upstream direction.

**What to do with this (stage B item 2, and possibly earlier):**
1. `--ngram-placement ram` may buy **decode** time too, not just prefill: ~2 ms of a 26.6 ms step is that
   one host read. Measure it as its own A/B (M-pair on the same image, MTP off and on) before any prefill
   kernel work, since it is a flag, not code.
2. **Measure the memory first.** intel-host has 121 GiB total, ~116 available. The table is 47.7 GiB pinned on
   top of two ~16.6 GiB pinned pools, and H3/H4 saw transient driver-side dips of up to 53 GB during a new
   request's prefill. 47.7 + 33 + 10 = ~91 GiB leaves ~25-30 GiB for those dips against an 8 GiB guard
   floor -- it may not fit. Check with the memory sampler running, and if it does not fit, say so and keep
   the table on disk rather than risking the guard.
3. If RAM does not fit, the cheap alternative is the issue's own: put the table where reads are fast. Note
   intel-host's second drive (`nvme0n1`, Crucial P310) in the report as an option for a later `disk=PATH`.
4. If the flag turns out to help decode measurably, say so plainly in the report: production on amd-host runs
   the same placement on its own drives, so it would be a candidate for an owner-run production experiment
   in a later window. Do not touch production yourself.

## 38. What 1.12.0 holds for this port (owner, 2026-10-27 06:05 UTC)

The port is already on 1.11.0, so tonight's production window brings it nothing but like-for-like
agreement and prompt 10's answer. **1.12.0** (v1.12.0 = `5a9ce4912bb4`, 70 commits over `9eaf10d`; the
analysis is `~/code/radiance-test/keyenv/1.12.0-notes.md`, the tree with tags is
`~/code/radiance-test/src-1.11.0`) does hold three things for us. Read, do not merge wholesale: the port
branch stays on 1.11.0 for this milestone.

1. **The MXFP4 GEMM tuning series is the shape library we need now** (commits `168a26a a85f87e 6249462
   824e238 b90723e 58142a6 75b4d0b b66752a 2382686 0fb3bad 9daf738 d4eacec`): `gemv_wide`, `shortk`,
   slab-ahead, `TN=4` tiling, and fp8/int8 heads **past 64 rows**. It is HIP and a different quant, so
   none of it ports directly, but it is the engine's own answer to exactly our open question -- the
   multi-column trunk GEMV (`gemv_i8a8<4>`, 16.0% of the MTP step) and the expert GEMVs at 4 tokens
   (`moe_planes<1>/<5>`, 25.8% together). Read these before writing the next kernel, alongside owner
   note 32's llama.cpp PR, and say in the commit message which shape you took from where.
2. **`f2c8927` is a correctness fix we should carry** (5 lines, `core/mem/prefix.cpp`): with a host KV
   tier and `keep-all`, `PrefixCache::insert` could replace KV still referenced by an older recurrent
   checkpoint of a *stateful* model -- which Flash-Next is, and which the port runs with the same prefix
   cache and snapshot policy. Cherry-pick it onto `intel-l0` **before stage D's 24-hour soak** (a soak is
   exactly the long multi-session run that would expose it), verify the tape and device tests plus one
   rotation replay, and record it in the report as an upstream pick with its commit id.
3. **Prior art for stage B and a later kernel:** the pipelined prefill series (`d39e05...`: prefill in two
   pieces, `rad_lane1_window_begin/_end`, lane-1 scratch sizing, `RADIANCE_PREFILL_PIPELINE`) is for
   qwen35/gemma3, not our arch, but it is the engine's pattern for overlapping a prefill -- worth reading
   when stage B's XMX GEMMs land. `4d68a01` (qwen35's GDN state cache at f16, `RADIANCE_GDN_STATE_F16=1`)
   is the idea our fp32 GDN state (4.7% of the MTP step, 74.6 us a launch) could borrow later.

Not for us now: `d912c69` (linear MTP coupling and cooperative verify, opt-in, a third-party port landing
in the MTP path) -- interesting once our MTP is stable and measured, but it needs the full feature-check
and soak protocol, and our acceptance is already at the drafter's ceiling. Nothing in 1.12.0 touches
`core/place` (the mover), the `.rad` loader, or the exact 2-rank all-reduce.

## 39. What the quiet window actually forbids, and what to do in it (owner, 2026-10-27 06:30 UTC)

Note 30's quiet window exists so the owner's production run on amd-host is not slowed by ours. It forbids
**heavy** amd-host CPU: `build-l0.sh`, `make-rt.sh`, cmake/oneAPI compiles, the CPU test tier, container
builds, anything with `-j`. It does **not** forbid light work on amd-host: reading git history, writing and
editing source files, comparing saved outputs with a small script, writing STATE.md or report prose.
intel-host-side card steps stay allowed throughout (production is on amd-host's cards, not intel-host's).

So do not idle until 07:45. In this window, in this order:
1. **Compare the 11 coherence answers** from C1 (run 30) against milestone 2's, since they are already on
   disk and only need a diff: content, tool calls, the medical prompt. Record it; it is a report item.
2. **Read the 1.12.0 MXFP4 GEMM series** (note 38's commit list) in `~/code/radiance-test/src-1.11.0`
   (`git show`, no build), and the llama.cpp PR's `esimd.hpp` if it is already fetched. Take what applies
   to a 4-column int8 GEMV and a 4-token expert GEMV.
3. **Write both kernels** (the multi-column trunk `gemv_i8a8<4>` at 8.9 ms and the expert GEMVs at 14.3 ms)
   so that at 07:45 the only thing left is `build-l0.sh`, the oracle and the measurement. Put the chunked
   `mtp_enter` fix in the same build.
4. If time remains, draft the report's targets table from `progress.tsv` and the per-kernel tables.

The point is that the build window at 07:45 should open onto finished code, not a blank editor.

## 40. Production is on 1.11.0; prompt 10 is settled as a port divergence; heat (owner, 2026-10-27 12:20 UTC)

- **Production moved to 1.11.0-keyenv at 06:00 UTC and was promoted** (features 17/17, replay 9/9, first token
  0.48/2.08 s, per-turn 2.87/8.62 s, cache 99.3 %, 0 errors; report `/srv/models/test/reports/2026-10-27-
  amd-host-radiance-1.11.0.md`). From now on agreement checks are **like-for-like** (same version both sides);
  drop the "possibly version-related" caveat.
- **Prompt 10 is answered, and it is ours.** Production on 1.11.0 (same commit as the port) says **'The'
  at logprob -0.050**, identical to 1.9.0; 'We' sits at **-6.649** in its top-4. The card said 'We' at
  -0.32. So the 6-nat divergence is a **port numerics difference**, not a version difference. Reopen it as
  an open correctness item for the report with the suspects named: the E4M3 hc mixes and FP8->FP16
  conversions, the INT8 trunk GEMV's rescale, the 4-bit expert dequant, the n-gram lookup path. Milestone
  1b's CPU-backend comparison at the port's commit is the instrument (`m1/prompt10_host.py`); the
  per-layer first-divergence between libref and libxe on this prompt is what the report should state.
  Not a speed item: do it after the concurrency fix, before the soak, time-boxed to one run.
- **Production's own heat counters under the soak (1.11.0, cold start, varied prompts): 9.1 promotions
  and 9.1 readmits per step, miss 0.95 %, stream 63 MB/step, h2d 141 MB/step** -- against 1.9.0's 7-day
  live average of 1.9 / 1.9 / 0.25 % / 12.5 / 25. So under varied traffic the production mover churns much
  as ours does on the rotation (8.3 misses/step = 1.5 %). Use these as the comparison row in stage C and
  in the report's §5.6 successor: the policy finding is about the mover, not the port.
- **Production's draft acceptance ran 0.57-0.59** on the soak's traffic; our 46 % is on one repeated prompt.
  Measure ours on the rotation too before the report quotes a single number.
- **84 C.** O5 (the expert-op oracle on the big model) took intel-host to 84 C in 86 s -- above the 80 C guard
  line, caught only because the agent guard fires at 75 with a 2 s window. The decision to gate expert
  kernels on the tiny model (E4) stands: **no host-reference oracle on the big model again on intel-host.** Any
  step whose host load is unbounded runs at 1 CPU or not at all.
- The fault-7 pause lasted 3.5 h because the owner's watcher was down, not because anything was wrong
  with your review. Nothing to change on your side.
## 41. The build was refused because of its spelling, not a policy change (owner, 2026-10-27 13:50 UTC)

Run 36's `bash m3/build-l0.sh l0t '...'` was refused because the permission list has no rule for a bare
`bash ...` command; it has `Bash(timeout *)`. Run 34's build went through because it was written
`timeout 3h bash m3/build-l0.sh l0b` -- the same form your own Next section prescribes. So:
- **Every build, test or long command on amd-host goes through `timeout <limit> ...`** (`timeout 3h bash
  m3/build-l0.sh ...`, `timeout 3h bash m3/make-rt.sh ...`, `timeout 10 bash -n script.sh`,
  `timeout 20 grep ... /proc/meminfo`). That form is allowed; the bare form is not.
- `ssh intel-host podman images` is refused **by design** (the owner's deny list blocks `podman image*` on
  intel-host from your session); read image ids from your own `D/images.txt` records or from the step
  scripts' output as you did before.
- A refused command is never a reason to write PAUSED. Re-spell it in the allowed form first; if that
  is refused too, log it under the owner blockers and continue with what does not need it.
Nothing in the permission settings changed; your run-37 plan (build :39, C9) stands as written.

## 42. The C6 stall: what the log shows, and what to change before C9 (owner review, 2026-10-27 14:00 UTC)

Read the C6 log and the code. Your narrowing (C7: the history pass with no drafting row) holds. Four
things to change, the first two before C9:
1. **Your silence argument is unsound.** `ar_gave_up` prints with a device printf, which never surfaces
   under native launches (your own M10 finding), and the block "still arrives", so a timed-out collective
   completes with garbage and the host never knows. `xe_ar2_stalls` (seq_ctr[3]) is host-readable and the
   engine never reads it. Read it on both ranks after every step and log it; extend the 10 s play warning
   to name the tape, range and the event waited on. Until then no stall can be explained.
2. **Add a dangling-wait check at `tape_end`:** every recorded wait must have a signaller inside the tape
   or in the runtime's prior work. A no-draft history pass has an empty draft lane, and the M9 ordering
   records a wait on a lane event nothing signals -- the best-supported mechanism for "device 0's play
   never completes" while rank 1 finished. Make that pass record correctly rather than exempting it (your
   live fallback stays as the stopgap; `RADIANCE_TAPE_HISTORY_ONLY=1` as the repro).
3. **Record/skip/drop decisions must be rank-symmetric.** They are per rank now (the VRAM floor reads one
   card's free VRAM; "rank %d: ... issued in full" is per rank). Decide once per step, or agree at the
   barrier. Two ranks taking different paths in the same step is a latent hang in this backend.
4. **Never flip drafting in an admission step; add hysteresis** (two consecutive steps over or under
   `RADIANCE_SPEC_MAX_SEQS`). That is also the fix for the 3-user oscillation in C5.
The time-box stands: if C9 is not clean after 1-2, MTP off is the service setting, then prompt 10.

## 43. The owner is away; the runner is on Fable 5.1 at max effort; a system review runs on amd-host (2026-10-27 14:20 UTC)

- `run.sh` now launches `--model claude-fable-5-1 --effort max --fallback-model claude-opus-5-5` (the CLI falls
  back by itself when Fable's quota is exhausted). Nothing for you to do.
- A one-shot unit `ri-review-20261027` is writing a whole-system review to
  `briefs/2026-10-27-system-review.md` (read-only, light CPU; it shares amd-host with you). Ignore it; it will
  not touch your files or the cards. The owner reads it later.
- Pauses while the owner is away: an `autoresume.sh` exists (owner-enabled; it may or may not be running) that
  lifts a PAUSED after 15 min **only** when your PAUSED text says the cards are fine / no root action is
  needed / "To resume: rm PAUSED", or the pause is about a refused command, and intel-host's precheck is clean, at
  most twice in 6 h. So keep writing PAUSED exactly as you do: state plainly whether the cards recovered and
  whether a root action is needed, and name the condition (wedged, two faults, budget) when it is the owner's
  call. The supervisor's own watchers are down until the owner returns; your Telegram summaries are what the
  owner reads.

## 44. Production's K/V cache dtype is fp8 (owner, 2026-10-27 16:45 UTC)

Answer to run 41's blocker: production runs `--kv-cache-dtype fp8` (prod/radiance.container line 38, and
the running container's arguments say the same). So on prompt 10 the port's fp8 attention differs from
production's native-FP8 attention by about 0.4 nats on "The" while the top token agrees in every setting.
Record it in the report as a known numerics delta of the port's FP8 K/V path (E4M3 converted to FP16 on
Xe2), with the bf16 K/V figures beside it (The -0.0045, which matches production within 0.05 nats) as the
fidelity option that costs VRAM. Not a bug to chase now; the service keeps fp8 K/V. Prompt 10 is closed.

Also noted: the owner is present again and this chat is the single supervisor; the autoresume timer stays
as installed. Note 43's "owner is away" no longer applies.

## 45. Owner decision: test the engine's async table reads with io_uring permitted (2026-10-27 17:00 UTC)

Every engine start on intel-host logs `io_uring refused (Function not implemented)`: the engine's asynchronous
path for the n-gram table's ~15 reads a token is unavailable in the container, so the reads run one after
another at the drive's per-read latency. The owner reviewed this today and approved one measured test
with a **narrow** allowance only:
1. Make a copy of podman's default seccomp profile (`/usr/share/containers/seccomp.json`) under
   `m3/intel-host/seccomp-iouring.json` with exactly three syscall names added to the allow list:
   `io_uring_setup`, `io_uring_enter`, `io_uring_register`. Nothing else changes in the profile. Pass it
   with `--security-opt seccomp=<that file>` for the test containers only. Confirm the startup line no
   longer says "refused".
2. One A/B on the same image and args: single prompt MTP off and on (step time, tok/s) and the S5 prefill
   at 7K and 20K (first token), against M25/M26 and S5. The table's share today is about 2 ms a step and
   about 12 of 53 s at 7K; report what the async path recovers of each.
3. If it helps, the stage D service unit carries this same custom profile (the file beside the Quadlet),
   and the report's reference-doc section states the three syscalls and why. If it does not help, record
   the numbers and leave the default profile in place everywhere.
One run, after the current pool measurement; this is a runtime setting, not engine code.

## 46. The owner is away again from 17:05 UTC (2026-10-27)

The supervisor's watchers are down until the owner returns; the autoresume timer (note 43) is active and
lifts only the safe pauses; your Telegram summaries are what the owner reads, numbers first. Nothing else
changes: notes 44 and 45 stand, the service setting stands, the order is the tape's VRAM, then the io_uring
A/B, then stage B's prefill work. If a pause needs the owner (wedged, two faults, budget, a root action),
write it plainly and wait; the owner reviews on return.

## 47. Owner decisions: the mover policy (stage C) and cooperative verify, after B3 (2026-10-27 21:15 UTC)

The owner read runs 43-48 and chose two things for the decode number, in this order, after B3's commit
decision (the f16 expert GEMM's logprob check and timing):

1. **Stage C now, not last** (PROMPT's mover policy, one flag-guarded run): a minimum residency age or a
   frequency-weighted score so a unit demoted seconds ago is not brought straight back; the 5-minute
   rotation at the service config before and after, both measurements, plus misses a step and readmits.
   MTP steps route draft tokens to different experts, so this is expected to help MTP on more than off.
   Production's 1.11.0 counters (note 40: 9.1 promotions a step, miss 0.95 %) are the comparison row.
   The default stays production's unless the replay is better on both counts.

2. **Cooperative verify, port only** (at most two runs): cherry-pick upstream `d912c69` ("linear MTP
   coupling and cooperative verify", v1.12.0, in `~/code/radiance-test/src-1.11.0` with tags fetched) onto
   `intel-l0`. It is opt-in (`RADIANCE_MTP_COUPLED=1`) and refuses itself unless the plugin exposes the
   candidate head's reranked top-8; find out whether qwen4exp_fp8 does after the pick. Measure with it on
   and off on the same image: draft acceptance (single prompt and the rotation), tok/s, step time, and the
   5,604-token first-token logprobs against the control (the B3 rule). Keep it only if acceptance and tok/s
   both rise and the logprobs match; otherwise record the numbers and revert. Nothing of this touches
   production. If the pick conflicts beyond the scheduler and sampler files, stop and record why.

Then back to stage B's remaining prefill pieces (attention per query row, the GDN chunk path). The
expectation on record: cheaper passes and residency take MTP on from 42 toward 48-50; cooperative verify
is the only lever left that could add on top of that. No change to the fault, heat or budget rules.

## 48. Owner decision: measure the n-gram lookups' skew before deciding on a hot tier (2026-10-27 21:20 UTC)

The table stays on disk (note 37) and costs about 2 ms a decode step and ~11 s of the 5,604-token first
token. Before anyone codes a two-tier table (a pinned RAM tier for the hottest entries, the rest mapped
from disk), the owner wants the one number that decides it: how concentrated the lookups are.

- Observational only, no engine change beyond a logging switch: record the table offsets (or entry ids)
  that `embed_lookup_q` touches, per token, during **stage C's "before" rotation** (note 47 item 1 -- the
  same 5-minute replay, so it costs no extra card time) and during one 5,604-token prefill. Write them to
  `D/m3/<step>/ngram-offsets.bin`, nothing else changes.
- Off the cards, on amd-host: the hit rate a hot tier of 1, 2, 4, 8 and 16 GB would reach if it held the most
  frequently touched 4 KiB pages (LRU and frequency both), for decode and for prefill separately; and how
  many distinct pages the rotation touches in all. Put the curve in STATE.md and the report.
- Decision rule for later, not now: a tier is worth coding only if 8 GB or less catches at least 80 % of
  decode lookups; otherwise the drive is the only fix and the report says so. Do not build the tier in
  this milestone without the owner reading the curve first.

## 49. Owner decision: B3's matrix-unit expert form is ON in the service setting (2026-10-28 03:10 UTC)

The owner read run 58's table and the one-prompt ranking difference (" __" against "\n\n", a near-tie on
both forms, the matrix form being the exact arithmetic) and chose the speed: the matrix expert form is
the service default from now (7K first token 36 -> 31 s, 16K 65 -> 56 s; decode unchanged). The
production-ranking rule is not abandoned, it is measured instead:
1. Make the matrix form the default in the service args and the stage D unit; keep the switch so the
   GEMV form stays one flag away. Record the one-prompt near-tie in the report as a known divergence.
2. Run the 11-request agreement table (the review's item 8) once on the service setting with the matrix
   form on -- tool calls, top tokens and the logprob gaps against production, like for like on 1.11.0.
   Production requests remaining: 14 of 30; use at most 11. Put the table in the report beside the m2 one.
3. If that table shows the matrix form agreeing with production less often than the GEMV form did in m2
   (tool calls or top tokens), say so plainly in STATE.md and the notification and leave the default as
   the owner set it; the owner decides, not the rule. Do not revert on your own.
Order: after the bf16 GEMM lands (run 59's checks), before the GDN chunk-scan port.

## 50. Leads from a sweep of what others measured on B70s (owner, 2026-10-28 03:50 UTC)

The supervisor searched for current results on Battlemage MoE decode and prefill. Four are worth your
time, ranked by gain over cost; the rest are recorded so nobody re-searches them.

1. **Prefill chunk size (a flag A/B, same run as note 49's table).** Strata, a SYCL engine that runs
   this very model (Qwen3.8-Flash-Next, IQ3_S) on ONE B70 with only 11.5k of 24.6k experts resident,
   reports 980-1,000 tok/s prompt processing at 4K; ours is ~155 tok/s (5,604 tokens in 36 s on two
   cards). Their prompt chunk on a Battlemage card of >= 24 GB is 8,192 tokens; ours is 4,096. With the
   matrix-unit kernels in, each doubling of the chunk halves the number of expert-weight sweeps a long
   prompt pays. Measure the 7K and 20K first token at 8,192 (and, if the arena allows, 16,384) against
   36 / 56 s, with the staging ring and VRAM sampled; keep the largest chunk that fits the service's
   VRAM plan.
2. **Heat-engine decay and move budget (one run, no kernel).** Run 52's histogram says two thirds of
   readmits return 64-1023 steps after their demotion; LFU decay 0.95 a step forgets a unit in about 60
   steps (0.95^60 = 5 %), which is exactly how those units fall out and come back. Test decay 0.98 and
   0.99 (half-lives ~35 and ~70 steps) and `moves_per_step` 32 and 16 on the 5-minute rotation at the
   service config; read misses a step, readmits and the median. The vLLM RFC for Intel XPU expert
   tiering (issue 63666) found the same thing from the other side: seeding placement from the prompt's
   expert counts, rescaled to ~64 decode tokens and WITHOUT refreshing recency, took their B70 host-served
   lookups from 28/16/11 % to 10/2/0.1 % over three requests, while refreshing recency cost ~250 extra
   swaps a request. Check how our prefill chunks credit the heat (thousands of tokens against a credit cap
   of 512) before changing anything; the fix may be a rescale of prefill credit rather than a new policy.
3. **Prefill design input for the remaining ports.** Strata's Battlemage path reads experts in int8 for
   prompt processing (a prefill-only expert format, +12-21 % prefill in their notes) and runs the prompt
   reader on many host threads. Their SYCL source is MIT (github.com/Niko1221/Strata, `sycl/`,
   `docs/INTEL_ARC.md`); read it before the GDN chunk-scan and attention ports for shapes that worked on
   this silicon. Cite what you take.
4. **Platform facts for stage D (no action now).** Allocating up to the VRAM ceiling can leave the card
   "wedged" until reboot (drm/xe issue 7893): the VRAM floor (6ebd5d8) is also a safety feature, keep it in
   the service. Ahead-of-time compilation for `bmg-g31` with `ocloc` (Strata: `-DSTRATA_SYCL_AOT=bmg-g31`)
   removes the first-start JIT (~47 s in their engine); check how much of our model load is JIT and
   whether the stage D image should carry AOT kernels. Several B70 users report fewer Level Zero resets
   with the v1 adapter under sustained load, which is what we run. Raising the xe `job_timeout_ms` is
   reported not to fix real hangs; do not. Resizable BAR is on (32 GiB BAR2 on both cards, verified).
Looked at and not useful now: XeTLA (no Battlemage results), EAGLE-3 drafters (no Qwen3-MoE result,
needs training), the n-gram table and the SSD (closed by notes 48 and runs 52/57). Sources are in the
supervisor's session; the agent needs none of them beyond Strata's repository.

## 51. The report path is writable now (owner, 2026-10-28 04:15 UTC)

The owner added to `.claude/settings.json`: `Write` and `Edit` on
`//srv/models/test/reports/*radiance-intel-m3*`, and `Bash(bash m3/build-l0.sh *)` /
`Bash(bash m3/make-rt.sh *)` so builds no longer depend on the `timeout` prefix (it is still the
preferred form). Remove "report path not writable" from STATE.md's owner blockers. When the report is
ready, write it to `/srv/models/test/reports/<date>-radiance-intel-m3.md` and `.json` (the draft in
`m3/report-m3.md` stays as the working copy); the runner's DONE check looks for exactly those names.

## 52. The bottleneck analysis: fix the tape pool's collapse before the heat A/B (owner, 2026-10-28 05:30 UTC)

The supervisor ran an independent, adversarially checked analysis of every measured profile; read
`briefs/2026-10-28-bottleneck.md` (132 lines, every number sourced). Three findings change the order of work:

1. **The rotation's biggest measured loss is the tape pool, not misses or the mover.** In H11 the pool
   grows from 1,034 to ~8,000 live lists over requests 12-15, the 1 GiB floor fires at ~98 s (serve.log:
   1034-1035, one line per rank), and the drop rule (one still-played LRU tape per tape-less pass,
   ctx.cpp:1385-1391) empties the pool by request 22; from there every pass runs live on 2 CPUs at 149-254
   ms a step, median 17.0 against 35.75 while taped (`passes.issued_refused` 7,918 of 16,352). On any run
   longer than ~2 minutes the service converges to 17-22 tok/s -- that is what a soak would measure.
   **Do this before H16-H19** (the analysis puts decay/age changes at <= 1 tok/s, consistent with H12):
   (a) change the under-floor policy: remember a refused shape instead of dropping played tapes, and never
   drop a tape that was played in the last N steps; (b) bound the pool's VRAM -- record a decode pass as a
   few large lists rather than ~1,182 segment lists, since the per-list driver scratch (~3.8 MB) is the
   cost (A1-A4, K1). Measure with a rotation long enough to cross the old collapse (at least 6 minutes, or
   two 3-minute pieces in one server), reporting requests before and after request 22 separately.
2. **Decode is kernel-form-bound, not bandwidth-bound.** The MTP-off rank-step moves ~3.3 GB in 26.9 ms
   (123 GB/s, 21 % of 596). Largest items with measured same-card probe rates: the E4M3 hc mixes 8.0 ms
   at 14-16 % of peak against a 232-242 GB/s probe (round2 out-m1.txt:32-33), and the expert GEMVs 4.5 ms
   at 23 % against M6's 343. After item 1, the hc mix rewrite (#3 in the brief: norm once per connection,
   all waves streaming, an SLM or bit-trick E4M3 decode, fusing A+B and folding hc_write into ar_kernel)
   is the largest decode lever: ~4.4-4.9 ms, about +6-7 tok/s MTP off and -1.4 to -2.7 ms on the MTP step.
3. **Prefill's largest item is the GDN chunk scan** on the shuffle-emulated WMMA: 3,168 serial 64-token
   steps at 2.356 ms each, 7.47 s a rank of the 27.3 s (16.8 KB/thread spill); the brief sketches the
   parallel split (W/U and O parallel, only the 128x128 state update serial).
Order from here: item 1, then the report draft (it must say the rotation collapse and its fix plainly),
then item 2's hc rewrite, then item 3 if budget remains. H16-H19 may run alongside item 1's rotation only
if they cost no extra card time. The brief's realistic ceiling for this hardware: 15-17 ms a step
(59-66 tok/s MTP off, 47-50 MTP on at 0.46 acceptance), ~8 s first token at 7K.

Two corrections after the supervisor checked the brief against the files (2026-10-28 13:00 UTC):
- **The pool's mechanism is not established.** Item 1(b) above repeats run 42-43's reading that the
  +4.5 GB is per-list scratch for spilling kernels. K1-out.txt shows every kernel the decode pass
  launches spills 0 B (gemv_i8a8, hc_read_a, hc_up_mix, the GDN update at KT=8; K1:388, 528-537); the
  28-70 KB spillers are variants that are never launched (the GDN KT=2 family; hc tiled is disabled at
  r4d_hc_bf16.cpp:3795). A2 ruled out kernel objects, and A3/A4 proved nothing because the placement
  keys are ignored by this driver. So the cost is per-list but its source is unknown. Before building
  1(b), measure it directly: VRAM per card against the number of live lists at fixed shapes (record the
  same pass as 1,182, ~300 and ~75 lists by changing the block size, VRAM sampled), which tells whether
  fewer, larger lists is the lever without needing the driver's sizing rule. Item 1(a), the drop policy,
  is unaffected: its effect is measured (H11/H11t).
- The brief's "int8 matrix peak 177 TOPS measured, not 367" is wrong: 176.9 was round 1's probe
  (round1/STATE.md:59); round 2 measured 366.9 TOPS at acc8 (round2/STATE.md:67). Use 367 for prefill
  floors. Line numbers in the brief drift (the drop rule is now around ctx.cpp:1400-1430).

## 53. Owner decision: the CPU computes the experts that miss VRAM (sized now, built after the heatsink) (2026-10-28 13:45 UTC)

On varied traffic a decode step misses ~78-88 expert units a card (H11/H17), each read by the GPU over
PCIe at ~14.5 GB/s and exposed at ~150-220 us inside the step (the brief's fitted figure): roughly 13-19
ms of an ~80 ms rotation step. The same 1.24 MiB unit read by the CPU from DDR4 (~40-45 GB/s typical for
this platform, unmeasured on intel-host) is ~30 us, and the result is a small vector. The owner wants this
lever for the rotation's tok/s. It needs 4-8 cores, which intel-host's heat cap (2) forbids until the heatsink
(Nov 1), so:

**Now (one cheap step, inside the heat rules, after the current item 2 card checks):**
1. intel-host's host memory bandwidth at 1 and 2 threads (a STREAM-style read of a 4 GiB pinned buffer in the
   step container, a few seconds; guard as usual).
2. The CPU time for one missed unit: libref's host `moe_gemm_q` (w4nl64a8h) on one routed row against one
   1.24 MiB unit, 1 and 2 threads, read from the pinned host pool the engine already keeps (not a copy).
3. From the rotation counters (H17), the misses a step per layer and their distribution across the 48
   layers (does a layer's misses arrive together, so one hand-off per layer suffices?).
Write the three numbers and the projected step for 2, 4 and 8 cores into STATE.md and the report's
"what remains" section, sized in runs.

**After the heatsink (milestone 4 or the end of this one if time allows):** a flag-guarded split in the
expert dispatch -- resident units on the card, missed units on host threads in parallel, the partial sums
added on the card before the layer's next op; the card never waits on the PCIe read for a unit the host
computes. Gate it with the oracle (tiny model), the 11-request agreement, and the rotation A/B at 8 CPUs
against H17. Expected on the rotation +15-25 % (35 -> ~40-43 tok/s); nothing on the single prompt (2.5
misses a step). Do not start the implementation on 2 CPUs.

**Amended by the owner (2026-10-28 13:55 UTC): build it now, test its speed after Nov 1.** Replace the
last sentence above: write the split now, behind a flag (default off, so the service is unchanged), and
take it through every correctness gate that fits 2 CPUs this week -- the CPU-device tests, the tiny
model's oracle with the flag on (all 256-row/expert classes the host path serves), and the 11-request
agreement with the flag on at 1-2 host threads (slow is fine; it is a correctness run). Then stage the
speed A/B for the heatsink day: the 6-minute rotation in pieces (H17's form) with the flag off and on at
8 CPUs, plus the single prompt both ways, and a host-thread count sweep (2 / 4 / 8). No speed A/B, burst
test or CPU-ramp exception before the heatsink is in. Order: after item 2's card checks and the three
measurements above; it is worth more to the rotation than the pool-mechanism measurement, so it comes
before that.

## 54. Owner decision: fill the idle Xe-cores in decode (after the hc rewrite) (2026-10-28 14:05 UTC)

The bottleneck brief shows decode kernels that use part of the card. Queue these behind note 52 item 2
(the hc rewrite, which already includes "all 8 waves streaming"), each as a flag-guarded A/B on the
single prompt (MTP off and on, M-pair as M25/M26) with the oracle on the tiny model:
1. **GDN recurrent update: more work-groups.** 24 WGs on 32 Xe-cores, 57.9 us a launch, a serial chain
   inside each WG (brief section 1.2 row 4). Split the state by head and row block so >= 64 WGs run, and
   shorten the in-WG chain (barriers, the two LDS reductions). Target ~1 ms of the 2.08 ms a step.
2. **Expert GEMV grid shape.** gate_up runs 80-120 WGs = 1.25-1.9 waves on 64 slots (brief row 3 /
   ranked #7): try whole-wave grids (64 / 128 WGs) and more rows or K per WG for the down class. Target
   0.5-1 ms; it is also the first step of ranked #4 (the 4-row MTP expert form).
3. **Independent ops on the second lane.** The engine already records two lanes (the draft passes). Try
   the shared expert (and, if independent, the protected bf16 expert pair) on lane 1 alongside the routed
   experts within a layer, joined before the layer's sum; then the attention side-chain's small kernels.
   Measure that the tape keeps one list per lane per stretch (no new segmentation). Target 0.5-1 ms.
Together ~2-3 ms of the 27.9 ms step (+3-4 tok/s single user), stacking with the hc rewrite. Commit only
on token-identical tiny-model runs and a step that moved; record a null result as such.

## 55. Re-baseline the single-prompt decode on the current image (owner, 2026-10-28 14:15 UTC)

The single-prompt decode figures in progress.tsv (35.7 MTP off, 42.0 MTP on) are from image :36 (run 33);
since then the floor, gate, coupling, drop policy, matrix kernels and the architecture plugin itself
(:54) have changed, and no clean MTP-off/on pair exists on a current image. In the next card window,
before or with item 2's checks: the greedy single prompt (M25/M26's prompt and args) MTP off and MTP on,
on **:60** (the current service image without the E4M3 change) and on **:61** (with it), 2 runs each, step
lines and tok/s. That gives (a) the regression check of everything since :36 and (b) item 2's first
before/after. Update progress.tsv's two decode rows from it. From here on, every decode-path commit gets
the same M-pair on its image before it is called done.

## 56. Owner decision: expand the HyperConnection weights to FP16 once at load, as an A/B against the rewrite (2026-10-28 17:00 UTC)

The hc mixes are 8.0 ms of the 27.9 ms single-prompt step (brief section 1.2 row 2) and read E4M3
weights that every launch decodes value by value. Every E4M3 value is exactly representable in FP16, so
expanding the hc weights to FP16 **once, when the model loads** changes no number the kernels compute,
and turns the hc launches into plain FP16 GEMVs with no decode step. Try it as one A/B, after AG2 and the
rotation on :64, before the CPU-expert prototype:
1. A load-time option (flag, default off): the hc mixer weights (`hc_read_a` / `hc_up_mix` inputs and
   whatever `hc_write` reads, replicated per rank) are expanded E4M3 -> FP16 into a new VRAM buffer at
   load, the E4M3 originals released; the hc kernels get an FP16-weight path (the existing f32-accumulate
   bodies with the decode removed; keep the norm and all other arithmetic in the same order).
2. Correctness: bit-identical outputs on the tiny model with the flag on and off (the expansion is exact,
   so any difference is a bug), the CPU tests, and the 7K first-token logprobs equal to the flag-off run.
3. Cost and gain: VRAM per card before and after (expect ~+0.7 GB, i.e. ~550 fewer expert slots at the
   4 GiB headroom), the expert residency at load, then the M-pair (single prompt MTP off and on) and one
   6-minute rotation in pieces, flag off vs on on the same image.
4. Keep it only if the single-prompt step falls and the rotation median does not drop (the residency cost
   shows up there first); otherwise record the numbers and leave it off. If it wins, it can replace or
   simplify item 2's remaining hc rewrite -- say which in STATE.md rather than doing both.
Expected: the hc cluster from ~8 ms toward ~3 ms (FP16 GEMV at ~450 GB/s on twice the bytes), i.e. about
+5-7 tok/s MTP off and a few ms off the MTP step.

## 57. Owner decision: the new protected-expert kernel is the default (2026-10-28 17:25 UTC)

The owner read run 68's table (rotation median 35.5 -> 41.7, single prompt 34.9/45.8 -> 37.4/51.7;
agreement 2 of 11 identical answers against 3, the one change a near-tie at word 3 of the colors request
with 0.45 / 0.17 nat margins; texts 3 of 7 and tool calls 2 of 3 unchanged) and chose the speed, as in
note 49. Do both adoption steps:
1. Make the new form the code default (98ae19b's switch stays, so the old form remains one flag away),
   commit, and set it explicitly in `m3/service/run.sh` too, so the service does not depend on a default.
2. Rebuild the service image; on it, the M-pair (single prompt MTP off and on) and one 6-minute rotation
   in pieces as H20. If they reproduce run 68's figures, move the service image and the stage D unit to it.
   Record the near-tie in the report's agreement section beside note 49's.
Fit it in alongside note 56's FP16 work (same window if the rebuild allows); it needs no new code.

## 58. Owner decision: m3 fault 14 (H21's out-of-memory at the VRAM floor) -- fix in code, keep the 1 GiB floor (2026-10-28 20:45 UTC)

The owner read the pause and chose proposals 1 and 2 together, in this order:
1. Under the floor, drop stale tapes (least recently played first) until free VRAM is back above the
   1 GiB floor, instead of keeping every tape played in the last 2,048 passes while the card sits below it.
   Keep 89bf7fb's rule when the card is above the floor. Do not raise the floor to 2 GiB unless this
   is not enough; if it is not, say so in the report with the VRAM trace, then use the 2 GiB config.
2. A failed play append (zeCommandListImmediateAppendCommandListsExp out of device memory) must not hang
   the engine: both ranks issue that pass live, the tape is dropped, and the step continues. Log it once
   per tape. A test that forces the failure (a fake append error on the tiny model) proves both ranks
   agree and the text is unchanged.
3. Rebuild, then repeat M37's M-pair (it must stay at 37.3 / 51.8 with the same text) and H21 twice, plus
   H22 (note 56's rotation with FP16 hc). Record free VRAM through each rotation.
Note 56's decision waits for H22 against the repeated H21: M36 gave MTP off +6.6 % and MTP on -2.9 %
with ~500 fewer resident expert units, so the rotation (MTP on) decides whether FP16 hc stays.
Note 57's service-image move waits for a clean H21 on the rebuilt image.

## 59. Owner decision: an engine-owned submission backend for the B70 ("xe direct"), phased, design first (2026-10-28 23:20 UTC)

The tape VRAM problem (m3 faults 7 and 14, the floor, the drop policy, the ~4.5 GB a card behind the
recorded command lists that a dropped tape never returns) keeps coming back. The owner wants it settled the
way the AMD build settled its launch path in core/device/aql.cpp: the engine owns submission and the
memory of its recorded passes, and Level Zero keeps only what it must (compiling and loading kernels,
device properties). Finish run 74's items in flight first (the M-pair at the 2 GiB floor, H22 for
note 56, note 57's image move); then this goes ahead of the CPU-expert split, the occupancy items and the
hc / GDN work.

Phase 0 -- design brief, no new card work beyond read-only probes. Cap $40, then STOP for the owner.
1. Measure first what the driver holds per recorded command list (the pool measurement from note 52) and
   whether NEO's direct submission is active for our immediate lists on the B70 under xe; both feed the
   design.
2. Work out the minimum the engine needs from the xe kernel interface (the render node we already use):
   VM create and bind, exec queue create, exec with a sync, user fences; and the Xe2 commands it needs
   to write (compute walker, state base address, pipe control, batch buffer start, semaphore wait), with
   NEO's open-source Xe2 command definitions as the reference. How a kernel binary built today (SYCL
   AOT) is loaded and bound outside Level Zero.
3. Write briefs/<date>-xe-direct-design.md: the design (a new backend beside l0.cpp in the same dispatch
   table; l0.cpp stays the default and the fallback), what a recorded pass becomes and how freeing it
   returns its memory, the gates below, the risks (hangs, generation-specific commands), and an estimate
   of runs and dollars for phases 1-2. Also the expected tok/s, from measurements, not guesses: (a) the
   rotation with the recovered VRAM (more experts resident, no floor; M36's -500 experts gave ~+4 ms on
   the MTP step as a slope), (b) the single prompt from the gaps between dependent kernels (a timeline
   of one decode step: how much of 27 ms is idle between kernels that lighter barriers could recover).
   Then STOP: the owner approves the budget before phase 1.

Phase 1 (after approval) -- a standalone proof on card1 only: own VM, one buffer, one kernel, one
submission, checked result; then a GPU-polled ring (no system call per submission), timed per dispatch.
Phase 2 -- the backend: the tiny model's E-tests bit-identical; Flash-Next M-pair text identical to M37;
two clean rotations with a VRAM trace; the target is no recordings floor, or one well under 1 GiB.

Rules that do not change: intel-host-engine-guard and the 75 / 80 C heat rules; card1 only until phase 2's
gates pass on it; no kernel module, driver, firmware or root change (hand those to the owner); a GPU hang
the xe driver recovers is a fault row like any other, a card that needs a reboot pauses the runner for
the owner; production and the vLLM rollback untouched. If a new build script or allow rule is needed,
write it as a request for the owner, do not change .claude/settings.json.

## 60. Via Telegram (owner, 2026-10-28 23:43:15 UTC)

resume

## 61. Owner decision: xe direct phases 1 and 2 approved at $320 (2026-10-29 00:00 UTC)

The owner read the phase 0 brief (briefs/2026-10-28-xe-direct-design.md) and approved phases 1 and 2
together, $320 in all, so the runner does not stop between them. Note 60 ("resume") meant the same.
Order and gates:
0. First, account for the full ~4.5 GB a card the recordings hold. X1 measured 1.19 MiB a list, and
   lists of up to 128 launches reused that memory, so ~1,182 lists explain only ~1.4 GB of A1's 4.5-4.8
   GB. Measure, in the engine on card1, the VRAM (fdinfo) before and after each recording, split by what
   allocated it (the lists, the tape layer's own buffers, the USM pool). If less than ~2.5 GiB of it is
   memory xe direct would own and free, stop and tell the owner with the numbers before writing the
   backend: brief section 6(a)'s +10-15 % depends on it.
1. Phase 1 as the brief's section 4 (card1 only): proof (i), the ring (ii), the hang test (iii).
2. Phase 2 as section 4, card1 first, then both cards.
Stop for the owner also if: two GPU hangs in one run, a card that needs a reboot, the per-dispatch numbers
from (ii) show no gain over NEO's in-order lists and section 6(b) goes to zero, or spend passes $320.
Note 59's rules stay. Report the result as its own section of the m3 report.

## 62. Owner decision: resume xe direct phase 1 on documented interfaces only (2026-10-29 01:55 UTC)

The owner read run 77's pause and chose to resume phase 1, with these limits on method:
1. Work only from documented or published sources: the zebin format (sections, `.ze_info`, relocation
   entries) as Intel's compiler and runtime define it, and the Xe2 command definitions and field
   values in NEO's open-source code. Do not scan or copy this process's memory to find the runtime's
   own copies of kernels or commands (X8's capture mode): remove that mode from xe_direct_proof.cpp
   before committing.
2. Next: apply the zebin relocations to the ISA copy (the two __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0
   entries), set the walker and state fields as NEO's source sets them for this kernel (including
   ComputeOverdispatchDisable), rerun `xe_direct_proof 0 once` on card1. Commit when (i) passes, then
   (ii) and (iii) as planned.
3. Re-check the brief's 6(a) revision to +5-8 % before relying on it. The traces show ~25.5 GiB used a
   card after load (V2 / V3, ~7.2 GiB free of 32 GiB), not just the 4 GiB headroom; with tapes off
   (V3) ~5.3 GiB stays free after the first prompt's 1.8 GB. Find where that extra ~3 GiB is budgeted
   (the plan's reservations, KV, the tape pool allowance) and state how much of it the experts could
   take once recordings are MBs; correct 6(a) either way with the arithmetic shown.
If a response is interrupted again, pause as in run 77 and say which step it was on.

## 63. Owner decision: xe direct deferred; resume the rest of the queue (2026-10-29 02:05 UTC)

The owner read run 78's pause and deferred xe direct. Do no further xe direct work in the unattended
runner: no edits to its files, no card steps for it. Keep its uncommitted files as they are (do not
delete them) and record in STATE.md that phase 1 stopped at proof (i) with about $30 of note 61's $320
spent. Note 61's remaining budget is released back to the milestone.
The service keeps the 2 GiB recordings floor (note 58), which is stable.
Resume the order that stood before note 59: note 53's CPU-expert split build with its correctness gates
(the speed A/B waits for the heatsink, Nov 1), note 54's occupancy items, the hc rewrite remainder, the
GDN chunk scan, then the m3 report. Record the phase 0 brief and the item-0 measurements (V2, V3, H26)
in the report as findings; xe direct is listed there as deferred, with the brief's estimate and the open
question on the ~3 GiB beyond the headroom (note 62 item 3).

## 64. xe direct closed (supervisor, for the owner, 2026-10-29 02:20 UTC)

The attended attempt in the supervisor's session was also stopped by a safety classifier. As agreed with
the owner, xe direct is closed: no further work on it in any session. Note 63's queue stands.

## 65. Owner decision: cut the peak of recorded lists alive, ahead of the CPU-expert split -- as soon as possible (2026-10-29 02:45 UTC)

Read briefs/2026-10-29-xe-direct-findings.md section 6 first. The driver keeps the high-water mark of
command lists alive at once (~1.2 MiB each, X1 / V2) for the life of the process and reuses it. In H26 the
peak was 3,636 segment lists (+4.9 GiB a card) while the steady state needs 542 composed lists: the
segments of several new shapes' recordings stay alive together until each tape's first play composes them.
1. Park note 53's CPU-expert split at the next clean point: commit what builds and passes its CPU tests as
   opt-in (RADIANCE_HOST_EXPERTS stays off by default), or keep it uncommitted with a STATE.md line saying
   where it stopped. It resumes after this.
2. Find out why composition waits for the first play, and whether a tape can be composed (and its segments
   released) when its recording ends instead. If it can, do that; if not, bound the recordings in flight at
   once (segments not yet composed) so one tape's segments are released before the next tape records. Keep
   either behind a flag with the old behaviour one setting away; make it the service's setting only after
   the gates below.
3. Gates, with the VRAM trace (RADIANCE_L0_VRAM_TRACE): the tiny model's tests and the M-pair text
   identical to M43 / M44; the single prompt's peak lists alive and VRAM growth against V2 (2,265 / 2.74
   GiB); H26's rotation again: target peak about 900 lists or fewer, VRAM growth about 1.2 GiB or less,
   median not below 41.5, no out-of-memory, no retry. Report the size of a composed list.
4. If the gates pass: lower the recordings floor and the headroom by what the trace shows is no longer
   needed, give that VRAM to the expert pool, and run two rotations (H25's form) for the new median and
   misses a step. Also find where the ~3 GiB free after load beyond the 4 GiB headroom is budgeted (note 62
   item 3) and whether experts can take it.
5. Then measure (no build) how far below ~1 GiB this can go. The floor of this route is lists kept x
   ~1.2 MiB, and H26's steady state keeps 542 composed lists for ~11 tapes (~49 played ranges a tape).
   Report the played ranges per tape, why each tape is played in that many ranges (MTP acceptance, live
   passes between, host waits), and how many lists would remain if each tape were played whole, plus
   whether NEO's per-list allocation (~1.2 MiB for an empty list) has a documented setting that shrinks it.
   Write it as a proposal for the owner; do not build it without a note.
The heat rules, card windows and production rules are unchanged. Report it as its own section of the
m3 report, beside the xe direct findings.

## 66. Owner decision: the hc rewrite is second, right after note 65 (2026-10-29 03:19 UTC)

Order from now: (1) note 65 (the peak of recorded lists alive, its gates and item 5's measurement);
(2) **the hc decode rewrite** -- briefs/2026-10-28-bottleneck.md item #3: hc_read_a 4.21 + hc_up_mix 3.43
+ hc_write 0.37 = 8.01 ms a step at 81-96 GB/s, against 232-242 GB/s measured at the same E4M3 shapes on
this card (floor ~3.2 ms). Clean GEMV forms (norm once, 8 waves), an SLM E4M3 decode table instead of the
scalar convert, read_a and up_mix fused so the weights are read once, hc_write folded into the
all-reduce; the 4-row (MTP) forms too. Behind a flag first; gates: the tiny model's tests bit-identical
(or the documented near-tie rule of notes 49 / 57), M-pair text identical to M43 / M44, then the M-pair
speed and one rotation; then it becomes the default. Target: hc <= ~4 ms, MTP off >= ~42 tok/s.
(3) note 53's CPU-expert split resumes (its speed A/B still waits for the heatsink, after Nov 1);
(4) note 54's occupancy items; (5) the GDN chunk scan; (6) the m3 report.

## 67. Run 79 ended early by the supervisor for note 65 (owner, 2026-10-29 03:30 UTC)

The owner wants note 65 (the tape peak) started now, so the supervisor ended run 79 between steps, with
no card step or build running. That run is not a fault; its strike is the supervisor's. It had passed the
CPU split's gates (E16, M45, M45b on image :74) and was editing src/libxe/xe_moe_planes.cpp (note 54's
expert grid: rows per work-group, template parameter R). First, before note 65: look at that uncommitted
edit; if it builds and the CPU tests pass, commit it behind its flag, off by default; if not, keep it
uncommitted and write in STATE.md where it stopped. Record run 79's results (E16, M45, M45b) in STATE.md
if they are not there yet. Then note 65, then note 66's order.

## 68. Owner decision: make the early release and the 64 KB command buffers permanent if H28 passes (2026-10-29 04:35 UTC)

If H28 (RADIANCE_L0_TAPE_EARLY=1 with NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64) is clean --
no out-of-memory, no retry, no lost segment, median within noise of H25 / H27b, recordings' VRAM clearly
below H27b's +1,503 MiB -- then:
1. Check the M-pair under the same two settings: text identical to M43 / M44, speed within noise.
2. Make RADIANCE_L0_TAPE_EARLY=1 the code default (=0 keeps the old release, one setting away), commit.
3. Set both NEO settings explicitly in m3/service/run.sh (and wherever the stage D unit's environment
   lives), rebuild or re-tag the service image, and move the unit's digest to it, as for note 57.
4. Because OverrideCmdListCmdBufferSizeInKb is a NEO debug key, not a supported setting: at engine start,
   log whether it took effect (the VRAM trace's per-list cost on the first recordings, or NEO's own echo
   of the key), and keep running normally if it did not -- the 2 GiB floor still protects the card then.
   Record the NEO version it was measured on (1.14.37020) in the report; recheck it after any driver or
   runtime update before trusting it.
5. Then note 65 item 4: lower the recordings floor and the headroom by what the trace shows is no longer
   needed, give that VRAM to the expert pool, two rotations for the new median and misses a step.
If H28 is not clean, keep TAPE_EARLY alone (H27b passed its safety gates) and say why in STATE.md.
This is the B70 test service on intel-host only; amd-host's production is not touched.

## 69. Owner decision: test Intel's current GPU runtime and compiler, after note 68 and before the hc rewrite (2026-10-29 04:40 UTC)

Our images carry NEO 26.05.37020.3 (Feb 2026) with IGC 2.28.4; Intel's latest release is NEO
26.35.39758.10 (2026-09-17) with IGC 2.41.5 (github.com/intel/compute-runtime releases; Battlemage's
Level Zero support 1.14 -> 1.17). The release notes carry no changelog, so measure. Order: note 68's
adoption (and note 65 item 4) first, then this, then note 66's hc rewrite -- so the rewrite is tuned on the
runtime and compiler we will ship.
1. Runtime only: a test runtime image with NEO 26.35.39758.10 (its .debs from the release page, sha256
   checked against the release's sums), the same libxe build. On it: the list probe (X1's form: MiB a
   list, whether destroy returns memory, whether OverrideCmdListCmdBufferSizeInKb=64 still applies), the
   tiny model's tests, the M-pair (text identical to M43 / M44), one traced rotation against H28.
2. Compiler too: a test build image with IGC 2.41.5 (and the matching ocloc), the kernels rebuilt AOT. On
   it: the tiny model's tests bit-identical; the M-pair text and speed; one profiled decode step against
   P3 for hc_read_a, hc_up_mix and the expert GEMVs (does the new compiler alone move them); one rotation.
   A text or bit change is a finding: record which kernel, do not adopt until explained.
3. Also measure, on whichever runtime wins, NEO's ForceDefaultHeapSize (default 4 MB for the indirect
   object heap) with the list probe: does a smaller heap cut the per-list cost further without errors.
4. Adopt a newer runtime or compiler for the service only if its gates pass and it is no slower; keep the
   current images as the fallback and say which was chosen and why. Images and packages live inside
   containers only: no host package installs, no driver or kernel changes (those are the owner's).
Watch amd-host's disk (55 GB, the 50 GB floor): remove superseded test images as you go.

## 70. Owner: intel-host's GuC firmware update 70.58.0 -> 70.72.1, then a re-baseline (2026-10-29 04:45 UTC)

The owner is updating intel-host's xe/bmg_guc_70.bin (GuC 70.58.0 -> 70.72.1, from linux-firmware) and
rebooting intel-host; the kernel stays 7.2.6 (no newer prebuilt mainline build exists). The owner pauses the
runner for it and resumes it after. At the first run after the reboot, before other card work:
1. Read intel-host's kernel log for the GuC version on both cards (journalctl -k -b, "Using GuC firmware");
   record it in STATE.md. If it is still 70.58.0, say so and carry on as before.
2. If 70.72.1: re-baseline on the current service image -- the tiny model's tests, the M-pair (text
   identical to M43 / M44, speed within noise), one rotation in H25's form -- and check the kernel log
   for any new xe warnings or faults. A text change, a fault, or a clear slowdown: pause for the owner
   with the numbers (the rollback is the owner's: the backup firmware file is
   /lib/firmware/xe/bmg_guc_70.bin.zst.70.58.0).
3. Then continue the order: note 68, note 69, note 66.

## 71. Owner: the order from here (2026-10-29 05:05 UTC)

1. Finish note 70's re-baseline on GuC 70.72.1 now (the M-pair spaced for heat, then one rotation) --
   this is the owner's "tok/s right now".
2. The tape: note 68's adoption (early release + 64 KB command buffers as the service's setting, the
   startup check), then note 65 item 4 (lower the floor and headroom, the freed VRAM to the experts, two
   rotations).
3. Note 66's hc rewrite (the E4M3 decode in hc_read_a / hc_up_mix: 8.0 ms a step toward ~3.2-4).
4. Note 53's CPU-expert split: its remaining steps and gates; the speed A/B still waits for the heatsink.
Note 69 (the newer NEO and IGC) moves after item 4.

## 72. Owner: CPU boost is off on intel-host until the heatsink; resume note 70's re-baseline (2026-10-29 05:11 UTC)

The owner turned off intel-host's CPU boost (/sys/devices/system/cpu/cpufreq/boost = 0) to stop the
single-core spikes that trip the 75 C stop during model loads; it resets to on at the next reboot. The
heat limits are unchanged. The supervisor's reading of the pause: the stops are not new since the
reboot -- M38-M40, H27 and H28 (80 C at 04:43) stopped before it -- so this is intel-host's marginal cooling,
not the GuC update.
1. Check the boost value at the start of each run and write "boost off" (or on) beside every speed
   result from now on, so results compare fairly; if it reads 1 after a reboot, say so in STATE.md.
2. Keep spacing model loads (cool to <= 45 C first).
3. Resume note 71's order: note 70's M-pair and rotation (compare against M43 / M44 / H25 and note the
   boost and GuC differences), then the tape adoption, the hc rewrite, the CPU split.

## 73. Owner: R2c's hang -- a card0 health check, then the baseline (2026-10-29 05:28 UTC)

The supervisor's reading, which the owner accepted: R2c is m3 fault 6's family, not the GuC update -- the
heat-stop kills of R2 and R2b left card0's copy engine with a stuck job ("Fault response: Unsuccessful
-EINVAL"), R2c's load waited behind it (2 of 3 at the first step's barrier), and its timeout kill logged
the CAT:bcs. The same sequence happened on GuC 70.58.0 (faults 3, 5, 6, 9-11). Write the reviewed-fault
line for it as that family.
1. First, card0 alone: the tiny model (m3-e2e.sh's native form, 1 CPU, on card0). Clean -> 2. If it
   hangs or the kernel log shows a new fault: pause for the owner (a intel-host reboot clears card0; boost must
   be turned off again after it).
2. Note 70's M-pair (MTP off, then MTP on, cooled and spaced), then one rotation; boost off, GuC 70.72.1
   beside each result; compare with M43 / M44 / H25.
3. Then note 71's order.

## 74. Owner: intel-host rebooted to clear card0; boost off permanently until the heatsink (2026-10-29 05:50 UTC)

The owner rebooted intel-host at 05:45 to clear card0's stuck copy job (R5's hang) and installed
cpu-boost-off.service (enabled), so boost reads 0 after every boot until the heatsink; the owner
removes it then. Resume note 73: the card0 health check (R5 again, fresh after the reboot), then note
70's M-pair and rotation (boost off, GuC 70.72.1), then note 71's order. Record R5's hang as m3 fault
6's family (reviewed-fault line), cleared by the reboot.

## 75. Owner: test GuC 70.72.1 with a 100 us compute timeslice before any rollback (2026-10-29 06:06 UTC)

The supervisor's reading of the pause: X10's ~1.3 ms per exchange is about one 1,000 us timeslice
(intel-host's /sys/class/drm/card*/device/tile0/gt0/engines/ccs/timeslice_duration_us = 1000, the xe default,
unchanged by the update) -- the new GuC appears to hold the spinning exchange kernel's queue for a full
quantum where 70.58.0 did not. The owner sets timeslice_duration_us = 100 on both cards' ccs engines
(root, resets at reboot). Then:
1. At the run's start, read timeslice_duration_us on both cards and record it with the GuC version and
   boost; if it is not 100, pause and say so.
2. X10 (the two-rank all_reduce test, no model). ~90-200 us a launch pair = the timeslice was the cause:
   go on to 3. Still ~1.3 ms: pause for the owner (the rollback to 70.58.0 is theirs).
3. Note 70's M-pair (MTP off, then on, spaced) and one rotation, against M43 / M44 / H25, with GuC
   70.72.1, boost off and timeslice 100 us beside each result. Watch for any cost of the shorter slice
   (prefill time-to-first-token, the rotation's median).
4. If the M-pair and rotation are within noise of before: record it as the configuration (the owner will
   make the timeslice persistent in intel-host's boot service) and continue note 71's order. If clearly
   slower: pause with the numbers.

## 76. Owner: GuC 70.72.1 rolled back; pin the stack and add a startup self-check (2026-10-29 06:22 UTC)

GuC 70.72.1 is ruled out for this engine: X10 / X11 measured ~1.3 ms an exchange (89-101 us on
70.58.0) at either timeslice, and X11's "peer dt2 numel 65536" case gave rank-divergent results (stale
peer data). The owner reinstalled 70.58.0 (from linux-firmware commit 86725d73, sha256 of the raw file
cc5dbf7d...ef95; the earlier ".70.58.0" backup on intel-host was really 70.72.1 and has been replaced) and
rebooted. Boost stays off (cpu-boost-off.service); the timeslice is back to its 1,000 us default.
1. At the run's start: confirm "version 70.58.0" on both cards in intel-host's kernel log, boost 0. If not,
   pause and say so.
2. X10 again: every case must pass and a launch pair must be back at ~90-100 us. Then note 70's
   M-pair and rotation (boost off, GuC 70.58.0 beside each result) against M43 / M44 / H25.
3. Write STACK.md at the repo root: the validated stack -- DPC++ 7.2.0 (build image), IGC 2.28.4, NEO
   26.05.37020.3, Level Zero loader 1.29.0, kernel 7.2.6-070206, GuC 70.58.0, HuC 8.2.10 -- each with the
   date and the steps that validated it, and the known-bad entry: GuC 70.72.1 (X10 / X11, 2026-10-29;
   70.60 / 70.65 untested). Any update to a line goes through a test like note 69's first.
4. A startup self-check in the engine on Level Zero, before serving: log the GuC version (from the
   kernel's sysfs / debugfs where readable, else "unknown"), NEO and IGC versions; then a short two-rank
   exchange self-test (X10's peer and mailbox forms at two sizes, a few iterations, under a second): exact
   results and a time per exchange under a set bound (say 300 us). On failure: log why and refuse TP2
   serving with a clear error (no silent fallback). Behind a flag first, then on in the service once it
   passes on 70.58.0 with no measurable startup cost beyond ~1 s. Fits beside note 68's startup check of
   the 64 KB setting.
5. Draft (do not send -- nothing is posted anywhere) a short Intel bug report in briefs/: hardware,
   kernel, NEO / IGC, the two GuC versions, X10 / X11's numbers, the divergent case, and a minimal
   reproducer description; the owner decides whether to file it.
Then note 71's order (tape adoption, hc rewrite, CPU split), with items 3-4 done before the tape
adoption's service change.

## 77. Owner: xe driver tuning -- a GPU clock-lock A/B after note 76's baseline (2026-10-29 06:29 UTC)

The cards are different boards: card0 (0b:00.0) ASRock B70, stock cap 275 W; card1 (0f:00.0) Intel's
reference B70, stock cap 230 W. **card1's power cap stays at 230 W (owner: it cannot go up); no power cap
changes on either card.** The one driver setting to test is the clock floor: today both cards run
min_freq 1200 / max_freq 2800 MHz and drop toward 1200 in idle gaps.
1. Arm A = note 76's baseline (M-pair, rotation, first token at 7K), as measured now.
2. After it, pause with "ready for arm B" in PAUSED: the owner sets min_freq = 2800 on both cards
   (root sysfs, resets at reboot) and resumes.
3. Arm B: at the run's start, read min_freq / max_freq / act_freq on both cards and record them; then the
   same M-pair, rotation and first-token measure, plus the cards' own temperatures (hwmon) and power
   during the rotation. Report A against B per number with the noise band; keep B only if it is faster
   beyond noise and the cards stay within their normal temperatures. The owner then adds it to intel-host's
   boot service, or sets min_freq back to 1200.

## 78. Correction to note 77: the clock lock is already on -- measure arm B first (2026-10-29 06:30 UTC)

The owner set min_freq = 2800 on both cards already (read 06:30: 2800 / 2800). So swap note 77's order:
note 76's checks (GuC 70.58.0, boost 0, X10) and then the M-pair, rotation and first token **with
min_freq 2800 = arm B**. Record min_freq beside every result. Then pause with "ready for arm A": the
owner sets min_freq back to 1200 on both cards and resumes; then the same measurements = arm A, and the
comparison. Everything else in notes 76-77 stands.

## 79. Clock floor is back at 1200 -- whatever is set is the arm (2026-10-29 06:32 UTC)

The owner set min_freq back to 1200 on both cards at ~06:31, after note 78 was written. Rule for notes
77-78 from here: the arm is whatever min_freq reads at the time of each measurement -- 1200 = arm A,
2800 = arm B. Record the reading beside every result. Measure the arm that is set now (A), then pause
with "ready for arm B (set min_freq 2800 on both cards)"; the owner changes nothing until that pause.

## 80. Supervisor: the slow exchange is not the GuC -- find what the reboots changed (2026-10-29 06:37 UTC)

X12 / X12b on GuC 70.58.0 (06:3x) still give ~1.3 ms a launch pair, and R6 hangs at the first step as
R2c / R2d did. So note 76's reading (GuC 70.72.1 = the cause) is wrong for the speed; the divergent case
in X11 stays an open question on 70.72.1. Every ~1.3 ms measurement so far was taken with CPU boost OFF
(notes 72 / 74), after one or more reboots; the last fast in-engine exchanges were before the 04:54 reboot
(M43 / M44 / H25-H28: a 26.5 ms decode step with 97 exchanges cannot hold 1.3 ms each). Diagnose before
any more model loads:
1. Pause with "ready for a boost-on X10": the owner sets boost to 1 on intel-host for one X10 (no model, about
   a minute, no heat risk), then back to 0. If X10 drops to ~90-100 us with boost on, the exchange's host
   side depends on the CPU's clock behaviour (find which wait -- a sleep, yield or timer-bound poll -- and
   fix it in the engine so boost off cannot starve it).
2. If boost on does not help: compare host settings against what the pre-reboot runs logged (serve.log
   headers, ri-sample output: CPU frequency, governor, EPP, cpuidle, ASPM, ulimits, memlock, hugepages,
   NUMA, container CPU pinning), and the engine's own per-exchange timing in the pre-reboot serve logs
   (H28 / M47 / M48) against R6's; report the first difference that explains it.
3. Write the conclusion in STATE.md and note 76's STACK.md (correct the GuC 70.72.1 entry to what is
   actually shown: the divergent peer case only, speed not attributable).
No more Flash-Next loads on two cards until X10 is back near 100 us.

## 81. Owner away: keep working without owner actions; this replaces note 80's pause (2026-10-29 06:40 UTC)

The owner is away and cannot run root commands on intel-host. Do not pause to ask for owner actions (no
"ready for a boost-on X10", no "ready for arm B"). Create PAUSED only for UNATTENDED.md's four hard
conditions (a wedged card, a card not recovered after 10 minutes, a fault recurring after its
fix-claimed line, two faults in one run) or if the milestone spend reaches $800. Boost stays off and
min_freq stays as it is; record both beside every result.
Order:
1. The slow exchange, root-free: no two-card Flash-Next loads until X10 is back near 100 us a launch
   pair. Diagnose with X10 variants only (no model, no heat risk): time each phase of a launch pair (host
   submit, GPU kernel, the peer write's visibility, the host's wait) to find where the ~1.3 ms goes; read
   the engine's wait paths for anything bound to a timer, sleep, yield or the scheduler tick (CONFIG_HZ is
   1000 on intel-host); try fixes in the engine (a pure spin on the host side, the waits without sleeps, a
   different wait primitive) and keep any that bring X10 back with boost off. Compare host settings
   against the pre-reboot runs' logs (note 80 item 2). Commit fixes behind a flag first.
2. If X10 is fixed: note 76 items 1-2's baseline (M-pair, rotation, first token) with the settings as
   they are, then note 68's tape adoption and note 65 item 4 (freed VRAM to the experts), then note 66's
   hc rewrite, then note 53's CPU split steps -- note 71's order. The clock-lock A/B (notes 77-79) waits
   for the owner.
3. If X10 cannot be fixed root-free: write what was found and what root step would test it in
   WAITING-FOR-OWNER.md (not PAUSED), then do the work that needs no two-card model load -- note 76's
   STACK.md (correct the GuC 70.72.1 entry per note 80) and the startup self-check (behind a flag), the
   hc rewrite's kernels with the tiny model's bit-identical tests and single-card op checks, the CPU
   split's remaining code with its CPU tests -- and leave the two-card speed checks queued.
Keep the daily runs going; the owner will read Telegram and WAITING-FOR-OWNER.md on return.

## 82. Supervisor, with the owner away: debug the first-step hang with boost off -- it reproduces (2026-10-29 06:45 UTC)

Run 86's pause is answered by note 81: no boost-on test while the owner is away. The agent's finding
stands (runs 84-85 compared against milestone 2's build of the test; the GuC is not implicated; every
hang has boost off). A hang that reproduces every time is the best condition to find the race in, so:
1. Debug the first-step hang with boost off as it is: trace the cross-rank handshake and the step
   barrier (which rank thread does not arrive, where it waits, on what flag or event, with which
   generation counters), using the tiny model at --tp 2 if any config allows it, else Flash-Next with one
   load at a time (the hang costs a load, not heat: R6 peaked at 51 C), plus X12 for the divergent
   peer-form case.
2. Fix it in the engine so a slower host cannot deadlock or read stale data (ordering, fences or
   visibility on the peer writes and the mailbox, no timing assumptions); behind a flag first; prove it
   with X12 clean and the M-pair completing with boost off.
3. Then note 81 item 2's order. GuC stays 70.58.0 for now (validated for the whole milestone); a clean
   70.72.1 A/B can come after the fix, when the owner is back.

## 83. Owner: GPU min_freq is 2800 on both cards again (2026-10-29 06:44 UTC)

The owner set min_freq back to 2800 on both cards (the clock lock, arm B of notes 77-79). Leave it; record
min_freq beside every result. Results from here are arm B; the arm A numbers (min_freq 1200) are still
to be taken when the owner is back -- note that in STATE.md, and do not change the setting.

## 84. Owner: resume with the engine-side prebind (2026-10-29 12:54 UTC)

The owner read run 87's pause and WAITING-FOR-OWNER.md and chose the root-free route. Faults 19-21 are
reviewed as the agent described (19/20 kill artifacts on card0, cleared without a reboot; 21 caused by
NEO's MakeEachAllocationResident=2, never to be used again). Settings unchanged: boost 0, GuC 70.58.0,
min_freq 2800 on both cards (arm B), timeslice 1000.
1. Build RADIANCE_L0_HOST_PREBIND=1 as WAITING-FOR-OWNER.md describes (after the weights load and before
   the step loop: one small copy from every host allocation on every card's copy queue, one card at a
   time, each logged first), off by default.
2. One two-card load with it: if a bind still fails, the log names the allocation and order -- report and
   try the next engine-side change (for example making card1's imports resident at import time, or giving
   each card its own host allocations instead of importing card0's). If none fails and the first step runs:
   note 82's proof (X12, the M-pair completing with boost off), then note 81 item 2's order (baseline,
   tape adoption with the freed VRAM to the experts, hc rewrite, CPU split).
3. Notes 81-83's unattended rules stay (pause only on the hard conditions or at $800).

## 85. Owner: a root-side EBUSY trace runs on intel-host during the next hang (2026-10-29 13:07 UTC)

The owner runs ~/ri-ebusy-catch.sh as root on intel-host (bpftrace waits for the first xe_vm_bind_ioctl that
returns -16, then trace-cmd records 3 s of its call graph with return values; output in
~/ri-ebusy/<MMDDTHHMM>/ebusy-context.txt, readable by user). It catches one EBUSY and
exits. So in this run, besides note 84's prebind test:
1. Make sure one two-card Flash-Next load reproduces the hang while the catcher is waiting: if the
   prebind load hangs, that is it; if the prebind load works, do one more load with the prebind off
   (R10's form) so the hang happens once more. Check that ~/ri-ebusy/ has a new folder afterwards.
2. Read ebusy-context.txt: name the innermost xe (or dma-buf / TTM) function that returns -16 first in the
   failing call graph and what it was waiting on, and record it in STATE.md and WAITING-FOR-OWNER.md.
   That names the fix (in the engine, or a kernel-side issue to report).

## 86. Supervisor: the root catcher missed R15 (a filter bug, fixed); it is re-armed (2026-10-29 13:35 UTC)

The first catcher matched -16 as a 64-bit value; xe_vm_bind_ioctl returns a 32-bit int, so R15's EBUSY
went past it. Fixed in ~/ri-ebusy-catch.sh on intel-host; the owner restarts it. Whatever HB1's matrix shows,
reproduce R15's stall once more after the owner's restart (R15's form: prebind on, the one load), then
read the new ~/ri-ebusy/<MMDDTHHMM>/ebusy-context.txt per note 85 item 2. R15's own finding stands: card1
cannot bind host allocation 2 of 816 (16,991 MiB, kind 3, allocated with card 1 current) while
allocation 1 (1,546.78 MiB) binds; card0 binds all 816 (34.81 GiB) in 12 ms.

## 87. Owner: an xe kernel bug found -- split the host pool under 4 GiB; never bind >4 GiB imports again (2026-10-29 13:35 UTC)

HB1 (m3 fault 22): binding an 8 GiB host allocation imported from card0 on card1 oopsed the xe driver
(WARNING xe_res_cursor.h:167 xe_res_next in xe_vm_bind_ioctl -> xe_pt_stage_bind -> xe_pt_walk_range, a
NULL dereference, then a recursive fault at close); 1.5 and 4 GiB passed; card0 bound 8 GiB in 0.9 ms. The
engine's 16.6 GiB host pool imported on card1 is the same bug's other face (EBUSY forever = the first-step
hang). The owner reboots intel-host; min_freq returns to 1200 (arm A), boost stays 0, GuC 70.58.0.
1. Never again bind, copy from or probe a host allocation of more than 4 GiB on the importing card (it can
   crash the kernel). The catcher is no longer needed.
2. Build the fix behind a flag: every host allocation the engine makes (the pool, the n-gram table, the
   mover's staging, the host experts) in pieces of at most 2 GiB, or each card allocating its own host
   memory instead of importing card0's -- whichever is the smaller change; check on one card first, then
   R15's two-card form with the prebind logging every allocation's size.
3. If the first step runs: X12, then note 76's baseline (M-pair, rotation, first token) with min_freq 1200
   = arm A of notes 77-79, then note 81 item 2's order. Make the fix the default once it passes.
4. Draft (not sent) a kernel bug report for drm/xe: kernel 7.2.6, BMG G31 x2 (ASRock + Intel boards),
   NEO 26.05, the 1.5 / 4 / 8 GiB results, the oops text from logs/HB1-oops*-klog.txt, and a minimal
   reproducer description (a host allocation in a two-device context, imported and bound on the second
   device). Explain in it why it surfaced only after today's reboots if you can tell.

## 88. Owner decision: yesterday's settings back -- one two-card load, then the baseline (2026-10-29 14:07 UTC)

The owner wants to know whether the cards still run the model as they did until 04:43 today. The owner
turned CPU boost back on (boost 1, cpu-boost-off.service disabled); with GuC 70.58.0 and min_freq 1200
that is yesterday's configuration, apart from the fresh boot. Faults 23 / 24 (run 89's import-route cases
b and d) are reviewed as run 89 describes; that route is ruled out and is not run again.
1. One two-card Flash-Next load in M43's form (the service image :70, the M-pair's MTP-off half), after a
   cool wait, with loads spaced for heat as before 04:43 (boost on: the guard's 75 C stop still applies).
   No probes and no host-bind tests before it.
2. If the first step runs: that is the answer -- record it, then the baseline (M-pair MTP off and on,
   one rotation, first token at 7K) against M43 / M44 / H25, with boost 1, min_freq 1200 and GuC 70.58.0
   beside each result. Then note 81 item 2's order; note 87's pool-in-pieces fix still gets built, behind
   a flag, as protection for boost off and fresh boots.
3. If it hangs as R6 did: record it, then go on with run 89's plan (HB2 case c alone, then the
   pool-in-pieces change and R15's two-card form).

## 89. Owner decision: the baseline on the new card firmware (2026-10-29 16:48 UTC)

Answers to run 90's questions, and what changed since:
- The reboots at 16:07, 16:35 and 16:46 were the owner's, for a B70 card firmware update (fwupd from the
  snap, then Intel's igsc). No cooling or power change was made. Both cards now run card firmware
  **31.1062** (was 31.1058; flashed with igsc from LVFS's signed bmg_g31_fwupdate.bin, main firmware only --
  board data and OptionROM unchanged). Faults 26 / 27 are reviewed as run 90 describes: kill artifacts of
  the heat stops, during which the owner's first fwupd attempts also ran; the cards are healthy (xe bound,
  32 GiB BARs, firmware interface normal).
- Settings now: boost 1 (keep it; boost 0 hangs two-card loads until note 87's fix exists), min_freq 1200
  on both cards (it reset at boot), GuC 70.58.0, timeslice 1000.
1. Note 88 item 2's baseline again, on card firmware 31.1062: the M-pair (MTP off, then on), one rotation,
   the first token, against M51 / M52 / H55 / B30 (today's, firmware 31.1058), with the card firmware
   version beside each result. Space loads for heat: cool to 40 C or less before each load and leave
   several minutes between loads; the guard limits do not change. Note whether peaks stay near 65-67 C as
   before 16:07 or near 75 C as after it.
2. If the numbers hold: continue note 81 item 2's order (the freed VRAM to the expert cache, then note 87's
   pool-in-pieces fix behind a flag, then the hc rewrite). If a load hangs or a number drops beyond noise:
   record it and pause with the numbers.

## 90. Owner decision: build the pool-in-pieces fix now, then the baseline (2026-10-29 17:02 UTC)

The owner read run 91's pause and accepts its reading: the first-step hang depends on the boot (good at
13:40, bad at 16:07 / 16:35 / 16:46), not on boost or the card firmware; the afternoon "heat stops" were the
hang's spinning thread. Fault 28 (card0's CAT:bcs at M62's kill) is reviewed as run 91 describes. Run 91's
proposal is the plan, root-free:
1. Build the fix behind a flag: the host pool and the mover's class pools within it as allocations of at
   most 2 GiB (and any other host allocation over 2 GiB the prebind log lists). No card loads until it
   builds and passes its CPU tests.
2. One card: the tiny model, then HB2 case c alone (16.6 GiB as 9 x 2 GiB), after card0's stuck copy job
   from M62 has cleared (check the kernel log; if it has not cleared, use card1 alone first).
3. Two cards: R15's form with the fix and the prebind logging every allocation's size.
4. If the first step runs: note 89's baseline on card firmware 31.1062 (M-pair, rotation, first token).
   Then make the fix the default (it must also pass with boost 0 once, as the proof that the boot-dependent
   trigger is gone), then note 81 item 2's order.
A hang or fault in 2-3: record it and pause with the numbers.

## 91. Owner decision: R19 again with the fix alone, then the baseline (2026-10-29 17:52 UTC)

The owner read run 92's pause and accepts its proposal. Faults 28 / 29 are reviewed as run 92 describes.
1. R19's two-card form with RADIANCE_HOST_POOL_PIECE_MIB=2048 and the prebind OFF, so card1 binds only what
   rank 1 uses (~18 GiB). Space it for heat as before.
2. If the first step runs: note 89's baseline on card firmware 31.1062 (M-pair MTP off and on, one rotation,
   the first token) with the fix on, against M51 / M52 / H55 / B30. Then note 90 item 4 (one pass with
   boost 0 as the proof, then the fix as the default) and note 81 item 2's order.
3. If it hangs: record the stack and which bind, then build the next change (each card allocates its own
   host memory, so nothing is imported card to card) behind a flag, and test it the same way.
4. Root-free and optional, while the cards are busy elsewhere: limit the prebind to the allocations a rank
   owns, so it stays usable as a diagnostic. Record the ~26 GiB total-import limit on card1 for the xe bug
   report draft.

## 92. Owner: more VRAM to the experts next, before the hc rewrite (2026-10-29 20:42 UTC)

The owner read H61 (the full candidate setting with the headroom 3072: rotation 44.0, misses 66.4) and wants
the experts to get more of the VRAM the 64 KB key freed. Pause the hc rewrite's preparation and do this
first, on the candidate setting (the fix, the key) with the VRAM trace on:
1. Lower the recordings floor and the headroom in steps, each step one rotation in H61's form, the trace on:
   - step A: floor 2048 -> 1024 MiB, headroom 3072 -> 2560;
   - step B: floor 1024, headroom 2560 -> 2048;
   - step C (only if B is clean and gains): floor 1024 -> 768, headroom 2048 -> 1792.
   Keep a step only if: no out-of-memory, no append retry, no lost segment, the floor fires at most rarely,
   free VRAM at the rotation's end stays above the floor, and the median rises or holds against the step
   before (beyond noise for a gain). Stop at the first step that fails one of these and keep the one before.
2. For the step kept: the M-pair text check (MTP off and on, text identical to M43 / M44) and a second
   rotation to confirm the median.
3. Report the expert units resident, misses a step and median per step. That setting becomes the service
   candidate (with the fix and the key), replacing headroom 3072.
4. The boost-0 proof (note 90 item 4) and the service adoption still wait for the owner's answer; the hc
   rewrite (note 66) comes after this.

## 93. Owner: in note 92, skip step A and start with step B (2026-10-29 20:46 UTC)

Go straight to step B (floor 1024 MiB, headroom 2048) with the same keep-rules, the trace on. If B fails a
rule, try A (floor 1024, headroom 2560) as the fallback before giving up. If B is clean and gains, step C
(floor 768, headroom 1792) as written.

## 94. Owner: push the floor lower after B (2026-10-29 20:50 UTC)

The owner thinks the floor can go lower now that a recorded list costs ~0.2 MiB and an out-of-memory append
is retried or fails its step visibly (note 58). Replace note 92's step C with:
- step C: floor 512 MiB, headroom 1536;
- step D (only if C is clean and gains): floor 256 MiB, headroom 1280.
Same keep-rules, the trace on; also watch the lowest free VRAM a card reaches in each rotation and any
"failed out of device memory ... retrying" line (one retry that recovers is acceptable, a lost segment or a
failed step is not). Stop at the first step that fails a rule and keep the one before. Do not go below a
256 MiB floor.

## 95. Owner decision: compress everything else first, then give the experts what is left (2026-10-29 21:39 UTC)

The owner's direction replaces the headroom ladder of notes 92-94 (stop it where it is; H61's headroom
3072 stays the best measured setting until this lands). Instead of guessing a headroom, shrink every
non-expert use of VRAM, allocate all of it before the expert cache, and size the expert cache last from
the VRAM actually free, minus a small margin -- the shape production has on the R9700 (0.1 GiB spare).
1. Measure first (no rewriting yet): one card's VRAM at steady state after a rotation, split by owner --
   static weights, arena, KV cache, the checkpoint store (1.54 GiB, H63 / H64), the prefill staging, the
   mover's slots, the recorded passes (the 64 KB key), the driver's per-list memory, and the first-use
   ~1.8 GB of V3. For the first-use part: the compiled kernels' spill and private-memory sizes from their
   zebin metadata (xe direct's zebin reader), the first prompt at 16 / 1K / 5.6K tokens with the VRAM
   trace, and the prefill kernel forms switched one family at a time, to name what allocates it.
   Write the table in STATE.md and briefs/.
2. Compress what the table shows, each behind a flag with the text checks: the checkpoint store in pieces
   or sized to need; the first-use memory (stop the spills or reuse one scratch; the GDN chunk scan if it
   is the cause); the recorded passes (the early release without its 5% cost, or fewer play ranges); the
   arena and KV sized to the configured context; staging at what prefill needs.
3. Order: allocate every non-expert buffer at startup, including a warm-up that triggers the first-use
   allocations, before the expert cache is sized; then size the experts from free VRAM minus a margin
   (start at 512 MiB; the recordings floor below that). This also removes step B's stall at the end of the
   load, since nothing large is allocated after the experts.
4. Gates: the M-pair text identical to M43 / M44, two rotations with the VRAM trace (no out-of-memory, no
   retry, no lost segment), and the median against H61's 44.0. Report experts resident and misses a step.
5. The hc rewrite (note 66, V2 step 2 onward) follows this. The H61 adoption and the boost-0 proof still
   wait for the owner.

## 96. Owner: no service adoption for now; the boost-0 proof goes last (2026-10-29 21:42 UTC)

The owner wants to see the improvements land before adopting anything. So:
- Do not change the service (m3/service/run.sh, the unit, its image) until the owner says so: not the
  H61 setting, not the two-card fix, not hc V2. Keep measuring each change against the references
  (M43 / M44 / H25, and the newest best measured setting) and keep it behind its flag.
- Note 90 item 4's boost-0 proof moves to the end of the queue; nothing waits for it meanwhile.
- The order is note 95 (compress, then size the experts last), then note 66 (the hc rewrite). Report
  each gain as it is measured, against the service's numbers (37.8 / 51.5 / 41.4) and the newest best.

## 97. Owner: run to the $800 pause, then a decision summary (2026-10-30 00:36 UTC)

Keep going in the current order (note 95 item 3 with the pageable checkpoint store, then note 66's hc
rewrite, then the rest of the queue) until the milestone reaches $800, where note 81 pauses the runner.
In that pause, write briefs/<date>-m3-at-800.md for the owner's decision: every gain measured since note
95 against the service's numbers and the newest best, what is committed and behind which flag, what is
left in the queue with an estimate of runs and dollars for each item (the hc rewrite's remainder, the
recorded lists' count, the trunk and expert GEMVs, the GDN chunk scan, note 69's update test, the CPU
split's speed A/B, the boost-0 proof, the m3 report), and the options: raise the cap, open a milestone 4
for the kernel and prefill work, or write the m3 report and stop. Change nothing in the service.

## 98. Owner decision: bind card1's imports before the expert slab, then the hc rewrite (2026-10-30 02:26 UTC)

The supervisor's reading of H63-H70, which the owner accepts: every load stall is card1 binding memory it
imports from card0 (the checkpoint store, then the host pool's pieces) while card1's VRAM is nearly full.
Production's ROCm path maps host memory for every card at allocation, while the cards are empty.
1. After H71 (the host pool at 9 GiB), and whatever it shows: change the bring-up order behind a flag so
   that every host / imported allocation each card will use (the host pool's pieces, the pageable or pinned
   stores, staging) is allocated AND made resident on every card that uses it before the weights pool and
   the expert slab are allocated; then size the slab from the free reading (note 95 item 3, the probe fix).
   Watch with the VRAM trace whether the driver later re-binds those imports (a stall mid-run would show it).
2. Gates as note 95 item 4: the M-pair text identical, two traced rotations, no out-of-memory / retry /
   lost segment, the median against H61 / step A's 44.0; report experts resident and misses a step.
3. Also explain H69's -8 % with one control rotation on the same image with step A's settings (H65's
   form), before trusting rotation numbers from the new images.
4. Then note 66's hc rewrite (V2 step 2 onward, the E4M3 lookup table, the fusions, the 4-row MTP forms).
5. Add to the $800 decision summary (note 97): "one process per card" (no imports at all; the cross-process
   all-reduce to be built and measured) as a milestone-4 candidate.

## 99. Owner: the hc rewrite first; the VRAM goal stays (2026-10-30 02:29 UTC)

The goal is unchanged: experts ~23.3 GiB a card, everything else allocated first and the slab sized last
(figure 19's target). But the owner wants the hc rewrite next, ahead of note 98's items 1-3:
1. Note 66's hc rewrite now, from 5a7e0e2: level 2's staged W_up rows as the base (M71: 39.45 tok/s), then
   the E4M3 decode lookup table in local memory for hc_v2_down and hc_up_mix, the fusions (the norm into
   the down product; hc_write into the all-reduce), and the 4-row (MTP) forms. Each step behind
   RADIANCE_XE_HC_V2, rad-kbench byte-identical first, then the M-pair (text identical, speed against
   M71 / M68), with a profile (P-run) when a step changes the hc split. Use image :86's settings (H61's
   service candidate) or the service's, not the new images whose rotation is unexplained, for any speed
   comparison.
2. Then note 98 items 3, 1 and 2 in that order (the control rotation for H69 / H71's slowdown first, then
   binding the imports before the slab, then its gates).
H71 (the host pool at 9 GiB, the slab sized from free VRAM): it loaded and ran 71 requests, all 200, with
a median near 39.5 -- the same unexplained slowdown as H69; record it with that caveat.

## 100. Owner: update progress.tsv as soon as a result is measured (2026-10-30 03:28 UTC)

The owner watches watch.sh and wants it to show each finding as it happens. From now on, update
progress.tsv's "now" column right after any measured result that beats the row's current value, in the
same step as the STATE.md row -- not at the run's end. A result counts if it passed its gates (text
identical where the row has a text gate; no out-of-memory / retry / lost tape / failed step); put the step
id, image and the flag or setting at the start of the row's note (for example "H74/H75 :95, the candidate
of note 95: 47.5 / 47.3"). Keep the file's five tab-separated columns and its baseline and target columns
unchanged. A result that is opt-in and not in the service still counts (note 96 keeps the service as it
is; progress.tsv records the best measured). Now: the rotation row from H74 / H75 (47.5), the misses row
(43.2), and the tape row if it moved.

## 101. Owner: test the adaptive draft window after the hc rewrite's 4-row MTP forms (2026-10-30 03:44 UTC)

From the supervisor's comparison of Radiance and vLLM on amd-host: most of Radiance's lead there is MTP
acceptance, and Radiance has a setting the Intel service never used -- `--draft-window adaptive`
(GUIDE.md "Adaptive window"): with `--num-speculative-tokens` as the ceiling, the engine measures each
draft depth's step cost and each position's acceptance and picks the depth that maximises tokens per step
time. On the B70 the verify step is costly (the 4-row forms), so a fixed depth of 3 may not be the best.
Right after note 66's 4-row (MTP) forms land (or, if those are not reached this milestone, after note 99's
other hc steps), on the best measured setting (note 95's candidate plus the hc flags):
1. The M-pair's MTP-on half and one rotation at `--num-speculative-tokens 4 --draft-window adaptive`, then
   at 6 if 4 gains; against the fixed 3 on the same image. Record the depth the engine settles on
   (`radiance:draft_depth_adaptive`, `radiance:draft_verify_depth_total`), the acceptance per position,
   the tape count / recordings' VRAM (each new depth is a new recording), and tok/s.
2. Gates: text identical at temperature 0 against the fixed depth where depths agree (GUIDE: the text is
   that of a fixed depth wherever fixed depths agree); no out-of-memory / retry; the rotation median.
3. Keep it behind its settings; record in progress.tsv per note 100 if it beats a row. Startup must say
   whether the adaptive window is active (it needs a serial draft head; GUIDE says when it stays fixed).

## 102. Owner: the prefill hc right after the decode hc rewrite (2026-10-30 03:58 UTC)

The order from here: note 66 / 99's decode hc rewrite (to its 4-row MTP forms), then this, then note 101's
adaptive draft window, then note 98's VRAM follow-ups.
- The prefill HyperConnection: hc_read_a 388 x 12.95 ms + hc_up 1.48 = 6.51 s a rank at the 7K prompt
  (briefs/2026-10-28-bottleneck.md, the B10 table; 0.6 % of the f16 pipe; the brief's item 8: "matrix-unit /
  batched-row form"). At prefill's M rows the mixes are matrix products: run them as one batched product
  per launch on the matrix units (the expert and trunk products' joint_matrix forms are the model), with the
  E4M3 decode the decode rewrite builds (decode once per tile into local memory, not per row), behind its
  own flag.
- Gates: rad-kbench against the reference at prefill shapes (tolerance as the matrix expert form's, or
  byte-identical where the order allows); the tiny model's text; B-runs (the 7K and 20K first tokens, the
  20K reusing the 7K's KV) with answers identical to B32 / B35 or a documented near-tie (notes 49 / 57),
  on the best measured setting; a profile (P-run) of one prefill for hc's new share.
- Target: the 7K first token from ~26 s toward ~20 s (hc 6.5 s -> ~0.5 s). Record per note 100.

## 103. Owner: the recorded passes down to ~1 GiB, before note 98's VRAM follow-ups (2026-10-30 04:43 UTC)

Order from here: the decode hc rewrite, the prefill hc (note 102), the adaptive draft window (note 101),
THIS, then note 98's follow-ups (so the expert slab is sized after the tapes have shrunk).
- The lever is the early segment release (RADIANCE_L0_TAPE_EARLY=1; H28: +505 MiB with the 64 KB key against
  +1.9 GiB without) without its ~5 % MTP-on single-prompt cost (M54 / M56). That cost is host work: once the
  segment lists are released, every play composes, and the partial ranges (an MTP verify that accepts fewer
  than all drafts) are composed again on demand. Keep the composed partial-range stretches -- a few kinds a
  tape, one per accepted count (and per depth, if the adaptive window is on) -- in the bounded stretch cache
  so each is built once, and account their lists in the VRAM trace.
- Gates: the M-pair text identical and MTP on within noise of the fixed-release figure (no 5 % loss); two
  traced rotations: the recordings' VRAM (target <= ~1 GiB a card, against 1.6-1.9), the lists' peak, the
  median against note 95's candidate (47.5), no out-of-memory / retry / lost tape.
- Also measure "fewer, longer segments" if the code shows cuts that need no host step; record progress.tsv's
  tape row per note 100.

## 104. Owner: the 4-bit expert kernels, right after the decode hc rewrite (2026-10-30 06:51 UTC)

Order from here: the decode hc rewrite (note 66 / 99, through its 4-row MTP forms), THIS, then the prefill
hc (note 102), the adaptive draft window (note 101), the recorded passes to ~1 GiB (note 103), and note 98's
VRAM follow-ups.
- Why: after hc level 5, the routed experts (IQ4R = w4nl64a8h) are the largest single item of the decode
  step. In P36 (level 5 at note 95's candidate), per card and step: moe_planes_v2<5> (gate / up, K 2560)
  57.7 us a launch, about 142 GB/s; moe_planes_v2<1> (down, K 320) 51.8 us, about 79 GB/s -- together ~5.0 ms
  of the 23.1 ms step. Round 2's M6 probe ran the same format at 343 GB/s on this card. These byte rates are
  derived from the unit size (1,268,480 B = an expert width of 320 a rank); check them against the kernel's
  real shapes first.
- Where to look (owner's reading, to be confirmed by measurement):
  1. **down (K 320):** a row is 160 bytes, so 6 of the 16 lanes of a 16-byte-a-lane team are idle, and each
     row pays the full butterfly reduction and scale step. Several rows a sub-group (lanes sized to the row),
     or another split, so no lanes idle.
  2. **gate / up:** one routed row a work-group stages the 2,560-value activation and the 256-entry table for
     only 64 rows (~100 work-groups on 32 cores), so the prologue and latency dominate. More rows a
     work-group (the prologue paid once), more loads in flight.
  3. **router_topk_scatter** at 18.3 us a layer (~0.8 ms a step) for top-10 of 512.
  4. **Fusions:** silu_mul / gated_quant / quant_act into gate / up's output (~16 us a layer of small
     launches today); one launch less where the order allows.
  5. The same changes to the 4-row form (moe_planes_v2m) for MTP on.
- Gates: every form behind its own flag. moe_planes_bench's CHECK against libref at the real shapes (the
  kernel's existing tolerance, the same summation order where possible); the tiny model's text; the M-pair
  at note 95's candidate with text identical to M43 / M44, or a documented near-tie; a P-run for the new
  per-launch times; one traced rotation (misses must not rise). Nothing becomes a default without the owner
  (note 96).
- Target (owner's estimate, not measured): the expert GEMVs from ~5.0 toward ~2 ms a step, i.e. ~2.5-3.5 ms
  off the step: MTP off ~43 -> ~48-50 tok/s, MTP on up as well. Record progress.tsv per note 100.

## 105. Owner: the 4-row MTP verify on the matrix units, the way radiance does it on AMD (2026-10-30 07:04 UTC)

Order from here: the decode hc rewrite now running (note 66 / 99, level 7's 4-row down form included), THIS,
then note 104's 1-row expert kernels (its item 5, the 4-row expert form, moves into this note), then the
prefill hc (note 102), the adaptive draft window (note 101), the recorded passes to ~1 GiB (note 103), and
note 98's VRAM follow-ups.
- Why: P41 (run 99, level 5 MTP on) shows the 4-row verify paying per-row work on unchanged weights:
  gemv_i8a8<4> 40.9 us against 19.7 at one row (18 % of the MTP step), moe_planes_v2m 140 / 138 us (27 %),
  hc_v2_down 45.7 us against 12.6 (level 7 addresses this one). libr4d on the R9700 does not use per-row
  GEMVs for these: audit.md's map has the trunk as gemm_i8a8_nt_m16 (iu8 WMMA, 16-row tiles), the experts as
  codes -> E4M3 -> fp8 WMMA, the MTP draft head as iu8 WMMA, and at tp2 hc's write fused into the all-reduce
  (ar_hc_write). A tile reads its weights once for every row it holds, so AMD's 4-row verify costs about
  the weight traffic of one row. This is the same finding as the research agent's "make verification
  cheap", now with the mechanism and P41's numbers.
- The Xe2 equivalent, per form, each behind its own flag:
  1. **Trunk W8A8 at 2-8 rows on XMX (DPAS) in INT8**: exact, the same integer products (owner note 32's
     multi-column XMX GEMV; the prefill's gemm_i8a8_dpas is the starting point). The weights relaid once at
     load if the tile needs it, within the VRAM ledger.
  2. **Experts at 2-8 rows**: the codes to f16 through the 16-entry table as now, then f16 XMX with f32
     accumulate (audit.md section 6: the products equal the fp8 WMMA's; only the summation order differs);
     the prefill's matrix expert form (RADIANCE_XE_MOE_MX) is the model. Includes note 104's item-5 ideas.
  3. **hc at 4 rows**: if level 7 leaves hc_v2_down / hc_up_mix well above their 1-row cost, the same
     decode-once-per-tile on XMX.
  4. **hc_write fused into ar_kernel** at tp2, as libr4d's ar_hc_write: one launch less, 96 times a step.
  Keep the 1-row path on the GEMVs unless a measurement shows the tile form wins there too.
- Gates: kbench / bench CHECKs against libref at the 4-row shapes (exact for INT8; the existing tolerance for
  f16); the M-pair MTP on at note 95's candidate with text identical to M44 / M91 (the tiny model has no MTP
  head, run 99), MTP off unchanged; a P-run (P41's form) for the new per-launch times; one traced rotation.
  Nothing becomes a default without the owner (note 96).
- Target (owner's estimate, not measured): the 4-row trunk, experts and hc near their 1-row weight cost --
  MTP on from 52.9 toward ~60-70 tok/s. Record progress.tsv per note 100.

## 106. Owner: the cap is raised to $1,000 -- option 1 of the $800 summary (2026-10-30 12:24 UTC)

The owner read briefs/2026-10-30-m3-at-800.md and chose option 1: milestone 3 continues to $1,000. Note 81's
$800 pause is lifted; the $1,000 limit stays a hard stop. Everything stays opt-in (note 96).
Order, cheapest and most certain first:
1. **Note 105 items 1 + 4** (the trunk's rows form on the matrix unit, RADIANCE_XE_GEMM_DPAS_ROWS=1, and the
   residual write fused into the all-reduce, RADIANCE_XE_AR_HC_FUSE=1): BK12c, the M-pair with both flags, a
   P-run. Commit each only after its gates.
2. **Note 103's second traced rotation** (H77 again), to confirm H76's 49.2.
3. **Note 105 item 3** (hc at 4 rows).
4. **Note 104** (the 1-row expert kernels), as far as the budget allows.
5. If budget remains: the 6-7-row text finding (which kernel's result depends on the verify's row count).
- Skip note 105 item 2 (experts on XMX at the verify) for now, per run 99's reading that an expert's run is
  ~1 row there; it goes to milestone 4 with note 98, the GDN chunk scan and "one process per card".
- **Reserve ~$20 for the m3 report.** Start it no later than ~$975 and finish it before $1,000, whatever item is
  open; the report lists what was left with its state.
- Run 99's lesson stands: after a heat stop's kill, run a small step before a long one. intel-host idles at 41-42 C;
  keep note 89's cool-down rule as it is.

## 107. Owner: the proper fix for the load stall now -- note 98 item 1 comes forward (2026-10-30 13:18 UTC)

M100 / M100b stalled in card1's host-pool import binds again (xe VM_BIND EBUSY), on a boot where the same
8 GiB setting loaded this morning; the change was most likely card1's reset (fault 32). This bug has now
stopped two-card work on Oct 29, at M83 and at M100. Smaller pieces only move the line, so the owner wants
the proper fix before more two-card measurements. This overrides note 106's "note 98 to milestone 4".
Order from here:
1. The 1 GiB-pieces step already in flight may finish as a diagnostic; do not tune piece sizes further.
2. **Note 98 item 1:** behind its own flag, change the bring-up order so that every host / imported
   allocation each card will use (the host pool's pieces, the checkpoint store, staging) is allocated AND
   made resident on every card that uses it while the cards are still nearly empty -- before the weights
   pool and the expert slab -- then size the slab from the free reading (RADIANCE_VRAM_CAP_FROM_FREE).
   Production's ROCm path maps host memory at allocation, while the cards are empty; this is the same order.
   Watch with the VRAM trace whether the driver later re-binds those imports (a stall mid-run would show it).
3. **Gates, on this boot first (no reboot):** the 8 GiB candidate loads with the flag where M100 / M100b
   stalled; then the 16 GiB host pool (the service's form, M83's stall) loads too; then note 98 item 2's gates
   (the M-pair text identical to M43 / M44, a traced rotation against H76's 49.2, no out-of-memory / retry /
   lost segment, experts resident and misses a step). Note 98 item 3 is already answered (run 97).
4. If the flag still stalls on this boot, stop two-card steps and write in WAITING-FOR-OWNER.md that intel-host
   needs a reboot (root, the owner); single-card steps (kbench, the tiny model) may continue meanwhile.
5. Then note 106's order resumes: note 105 items 1 + 4 (the M-pair with both flags, a P-run), H77, note 105
   item 3, note 104. The ~$20 reserve for the m3 report stands.
- **Owner, 13:42 UTC: keep this boot for the proof.** The owner read run 100's reboot request
  (WAITING-FOR-OWNER.md, 13:25) and will NOT reboot intel-host yet: this boot (card1 binding only 2-6 GiB since fault
  32) is the case the fix must cure, and a healthy boot cannot show that. So do not wait for a reboot: build note 98
  item 1's flag first in the next run, and try it on this boot (a small load first, then the 8 GiB candidate where
  M100 / M100b / M100c stalled). Ask for the reboot only per item 4, after the flag has been tried here. Until
  then the single-card work (note 105 item 3's kbench, note 104's bench) continues as run 100 planned.

## 108. Owner: the GDN recurrent update after note 104; the rest to milestone 4's list (2026-10-30 14:21 UTC)

Order: note 107 (the load fix, proved on this boot), note 106's items (note 105 items 1 + 4, H77, note 105 item 3,
note 104), then THIS, within the budget; the ~$20 m3-report reserve (start by ~$975) still comes first.
1. **gdn_recurrent_update in decode:** ~58 us a launch x 36 layers = ~2.1 ms of the 23.1 ms MTP-off step (P36),
   ~10 % of bandwidth by the bottleneck brief's reading (item 13: a fixed dependent chain of ~53 us, the per-token
   body <= ~5 us). Find what holds the chain (the reductions in series, the state's read-modify-write, the
   work-group count against 32 Xe cores) on its own kbench at the decode shape, then shorten it behind a flag:
   merged reductions, more work-groups over the state's heads / blocks, fused gates. Owner's estimate (not
   measured): ~1 ms a step. Gates: kbench against libref (the existing tolerance; bit-identical where the order
   allows), the tiny model's text, the M-pair text identical to M43 / M44, a P-run.
2. **For milestone 4's list, in the m3 report with these estimates (owner's, not measured), not to do now:**
   - the all-reduce protocol: 96 calls x ~25 us = ~2.4 ms, mostly handshake latency -- ~0.5-1 ms;
   - the 1-row INT8 trunk GEMV at ~53 % of bandwidth against the probe's 530-573 GB/s -- ~1 ms;
   - the small bf16 GEMVs (~145 launches, 1.2 ms, launch-bound): batch / fuse -- ~0.5 ms;
   - hc_v2_down at one row (~24 us, five layouts tried in run 98): needs a new idea;
   - the 6-7-row text finding (deeper MTP windows), if note 106 item 5 does not reach it;
   - the prefill GDN chunk scan (42.5 % of the 7K prefill, P43): first token ~21 -> ~15 s;
   - with note 105 item 2, note 98's "one process per card" and the GDN chunk scan already listed.
- Owner's sum of the possible MTP-off gains (not measured): the step ~23 -> ~17-18 ms (~55-58 tok/s), near the
  bottleneck brief's realistic ceiling (59-66). Record progress.tsv per note 100.

## 109. Owner: the milestone budget is $2,000; milestone 4's list is done inside milestone 3 (2026-10-30 14:22 UTC)

The owner raised budget.txt from 1000 to 2000, to go on without another stop. This replaces note 106's $1,000
limit and its report timing, and note 108 item 2's "not to do now".
Order: note 107 (the load fix, proved on this boot), note 106's items (note 105 items 1 + 4, H77, note 105 item
3, note 104), note 108 item 1 (the GDN recurrent update), then note 108 item 2's list in this order:
1. the all-reduce protocol (~2.4 ms a step);
2. the 1-row INT8 trunk GEMV (~53 % of bandwidth);
3. the small bf16 GEMVs (batch / fuse);
4. the 6-7-row text finding (then note 101's deeper windows may be measured again);
5. the prefill GDN chunk scan (first token 7K ~21 -> ~15 s);
6. note 105 item 2 (experts on XMX at the verify), only if a measurement shows a tile shares rows there;
7. hc_v2_down at one row, if a new idea comes up;
8. "one process per card" (no imports; a cross-process all-reduce): design and measure last, as a brief first.
- Every item keeps its own flag and gates; nothing becomes a default without the owner (note 96).
- **The m3 report: start it by ~$1,950 and finish it before $2,000**, whatever item is open. Also write a short
  progress summary into WAITING-FOR-OWNER.md at each $250 step ($1,250, $1,500, $1,750) without pausing.
- No other $ pause: PAUSED only for UNATTENDED.md's hard conditions.

## 110. Owner: intel-host was rebooted at 15:16 UTC -- run the waiting two-card gates now (2026-10-30 15:36 UTC)

The owner rebooted intel-host at 15:16:11 UTC, as run 101 asked (WAITING-FOR-OWNER.md, after M103). Run 101's
"two-card steps stop on this boot" no longer holds: this is a fresh boot.
1. Finish the step in flight (the GDN diagnostics) and record it, then switch to the two-card gates.
2. The two-card gates, in this order: run 100's waiting M-pairs (note 105 items 1 + 4: rows + fuse; note 104's
   moe8 down kernel; note 105 item 3: hcsr), each against M91 / M90 at note 95's candidate; then H77 (note 103's
   second rotation); then note 98 item 2's gates for RADIANCE_HOST_FIRST + the load reserve on this healthy boot
   (it did not cure the bad boot -- M103 -- but should cost nothing here: check it does no harm, then the 16 GiB
   host pool with it).
3. Then back to note 108 (the GDN update) and note 109's list.
- Kernel: intel-host is on 7.2.6. Stable 7.2.9-7.2.13 carry xe fixes on the import / external-object paths (7.2.9
  "vm: nuke PTs only after unlinking contested VMAs" -- the stack of the Oct 29 oops; 7.2.11 shrinker accounting
  on nested external pins; 7.2.13 LRU bump for external objects, eviction of a BO with no resource). The owner
  will install a newer kernel when Ubuntu's mainline 7.2.14 amd64 build exists; until then nothing changes here,
  and after it a baseline M-pair comes first (as after the card firmware).
- The guard's --precheck now prints intel-host's boot time and flags a change since the last check (owner change).

## 111. Owner: audit the ported kernels for AMD-only tunings, after note 110's gates (2026-10-30 16:11 UTC)

GB4 showed one AMD tuning (r4d's RU_NT_MB non-temporal hint) costing ~7x on Xe in gdn_recurrent_update (97.5 -> 13.2
us, bit-identical). Other kernels came from libr4d with choices made for RDNA4. A cheap, quick audit:
- Order: after note 110's two-card gates (with the `gnt0` M-pair in them), before note 109's list.
- Scope 1, the rest of gdn_recurrent_update's RU_* knobs (RU_K, RU_V, RU_LAUNCH, RU_TMAX, RU_PICK*, ...): which
  values are AMD's, and the Xe alternative for each on its bench.
- Scope 2, the other ports in src/libxe/ports (wave-32 lane math, work-group sizes, unroll factors, LDS staging,
  prefetch and cache hints), by their share of the decode step (P36 / P41): router_topk_scatter, the moe gather,
  rowtopk, quant_act_fp8, gdn conv / gated norm, the qsa_* and attention split / merge kernels, the sample chain.
  For each: list the inherited choices, try the obvious Xe alternative on its kbench / bench; keep only results that
  are checksum-identical or within the kernel's existing tolerance; drop a kernel after a first try that moves it
  under 5 %.
- Budget about one run (~$30-35) for the whole audit; the wins join an M-pair together, each behind its own flag.
- Correction to note 110's kernel bullet (owner, from a source read of 7.2.6): the 7.2.9-7.2.13 xe fixes do not
  touch the import mapping path, and 7.2.9 does not match the Oct 29 oops. The EBUSY is xe_dma_buf.c:150 -- any
  dma_map_sgtable failure of card0's export for card1, through the AMD IOMMU in translated mode -- turned into
  EBUSY. The owner is considering IOMMU passthrough (iommu=pt) after note 110's gates; a note will say when, and
  PAUSED will be set around that reboot.

## 112. Owner: fail fast on a stuck bring-up, so a stall never becomes a card reset (2026-10-30 16:35 UTC)

Today's three resets (faults 32, 33 and the 10:08 one) came from killing processes stuck at bring-up, not from the
stalls themselves: M102 waited 15 minutes for the step's limit with the loader's copies queued behind the stuck
bind, and its kill reset card0; M103 was stopped after ~4 minutes, before any copy was queued, and left both cards
clean. So:
1. **Engine, behind a flag that the m3 configs turn on:** when bring-up logs nothing for ~60 s while the startup
   thread is inside a driver call (the existing "nothing has been logged for N s" watchdog), exit the process with a
   clear message and a nonzero code instead of waiting -- before the loader queues copies, where possible. Keep the
   host-first order (RADIANCE_HOST_FIRST and the load reserve, run 101) as the default for these configs once note
   110's no-harm gate passes: it puts every import bind ahead of the copies, which is what made M103's stop clean.
2. **Runner:** any step whose bring-up shows that watchdog line for ~90 s is stopped with
   `intel-host-engine-guard.sh --stop` (as M103), never left to its deadline. Record it as a stall, not a fault.
3. Gates: a forced stall on a healthy boot is not available, so check the code path with a unit test or a
   tiny-model run with an injected wait (a debug variable), and that the normal M-pair is unchanged.
- Small and cheap: do it right after note 110's gates, before note 111's audit.

## 113. Owner: the trunk's rows form (note 105 item 1) -- time it first, then hunt the bit (2026-10-30 17:14 UTC)

Run 101 did not adopt the rows form (RADIANCE_XE_GEMM_DPAS_ROWS): BK12c-g pass the tolerance, but the bits differ
from the GEMV's, which flips the M-pair prompt's first-token near-tie (M105 / M108, E31); the fma hypothesis was
wrong (BK12g). M105's 46.22 says nothing about its speed (different text, different work). Order: after note 112,
before note 111's audit.
1. **Time it first (cheap):** a P-run or the kbench timing at the verify's 4 rows and prefill's 8-row chunks, per
   launch against gemv_i8a8<4>'s 40.9 us (P41). If it is not clearly faster (say < ~30 us), record it and stop
   here.
2. **If it is faster, find the differing bit:** the int32 group sums are exact in both, so the difference is in the
   float epilogue (the scale products and their accumulation order across groups / lanes / sub-groups). For one row,
   dump each 128-K group's partial float sum and the running accumulator from both kernels (a debug variable, the
   bench only) and compare in order; the first differing value names the step. Then make the rows form reproduce
   the GEMV's float order exactly, and re-run BK12 (md5 against the GEMV), the tiny model, and the M-pair text gate.
   Budget about half a run.
3. **Only if the bits cannot be matched:** stop and write the case for the owner in WAITING-FOR-OWNER.md -- the form
   as a tolerance form like the prefill hc (RADIANCE_XE_HC_PF_DPAS=1), judged by agreement with production on the 11
   requests (AG) and documented near-ties instead of identical text. That is the owner's decision, not the runner's.

## 114. Owner: a second look at hc level 7 (the 4-row down form on the row path), time first (2026-10-30 17:43 UTC)

M114 (hc at 4 rows on the matrix units) was 19 % slower with the text changed; on Xe the verify's 4 rows do not pay
for the decode and staging of a tile. The form that fits is run 99's level 7: the per-row V2 path, each sixteen codes
read and decoded once for the four rows. Its record: M92 identical text but slower (50.34; the 4-row instantiation
spilled 2 KB a thread), M93 (no spill, two loads in flight) diverged in run 2 with its control M94 identical; then
removed, never committed. M104b showed today that the default path itself can flip a run-2 near-tie, so M93 alone
does not settle it. Order: after H77 and note 112, beside note 113 (time first, then text).
1. Rewrite level 7 (RADIANCE_XE_HC_V2=7) from run 99's description in STATE.md, without spills (zeinfo: scratch 0).
2. **Time it first:** kbench / bench hc_v2_down at 4 rows, against P41 / P44's ~45.7 / 42.5 us. If it is not clearly
   faster (say < ~30 us), record and stop.
3. If faster: BK (md5 equal to level 5's where the order allows), then the M-pair MTP on **twice** (two loads), with a
   level-5 control on the same image; a divergence that repeats is a defect, one that does not repeat while the
   control also flips is a near-tie of the replay. Commit as opt-in only if the text holds in both loads.
- Budget: about half a run; stop at step 2 if there is nothing to gain.

## 115. Owner: amd-host pruned -- image builds may resume (2026-10-30 18:16 UTC)

The owner ran `podman image prune` on amd-host at ~18:15 UTC, as run 101 asked (WAITING-FOR-OWNER.md). `df ~/code`
shows ~62 GB free; :117 and the vLLM rollback image are still there. Builds may resume (note 112's card gate, note
114, note 111).

## 116. Owner: intel-host moves to IOMMU passthrough (iommu=pt) -- a baseline first, then the reset proof (2026-10-30 18:42 UTC)

Why: the import-bind stalls end in xe_dma_buf.c's EBUSY for any dma_map_sgtable failure of card0's export mapped for
card1 through the AMD IOMMU in translated mode (note 111's correction; briefs/2026-10-29-intel-xe-import-bind-draft.md,
"Where the EBUSY comes from"). With iommu=pt the cards get identity domains and that mapping code does not run.
Production on amd-host has run iommu=pt on its four R9700s all along.
- The owner reboots intel-host with `iommu=pt` added to GRUB_CMDLINE_LINUX_DEFAULT during run 102's image build (~18:5x
  UTC). The guard's precheck will print "INTEL-HOST REBOOTED". If intel-host is unreachable at a step, wait and retry; do not
  count it as a fault.
- After the reboot, before anything else on two cards:
  1. Check both cards' IOMMU group type is `identity` (`/sys/bus/pci/devices/0000:0{b,f}:00.0/iommu_group/type`)
     and record /proc/cmdline. If not identity, stop two-card steps and say so in WAITING-FOR-OWNER.md.
  2. The baseline on passthrough, the same image and configs as this boot's: the M-pair (base, MTP on / off) against
     M104c / M113 (52.91 / 38.31, text identical); the 16 GiB host pool (`hf-hp16`) against M117; one rotation
     (H78) against H77b / H76; the 7K / 20K first tokens (a B-run) against B38. Host-memory reads skip IOMMU
     translation now, so record any change in the rotation, misses a step and first tokens.
  3. The reset proof: the owner will force a clean GT reset of card1 through xe's debugfs (force_reset_sync, with
     nothing running) when the runner writes in WAITING-FOR-OWNER.md that it is ready; then load the 8 GiB
     candidate, then the 16 GiB pool. Card1 binding them all after the reset is the proof.
- Then the order resumes: note 112's card gate, note 114, note 111's benches, note 109's list.

## 117. Owner: card1's GT reset is done -- run the reset proof now (2026-10-30 19:48 UTC)

The owner forced card1's GT reset at 19:48:24 UTC through
`/sys/kernel/debug/dri/0000:0f:00.0/tile0/gt0/force_reset_sync`, with nothing running. The kernel log:
"trying reset from force_reset_sync_write", "reset queued", "reset started", "reset done" (dmesg 3907.144 ->
3907.157, ~13 ms), with no error lines. (The AppArmor "who" denials at the same second are unrelated.)
- Note 116 item 3 now: load the 8 GiB candidate (`svcmtpx-base 2`), then the 16 GiB pool (`svcmtpx-hf-hp16 2`),
  under note 112's rule (stop at ~90 s of the watchdog line). Record card1's binds, the text against M97 and the
  speed against M118 / M120.
- If both load: the proof holds for a clean reset. Say in the result that today's earlier resets were teardown
  resets of killed processes ("Schedule disable failed to respond"), which this forced one does not reproduce; the
  IOMMU mapping path that stalled is gone either way under iommu=pt.
- If either stalls: stop two-card steps, record where, and write it in WAITING-FOR-OWNER.md.
- Then the order resumes as note 116 says.

## 118. Owner: the milestone budget is $4,000; daily cap $1,000 through Nov 15 (2026-10-30 20:08 UTC)

The owner raised budget.txt from 2000 to 4000 (first 2500, then 4000 the same minute) and set daily-cap-2026-11-02 .. -15 to 1000 (Nov 1 and earlier were
already 1000). This replaces note 109's $2,000 and its report timing.
- **The m3 report: start it by ~$3,950 and finish it before $4,000.** The progress summaries in WAITING-FOR-OWNER.md
  continue at each $250 step ($1,000, $1,250, ... $3,750), without pausing.
- Order unchanged: note 116's order (note 112's M-pair, note 114, note 111's audit), then note 109's list. The owner
  asks that the **prefill GDN chunk scan** (note 109 item 5, ~42 % of the 7K prefill) not be squeezed out by the
  decode items: if the decode list is not done by ~$1,700, start the chunk scan there and return to decode after.
- Everything stays opt-in (note 96); PAUSED only for UNATTENDED.md's hard conditions.

## 119. Owner: try run 102's column-split grid for the 4-row hc down form (2026-10-30 20:45 UTC)

P46 recorded level 7 at 55.2 us a launch against level 5's 45.7 (4 rows): one workgroup walks four rows serially and
three of four column workgroups return, so a read-latency-bound kernel lost 3/4 of its workgroups in flight. Run
102 named the untried alternative; the owner wants it tried, cheaply:
1. After note 111's lane-layer gates (the XE_NATIVE_XOR M-pair), write the column-split form: four rows a workgroup,
   each sixteen codes decoded once for the four rows, over ceil(lr / 2) (or finer) column slices, so the
   workgroups in flight stay near level 5's; no spills (zeinfo).
2. **Time first** (a P-run at MTP on): keep it only if hc_v2_down at 4 rows is clearly under level 5's 45.7 us
   (say < ~35 us); otherwise record and stop.
3. If faster: kbench md5 equal to level 5's, then the M-pair MTP on twice with a level-5 control on the same image
   (as note 114 item 3). Commit as opt-in only if the text holds in both loads.
- Budget: well under half a run; stop at step 2 if there is nothing to gain.

## 120. Owner: the all-reduce's large-message race must be fixed too, not only decode's speed (2026-10-30 21:24 UTC)

AB1 / AB2 (and X11 / X12 / X16 in runs 84-85) show the peer all-reduce giving different sums on the two ranks for
large messages (f32 65,536 into y; bf16 262,144 in place), in the shipped form 0 and in forms 1-3. Decode's 2-20 KB
messages pass; prefill's large ones are exposed, silently. The LL form (RADIANCE_XE_AR_PROTO=4) falls back to form 0
when a message does not fit twice in its slot -- which is exactly the large, failing case. So:
1. **Correctness first, for every size the engine sends:** make the large path race-free too -- e.g. send a large
   message as a series of LL-sized pieces through the same slot, or give it a correct handshake (each payload store
   visible before the flag: per-thread release at system scope plus a work-group barrier before the single flag
   store, or per-thread flags). The fix must make allreduce_test's peer cases pass at every size, repeated (say 20
   repetitions of the large cases on this boot), on both cards.
2. **Then find what it touched:** which engine all-reduces are large (prefill chunks, the prefill hc / expert
   paths), and whether any served result depended on it -- re-run the B-run (7K / 20K first tokens, B38 / B39's
   answers) and the agreement set against production (AG) with the fixed form. If an earlier result changes,
   record which.
3. Then the LL speed work for decode (AB3 onward), with the M-pair text gates as usual.
- The fix for the large path is a correctness fix. Once it passes items 1-2, the runner may make it the m3 configs'
  default (like note 112's flags); making it the engine's own default stays the owner's call (note 96).

## 121. Owner: load at <= 43 C until the new heatsink; the LL all-reduce becomes the engine's default (2026-10-31 03:41 UTC)

Answers to runs 103-104's two questions (WAITING-FOR-OWNER.md).
1. **Heat:** intel-host idles at 41-43 C now. Until the owner says the new heatsink is in, note 89's load-start bar is
   **<= 43 C** (was 40). Everything else in the heat rules is unchanged: the 75 C step guard (`--hot 75`), the 80 C
   hard guard, note 16's halving after a peak above 72, and the small step after any heat stop. If an idle reading
   goes above 43, wait as before; do not raise the bar further.
2. **The LL all-reduce is the engine's own default** (owner's decision under note 96, for this item only):
   RADIANCE_XE_AR_PROTO defaults to 4; `RADIANCE_XE_AR_PROTO=0` keeps the old form as an opt-out. It is a correctness
   fix (AB5: 20 / 20 race-free at every size; AG3-AG6: the old form changed chat requests 05-07 load to load, LL is
   bit-repeatable). Gates for the change: allreduce_test at every size with the default build, one M-pair (MTP on /
   off) with no RADIANCE_XE_AR_PROTO set, text identical to M128 / M129's; then commit with these results in the
   message. Record in progress.tsv's note that every result from here uses LL.

## 122. Owner: deeper MTP drafts -- re-test the adaptive window on the new verify kernels (2026-10-31 03:59 UTC)

Run 99's adaptive window lost at a ceiling of 4 (M95 46.60 against 52.94) and changed the text at 6 (39.31). Since
then the verify rows got cheaper (hc level 8, the LL all-reduce's 4-row exchange -15 to -22 %, the router pair), so
the trade may have moved. Order: right after M136 and note 121's engine-default gates, before note 109 item 4.
1. With M136's combined settings (the best single-user form), run the MTP-on single prompt at the shipped 3 drafts
   and with `--num-speculative-tokens 4 --draft-window adaptive` (note 101's form) on the same image and load
   conditions.
2. Do the same on **one code / JSON / tool-call prompt** (e.g. one of the agreement set's tool requests, or a short
   "write a function" prompt with greedy decoding), where draft acceptance is much higher than on the prose prompt.
3. Record for each: tok/s, tokens a step, the window the controller chose, and acceptance by draft position.
   Gate: each prompt's text identical between 3 drafts and the adaptive window (greedy verification must not change
   the output at <= 5 rows).
4. Decision: if the adaptive window at 4 gains >= ~2 % on either prompt with identical text, move note 109 item 4
   (the > 5-row text change) up next, then test ceilings 5 and 6 the same way; 5-8-row kernel forms (hc level 8 and
   the expert v2m groups of four) only if those deeper windows win. If not, record it and go on with note 109's
   order.
- Budget: ~a third of a run for items 1-3.

## 123. Owner: give the idle VRAM to the experts -- lower the headroom step by step (2026-10-31 04:08 UTC)

M136's VRAM trace (card1, steady state): weights pool 27.5 GiB (static 3.5 + the slab 23.4, 19,755 slots), KV 0.7,
arena + staging 1.5, the driver's lists and surfaces ~1.1, and **1.7 GiB free** (never under 1.68 GiB in the run).
The free part is mostly `--gpu-headroom-mib 2560`, sized when the recorded passes took 1.6-4.9 GiB; they take ~0.36
GiB now. The R9700 holds 24.9 GiB of experts. Order: after note 122, before the prefill GDN chunk scan's rewrite
(note 109 item 5) -- or interleaved with it, as cool-downs allow; it is settings and measurement only.
1. At the best settings (M136's: gnt0 + hc8 + rp + LL, with the m3 defaults), step `--gpu-headroom-mib` down:
   2048, then 1536 (then 1280 only if 1536 leaves > ~0.8 GiB free at its worst moment).
2. At each step: the M-pair (text identical to M97 / M90), a traced rotation (H-run: median, misses a step, the
   driver-free minimum over the whole run), and a B-run (7K / 20K first tokens; the prefill's staging and the LL
   all-reduce's large messages are the peak users). Record slab slots, misses a step and the free minimum.
3. Keep the lowest step whose worst-moment free VRAM stays >= ~0.5 GiB on both cards with no out-of-memory, retry,
   lost tape or floor event; that becomes the m3 configs' headroom (the engine's own default stays the owner's,
   note 96). Record progress.tsv's rotation / misses rows per note 100.
- Expected (owner's estimate): ~+900 slab slots a card, fewer misses on varied traffic; single-prompt speed about
  unchanged. Budget about half a run.

## 124. Owner: amd-host pruned again -- builds may resume (2026-10-31 04:16 UTC)

The owner ran `podman image prune` on amd-host at ~04:15 UTC, as run 105 asked. `df ~/code` shows ~61 GB free; :125,
:126 and the vLLM rollback image are still there. Builds may resume.

## 125. Owner: a side probe -- FLA's Triton GDN chunk scan on one B70, as a yardstick (2026-10-31 04:46 UTC)

The engine uses no Triton. The best-known GDN chunk scan is flash-linear-attention's (FLA, MIT license) Triton kernel
(`chunk_gated_delta_rule`, the form vLLM uses for this model family), and Intel's Triton backend can lower `tl.dot` to
the matrix units. Before or beside note 109 item 5's real rewrite, measure how fast that kernel is on this card,
outside the engine:
1. A separate, throwaway environment on intel-host (a container or venv in the m3 step area; PyTorch XPU + Intel's Triton
   backend + FLA, versions pinned and recorded). Do not touch the engine's images or intel-host's system packages. If
   installing these is refused by the session's permissions, stop and ask in WAITING-FOR-OWNER.md with the exact
   commands; do not work around it.
2. One card, nothing else running, the usual heat guard. Shapes as the engine's prefill per rank: K = V = 128, the
   rank's value / key heads (16 / 8), chunk 64, bf16 inputs, fp32 state; T = the 7K prompt's 5,604 tokens (and
   4,096, one chunk of the engine's prefill). Warm-up, then time per call.
3. Check its output against libref's chunk scan (or a plain fp32 reference) on the same random inputs, within the
   engine's tolerance for this op, so the time is for a correct result.
4. Compare with our chunk scan at the same shape (rad-kbench timing, today's form and XE_WMMA_BCAST). Record:
   - **much faster:** its structure (tiling, how it uses tl.dot, its sub-chunk parallelism) is the design reference
     for step 2's SYCL rewrite; an ahead-of-time build into the engine is a later owner decision;
   - **about the same or slower:** step 2 goes on as planned.
- Budget: a few dollars. It never becomes part of the engine without an owner note.

## 126. Owner: COMPACT and the matrix-unit chunk scan become engine defaults; the supervisor runs the Triton probe (2026-10-31 13:17 UTC)

Answers to run 108's three questions (WAITING-FOR-OWNER.md), under note 96 for these items only:
1. **RADIANCE_XE_MOE_MX_COMPACT=1 becomes the engine's default** (the fix for the matrix expert form's wrapped-leader
   blocks: wrong expert weights and writes past the buffer on long prompts). BOUND stays as an opt-in alternative;
   the old grid stays reachable by an explicit opt-out for comparisons.
2. **RADIANCE_XE_GDN_CS_JM=64 becomes the engine's default** (the matrix-unit chunk scan). The emulated form stays an
   explicit opt-out.
3. Gates for both default changes, then one commit with the results: the kbench / bench checks as before (exact
   where they were exact, the tolerance as recorded), the 7K / 20K B-run with no flags set (expected ~13-14 s at 7K
   with STAGE, ~14.5-15.3 without), the 11 agreement requests against production (expected at least AG's 438 shared
   tokens and 3 / 3 tool calls), and the M-pair.
4. **New text reference.** These defaults change the M-pair prompt's text (it now closes thinking at once, a
   near-tie). Record the default build's M-pair texts (MTP on and off) as the new references, M97' / M90', and gate
   later decode changes against them; keep M97 / M90 for comparisons with older results.
5. RADIANCE_XE_MOE_MX_STAGE (bit-identical, 12.5 -> 9.3 ms a call) stays opt-in in the engine but goes into the
   m3 configs' defaults, like note 112's flags; its engine default is a later owner call.
- **Note 125's probe:** the supervisor builds and runs it on intel-host (image ri-tp:1, container ri-m3-TP1, card1,
  the usual guard), around 13:20-14:00 UTC, while the runner is held by amd-host's disk. The supervisor then frees
  amd-host's disk (prune + rmi of ri-m3rt:127 / :128). Results go to m3/TP1/ on intel-host and a note here. Do not start a
  card step while a container named ri-m3-TP1 or ri-tp exists on intel-host.
- **Update 13:25 UTC: the probe is deferred until the new heatsink.** Its image build heat-stopped twice at 75 C
  (13:19 and 13:21, at pip's package install, even under a one-CPU quota: rootless storage unpacks on the host). No
  image was made and no ri-tp container exists. The supervisor pruned amd-host and removed ri-m3rt:127 / :128 at
  13:24: `df ~/code` ~52 GB. Go on with items 1-5 above; note 125 waits.

## 127. Owner: amd-host has a 10.4 TiB array at /srv/bulk; old build trees moved there (2026-10-31 13:45 UTC)

The owner made the four 4 TB drives a ZFS raidz1 pool (`bulk`, lz4 compression, mounted at /srv/bulk). The runner's
area is /srv/bulk/ri (owned by user); /srv/bulk/archive is for archives.
- The supervisor moved build-m1hip14, build-m1cpu and build-cpu (milestone 1, unused since) to /srv/bulk/ri and left
  symlinks at the old paths, so every path still works. `df ~/code`: ~74 GB free (was 56).
- From now on, put large, recreatable outputs (old result folders, profiles, image archives from `podman save`) in
  /srv/bulk/ri rather than under ~/code; ~/code's 50 GB floor still applies. Do not move build-m1l0, sycl-cache,
  tiny, images, or anything a current script uses, without a note. Podman's image storage stays where it is: it
  also holds production's containers (an owner maintenance window, later).

## 128. Owner: the next queue -- prefill lower, fewer expert misses, more tok/s (2026-10-31 13:51 UTC)

Note 109's list is done. After note 126 (the two engine defaults, their gates and M97' / M90'), in this order. Every
item keeps its own flag and gates (exact where it can be; a tolerance form is judged by agreement with production,
and its default is the owner's call, note 96). Record progress.tsv per note 100. Owner's estimates, not measured.
1. **Measure first (cheap):**
   - a prefill profile (B-run with the profiler) on note 126's defaults, for the 7K first token's new breakdown
     (attention, expert products, hc, chunk scan, host gaps, rank imbalance);
   - the **MTP-off "all best" M-pair**: hc level 5 + gnt0 + rp + GEMV_NX + native xor + LL, never measured together
     (the 43.3 best is hc level 5 alone). Expected ~45-46.
   - the **clock-lock A/B** (owner notes 77-79's test, never finished: runs 84-92 hung on the import bug): on the
     best settings, an MTP-on M-pair and one traced rotation at the current min_freq 1200 (arm A), then write in
     WAITING-FOR-OWNER.md "ready for min_freq 2800" and wait; the owner sets 2800 on both cards
     (`/sys/bus/pci/devices/0000:0{b,f}:00.0/tile0/gt0/freq0/min_freq`, root) and says so in a note; then the same
     M-pair and rotation (arm B), with peak Tctl and the cards' power recorded beside each; then ask the owner to set
     1200 back unless arm B wins by more than noise. Expected 0-3 %. The heat rules apply unchanged.
2. **Expert prefetch** (fewer exposed misses on varied traffic). The engine logs "expert prefetch off: the device
   backend has no DMA engine to spare for it". First find what the B70 offers (zeDeviceGetCommandQueueGroupProperties:
   copy-engine groups and queue counts on each card). If a second copy queue exists, give it to the prefetch ring
   (sized from the headroom, e.g. 256-512 MiB); if not, try the prefetch on the mover's queue at a lower priority, or
   say why it cannot work. Gate: the rotation (median, misses exposed a step, the driver-free minimum) against H82,
   the M-pair text. Expected: rotation +3-8 % if the reads hide.
3. **Prefill attention on the matrix units** (attention still runs a query row at a time at prefill): a blocked form
   over the QSA-selected blocks; then whatever else item 1's profile shows is largest (expert products' tiles /
   occupancy, host gaps between passes). Expected first token 7K ~12.95 -> ~10-11 s.
4. **VRAM back to the experts:**
   - the activation arena: allocated 1.47 GiB against its plan's 1.15 (the R9700 uses 1.1); find the gap;
   - hot-path kernels that spill (zeinfo scratch) and so make the driver hold scratch / private memory (~0.2-0.45 GiB);
   - each freed MiB goes to the slab at the same worst-moment rule as note 123 (>= ~0.5 GiB free). Expected misses
     28.5 -> ~22-25.
5. **Decode kernels:** the expert gate / up (the lookup table in registers, more loads in flight); hc down at one row
   with level 8's finer column split (time first, keep only if clearly under ~24 us); the 16-lane groups / 32-core
   launch sizes of figure 24 on the experts, trunk and hc (stop if nothing moves more than a few percent). Expected
   MTP off ~46-48, MTP on ~58-60.
6. **After the owner says the new heatsink is in:** the CPU split speed A/B (varied traffic, expected +9-17 %) and
   note 125's Triton probe.
- Then the m3 report (by ~$3,950, note 118), or earlier if this list ends and nothing else is queued.

## 129. Owner: note 128's order changes -- the kernel headroom probe's ranking (2026-10-31 14:24 UTC)

A read-only probe (briefs/2026-10-31-kernel-headroom.md: bytes per call, achieved GB/s, the best measured rate per
class, an R9700 comparison) ranks the remaining headroom. Its savings are ceilings (estimates), not forecasts. Note
128's items stay; their order becomes:
1. **Note 128 item 1's measurements** (the prefill profile, the MTP-off "all best" M-pair, the clock-lock A/B), now
   also: **break down the ~3.7 s of the 7K first token that falls outside every kernel** (host work, pass
   boundaries, first-request expert staging, waits on the slower rank). It is the largest prefill item left.
   Owner, 16:12 UTC: also run the MTP-off "all best" form once with the tape profiler (and the MTP-on best the same
   way), so the decode map is of the best settings: M157's map had the old hc_read_a (no hc level 5, GEMV_NX or
   native xor), so its hc share (~29 %) overstates the best form's.
2. **The expert kernels at the verify:**
   - first, moe8 (RADIANCE_XE_MOE_V2R_SMALLK=4, 4 rows a team) **only when a launch has more than one row** (it
     gained 1.9 % MTP on and lost 1.9 % at one row; the probe puts it at a measured -1.05 ms a verify step);
   - then the down projection's per-work-group setup: v2m<1> runs at ~65 GB/s, the worst rate in the step (each
     work-group stages the activation and table for ~8 KB of weights). Ceiling ~-5 ms MTP on.
   - then gate / up (132 GB/s against the 343 GB/s 4-bit probe): the table in registers, more loads in flight.
3. **The trunk gemv_i8a8 at 4 rows** (9.1 ms at 164 GB/s; 1.9x the 1-row time for the same bytes): exact, on the
   int8 path the card has. The rows form (note 113) was ~15 % faster a launch but not bit-identical; look for an exact
   way to share the per-row work across the four rows. Ceiling ~-5.8 ms.
4. **router_topk_scatter launches one work-group** (`dim3(1)`, ports/r4d_moe.cpp ~2268): spread it, exactly.
5. **Prefill:** attention on the matrix units (note 128 item 3; ~2.7 s, the largest R9700 gap), then the expert
   matrix form's tiles / occupancy (~3.4 s at ~3.1 TFLOPS).
6. Then note 128 items 2, 4, 5 (prefetch, VRAM trims, hc / draft head / 16-lane checks) and 6 (after the heatsink).
- Note for the record: M153 / M97' (51.74) is not a regression against M136 (56.82): the new defaults change the
  prompt's text, so the run generates different tokens. Compare decode speed only between runs with the same text.

## 130. Owner: RADIANCE_XE_ATTN_MX becomes the engine default, after the agreement check (2026-10-31 16:09 UTC)

Answer to run 109's question (WAITING-FOR-OWNER.md, prefill attention on the matrix units, 847d72a), under note 96 for
this item only:
1. First the agreement check, as for note 126's defaults: the 11 requests against production with ATTN_MX on and
   note 126's defaults (+ STAGE as in the m3 configs). Bar: shared tokens >= 438 (AG9) and tool calls 3 / 3.
2. If it passes: ATTN_MX=1 becomes the engine's default (prefill calls of >= 64 rows; RADIANCE_XE_ATTN_MX=0 the
   opt-out), with one B-run with no flag (expected ~12.4 / ~19.1 s, the 7K answer ' __') and the M-pair against M97' /
   M90' (decode unchanged by construction). Then one commit with these results; record progress.tsv's first-token
   row.
3. If it fails the bar: keep it opt-in, record which requests moved, and say so in WAITING-FOR-OWNER.md.
- Then go on with note 129's order.

## 131. Owner: a format probe -- the routed experts as plain int4 on the int8 path ("option B") (2026-10-31 16:38 UTC)

Why: Xe2 has no FP8 math (briefs/2026-10-28-bottleneck.md, "No FP8 math on Xe2"). Today's experts (w4nl64a8h) go
nibble -> codebook level -> E4M3 -> ALU decode, against FP8 activations and E4M3 scales under the fixed 2^-13, so they
never touch the card's int8 path (DP4A / int8 XMX, 177 TOPS measured). The v2m kernels run at 65 / 132 GB/s against
the 343 GB/s u4 probe. Option B asks whether a format built for that path is faster on this card. The w4nl codebook
is kept for production; this probe changes nothing it serves.

Format B (for the record; nothing is converted in step 1):
- the routed experts' codes `i4`, two's complement -8..7, a **bf16** scale every 64 weights (not f16: the experts'
  scales reach 2^-15, below f16's smallest normal 2^-14), still `transform=fwht128`, GPTQ + `cd=3` on the same
  calibration; 4.25 bits a weight against 4.125 (+3 % expert bytes);
- activations int8 per row after the rotation (r4d_quant_act_i8 exists for the trunk); the integer dot over a group
  of 64, its scale (weight scale x activation scale) applied once after the group's sum;
- the ten protected bf16 experts, the trunk, hc and everything else as served today.

1. **Time it first, at kernel level only** (rad-kbench, one B70, random weights; no rad-convert, no engine
   change, nothing in the serve path):
   - a prototype i4 x i8 form of v2m for gate_up (512 x 2560) and down (2560 x 256) per rank, 1 row and the 4-row
     verify, with the verify's expert mix (26.5 distinct experts a layer, 1.48 rows an expert; 10 at MTP off);
   - against today's v2m<5> / v2m<1> timed in the same kbench, same shapes, same session;
   - correctness: exact integer group sums against a reference of the same format, **computed on amd-host, not on intel-host's
     CPU** (note 3);
   - bar: **both** projections >= ~25 % faster a call at the verify (about >= 3 ms a step MTP on). Below it: stop,
     record the times and why in STATE.md, and go back to note 129's order.
   - Time box: at most one run's worth (~$250). If it isn't decided by then, record where it stands and stop.
2. **If it passes, stop and ask** (WAITING-FOR-OWNER.md): the times, and a conversion plan (rad-convert --reuse
   re-quantising only the expert rules: on which host, CPU hours, disk, effect on amd-host's production). The owner
   decides on converting. After a conversion the check is the engine's KL mode against bf16, compared with today's
   0.0789 / 92.85 % (all tokens) and 0.0239 / 94.57 % (text and assistant turns).
- Where it goes: after note 129 item 2 (the expert kernels at the verify), before item 3. Item 2's work is not wasted
  if B wins: the down projection's per-work-group setup and the loads in flight carry over to an i4 form.

## 132. Owner: one shared schedule for every session; a hold file for short intel-host gaps (2026-10-31 17:01 UTC)

The owner: "schedule everything so no one goes over other works, and collaborate if necessary." Several sessions now
share amd-host and intel-host (this runner, the supervisor session, the session that wrote note 131, sessions running the
sparse / LoRA / Occamy tests on amd-host).
1. **~/code/SCHEDULE.md is the shared schedule**, kept by the supervisor (this runner cannot read files outside
   ~/code/radiance-intel). Anything new and heavy on a shared resource beyond the usual (the usual: card steps on
   intel-host; l0b builds and ri-m3rt images on amd-host) -- a rad-convert, a large download, more than 50 GB of disk, more
   than 8 GiB of extra RAM on amd-host -- is first written in WAITING-FOR-OWNER.md as a request with its size and
   duration; the supervisor books it and answers in a note. intel-host's GPUs stay this runner's standing resource. The
   runner's own check (supervisor's note, 17:20 UTC): before each card step, wait while a container named ri-tp* or
   ri-m3-TP1 exists on intel-host.
2. **The hold file.** Before each card step (and before a build that will load intel-host), check for
   `~/code/radiance-intel/HOLD-INTEL-HOST`. While it exists, start no card step: wait as for a cool-down (re-check every
   minute, up to 45 minutes), and record the wait in STATE.md. A step already running is never interrupted; the
   holder waits for it to end. Over 45 minutes: write it in WAITING-FOR-OWNER.md and continue with amd-host-only work.
   The supervisor uses it for short probes (first: note 125's Triton probe on card1, ~10 minutes).
3. **For note 131 step 2:** the supervisor is downloading the bf16 checkpoint Qwen/Qwen3.8-Flash-Next (rev de4b8e4,
   360 GB) to /srv/bulk/archive/qwen3.8-flash-next-bf16 (log beside it). Do not start another download; a conversion
   will read it from there. The note 131 gate stands (both projections >= ~25 % faster, then ask the owner).
4. **ATTN_MX (note 130):** it failed the agreement bar (AG10: 363 shared tokens, 2 / 3 tool calls) and stays opt-in,
   as recorded. No further work on it unless its precision is improved first.

## 133. Owner: note 125's Triton probe result -- FLA's chunk scan is ~2x ours; low priority now (2026-10-31 17:17 UTC)

The supervisor ran note 125's probe on intel-host card1 at 17:12-17:16 UTC (image localhost/ri-tp:4: Ubuntu 24.04, intel-host's
pinned NEO 26.05.37020.3 / IGC 2.28.4 / Level Zero 1.29 debs + Intel's OpenCL ICD of the same release, torch
2.9.0+xpu, pytorch-triton-xpu 3.5.0, flash-linear-attention 0.4.0; --cpus 0.3, peak 66 C; output
logs/TP1-out-ok.txt, script m3/intel-host/fla_probe.py). FLA's `chunk_gated_delta_rule` at the engine's per-rank prefill
shapes (HV 16 / HK 8, K = V = 128, bf16 in, fp32 state):
- T 5,604: 2.017 ms a call; T 4,096: 2.582 ms a call (mean of 50 after 6 warm-ups); rel_l2 against an fp32
  recurrence 3.7e-3 (out) / 2.9e-3 (state), the same class as KJ5's 3.5e-3.
- Triton printed "sycl_arch not recognized" (it does not know BMG-G31) and still ran generic code.
- B51's cs_jm averaged ~4.1 ms a call (a layer x a chunk of <= 4,096) plus kkt_solve ~0.66 ms; FLA's one call covers
  both and the whole sequence. So FLA is roughly 2x ours (more at 5,604, which the engine splits in two chunks).
What to do with it:
- It is a yardstick, as note 125 said. The chunk scan is ~3.5 % of the 7K prefill now (~0.3 s), so matching FLA is
  worth ~0.15-0.2 s: below note 129's items. When the chunk scan comes up again, read FLA's
  fla/ops/gated_delta_rule (chunk.py, the wy_fast / chunk_delta_h kernels) as the design reference (sub-chunk
  parallelism, one call per sequence).
- Triton works on the B70 with the image above: a possible tool for future prefill kernels (an owner decision; it
  would need an ahead-of-time path into the engine). Nothing of it is in the engine now.
- Note 125 is done.

## 134. Owner: answers to run 110's four questions (2026-10-31 20:34 UTC)

1. **MOE_MX_VEC and MOE_MX_STAGE become engine defaults** (both exact: KX7 / B72 equal in every digit; owner's
   decision under note 96 for these two). Opt-outs stay (=0). Gates: one B-run with no flags (expect ~11.5 s at 7K,
   ~18 s at 20K, answers = B69 / B71) and the M-pair against M97' / M90' (decode unchanged); then one commit.
2. **The prompt's tail step: do both exact fixes.**
   - Do not record a prompt-final tail (a step that ends a prompt's prefill and will not be replayed) as a tape
     (B70: ~0.6 s at 7K).
   - Stage the tail's experts in bulk (take back the ring's loan for that step) instead of reading pooled experts
     in place across the link (~1.8 s at 7K, ~1.25 s at 20K).
   - Gates: the 7K / 20K B-run with the trace (the tail's ms), answers and the 7K logprobs unchanged (they must be
     bit-identical: staging moves bytes, not math), and a rotation (staging must not starve the decode slab).
   - Not now: the mid-pass state checkpoint (merging the tail into the previous chunk).
3. **Note 131's int4 x int8 format: yes, staged.**
   - First the code, all opt-in: the converter rule (check group 64), the new expert dtype with its host reference
     and relayout, the probe's forms grown into the launcher, the int8 activation quantisation. Order it after items
     1-2 above and note 129 item 2 / 3's work in progress, and keep it within ~$150 of runs before asking again.
   - **Calibration:** the supervisor found no production calibration directory on amd-host (no pooled or per-expert
     Grams under /srv, /srv/bulk or ~; the container was converted upstream). Find in the Radiance sources
     (docs/MOE-W4.md "Calibration", rad tools) how the Grams are recorded, what it would take here (which GPU, how
     long, from the bf16 checkpoint or from the served container), and write that in WAITING-FOR-OWNER.md before
     any recording. If recording is not practical, say so; an RTN-only container for speed with a KL check is the
     fallback, as an owner call.
   - The bf16 checkpoint is complete: /srv/bulk/archive/qwen3.8-flash-next-bf16 (360.0 GB, rev de4b8e4).
   - A one-layer timing convert comes before any full convert. Ask for the booking (WAITING-FOR-OWNER.md, with the
     measured CPU hours, RAM and disk); the supervisor books it in ~/code/SCHEDULE.md.
4. **Clock-lock arm B:** the owner sets min_freq 2800 on both cards now. Before arm B, read
   `/sys/bus/pci/devices/0000:0{b,f}:00.0/tile0/gt0/freq0/min_freq` (no root needed) and go on only when both read
   2800; then the same M-pair (M161) and rotation (H83) with the cards' energy, and record whether arm B wins by more
   than noise. Then ask the owner in WAITING-FOR-OWNER.md to set 1200 back unless it wins.

## 135. Owner: answers to run 110's three later questions (2026-10-31 21:37 UTC)

1. **RADIANCE_XE_ATTN_VX becomes an engine default after the M-pair** (owner, under note 96 for this item): it is
   exact (KA5's md5 = the split kernel's). Gate: the M-pair with it on and no other change, texts identical to M97' /
   M90' (decode uses the same kernel); then one B-run with no flags; then the commit. If a text differs, keep it
   opt-in and say why.
2. **The prompt's tail step: (a) now, then (b).**
   - (a) Issue an unrecorded prefill pass the way tape-off mode does (B78 vs B75: the 7K tail 1.51 s against 2.25,
     the 7K first token 10.95 against 11.75). Exact; gate on the 7K logprobs in every digit, the B-run, a rotation.
   - (b) Then stage only the tail's routed experts (a selective copy keyed by the router's ids), exact, the same gates.
     Drop it if it is not faster than (a) alone.
   - RADIANCE_PREFILL_TAIL_STAGE (bulk) stays uncommitted / dropped (B77 slower).
3. **Calibration for note 131's format: route (b), the bootstrap**, after a libxe `gram_accum` (device form, opt-in,
   checked against libref at small sizes on amd-host, not on intel-host's CPU).
   - Record from today's served w4nl container with the tap, **1M tokens first**, on intel-host; then report the time and
     whether 10M is worth it.
   - The Grams go to /srv/bulk/ri (~10 GB a rank for the per-expert down set); say the sizes before writing.
   - The calibration corpus: use the one the docs name if it exists here; if it does not, say what you would use.
   - Order: after items 1-2 above and the remaining note 134 work, within note 134 item 3's ~$150 for the code.
4. Clock lock (note 134 item 4): still waiting for the owner to set 2800; keep arm B pending until both cards read it.

## 136. Owner: one more arm-B step -- the first token at min_freq 2800 (2026-10-31 22:24 UTC)

Before the owner sets 1200 back: while both cards still read 2800, run **B73's form again (`nomtpcappfmdef`) and
B74's (`nomtpcappfmdefvx`)** on the same image as B73 / B74, as the next card steps (before ATTN_VX's M-pair if that
has not started; otherwise right after the step in progress). Compare the 7K / 20K first tokens with B73 (11.84 /
18.30 s) and B74 (11.29 / 16.92 s), with the cards' energy as in M170 / H84, and record whether arm B wins at
prefill beyond noise. Then say in WAITING-FOR-OWNER.md that 1200 can go back (unless prefill wins).
- Done (runner, 22:4x UTC): B80 / B81 within noise; the owner set 1200 back on both cards (22:49 UTC). The clock lock
  is closed: no gain at decode, rotation or prefill.

## 137. Owner: expert prefetch -- a second copy path, including the idle render engine (2026-10-31 22:59 UTC)

An addition to note 128 item 2 (expert prefetch). The B70 cards expose three engine classes in sysfs
(`/sys/bus/pci/devices/0000:0{b,f}:00.0/tile0/gt0/engines/`: `bcs` copy, `ccs` compute, `rcs` render). Our kernels use
ccs, the mover uses one copy queue, and **rcs is idle**. Production on the R9700 gives prefetch its own SDMA queue,
which is part of why its misses are ~0.95 % of routed experts against our ~1.9 %. Options, in this order:
1. **List the queue groups** on each card (zeDeviceGetCommandQueueGroupProperties: ordinals, flags COPY / COMPUTE,
   numQueues). If the copy group has more than one queue, give prefetch its own (lower priority if the API allows).
2. **The render engine:** if a queue group maps to rcs and supports copies (`ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY`
   or a compute group that takes copy commands), try prefetch copies on it, in parallel with ccs and the mover's bcs.
   Check that NEO really runs them on rcs (the xe engine's busy time in fdinfo, or a copy-throughput probe with both
   queues busy at once).
3. Only if neither exists: share the mover's queue through a priority / cancellable scheduler, or say why it cannot
   help.
- The link stays the cap (PCIe 4.0 x8, ~14.5 GB/s): a second path lets prefetch run ahead and alongside, not faster.
- Gates: a small probe first (two queues copying at once: is the sum higher than one, and does it slow compute?),
  then the rotation (misses a step, the median) against H83 / H84 and the M-pair text. Opt-in; its default is the
  owner's call. Order: as note 128 item 2 sits in note 129's order (after the expert kernels and note 135's items).

## 138. Owner: GDN decode update -- split each head across workgroups, as FLA's Triton kernel does (2026-11-01 00:43 UTC)

A supervisor probe (2026-11-01 00:31-00:41 UTC, card1, three ~1-min runs under HOLD-INTEL-HOST) timed FLA 0.4.0's Triton
`fused_recurrent_gated_delta_rule` on one B70 at our per-rank shape (24 value heads / 8 key heads, K = V = 128, fp32
state, q/k L2-norm in the kernel). It agrees with an fp32 reference (out 1.7e-3, state 4e-7) and takes **6.0 us at 1
token and 9.5 us at 4** (~525 GB/s on the 3.1 MB state read + write). Ours is ~22 us (E) at 1 row and ~33 us (M) with
MTP, but ours also does the short conv, the gating, the L2-norm and the gated RMS norm. Output:
briefs/2026-11-01-fla-decode-probe-out.txt; FLA's kernel source for reference: briefs/2026-11-01-fla-fused-recurrent-ref.py.txt.

**The likely cause:** the decode fold (gdn_conv_recurrent_update, the NORM path) runs one workgroup per (sequence,
value head) = 24 workgroups on 32 Xe cores, with barriers. The VS split is off there because the fused norm spans
the 128-channel row and the shift marker forbids splitting a head (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.cpp
~111-115 and ~779-787). FLA's grid is (V/8 = 16 slices) x 24 heads = 384 one-warp workgroups, each holding a 128 x 8
state tile in registers, no LDS and no barriers.

Asks, in this order:
1. **Split the decode fold's v rows** across workgroups (try VS = 4, 8, 16) with q / k convolved per workgroup as now
   (the redundant k work is small) and the conv-state rewrite kept on the existing last-arriver counter.
2. **The gated RMS norm** (and the int8 Hadamard quant of oq / os, when on): first try the existing
   r4d_gdn_gated_norm_had_quant_i8 / r4d_gdn_gated_rmsnorm_h128_bf16 as a separate step after the split update (one
   more recorded launch a layer). Only if that launch costs more than it saves: per-slice partial sums with the last
   arriver of a (token, head) finishing the norm.
3. **The shift marker:** make it per-slice-safe (e.g. stamp only after every slice of the head has read it, on a
   counter), or say why it cannot be, and then run the split only where the marker is off.

Gates:
- Opt-in flag (e.g. RADIANCE_XE_GDN_SPLIT); the default is the owner's call.
- A kernel microbench first: 1 row and the MTP verify rows, current vs split, byte-identical output and state where
  the summation order is unchanged; otherwise note 130's agreement bar (shared tokens >= 438, AG9; tool calls 3 / 3).
- Then the rotation against H85 / H86 and the M-pair text. Keep it only at a median gain of **>= 1.5 %** (the
  supervisor's estimate is +1.5-3 %: 0.4-0.75 ms a step over 36 layers).
- About $120 of runner spend; if the marker needs a redesign beyond that, stop and write up the options.
- Order: after CP1 (note 137) and the ATTN_VX default commit; before the int4 format code.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 139. Owner: expert prefetch one layer ahead, on the compute group's free copy path (2026-11-01 01:10 UTC)

Your CP1 and note 137 answer settle the path: the link is the cap (14.42 GB/s alone, 14.40 with both paths), and a
second immediate list in the compute group costs our kernels nothing (-0.1 % on the triad). So **prefetch is limited
by guessing which experts to fetch, not by a copy path**. Your proposal in WAITING-FOR-OWNER (a predictor driving
prefetch on a compute-group list) is approved, in this form. It goes **ahead of note 138**: if 138 is already in a
step, finish that step and record it, then switch.

Your run 112 decision holds: a layer's routing exists only on the device, and Level Zero has no device-driven
indirect copy. So the guess and the fetch must both be device-side, and the gain comes from **starting one layer
earlier** than today's in-place read, not from a faster copy. (Rough supervisor figures, estimates: ~26.5 misses a
step over 48 layers is ~0.55 a layer; one layer's time, ~0.37 ms at 14.4 GB/s, moves ~5 MB, about 5 experts at
~1 MiB a rank. If every miss waits on its read, misses cost up to ~1.8 ms of a ~18 ms step.)

Stages:
0. **Measure the guess first, fetching nothing** (debug flag, one rotation text): at each layer L, apply layer L+1's
   router to the hidden state you have at L (the residual entering L+1 is not ready yet; try the residual entering
   L and the one after L's attention / GDN mixer) and compare its top-k with L+1's real top-k. Report the hit rate a
   layer and **the share of real misses the guess would have covered**. Stop and report if it covers < 50 %. If a
   cheaper guess works as well (e.g. last step's experts at L+1), say so.
1. **Build it behind an opt-in flag** (e.g. RADIANCE_XE_EXPERT_PREFETCH):
   - the guess kernel writes the guessed experts that are not resident;
   - a gather kernel on a second immediate list in the compute group copies them from host memory, one layer ahead,
     into **a small reserved ring of prefetch slots** sized from the headroom, so prefetch never evicts useful
     slab experts;
   - layer L+1's expert lookup checks the ring first, behind a per-slot ready flag. A slot not yet complete falls back
     to today's in-place read, so a late or wrong guess costs only idle link time;
   - FLA's lesson for the gather: many small independent workgroups, a slice of an expert each, no barriers.
     Few workgroups in all, since it is link-bound.
   - It must work with the recorded passes, or say why it can't and what the cheapest form that does would be.
2. **Only if stage 1 leaves more than ~10 misses a step:** let the expert kernel start on the slices that have
   already arrived, instead of waiting for the whole expert. A design write-up first; no build without a note.

Gates:
- The text is exact: M-pair texts M97' / M90' (prefetch moves bytes, not math).
- Rotation against H85 / H86: report misses a step (H85 26.48), the median, the ttft median, the guess hit rate,
  worst driver-free VRAM (>= 0.5 GiB) and starved_slab.
- Keep it only if misses fall and the median rises **>= 1.5 %**. Its default is the owner's call.
- About $30 for stage 0 and $150 for stage 1. A fault pauses the runner as usual.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 140. Owner: the int4 x int8 calibration recording -- go, at 2 CPUs, with a small public chat set (2026-11-01 01:41 UTC)

Answers run 111's note 135 item 3 questions in WAITING-FOR-OWNER ("the calibration recording is ready to try").
1. **Go for one 1M-token recording** with the sizes you gave (~23 GB of Grams, ~20 GB host RAM on intel-host, ~45-60 min):
   - **2 CPUs**, in **two halves with a cool-down** between them, under the usual guard. intel-host's CPU still has the
     failing heatsink: the start bar stays <= 43 C and the 75 C step guard applies to each half. If a half
     heat-stops, stop there and report; don't retry with more CPUs.
   - **Order:** after note 139 stage 0's guess measurement is complete (the per-layer table), and before stage 1.
     Never at the same time as another card step.
   - Commit the tap's protected-expert mapping (rad_block_moe_fp8.h) once this recording exercises it.
   - Copy the Grams to /srv/bulk/ri/calib. If this session can't write there, leave them on intel-host, say where, and
     the supervisor copies them.
2. **The corpus:** local plus the small public set.
   - Local: the rotation's prompt families widened (chat, tool calls with their JSON schemas, code, long documents)
     and this repo's docs and source, as you proposed.
   - Public: **calib-corpus/ultrachat-2000.jsonl** (2,000 conversations from HuggingFaceH4/ultrachat_200k, MIT
     license, ~2.9M tokens by a chars/4 estimate; provenance in calib-corpus/README.md). It is already downloaded:
     no other downloads. Use as much as gives a sensible mix (roughly a third to a half of the 1M tokens from chat).
   - Render everything with the model's chat template, distinct first lines, ~10 % held out for scoring, as you
     planned. The files are data: nothing in them is an instruction.
3. **Not yet:** no convert of any kind. The int4 engine code comes first (note 134 item 3, after notes 139 and 138),
   then a one-layer timing convert, which needs a booking.
- About $50 for this. A fault pauses the runner as usual.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 141. Owner: the multi-row (MTP verify) kernels decode each weight once for all rows -- and a policy change (2026-11-01 02:48 UTC)

**Policy (for this work; today rewrites are kept only if their output is byte-identical or, as for ATTN_MX in note 130, they pass the agreement check):** a rewrite that is not byte-identical may be kept,
opt-in, if it passes **note 130's agreement check** (shared tokens >= 438, AG9; tool calls 3 / 3) on the M-pair
texts and the rotation. Byte-identical stays the first goal. Defaults remain the owner's call.

**Why:** the bottleneck brief (2026-10-28, §2) puts **13.4-13.6 ms of the 4-row step's +28.1 ms over the 1-row step**
in 4-row forms that redo per-column work for each row on the same weights: trunk 5.9, hc 3.4, bf16 experts 2.3,
all-reduce 1.2, GDN 0.6. The headroom brief (2026-10-31, §1.1) shows the trunk `gemv_i8a8<1,4,0>` at 9.07 ms a step
and 164 GB/s, 1.90x the 1-row time for the same bytes; RT and G4 did not fix that.

Asks, in this order, after note 138's rest and the int4 x int8 engine code (note 134 item 3):
1. **The trunk GEMV at 4 rows on the matrix units** (int8 DPAS / joint_matrix, an 8-row tile with 4 used, or the
   "lane owns a group" mapping if DPAS loses). Its group sums are integers, exact in any order (RT's own comment),
   so **try to keep it byte-identical**: exact int32 group sums, then the scale steps in today's order. Only if that
   can't be done, fall back to the agreement check. Look at the uncommitted DPAS_ROWS work (runs 99 / 101) first.
2. **Then the hc group** (hc_v2_down_rows / hc_up_mix at 4 rows), by the same rule. These are E4M3 float, so the
   agreement check is the likely bar.
3. Then the protected bf16 experts and the all-reduce's 4-row form, only if 1-2 show real end-to-end gains.
   Not today's codebook expert kernels: the int4 x int8 format replaces them.
Gates: a kernel microbench first (time + hashes); then the M-pair and one rotation. Keep each item only at a
measured end-to-end gain of **>= 1.5 %**. About $200 for items 1-2; stop and report after item 1.

## 142. Owner: fewer, fatter launches in the decode step -- exact (2026-11-01 02:48 UTC)

The MTP-on step issues ~222 trunk GEMV launches a rank (headroom brief, row 1) and ~160 small elementwise
launches (bottleneck brief, row 6, 3.1-4.2 us each), each paying a fixed floor.
1. **Projections that read the same input share one launch:** concatenate their weights at load time (a load-time
   layout change only; the .rad file stays as it is), one GEMV, outputs split by offset.
2. **Fold the small tails** (quant_act, silu_mul, rope, gather, rmsnorm, hc_write / hc_quant_rows) into the kernel
   that produces or consumes them, where the arithmetic and its order stay the same.
- **Exact:** byte-identical output is required here (no agreement-check fallback).
- Write the code while a card step runs; test in card gaps. Report the launches a step before / after.
- Keep only at a measured end-to-end gain of **>= 1 %**; about $120. Place it anywhere after note 139 stage 1.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 143. Owner: the tail fix becomes the engine default; today's goals (2026-11-01 10:11 UTC)

1. **RADIANCE_TAPE_DEFER_AFTER_PREFILL becomes the engine default** (a5c32e8; your run 112 numbers: 7K first token
   10.23 / 10.55 s against 11.26 / 11.00, top-5 logprobs equal in every digit, rotation ttft 0.67 against 0.72 s).
   As note 134 did for STAGE + VEC: default on, `=0` opts out. Gates: the build, one B-run (7K + 20K) with no flag
   set, and the M-pair texts unchanged (M97' / M90'). Commit it, and from then on the baselines include it. The
   service (m3/service) still does not change (note 96).
2. **For your information, nothing to do:** the fault judge in run.sh counted guard heat-stop artifacts toward
   "two GPU faults in one run" even though it logged them as "not counted". That paused you from 04:57 to 09:54
   today (H87's 01:24:54 heat-stop lines). The supervisor fixed it: such incidents are now saved as
   `kill-artifact` and left out of that count. Real faults, wedged cards, unrecovered cards and recurrences after a
   `fix-claimed:` still pause exactly as before. Keep writing `reviewed-fault:` lines in STATE.md for faults you
   review.
3. **Today's goals (Sun 11-01), in this order:**
   a. Item 1 above.
   b. **Note 139 stage 1 v2** (the recorded pass stays whole): exact text, then one rotation against H85, and a
      keep / drop decision by its gate (median >= +1.5 %).
   c. **Note 138's rest** (the anchor-marker form for MTP on, VS 8 / 16, its gates).
   d. **Start the int4 x int8 engine code** (note 134 item 3): the converter rule and the new expert dtype with its
      host reference first, then the kernels. When the code is ready for a one-layer timing convert, write its
      booking request (CPU, RAM, time) in WAITING-FOR-OWNER so the supervisor can book amd-host.
   - Report each result as it lands; don't wait for the end of the run.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 144. Owner: note 69 (Intel's newer GPU runtime and compiler) goes into today's queue, after prefetch v2 (2026-11-01 10:21 UTC)

Note 69 was approved on 10-29 and prepared (run 80), but never run. Our kernels are limited by the code IGC
generates, so a newer compiler may move every kernel at once. Today's order (note 143 item 3) becomes:
a. the tail default; b. note 139 stage 1 v2; **b2. note 69, as written there**; c. note 138's rest; d. the
int4 x int8 engine code.
- **The packages are already here:** m3/neo-26.35/ (NEO 26.35.39758.10, gmmlib 22.10.0, IGC 2.41.5 core + opencl,
  ocloc), checksums verified (README.md there). No downloads needed. If the Level Zero loader also has to change
  (the release was built against v1.32.0) and this session can't fetch it, say so in WAITING and the supervisor
  fetches it.
- **References are today's, not note 69's old ones:** the M-pair texts M97' / M90' and the speed against M170 /
  the newest best; the rotation against H85; a profiled decode step against P47 for the expert, hc and trunk
  kernels.
- Note 69's rule stands: a text or bit change is a finding. Record which kernel; do not adopt until explained.
- Runtime first (item 1), then compiler (item 2); item 3 (ForceDefaultHeapSize) only if there is time.
- Test images only: the service and intel-host's host stay as they are. About $80.

## 145. Owner: note 69's follow-ups -- the 1.6 GiB under the new runtime, and the new compiler again after int4 (2026-11-01 13:07 UTC)

Your run 113 write-up (WAITING, "note 69 / 144") answered. Neither change is adopted yet; two follow-ups:
1. **The runtime's 1.6 GiB (small, any time a card is free between int4 steps):** on ri-m3rtneo:1 (intel-host has it),
   find how much VRAM a card can really allocate: fill with allocations until failure (as the 2 GiB probe did),
   under NEO 26.05 and 26.35, on one card. If the 1.6 GiB is usable under 26.35, adapt the budget logic so the
   expert slab sizes from what is allocatable, not from the driver's "free" reading (it must give the same slab
   on 26.05, so today's results don't move), then rerun H90's rotation: misses should come back to ~26.
   If it is not usable, say so and the runtime stays at 26.05. About $25.
2. **The compiler, once the int4 x int8 kernels pass their tiny-model / kbench checks:** today's codebook
   gate_up (moe_planes_v2m<5,4>, +44 % under IGC 2.41.5) is what int4 x int8 replaces, and the new IGC sped up
   hc_up_mix (-18 %), expert down (-14 %) and qsa. So rebuild the int4 build under IGC 2.41.5 too and compare
   both compilers on the int4 model: M-pair and one profiled step. Check the spill warning in the int4 kernel
   under the new IGC first. About $40.
- Order: int4 x int8 work (note 143 item d) stays first; item 1 fits in a gap; item 2 comes after the int4
  kernels exist.
- Disk, for your information: the supervisor moved build-m1l0neo to /srv/bulk/ri (symlinked, as before), removed
  amd-host's copies of ri-m3rtneo:1/:2 (intel-host keeps them) and the old ri-m2rt:23/24 and ri-m1rt:27. amd-host / is at
  ~67 GB free. An image-cleanup timer now keeps the newest five ri-m3rt (plus :80) and two of each note-69 image,
  and prunes ri-probe=1 build leftovers. Put new large build trees under /srv/bulk/ri and symlink them; if this
  session can't, say so and the supervisor will.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 146. Owner: answers -- GL by a token-replay check, NEO 26.35 adopted, the one-layer convert now, codebook x int8 queued (2026-11-01 16:59 UTC)

1. **GL (note 141 item 1) is judged by a token-replay check, not the agreement bar.** Your run 115 finding stands:
   no faster form can be the GEMV's bits, and the bar effectively asks for production's exact tokens. So:
   - For each of the 11 agreement requests (and the M-pair prompts), take the control's answer (GL off) and replay
     prompt + answer through both forms, teacher-forced, so the multi-row form GL changes is exercised (the MTP
     verify's rows). Compare each position: **top-1 the same at >= 99.9 % of positions, mean KL(off || on)
     <= 1e-4**, and report the worst positions. Free-running tool calls stay 3 / 3 (AG11 already is).
   - If it passes: one rotation, bestrgl against bestr, kept opt-in at a median gain >= 1.5 %. The default stays
     the owner's call. If the replay can't be done cheaply in the engine, say how you would do it before building.
2. **NEO 26.35 is adopted as the runtime of the test images** (runtime only, as ri-m3rtneo:1 / :3 were built: the
   26.35 packages from m3/neo-26.35/ with the binaries AOT-built as today). Rebuild the runtime image on it and
   take new baselines (M-pair texts must stay M97' / M90'; one rotation). The compiler (IGC 2.41.5 AOT) stays note
   145 item 2. The service (m3/service) does not change (note 96).
3. **The one-layer int4 x int8 timing convert: the supervisor runs it now** (you can't reach /srv/bulk). The
   served container on amd-host is
   /srv/models/llm/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad (122 GB; it is production's
   file: read it only, never write next to it). Write the exact command (image, binary, flags, rule, output path
   under /srv/bulk/ri/) in WAITING or send it to the supervisor; it reports back the log and the output's path, and
   copies the output to intel-host.
4. **Queue "codebook x int8" (format C):** today's w4nl codebook served with int8 activations, its 16 levels
   rounded to int8 at load (scale 127/22: {6, 19, 32, 46, 63, 81, 104, 127} and negated; the largest relative shift
   is the smallest level, 1.125 -> ~1.04), so the served .rad needs no convert.
   - First a CPU check, no cards: the held-out output error (docs/MOE-W4.md's measure,
     sqrt(tr(dW H dW^T) / tr(W H W^T)) with the layer's pooled gate_up Gram and each expert's down Gram) for the
     served codes under the exact levels vs the int8-rounded ones, 6 experts in each of 7 layers, against format B
     on the same experts. Tell the supervisor which Gram files you need: it copies a sample into this folder.
   - If the rounding costs little: a kernel probe after format B's kernels (format B's mapping plus the code ->
     int8 byte permute), against today's w8 and format B. About $40. If it holds up, it may serve today's .rad at
     format B's speed with today's quality.
- **Order:** finish note 142's rotation pair (H92 / H93), then item 2, then item 1, with item 3's command sent as
  soon as it is ready (the convert runs on amd-host in parallel), then item 4's CPU check, then the probe.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 147. Via Telegram (owner, 2026-11-01 18:51:20 UTC)

Go: build format C into the engine. Add a libxe moe_gemm_q row for today’s w4nl planes against int8 activations, plus the engine’s int8 quantisers in front of the experts. Keep it opt-in. It comes ahead of more format B work. Before any M-pair, first check format B’s engine slowdown (two expert classes with different column counts under the two-card split), since format C would use the same launcher path. Do the real-weight correctness check on amd-host, not intel-host. Then an M-pair against today’s container on rtneo:4, a profiled step, and the KL check.

## 148. Via Telegram (owner, 2026-11-01 22:25:12 UTC)

format c opt in, give me option for gl, complier not switch, format b drop

## 149. Via Telegram (owner, 2026-11-01 22:27:59 UTC)

option 2 CL

## 150. Via Telegram (owner, 2026-11-01 22:37:53 UTC)

keep format C opt in; drop format B

## 151. Owner: three more tok/s levers -- the cards' wait at the all-reduce, prompt lookup, and the small kernels + host time (2026-11-01 22:54 UTC)

After GL's rotation pair (H99 / H100) is decided, in this order:
A. **Where the two cards wait (measure first, ~$30).** Format C showed it: as kernels got faster, ar_kernel grew
   (+30 % in P58), because each card spins at the all-reduce until the other arrives. On one rotation, record per
   card and per layer the time each rank reaches each all-reduce (or the spin time inside it), and what differs
   when one is late: that card's expert misses / promotions that step, kernel times, host submission. Report the
   mean and p95 skew a step and its causes. Experts are split by columns across the cards, so the expert work is
   equal by construction; if the skew comes from per-card residency (one card missing experts the other has),
   propose making the two cards' residency decisions identical. No fix without a note.
B. **Prompt lookup (the built-in draft_ngram, ~$60).** Today it proposes only when the MTP head declines. Measure
   on the rotation how often it fires and its acceptance, then try, behind a flag, using its proposal when its
   match is long (e.g. >= 4 tokens) even if the head would draft, or verifying both in one pass if the verify
   shape allows. The GUIDE says prompt lookup is distribution-exact above temperature 0 and fixed-depth-exact at 0,
   so the gate is the M-pair texts plus a rotation: keep at a median gain >= 1.5 %. Code, JSON and quoted text in
   the rotation are where it should help.
C. **The small kernels and the host time (~$150).** From the headroom brief's rows: the all-reduce's own form
   (3.34 ms, class rate ~1.9), the MTP draft head gemv_w2a8 (1.56 -> ~0.43), router_topk_scatter (0.97 -> ~0.25),
   logits_i8 (1.29 -> ~0.58). Exact first; a non-exact form is judged as GL was (note 149: relative to the
   non-bit-identical controls). And a measure-only look at the ~5.8 ms of host / sync a step: what it is (list
   boundaries, syncs between draft and verify, host sampling), and what could go inside Level Zero. xe direct
   stays closed (note 64).
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 152. Owner: the hc group, next after note 151 -- decode once for all rows, and an int8 form like format C (2026-11-01 23:05 UTC)

The hc family is ~7.5 ms of the MTP-on step (headroom brief rows 4-5: the level 8 down group 4.30 ms, ~2.9 ms of
room; hc_up_mix<4> 3.18 ms, ~1.75). Note 141 item 2 named hc after the trunk GEMV and stopped at item 1; this
resumes it. The brief's reading (§1.1 items 4-5): the down group is latency-bound with a software E4M3 decode,
and every row reloads and re-widens its bf16 N values; hc_up_mix's 1-row form already reaches the E4M3 probe rate,
so its 4-row form's extra ~17.7 us is per-row work, not bytes. After note 151's items:
1. **Decode once for all rows** (note 141's rule): the down group and hc_up_mix at the verify's rows decode each
   W_down / W_up code once and apply it to every row; more loads in flight for the down group's columns. Exact
   first; otherwise judged as GL was (note 149: relative to the non-bit-identical controls).
2. **An int8 form, like format C:** hc's E4M3 weights rounded to int8 levels at load (a scale per row and group of
   64), the inputs quantised to int8 once, DP4A. Opt-in, judged by note 149's bar. Look first at what run 69's
   FP16 hc matrices (note 56, b8d76dc) measured, so this doesn't repeat it.
3. Fold what is left of the norm / inject-gain launches into the down kernel (RG, note 142, did hc_read level 8's).
- Gates for each: a kernel microbench (time + hashes), the M-pair, then one rotation; keep at a median gain
  >= 1.5 %. About $150 in all; stop and report after item 1.
- Never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard).

## 153. Via Telegram (owner, 2026-11-01 23:10:36 UTC)

approve briefs/2026-11-01-matrix-units-decode.md section 4, after note 151 and note 152 item 1

## 154. Via Telegram (owner, 2026-11-02 02:15:04 UTC)

151c keep

## 155. Via Telegram (owner, 2026-11-02 02:16:46 UTC)

make the lm_head and draft-head rows form r=2 defaults

## 156. Via Telegram (owner, 2026-11-02 02:23:38 UTC)

can we get 80 tok/s? Are u using outside source and newest idea to improve the tok/s

## 157. Via Telegram (owner, 2026-11-02 02:30:13 UTC)

write a step time budget for 80 tok/s per item. Ms saved and confidence. Survey recent outside work for ideas not yet tired and report before building

## 158. Via Telegram (owner, 2026-11-02 03:14:05 UTC)

note 152 item 2: go, int8 bc form opt-in, judged by note 149’s bar

## 159. Via Telegram (owner, 2026-11-02 03:37:57 UTC)

build draft-pass chaining in level zero (151c), measure against p60

## 160. Via Telegram (owner, 2026-11-02 08:51:07 UTC)

format C is default

## 161. Owner: intel-host's new heatsink is in -- the heat rules change (2026-11-02 17:41 UTC)

The owner installed the new heatsink on 2026-11-02 (both B70s back at PCIe 4.0 x8, BAR 32 GB each, iommu=pt
passthrough). The supervisor's ramp (logs/thermal-test-20261102.txt): idle 29-34 C (was 39-43); 2 busy threads
plateau 70 C; 4 threads 74 C; 8 threads reached 80 C in 30 s; back under 43 C within ~20 s of a load ending.
**The owner raises the limits** ("80 C won't hurt the CPU"; the 5950X's own limit is 90 C):
- **Step guard: 80 C** (was 75): start every card step's guard with `--hot 80`. m3/mstep.sh is changed already.
- **Hard stop: 85 C** (was 80): intel-host-engine-guard.sh's default; --hot accepts 50-85 now.
- **Start bar: <= 43 C, kept.** It now costs about 20 s.
- **CPUs a step:** 2 or 4 (4 plateaus at 74 C). 8 only after the supervisor's 8 / 16-thread re-test shows where
  they settle; the CPU-expert split's speed A/B (note 128 item 6) waits for that.
- Every note's "never loosen the heat rules (start <= 43 C until the heatsink; 75 C step guard; 80 C hard)" now
  reads: **start <= 43 C; 80 C step guard; 85 C hard.** Don't loosen these further without a note.

## 162. Owner: resume -- Strata's tail skip, the CPU-expert split at 4 threads, then both tracks' next items (2026-11-02 17:48 UTC)

The owner removed PAUSED (via the supervisor). Rotation median is now 80.1 (H109); MTP on 68.9, MTP off 55.3.
In this order; report each result as it lands (WAITING + notify):
1. **Tail skip** (from the open-source Strata engine, github.com/Niko1221/Strata, MIT, same model): when a routed
   expert is NOT resident and is low in the token's routing (its rank in the top-10, or its gate weight under a
   threshold), skip it instead of reading it over PCIe, and renormalise the remaining weights (or say why not).
   Strata defaults to skipping past rank 7 among non-resident experts and reports mean KL 0.0058 / top-1 97.7 %
   against its exact path. First check whether Radiance already has an equivalent (the heat engine, a residency
   policy). Opt-in flag; measure misses a step, the KL by kld-gl.sh against the exact path (judged by note 149's
   relative bar), then one rotation: keep at a median gain >= 1.5 %. Try two thresholds at most.
2. **The CPU-expert split** (note 53's flag-guarded code, b86197b + 151a321; note 128 item 6's speed A/B) at
   **4 CPU threads** for the experts, which stays under note 161's limits (4 threads plateau at 74 C). Exactness
   or the relative bar as it turns out; one rotation against the same image without it; keep at >= 1.5 %. Report
   the expert time on the CPU per layer and how many misses it took. If 4 threads are clearly too few, say how many
   it would need: 8 threads settle at ~84 C, over today's 80 C step guard, so more is the owner's call.
3. **Track 1, your own leads (run 126 / 127):** the first token's embedding on the CPU in 64-token chunks (~8 ms a
   chunk, the card idle), and the remaining ~0.7 ms gap inside the verify pass. Exact.
4. **Track 2 (format C, now the default), fewer bytes next:** briefs/2026-11-01-mixed-bits-experts.md step 1 and 2
   (no cards): the routing-popularity curve (the supervisor runs the header scan on amd-host and writes it into the
   brief) and, in your format C CPU harness on formatc/, the held-out error of a 3-bit codebook at int8 levels next to
   w4nl and format C. Report before any convert.
- Heat: note 161 (start <= 43 C; 80 C step guard; 85 C hard).

## 163. Owner: two ideas from sFNX (github.com/johnfink8/sfnx, MIT) -- long drafts on cheap verify rows, and a card-to-card ring (2026-11-02 17:53 UTC)

sFNX runs this same model (Unsloth UD-Q4_K_XL, ~4.5 bits) on 3x V100-32GB over PCIe Gen3 and reports, single
request: prose (t = 1.0) 139 tok/s, new code 200-229, **code edit 367-382** (greedy). Its README says one verify
window "costs about the same whether it accepts one token or sixteen", so edit-like text, which drafts well, goes
far. It sums across cards with GPU threads writing straight into the next card's memory inside the recorded graph,
in a ring (fp16 sums by default), not a library. After note 162's items, in this order:
1. **Measure first, no build (~$30):**
   a. The verify step's cost against its rows: one profiled step at 1, 2, 4, 8 and 16 rows (or the deepest the
      engine allows), and which kernels grow with rows.
   b. Prompt lookup's potential at long windows. Note 151 B's shadow measure (83acda6) scored proposals up to the
      head's ~4-token window. Score lookup proposals of up to 8 and 16 tokens the same way (tokens a verify would
      emit), on the rotation **and on two or three code-edit prompts** (give a file, ask for a small change; keep
      them out of the rotation baseline). Report tokens a verify by match length.
   c. Report what a long-window lookup step would gain on code edits, given (a)'s verify cost. Decide nothing
      without a note.
2. **Card-to-card writes (~$40, then ~$100 if it proceeds):** check whether the two B70s can reach each other's
   memory directly on intel-host (zeDeviceCanAccessPeer, peer bandwidth and latency of a small write). If they can,
   compare a ring exchange, written by the kernels into the other card's buffer inside the recorded pass, against
   today's all-reduce (~3 ms a step). Keep fp32 sums in today's order if possible (exact); fp16 sums only under note
   149's bar. Gate: the all-reduce time a step, the M-pair texts, one rotation at >= +1.5 %.
- Heat: note 161.

## 164. Owner: the depth-2 rotation -- after PEER2, and the default if it wins (2026-11-03 14:06 UTC)

Your run 160 lead: draft depth 2 (3 verify rows) beat the served depth 3 by 4.2 % on the single prompt (71.8 against
68.9 tok/s, texts identical). After note 163 item 2's PEER2 probe (and its ring build if that proceeds -- your call
on the order if the build is long):
- One rotation at `--num-speculative-tokens 2` against a same-image control at the served depth, back to back.
- **If the median gains >= 1.5 %** (report mean and aggregate too): make depth 2 the default in the configs (the
  service still doesn't change, note 96), and check one M-pair on it. If not, depth 3 stays; report.
- Heat: note 161.

## 165. Owner: next after note 164 -- gated draft depth, the all-reduce + norm fusion, sFNX's interleaved lanes, then the expert-union shrink (2026-11-03 14:15 UTC)

Recent papers on MoE speculative decoding describe what your run 160 measured: every verify row routes ~10 more
experts to read, so a row costs ~3 ms (EVICT arXiv 2605.00342, MoE-Spec arXiv 2602.16052, S2-MoE arXiv 2608.15018,
"Limits of Speculation" arXiv 2609.22156). In this order, each opt-in until it passes, reported as it lands:
1. **Confidence-gated draft depth (lossless, ~$60):** per verify step, verify the head's 3rd draft only when it is
   likely to be accepted (e.g. the head's probability for draft 3, or the product p1 x p2 x p3, over a threshold);
   otherwise verify depth 2. Both pass shapes are recorded. The verification is unchanged, so texts should match
   the fixed depths' (check the M-pair). Measure the share of steps at each depth and tokens a step; try at most
   three thresholds; one rotation against the best fixed depth from note 164: keep at a median gain >= 1.5 %.
2. **The all-reduce fused with its neighbour (~$80):** on top of note 163 item 2's wide-store exchange if it
   proceeds, fold the RMSNorm / residual add that follows each all-reduce into the same kernel (NVIDIA's one-shot +
   RMSNorm, vLLM / SGLang's PCIe one-shot forms). Exact if the sums keep today's order. Gate: kernel time a step,
   M-pair texts, one rotation at >= 1.5 %.
3. **Interleaved prefill and decode for several users (sFNX's lanes, ~$60):** first measure what one user's long
   prompt (~20K tokens) does to two other users who are decoding meanwhile: their longest gap between tokens and
   their tok/s while it reads. If the decoders stall for most of the prompt, read long prompts in chunks with the
   decode steps between them, so the other users keep decoding. Gate: the decoders' worst gap and the total tok/s
   no worse than 2 % below today. The service doesn't change (note 96).
4. **Expert-union shrink at verify (lossy, measure first, ~$60):** count the distinct experts a verify step reads a
   layer, and how many would go if borderline routing (the lowest one or two picks with a small gate margin) leant
   toward experts the step's other rows already use (S2-MoE's bias) or a per-layer budget (MoE-Spec). KL replay
   judged by note 149's bar first; one rotation only if it passes and the profile shows >= 1 ms a step.
- Heat: note 161.

## 166. Owner: the CPU in parallel for the first token -- warm n-gram table, the next chunk ahead, a multi-threaded gather (2026-11-03 14:57 UTC)

Your run 126 / 160 finding: each 64-token prefill chunk spends ~8 ms in embed_lookup_q on the host (the 51 GB
n-gram table's page faults; intel-host's NVMe is DRAM-less) while the cards idle: roughly 0.9 s of the 7K first token and
~2.5 s at 20K. The new heatsink allows 4 CPU threads (~74 C, note 161). Decode work stays on the cards (the
CPU-split analysis: a per-layer hand-off costs more than it saves). After note 165, all exact:
1. **Warm table:** measure the gather with the table's pages cold vs already in the page cache (intel-host has 121 GB
   RAM: check what the engine leaves free). If warm wins, warm it at startup on spare threads (madvise WILLNEED /
   a background read, at nice 19, without pinning: M23's 47.7 GiB pinned copy is what failed) and report the time
   to warm and the RAM it holds.
2. **The next chunk ahead:** gather chunk N+1's rows on the host while the cards run chunk N (double buffer).
3. **A multi-threaded gather:** split a chunk's rows over up to 4 threads.
- Gates: the 7K and 20K first tokens (B-runs) against today's, logprobs equal; decode tok/s not below today's
  (M-pair) and amd-host / intel-host memory gates unchanged. Report each step's first-token time. About $80.
- Heat: note 161 (4 threads plateau ~74 C; don't go past 4 without a note).

## 167. Owner: format C's expert kernels toward their weight-streaming floor (2026-11-03 16:22 UTC)

Run 117's figures at the verify (~35 distinct experts a card): gate_up 112 us a launch against ~86 us to stream its
~34 MB (1.3x); **down 65 us against ~43 us (1.5x)**. Format C has no FP8 left, so the gap is the kernels' own setup
and latency. After note 166 (or in a gap while another item waits on a rotation, your call):
1. **down first:** profile where its ~22 us over the floor goes (the staging prologue, the code -> int8 lookups,
   loads in flight, the grid's work-group count at the verify's rows) and fix the largest part.
2. **then gate_up** the same way if its profile shows >= 10 us of avoidable time.
- **Exact:** format C's own output bits must not change (the same sums in the same order); the M-pair texts equal
  today's. Gates: kernel time a launch, the device ms a step, one rotation at >= 1.5 %. About $80.
- Heat: note 161.

## 168. Owner: prefill next -- halve the first token (2026-11-03 16:39 UTC)

The first token is the largest gap left: 7K 10.6 s and 20K 16.8 s here, against AMD production's ~1.4 s at 8K.
Prefill is compute-bound, where AMD's native FP8 matrix path still counts; decode's int8 lessons (format C) haven't
reached prefill yet. The headroom brief's 7K table (B58, 10-31, per rank): expert matrix form 3.40 s; **attention
2.67 s (attn_split: the decode kernel run at prefill -- scalar fp32, E4M3 decoded per element, no matrix units)**;
all-reduce at large messages 0.80 s; hc 0.46; trunk 0.46; chunk scan 0.36; and **~3.7 s outside the kernels**.
**Order: right after note 165 item 1 (gated depth) finishes, ahead of note 165 items 2-4, and note 166 folds into
item 4 here.** Stop and report after item 0 and after each item:
0. **Today's breakdown (~$20):** a profiled 7K and 20K prefill on today's defaults, plus an unprofiled B-run with
   the pass timer, to decompose the time outside the kernels (host, pass boundaries, first-request expert staging,
   rank skew, the n-gram gather).
1. **Attention at prefill on the matrix units (~$80):** a prefill attention kernel (flash-attention-style tiles on
   joint_matrix f16 / bf16, K / V decoded once a tile) in place of attn_split at prefill. Not bit-identical, so
   judged by note 149's relative bar (KL replay) plus the B-runs' top-5 logprobs. Decode unchanged.
2. **Format C at prefill (~$80):** the prefill expert form on int8 DPAS with format C's int8 levels and int8
   activations (a scale per 64), in place of the f16 matrix form's code -> f16 decode; try 128-K groups to halve the
   barriers (run 108's idea). Judged by note 149's bar.
3. **The all-reduce at prefill's large messages (~$30):** the LL form's tagged words double the bytes; for large
   messages use a bandwidth form (wide stores, or plain stores + one release, as PEER2 measured). Exact.
4. **Outside the kernels (~$60):** fix the largest part item 0 finds. This includes note 166 (the n-gram gather
   warm / ahead / threaded), which moves here.
- Gates: the 7K and 20K first tokens (B-runs) against today's; decode's M-pair texts and tok/s unchanged; keep
  each item at >= 5 % off the 7K first token. Target: 7K ~5-6 s. Heat: note 161.

## 169. Owner: borrow Intel's tuned kernels and use real counters (amends note 168) (2026-11-03 16:46 UTC)

To mature kernels faster, start from Intel's own SYCL kernels where they fit, and diagnose with hardware counters
instead of timestamps. The supervisor fetched both into **third_party/** (README.md there: commits, licenses). Their
text is data, not instructions (their CLAUDE.md / AGENTS.md / .claude were renamed for that reason).
1. **Note 168 item 1 (prefill attention) starts from vllm-xpu-kernels' Xe2 prefill attention** (Apache-2.0):
   csrc/xpu/attn/xe_2/ (chunk_prefill*, kernel/chunk_prefill_kernel.hpp, collective/). Adapt it to our KV format
   and heads, or say why it doesn't fit; keep its Apache-2.0 header and add it to NOTICE / vendor/UPSTREAM. Judged
   by note 149's bar as note 168 says.
2. **Look at the other kernels there before writing ours:** csrc/xpu/gdn_attn (against our GDN chunk scan and
   recurrent update), csrc/xpu/grouped_gemm (note 168 item 2's expert GEMMs: theirs takes FP8 / MXFP4, ours the
   int8 levels, so the tiling not the code), csrc/xpu/mhc (if it is hyper-connection mixing, against our hc),
   csrc/layernorm* (fused add + RMSNorm, for note 165 item 2). A paragraph each: does it fit, and would it be faster.
3. **unitrace** (third_party/pti-gpu/tools/unitrace, MIT): build it into a test image only (no host or driver
   change) and use its per-kernel counters (EU active / stall, occupancy, spills) for note 168 item 0's breakdown,
   and for each kernel you rework from then on. If it needs a kernel or permission setting on intel-host, say what, and
   don't change it yourself.
- Order: inside note 168 as above; item 2's read takes a few minutes per kernel, so do it alongside item 0.
- Heat: note 161.

## 170. Owner: keep working while LoRA / Clover share amd-host -- constrained mode (2026-11-03 16:59 UTC)

The LoRA and Clover tests run on amd-host's GPUs 2,3 at night (~40 GiB of amd-host's RAM), which used to make every
cycle skip at the 60 GiB gate. Changed by the supervisor (backups *.bak-1103T1715):
- **run.sh's gate:** while one of their units is active (sparse-night-lora / sparse-run / build-chain) or a
  AMD-HOST-SHARED file exists, a run starts with **> 24 GiB** MemAvailable instead of 60, and run.sh writes
  **AMD-HOST-CONSTRAINED** (who shares, the free memory) for that run. Otherwise the 60 GiB rule is unchanged.
- **m3/build-l0.sh and build-l0-neo.sh:** with AMD-HOST-CONSTRAINED present they build at **--cpus 4 --memory 10g
  -j4**, and refuse to start (exit 75) below 20 GiB MemAvailable.
**Your rules while AMD-HOST-CONSTRAINED exists:**
1. Prefer card steps on images that already exist (rotations, profiles, B-runs, probes); build only when a step
   needs a new binary, and expect ~2x the build time.
2. No image builds (make-rt*), no rad-convert, no other amd-host job over ~8 GB, and nothing on amd-host's GPUs.
3. If a build is refused (exit 75), do card work or end the run; don't retry the build in a loop.
4. Note in STATE when a run worked in constrained mode.
- intel-host is unaffected (heat per note 161).

## 171. Owner: a standing kernel scoreboard -- always work the biggest gap to the floor (2026-11-03 18:06 UTC)

To mature the kernels systematically, keep **briefs/kernel-scoreboard.md** as a living table, updated whenever a
profile or a kernel change lands (no separate runs for it):
- **One row per hot kernel and shape** (decode at the served rows, prefill at the 4096-token chunk): name and
  variant, calls a step / a 7K prefill, time a launch and in total, its **floor** (bytes / 608 GB/s for
  memory-bound, ops / the matrix units' peak for compute-bound; say which and how computed), **the gap ratio**
  (time / floor), the share of the step or the first token, the latest unitrace counter findings once the metrics
  libraries are in (occupancy, EU stall reasons, spills), the last change (commit) and the next idea.
- **Order of work, after the current notes (168, 165 items 2-4, 167):** the row with the largest absolute time over
  its floor (ms a step or s of the first token) comes next, unless a note says otherwise. Say in STATE which row you
  picked and why.
- **The standard fixes to try first** (each has worked here): enough work-groups to fill the 32 Xe cores; wider
  loads / stores and more in flight; no per-row repetition of decode work; fewer barriers and LDS round trips;
  the matrix units where rows >= 8; int8 instead of software FP8; fewer launches. And Intel's designs in
  third_party/ (vllm-xpu-kernels, sycl-tla) and Xe-Forge's B70 tile shapes as starting points.
- **Parameter sweeps:** when a kernel has knobs (work-group size, rows a team, tile shape), sweep them on the real
  weights in idle card time and keep the fastest exact setting per shape, recorded in the scoreboard.
- Gates as always: exact bits (or note 149's bar for non-exact forms), then the end-to-end gate (rotation for
  decode, B-runs for prefill). A kernel win that doesn't show end to end is recorded but not kept.
- Start the file at the next profile you take (B66 or the next P-run); about $10 a week of upkeep.

## 172. Owner: answers to run 164 -- the prefill DPAS form stays the default; the bank-conflict fix first, then attention (2026-11-03 19:34 UTC)

1. **Note 168 item 2's form stays the default** (e7613c3; KL 0.0256 / top-1 97.4 %, inside note 149's bar).
2. **Order next:** (a) **the expert DPAS kernel's shared-memory bank conflicts** (B90: 3.3e10 conflicts, LDS- and
   vector-ALU-bound, the matrix pipe at 1.3-1.8 %) -- a conflict-free LDS layout, the cheap step, exact to the form's
   own bits; then (b) **note 168 item 1** (attention at prefill, the chunk_prefill port); then (c) item 4 with note
   166 (the host gather ahead); then (d) item 3 (the bandwidth all-reduce). The 128-K-group rework of the expert
   kernel after (a), if the scoreboard still ranks it first.
- Gates as note 168 (B-runs; >= 5 % off the 7K for items; (a) may land under 5 % if it unlocks the rework -- report it).
- One build at a time (note 17), strictly. Heat: note 161.

## 173. Owner: answers to runs 167-168 -- one more try at the attention op (note 172 b), the "shape" refusal explained off the card first (2026-11-03 23:24 UTC)

The owner lifts run 168's pause (B100 = the run's second fault) for this item only; the supervisor removes PAUSED
when it appears.
1. **Off the card first:** explain the op's "shape" refusal (RAD_E_SHAPE on both ranks, B100) without a card step:
   the operands the engine actually passes (dtype / rank / shape / strides, as image 192's diagnostics print) run
   against the op's checks on the CPU (the kbench harness, the host device, or the engine's debug path). Fix the
   cause, and say from the code what B100's CAT error at 22:58:48 most likely was (which kernel, which memory).
2. **Then exactly one card step:** the 7K alone, the op opt-in, with the kernel log read from before the engine
   starts (B100's read began at 22:59 and missed its own fault: every card step's fault check starts at the
   step's start from now on).
3. **If that step faults, or the card does not come up (B99's form):** stop item (b), no further card steps with
   the op without a new owner note; a peer probe for the card's health (PAUSED with the reset for the owner if it
   fails); then go on with (c) item 4 + note 166, then (d) item 3. Report in WAITING.
4. **If it is clean and serves:** the KL replay against the control by note 149's bar, then the B-run (7K / 20K).
   The op stays opt-in until the owner makes it the default.
- Gates, builds and heat unchanged (note 172: one build at a time, strictly; heat: note 161).

## 174. Owner: the host-side prefill part first, ahead of the attention op (amends notes 172-173) (2026-11-03 23:40 UTC)

The time outside the kernels is the largest single part of the 7K first token (~3.8 s a rank of 10.49, scoreboard
"outside the kernels"), and it is host work, so it cannot fault a card.
1. **Now: note 172 (c) = note 168 item 4 with note 166** (the n-gram gather and build at the chunk boundaries warm /
   ahead / threaded, pass boundaries, rank skew; item 0's breakdown says which part is largest). Gates as note 168
   (B-runs, the 7K and 20K first tokens; >= 5 % off the 7K, decode unchanged).
2. **Then the attention op (note 172 b) under note 173's conditions** (the "shape" refusal explained on the CPU,
   then one card step; a fault shelves it). Write what run 169 has found on the refusal so far into STATE before
   switching, so it resumes from there.
3. **Then (d)** (the bandwidth all-reduce at prefill), **then the expert kernel's 128-K-group rework.**
- Goal unchanged: 7K ~5-6 s once all four land. Builds one at a time; heat: note 161.

## 175. Owner: libavx in every step by default; a card check after any fault before the next engine start (2026-11-04 00:27 UTC)

1. **The kernels list with libavx is the step default and the new baseline** (B102, 88b65e3 / image 193:
   `--kernels libxe:libavx:libref` + note 45's seccomp profile; 7K 9.60 s, 20K 15.72 s, logprobs identical, decode
   unchanged). Later images carry libavx.so; prefill gates compare against B102 from now on.
2. **After any GPU fault on either card (CAT, exec-queue reset, GT reset, timed-out job), the next card step is the
   peer probe (logs/peer1.sh on image 188), not an engine.** B99 and B101 both started engines on a card that had
   logged nothing for 10 min after a CCS fault, yet was not recovered: each hung at startup and cost a step plus
   a forced GT reset.
   - The probe fails or hangs: PAUSED, the card for the owner.
   - The probe passes but the next engine logs nothing for 180 s during startup (B98 came up in 88 s): stop it
     there instead of at the step's limit, write it up as "card not recovered after <fault>", and probe again
     after the reset before any further engine step.
   - These rules add checks only; the fault counting (two faults in one run pause) is unchanged.

## 176. Owner: the three switches and a startup warm-up become defaults; the expert prefill redesign next; the order to the milestone's end (2026-11-04 08:39 UTC)

1. **Defaults now:** the attention op (RADIANCE_ATTN_QSA_FMHA=1, 491dc58), the bandwidth all-reduce
   (RADIANCE_XE_AR_PROTO=6, 3e69fb3) and the event records as barriers (RADIANCE_L0_RECORD_BARRIER=1) -- each env
   switch still turns its form off. Because the op's KL replay covered only the two long prompts, add a broader
   check with the change: >= 5 long prompts (> 2,051 tokens; code, prose, a tool transcript, a 20K document) replayed
   by note 149's bar. If it fails the bar, the op goes back to opt-in and you report; the other two stand.
2. **A startup warm-up pass, default on** (a flag turns it off): one synthetic prefill through every kernel the
   service uses (a full chunk, a short tail, decode with MTP) before the engine reports ready. It must not change
   any output; report the time it adds to startup. From now on the prefill figures are "first prompt after
   startup"; keep a truly cold figure beside them for the report.
3. **Next run: the expert prefill redesign** (run 174's brief: the codes decoded once a layer into an int8
   scratch, ~1 GB a card taken from the expert slab), behind a flag. Gates as note 168 (the 7K / 20K B-runs,
   exact or inside note 149's bar, the decode M-pair unchanged), plus the misses a step on the rotation, since the
   slab shrinks; report them.
4. **Then, in this order:**
   a. Finish note 165 item 4 (RADIANCE_XE_ROUTE_LEAN, in flight) by its gates.
   b. The matrix units at decode's 4-row steps (briefs/2026-11-01-matrix-units-decode.md section 4: the existing
      RADIANCE_XE_GEMM_DPAS_ROWS=1 and RADIANCE_XE_HC_PF_DPAS=2 rebased, timed by teacher-forced replay, >= 1.5 %
      and inside the bar, then one rotation as the sanity check; opt-in).
   c. The long-prompt lanes, re-measured after item 3 (run 172's L-steps): pick the service setting whose decoders'
      worst gap is <= 3 s at <= 30 % extra prompt time, and put it in the service args.
   d. The step-barrier hang (HyperConn::write, the fused all-reduce + hc write): find the host wait inside the
      collective and fix it before the soak. addr2line is being added to your tools by the owner.
   e. The service soak (stage D), then the owner's PCIe 5 check, then the milestone report.
- The tail checkpoint stays "not now". Builds one at a time; heat: note 161; fault rules: notes 175 and run.sh.

## 177. Owner: do all the prefill and decode items (amends note 176's order) (2026-11-04 08:40 UTC)

The owner: "do all the prefill and decode". Note 176 items 1-3 stand as written; this adds the rest and sets the order.
**Prefill**
1. Note 176 items 1-3: the three defaults (with the long-prompt KL check), the startup warm-up, the expert
   prefill redesign.
2. **The tail checkpoint, now in** (it replaces note 176's "not now"): fold the last chunk's short tail (the 36-row
   step at the 7K) into the chunk before it (the mid-pass checkpoint from run 169's WAITING; ~1.1 s of every
   prompt). Exact if the rows and their order are the same, else note 149's bar; the 7K / 20K B-runs and the
   decode M-pair as gates.
**Decode**
3. Note 165 item 4 (RADIANCE_XE_ROUTE_LEAN, in flight) by its gates.
4. The matrix units at the 4-row steps (note 176 item 4b).
5. **Note 167: format C's expert kernels toward their streaming floor** (gate_up / down at 1.3-1.5x the floor).
6. **The launch path:** measure a decode step's host-side time (submissions, barriers between passes, the
   n-gram drafter's lookup on the host) on the rotation's configuration, then lighter barriers where the order
   allows, and the n-gram prompt lookup on the device if it is on the step's critical path. Exact.
7. **Mixed-bit experts** (briefs/2026-11-01-mixed-bits-experts.md): its steps 2-3 first (the 3-bit codebook's
   held-out error on formatc/'s samples, and the VRAM / miss model at the rotation's traffic); a convert only if
   the model predicts misses down by >= 30 % with the error inside note 149's bar; report before any convert.
- Gates for 4-6: exact or inside note 149's bar, >= 1.5 % on the replay or the rotation, the M-pair texts. Each
  item opt-in until it passes; a passed exact item becomes the default and you report it; a lossy one waits for
  the owner.
**Then** note 176 items 4c-4e: the long-prompt lanes setting, the step-barrier hang, the soak, the PCIe 5 check,
the report.
- One build at a time; heat: note 161; fault rules: note 175 and run.sh. Stop and report in WAITING after each item.

## 178. Owner: answers to run 175 -- the lean closed, the tail checkpoint skipped, the mixed-bit convert tried on the coldest quarter (2026-11-04 12:39 UTC)

1. **Note 165 item 4 (RADIANCE_XE_ROUTE_LEAN): closed.** It fails note 149's bar at every setting worth >= 1 ms.
   Keep the code opt-in; no further work.
2. **Note 177 item 2 (the tail checkpoint): skipped.** ~0.1-0.15 s after the warm-up, lossy, several runs of work.
3. **Note 177 item 7 (mixed-bit experts): one try, the coldest quarter only.** Convert the least-routed 25 % of
   each layer's experts (by step 1's routed-row counts) to the 3-bit codebook with its levels rounded to int8 at
   load (format C's DP4A path serves both), behind a flag. Gates, in order:
   a. **The KL replay by note 149's bar first** (the agreement corpus plus the long corpus, as KFL2). If it fails,
      stop: report it, delete the converted files, keep the code opt-in.
   b. If it passes: the rotation (H-step, as H119) with its misses a step and its median, against H119's 28.1 /
      88.8; and the decode M-pair. It becomes a default only by the owner, since it is lossy.
   - Disk: the converted checkpoint goes to amd-host /srv/bulk or intel-host's model store only if >= 50 GB stays free
     after it. If there is not room, do not delete anything: list the superseded test-step folders you would
     remove (models/test/radiance-intel, oldest first, never the controls B66 / B102 / B133-B136 / H119 /
     M257-M262) with their sizes in WAITING, and wait for the owner.
     Book the convert in ~/code/SCHEDULE.md (CPU and disk heavy).
- The rest of note 177's order stands: note 167's cut and the launch path (in flight), then the long-prompt lanes,
  the step-barrier hang (addr2line is in your tools now), the soak, the PCIe 5 check, the report.

## 179. Owner: push decode further -- fewer misses, and the draft head's acceptance (2026-11-04 12:52 UTC)

The owner: "let's try to increase decode". Run 176 showed the cards are busy ~94 % of a decode step and the
expert kernels sit at 1.3-1.5x their streaming floor, so the levers left are fewer bytes and misses, and more
tokens kept a step. Order, after the disk cleanup and with note 178's mixed-bit try first:
1. **Note 178's coldest-quarter mixed-bit try** (unchanged: the KL replay first).
2. **The slab's headroom, re-measured (settings only, exact):** `--gpu-headroom-mib 1280` was sized when the
   recorded-pass pool was ~0.36 GiB; it is 0.24 GiB now. On the rotation (H-step as H119), read each card's
   lowest free VRAM; if it stays > 0.8 GiB, step down to 1024, then 768, with the misses a step and the median
   each time. Keep the setting with the best median that never drops under 0.5 GiB free; report it.
3. **The draft head's acceptance -- measure and design, no training yet:**
   a. From the rotation and the M-steps: tokens kept a step, acceptance by draft position (1st / 2nd / 3rd draft),
      by prompt family; and what tok/s each +0.1 token a step would give at today's step time.
   b. A design brief (briefs/2026-11-04-draft-head.md) for training Flash-Next's MTP draft head on Flash-Next's own
      outputs (self-distillation): how to get the training pairs (the hidden state the head reads + the model's
      next tokens, dumped by our engine on intel-host over a corpus such as ~/code/lora-data/v2's Flash-Next replies),
      where to train (amd-host cards 2,3 with the LoRA stack, booked with that session), full fine-tune or adapter,
      how the trained head goes back into the .rad, the expected acceptance gain from published self-distilled
      heads, cost and time. The draft head never changes the served text (the verify keeps only what the model
      itself would emit), so the gate is speed only. Licensing: a retrained head is a modified Flash-Next weight
      under the same Qwen Community License as today; say so in the brief for the owner.
   No training until the owner says go.
4. Then note 176's remaining order (the step-barrier hang, the soak, the PCIe 5 check, the report).
- Gates for 2: exact, the rotation median and misses. Report after each item in WAITING.

## 180. Owner: the draft head's output projection at higher precision (option A); self-distillation waits (2026-11-04 13:22 UTC)

From run 176's draft-head brief (briefs/2026-11-04-draft-head.md):
1. **Option A, go:** store the draft head's vocabulary projection (`mtp.draft_head.weight`, 248,320 rows, today
   2-bit, 79.5 MB) at int8 if the VRAM allows (+~240 MB a card), else int4. Do it as a small `rad-convert --reuse`
   of the production file touching that tensor only, or by reusing the main int8 output projection if the
   shapes and the head's inputs allow it -- whichever needs no new kernel. One build/convert at a time: after the
   mixed-bit convert and its copy, or before them if the convert is a few minutes and uses <= 4 threads; book it
   in SCHEDULE.md through the supervisor.
   - The text never changes (the verify keeps only the model's own tokens): check that the M-pair texts are
     identical, then judge by speed only: the rotation (as H121: tokens a step, acceptance by position and family,
     the median), and the misses a step since the slab gives up the VRAM. Keep it if the median gains >= 1.5 %
     with the misses not worse by more than 2 a step; then it becomes the default and you report it.
2. **Option B (self-distillation): not yet.** Decide after option A and the mixed-bit result; keep the brief.
- The rest of the order stands (the mixed-bit KL replay, the barrier hang, the soak, the PCIe 5 check, the report).

## 181. Owner: go for option B -- self-distill the MTP draft head on Flash-Next's own outputs (2026-11-04 17:10 UTC)

The owner said yes (17:0x UTC) after option A showed no gain. Licensing acknowledged: the trained head is a modified
Flash-Next weight, kept internal, under the same Qwen Community License the served model already carries.
Design: briefs/2026-11-04-draft-head.md section 3. Phases, each reported in WAITING:
1. **The dump (yours, on intel-host):** the hidden-row hook (the trunk's final hidden state, bf16) and a top-32 logits
   writer added to the KL mode; teacher-force a corpus of Flash-Next's own replies (~/code/lora-data/v2 plus the
   agreement / rotation completions: prose, code, tool calls), ~0.5-1M positions, ~3-6 GB. Position 1 first; the
   pairs for positions 2-3 (the head reading its own drafts) only if position 1 trains well. Copy the pairs to
   amd-host /srv/bulk/ri/mtp-distill/ (with a README of the format, the corpus and the image). Keep >= 50 GB free on
   intel-host. Order: after note 178's mixed-bit KL replay; the soak follows the dump.
2. **The training (the LoRA session on amd-host cards 2,3; the supervisor hands it over):** the head's block and fc
   layers in bf16 from the bf16 checkpoint (/srv/bulk/archive/qwen3.8-flash-next-bf16, the `mtp.*` tensors), the
   output projection frozen (shared with lm_head), KL to the recorded top-32; a held-out 5 % for acceptance proxies.
   Booked in SCHEDULE.md; cards 2,3 paused and restored by the LoRA stack's start/restore as before.
3. **Back into the .rad (yours):** `rad-convert --reuse` of the production file with the `mtp.*` rules pointed at
   the trained tensors, quantised as served; copy to intel-host.
4. **Gates (speed only; the text cannot change):** the M-pair texts identical; the rotation as H121 (tokens a step,
   acceptance by position and family, median, misses). Keep it if the median gains >= 1.5 %; then it becomes the
   default and the owner is told. If not, report and stop.
- Production is not touched at any point; GPUs 0,1 stay production's. One build / convert at a time on amd-host.
- Supervisor addendum (17:30 UTC): the LoRA session's review corrects phase 1 -- the head reads the 4 hc streams (~20 GB
  for 1M positions), it is 2.61 B parameters with 2.52 B in frozen experts (train the ~0.09 B rest), and the dump must
  also record the engine's own draft top-32 for a parity check. Read briefs/2026-11-04-draft-head-lora-review.md first.

## 182. Owner: part 1 of the draft-head dump is enough; the copy and the parity / memory step now (2026-11-04 22:13 UTC)

1. **The two commands are allowed** (the owner added `Bash(bash logs/mtp_distill_copy.sh *)` and
   `Bash(bash logs/kdstep.sh *)` to .claude/settings.json at ~22:15 UTC).
2. **Part 1 (625,889 positions, documents 1-685) is the training set; skip part 2.** It is inside note 181's
   0.5-1M range, and part 2 would risk another memory-guard stop for little.
3. **Order:**
   a. the part-1 copy to /srv/bulk/ri/mtp-distill/flashnext-p1/ (with the README); tell the supervisor when it lands,
      and he hands it to the LoRA session;
   b. MD5 = the parity file + the 200-document dump with the memory sampler (`kdstep.sh MD5 a1868a409b1e mtpparity
      KDP2 logs/mtp-corpus-parity200.jsonl 3000`): name what grows (anonymous, shmem, kernel, the container) and
      whether the service's prefill path holds it too;
   c. then the soak (stage D), with the same 10 s memory sampler on, and a stop if MemAvailable falls under 20 GiB.
      If MD5 shows the growth is on the service path, report and fix it before the soak.
4. **The mixed-bit try is closed** (KCQ2: KL 0.110 / 93.0 %, fails the bar); the code stays opt-in. The owner may
   delete /srv/bulk/ri/flashnext-cold3.rad himself.

## 183. Owner: more decode -- a faster draft-head kernel, fewer hc kernels, and a megakernel design brief (2026-11-04 22:22 UTC)

The kernel scoreboard's decode table (P87) sums to ~32.5 ms against a ~9.5 ms byte floor (3.4x), mostly in many small
kernels that never reach full bandwidth. Order: after note 182 (the part-1 copy, MD5, the soak); re-profile first
(today's tree, the rotation's configuration) so the targets are today's, not P87's.
1. **The draft head's 2-bit projection kernel (`gemv_w2a8_rows_kernel`, ~1.55 ms a step, ~3.9x its floor):** decode
   the 2-bit codes through a small table (format C's approach) or registers instead of the software unpack. Exact
   (the same products and sums in the same order); judged by the kernel's time, then the M-pair texts and the
   rotation; it becomes the default if >= 1.5 % on the rotation, else opt-in and reported.
2. **Fewer hc kernels:** fuse the hc family (up mix, down rows, gains, norm, write, quantise rows) into fewer
   launches -- e.g. gains into down rows, norm with write, the quantiser into the producer -- keeping the arithmetic
   order so the result stays exact (bit-identical texts); one fusion at a time, each measured. Defaults by the same
   rule as item 1.
3. **A design brief only, no build: one persistent kernel ("megakernel") for a decode step** -- briefs/2026-11-0X-
   megakernel.md: what it would replace, how the two cards' all-reduces and the expert misses would work inside it
   on Level Zero / SYCL, the published examples (persistent-kernel LLM decoders) and what they gained, the risks
   (the driver's submit path, timeouts, debuggability, the B70's occupancy), a staged plan with a first measurable
   step, cost and time. The owner decides after reading it.
- Not now: the long prompt lookup for code edits (the service's traffic is tutoring and tool calls).
- Gates as before; one build at a time; heat note 161; report each item in WAITING.

## 184. Owner: resume after run 187's pause -- the probe first; the kernel-log read fixed (2026-11-05 06:58 UTC)

The owner allowed `Bash(bash logs/peer1.sh *)` and removed PAUSED (~07:00 UTC 11-05).
1. **First card step: the peer probe** (note 175), then GA3 on image 232 (the leak fix with the device-side wait),
   then the supervisor's gates for the leak fix (bit-identical 7K logprobs and M-pair, prefill vs B135, pools and
   MemAvailable flat), then the soak with the 10 s sampler.
2. **Supervisor addendum: the kernel-log read missed a fault again.** STATE says MD15 (01:21-01:26) had a clean klog,
   but run.sh's judge found 64 card0 lines at 01:26:33 (Fault response -EINVAL / -ENOENT at the out-of-device-memory
   abort); the same happened with B100 on 11-03. Find why each step's klog2 window misses lines at the step's end
   (a start time read too late, an end cut before the teardown, or a filter that drops card0), fix it so every step's
   check covers the whole step including teardown plus 20 s, and correct the MD15 line in STATE.

## 185. Owner: MTP drafts with several users at once -- measure first (2026-11-05 07:38 UTC)

Today MTP drafting is gated off as soon as 2+ sequences decode together (run 172's lanes trace: each decoder gets its
token + a 32-token n-gram draft, which rarely matches on fresh text), so 3 users get ~40 tok/s each against 69.7
alone. Real traffic (assistant, several users) shares the service, so this may be the largest per-user lever left.
Order: after note 183 (and after the soak); measurement first, nothing default without the owner.
1. **Find the gate** (the scheduler's MTP-at-concurrency rule; Radiance's own setting or ours) and what the verify
   step would look like with drafts for several sequences (rows = sequences x (1 + depth); the verify's ~3 ms a row).
2. **Measure on intel-host, the same harness as C13 (six seats, 16K context):** 2, 3 and 6 users with today's gate
   against MTP drafts kept on (at depth 3, and at depth 1-2 if the rows cost too much): per-user and aggregate
   tok/s, tokens kept a step, the step time, worst gaps. Texts identical by construction (greedy verify); check it.
3. **Report in WAITING** with a recommendation (which depth by number of users, if any); the owner decides the default.
- Gates: exact texts; no fault; heat note 161.

## 186. Owner: do all three decode ideas; the megakernel first as a design (2026-11-05 07:40 UTC)

The owner: "do all the improvement 1-3. I'm interested in 3". Gates as always: exact texts, no fault, heat note 161;
an exact item that gains >= 1.5 % on its measure becomes the default and is reported.
1. **MTP drafts with several users (note 185):** measure as written. If an exact setting gains >= 1.5 % per user at 2-3
   users with no loss at 1 user, make it the default (no longer "the owner decides"), and report it.
2. **Draft depth by content:** the measured acceptance runs from 0.53 (prose) to 0.89 (tool calls). Choose the depth
   per request from what the request is (tools / a JSON response format / a long tool transcript -> depth 4 if the
   chain allows; plain prose -> depth 2-3), or from the request's own running acceptance. Exact. Judged on the
   rotation and on a tool-heavy set; default by the rule above.
3. **The megakernel -- the owner's interest -- moves up:**
   a. **Now, desk work between card steps:** the design brief (note 183 item 3, briefs/2026-11-05-megakernel.md).
      Also say whether note 183 item 2 (fewer hc kernels) is a step toward it or would be thrown away; if thrown
      away, skip 183.2.
   b. **Then a first stage, behind a flag:** the smallest persistent-kernel piece that can be measured on its own
      (e.g. one layer's chain of small kernels as one persistent launch, with the device-side waits the tapes use),
      exact, timed against today's step. Report after it, with the cost to go further; larger stages wait for the
      owner. Stage 1 budget: about $150.
- **Order of card work:** the leak fix, then the new draft head's test, then the soak; then 183.1 (the draft-head
  kernel), the megakernel stage 1, 183.2 (if still useful), note 185 / item 1, item 2. The brief (3a) runs alongside.

## 187. Owner: keep pushing the megakernel -- stage 1b (2026-11-05 16:43 UTC)

The owner, after stage 1a's -17 / -21 %: "keep pushing the megakernel". 183.2's pairwise fusions stay the fallback,
not the next step.
1. **Profile first:** unitrace (or the tape profiler) on M268's decode step with HC_PERSIST on, against M262: how much of
   the persistent launch is the barrier's spin, and how much the phases' own bandwidth / occupancy. Decode-only steps;
   the tracer's known hang was at prefill's tail step, but stop at the first sign of a stall (note 175's rules).
2. **Stage 1b, behind the same flag:** occupancy-sized groups (several per Xe core, from the device query, refused if
   not co-resident) and **no device-wide barrier where the dependency is per row** (per-row ready flags in place of one
   global counter; bounded spins with an error word, as in the brief). Exact: the CPU row test bit-identical, the
   M-pair texts identical.
3. **Measure** the hc chain's microseconds a connection and the step, then (if it is no longer slower) the rotation.
   It becomes the default only at >= 1.5 % on the rotation; report either way, with what stage 2 (the dense chain
   between MoE layers) would cost and gain on these numbers.
- Budget for 1b: about $150 more. If 1b is still slower, report and stop: the owner decides between stage 2,
  183.2's fusions, or parking it.
- Order: after the new draft head's test and the soak's start; ahead of note 185 and the draft depth by content.

## 188. Owner: the draft-head project is closed; an expert weight-layout design brief next (2026-11-05 18:11 UTC)

1. **Note 181 (self-distilled MTP head) is closed.** H128 lost 2.9 % (acceptance 0.680 vs 0.709); the LoRA session's
   quantisation-aware eval showed the depth-1 gain survives quantisation offline (+1.2 pts), so the loss is a domain
   gap plus depth 2-3 drift; fixing it is 1-2 days for a few % at most. Keep the tools (dump, parity, merged-convert
   recipe) for the record; nothing else.
2. **A design brief, desk work, no card time: the expert weights' layout for decode** (briefs/2026-11-05-expert-
   layout.md). The format C expert kernels are the largest decode item (~9.3 ms a step, ~4x their byte floor at
   P87); note 167's exact rewrites didn't move them and concluded "only fewer bytes or another layout would". Fewer
   bytes failed the bar (mixed bits); another layout is untried. The brief:
   a. why the kernels sit at ~4x (from counters / the profile: load granularity, coalescing, cache lines a lane
      touches, the code->level table, occupancy at decode's ~4 rows and ~25-35 experts a card);
   b. what layout the B70 reads best for these kernels (a load-time repack of each expert's codes and scales into
      sub-group-contiguous tiles, the way format C already builds its table at load) -- values unchanged, so exact,
      and no convert of the file;
   c. the expected gain with its reasoning, the VRAM / load-time cost, the risks, and a staged plan with a first
      measurable step (a standalone kernel timing on one expert shape before any engine change) and its cost.
   The owner decides after reading it. The goal it serves: 1-user decode toward ~80 tok/s.
- Order: alongside the current card work (238's gates, megakernel 1b) and the soak; it uses no cards.

## 189. Owner: go for the expert-layout stage 1 probe (2026-11-05 18:20 UTC)

From briefs/2026-11-05-expert-layout.md (the load-time 16x64 tile repack, reordered nibbles, scales beside the codes,
the pair table precomputed; exact if the per-lane partial sums and their tree order are kept):
1. **Stage 1, the standalone probe on one expert shape** (~$40-60): bit-identical to today's run kernel, timed per
   launch. **Go to stage 2 (the relayout hook + the engine path behind a flag) only at >= 25 % off the launch.**
   Report either way, with stage 2's cost.
2. Card time: a short canary-sized step; it slots into the gaps around the soak, never displacing the soak.
- Goal: 1-user decode toward ~80 tok/s (brief's estimate for this item alone: ~74-75).

## 190. Owner: park the megakernel (a paper estimate of stage 2 only); port upstream 1.17.0's hc_read merge (2026-11-05 18:26 UTC)

1. **The megakernel is parked** after stage 1a (132.3 us) and 1b (114.2 us) against the three kernels' 42.7 us. The code
   stays opt-in (RADIANCE_XE_HC_PERSIST). **The owner is curious about stage 2, so: a desk estimate only, no build,
   no card time** -- a section 6.2 in briefs/2026-11-05-megakernel.md: what stage 2 (attention + hc + router between
   MoE layers as one persistent kernel) would gain and cost on the measured numbers (P87 / M269-M274), whether the
   register-footprint and code-generation cost that sank 1a / 1b grows with more and more varied bodies, and whether a
   different design (e.g. separate kernels that only overlap their loads) would avoid it.
2. **Port upstream's hc_read merge from Radiance 1.17.0 (released 2026-11-04; codeberg StillDeadcode/radiance; local
   clone ~/code/radiance-src-keyenv, tags fetched).** Upstream 1.14's "hc_read: kernels A and B as one grid at decode,
   their dots off LDS round trips (34.7 -> 25.2 us at 4 rows)" and its byte-for-byte comparison
   (r4d_hc_read_pin_decode) do at decode, as one ordinary launch, what our stage 1 tried with a persistent kernel.
   Read the upstream commits (git log v1.11.0..v1.17.0 -- libr4d hc_read / decode split law / gemm_fp8a8 narrow
   form), port the hc_read merge into libxe behind a flag, exact (bit-identical texts), timed per connection, then the
   rotation; default at >= 1.5 %. Also note in STATE which other 1.12-1.17 changes could carry to the port (the
   decode-hot expert pinning / hot VRAM cache for the misses; the MTP first-draft cache fix) for the owner later.
- Order: after the soak's start (the soak takes the cards for 24 h, so the port's desk work and builds run first and
  its card steps wait for the soak to end); the expert-layout probe (note 189) likewise.

## 191. Owner: three more levers for 1-user prose decode (2026-11-05 18:52 UTC)

Goal: 1-user prose decode from 70.2 toward 80+ (the single-prompt M-pair, MTP on). All card steps after the soak
(it ends ~18:46 UTC 11-06); desk work and builds now.
1. **First: the draft depth by content (note 186 item 2), moved to the front of the post-soak card queue.** Measured
   11-03: depth 2 beats depth 3 by 4.2 % on the single prose prompt (71.8 vs 68.9) but loses 3.5 % on the rotation
   (tool calls want depth). So choose per request: depth 2 for plain prose, 3 (or 4 if the chain allows) for tools / a
   JSON response format / long tool transcripts -- from the request itself, or from its own running acceptance after
   the first few steps. Exact. Gates: the M-pair (prose) gains, the rotation does not lose (>= -0.5 %), and a
   tool-heavy set. Default if the prose M-pair gains >= 1.5 % and the rotation holds.
2. **A short design note, then a probe: independent kernels side by side.** At 4 rows most decode kernels don't fill
   the card (2-5x their floor). Find the independent pairs inside a layer (e.g. the shared expert beside the routed
   experts; projections reading the same input; the MTP head's work where the order allows) and run them on two
   lists with explicit events (the GLUE_ALL=2 machinery already puts work on a second list). Each kernel keeps its own
   registers, so it avoids what sank the megakernel. Exact. Probe one pair first, timed; report before more.
3. **Add upstream's decode-hot expert pinning (Radiance 1.14: --expert-profile / --expert-pin-top /
   --draft-expert-placement; 1.15: the hot VRAM cache) to note 190's port list.** Read it in
   ~/code/radiance-intel/upstream-1.17.0, say how it maps onto our slab / heat engine, and what it would do to the
   rotation's ~28 misses a step and to prose.
- Order after the soak: (1) depth by content, then the hc_read merge (note 190), the expert-layout probe (note 189),
  then (2)'s probe, then (3). Report each in WAITING.

## 192. Owner: the soak is 12 hours, not 24 (2026-11-05 21:11 UTC)

The owner, 2026-11-05 ~20:55 UTC: "do it to 12 hrs". The soak (image 238, GLUE_ALL=2; healthy since 18:46 UTC) ends at
**06:46 UTC Fri 11-06**: the supervisor armed the user timer `ri-soak-stop-12h` on amd-host, which then stops the soak
client (both ends), the guard, the watchdog timer and ri-m3-flashnext.service on intel-host, and logs to
logs/soak-stop.txt. Then:
1. Write the soak's report (requests since 18:46 with status 200 only, the memory slope from the sampler, faults,
   temperatures, watchdog actions) into WAITING and the milestone report's draft as stage D at 12 h (the owner's
   choice; the plan said 24).
2. Start the post-soak card queue (note 191's order): image 239 onto intel-host; draft depth by content + MTP hist-ctx;
   the hc merge; the expert-layout and two-list probes; pinning.
- **Amended ~21:13 UTC: the owner chose "stop now"** (he wants the decode results first). The soak ran 18:46:35 -> ~21:13 UTC
  (~2.5 h; 3,720+ requests all 200, anon ~1.8 MiB/min and slowing, Tctl 44-45 C, watchdog quiet). Stopped by the
  supervisor: soak client, watchdog timer and ri-m3-flashnext.service on intel-host, the guard on amd-host; the 12 h timer is
  cancelled. Write stage D at ~2.5 h (the owner's choice) and **start note 191's post-soak queue now.**

## 193. Owner: adaptive draft depth per request (1-4), on top of depth by content (2026-11-05 22:09 UTC)

After depth by content (note 191.1: prose capped at 2; M276 prose +5.5-6.8 %, rotation H130 pending):
1. **Build an adaptive depth, behind a flag:** each request chooses its draft depth (1-4, 4 only where the chain
   allows) from its own running acceptance by position, with a cost model: keep draft k+1 only when the expected extra
   tokens (P(all of 1..k+1 accepted)) beat the extra verify row's cost (~3 ms of ~37 ms a step, measured per depth on
   today's image). Start from depth by content's rule for the first few steps, then adapt; re-evaluate as the request
   goes (prose turning into a table, and so on). The scheduler's round follows the deepest row in the batch, as now.
2. **Exact** (the verify keeps only the model's tokens): the M-pair texts identical. **Gates:** the prose M-pair >= depth by
   content's 74.2 tok/s, the rotation >= H130, and a tool-heavy set gains or holds. **Default** if it beats depth by
   content by >= 1.5 % on either the prose M-pair or the rotation without losing on the other.
- Order: after depth by content is settled (H130 / H131), before the hc merge's card steps, unless the hc merge is
  already running. Report in WAITING.

## 194. Owner: port Radiance 1.12-1.17's prefill changes where they apply (2026-11-05 22:38 UTC)

The owner wants the Intel 7K first token clearly under 5 s (today ~4.9-5.1 s; 20K ~10.0 s). From the upstream export
(~/code/radiance-intel/upstream-1.17.0, log-1.11-1.17.txt):
1. **Desk first: a short mapping in STATE** of each upstream prefill change and whether it applies to our engine:
   - libr4d "attn prefill h256: the small-launch KV split for every GQA geometry" / "gqa6: split the KV walk when the
     launch has too few workgroups" (1.13) -- our prefill attention is now the QSA FMHA op (491dc58) + attn_split for
     the rest; does the same too-few-workgroups problem exist in ours (chunk tails, the 36-row tail step)?
   - "fp8 prefill GEMM: the last band's upper waves leave when their rows lie past M" (1.15) and the gated prefill fold /
     short-prompt split row / rows forms of rmsnorm and rope (1.15);
   - core/sched: "a prefix hit resumes on the cold run's chunk grid", the preempted-prefill gate fixes, the reclaimed
     cache / state slots (1.14-1.15) -- engine-core code we share with upstream: port as is if it applies;
   - the pipelined prefill (1.12) is for > 2 ranks: note it as not applicable.
2. **Then port what applies, one item at a time, behind flags**: exact or inside note 149's bar; gates are the 7K / 20K
   B-runs (B135's order) and the decode M-pair unchanged; default at >= 5 % off the 7K, or >= 1.5 % with exactness.
- Order: after tonight's decode queue (adaptive depth, hc merge, probes); the mapping (desk) can start now.

## 195. Owner: build the top prefill lever overnight; draft depth is settled (2026-11-05 23:03 UTC)

1. **Draft depth is settled at today's fixed depth 3.** Depth by content (H130 -3.8 %), the adaptive window (H132 -3.8 %,
   M278 -4.9 %) and the MTP first-draft fix (H131 -2.6 %) all failed the rotation; their code stays opt-in. No more depth
   work unless the owner asks.
2. **After B150 (the 7K profile), build the #1 prefill lever it ranks, overnight, without waiting for the owner**:
   exact output (bit-identical 7K logprobs vs B135, M-pair texts identical), behind a flag, timed by B-runs in B135's
   order. **Default only if it cuts >= 5 % off the 7K first token** with decode unchanged; otherwise opt-in and reported.
   Report the top 3 with estimated gains in WAITING either way. Levers #2-3 wait for the owner.
- Order: after tonight's hc merge, the probes and B150.

## 196. Owner: a cooperative launch through Level Zero for the kernels whose groups wait on each other (2026-11-05 23:12 UTC)

Three designs lost today for one reason (megakernel 1a 132 us, 1b 114 us, the hc merge 179 us a connection against
42.7-67.3 us): work-groups that wait on other work-groups of the same launch hold slots the producers need, because
nothing guarantees co-residency. SYCL has no supported cooperative launch, but **Level Zero does**, and our backend
drives Level Zero directly: zeKernelSuggestMaxCooperativeGroupCount, zeCommandListAppendLaunchCooperativeKernel, and the
queue-group flag ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS (ze_api.h, build-unitrace/level_zero). Our src
uses none of them today.
1. **First, the limits (cheap):** on the B70, for hc_v2_ab_kernel<4> (RADIANCE_XE_HC_AB) and the megakernel 1b kernel
   (RADIANCE_XE_HC_PERSIST): the suggested max cooperative group count, which queue groups advertise the cooperative
   flag, and how that compares with the groups each launch uses now (~192 for the hc merge). A tiny standalone query or
   part of the next probe step; record it in STATE.
2. **If the cap covers the launch (or the kernel can be re-tiled to fit it):** a cooperative launch path in the L0 backend
   (core/device/l0.cpp; inside recorded tapes if the API allows, else noted), behind a flag, for those two kernels only.
   Re-time per connection (the -prof form), then the M-pair (texts identical), then a 10-min rotation. Default by the
   usual rule (exact and >= 1.5 % on the rotation).
3. If the cap is too small for either to gain, report the numbers and stop; the lesson in STATE stands.
- Budget ~$40 for 1 and the first timing of 2. Order: after the probes (note 189 / 191.2) and B150 / note 195.
- **Supervisor, 2026-11-06 00:29 UTC: note 196 STOPS at step 1** (the runner's recommendation, agreed). The cooperative cap is 128; the hc
  merge launches 216; 1b's 72 fit and still lose 2.7x. Mechanism: the hc merge's ticket order already prevents starvation;
  the cost is every block declaring the largest phase's LDS (up to 60 KB, ~4 groups a core) plus the ticket/fence protocol.
  Record in STATE and the megakernel brief: on the B70, fused phases must not share one LDS / register budget sized by the
  largest phase. Next: the probes (P97, pstep.sh now allowed) at the next session's start. Note 195's levers #2-3 wait
  for the owner.
- **Supervisor, 2026-11-06 00:31 UTC: after P97 (both probes under their bars, recorded), two APPROVED items remain -- do not end at a waiting point:**
  1. **Note 185 + note 186 item 1: MTP drafts with several users.** Measure 2 / 3 / 6 users, today's gate vs drafts kept on
     (depth 3, and 1-2 if the rows cost too much), on C13's harness; per-user + aggregate tok/s, tokens kept, step time,
     worst gaps; texts identical and the batch-invariance check (same prompts alone vs together). DEFAULT if >= 1.5 % per
     user at 2-3 users with no loss at 1 user (note 186).
  2. **Note 191 item 3: upstream's decode-hot expert pinning** (--expert-profile / --expert-pin-top, mapped in STATE):
     behind a flag; misses + median on a 10-min rotation; default at >= 1.5 %.
  Base image 244. Order 1 then 2.

## 197. Owner: resume and fix everything (2026-11-06 01:58 UTC)

The owner: "resume and fix everything". PAUSED is lifted by the supervisor on his word. Faults before now are not
judged again. In this order:
1. **Fix the startup warm-up's stale per-seat drafting state** (the warm-up's sequence leaves MTP / draft state that the
   next request in that seat inherits: 96/480 drafts kept, 45.6 tok/s against 70.1 in the 6-seat harness; and with drafting
   at 6 users it faulted the card twice at 0x00008000fffe0000, UR OUT_OF_RESOURCES). Reset that state fully when the
   warm-up's sequence retires (or run the warm-up outside the seat pool). Gates, in order:
   a. the 6-seat harness with the gate (default config), warm-up ON: the 1-user phase back to ~70 tok/s and texts identical;
   b. drafting at 2-3 users with the warm-up ON: clean, texts identical;
   c. only then ONE 6-user drafting phase with the warm-up ON. If it faults, stop (PAUSED by the rule) and report.
   **Until (a) passes, set the warm-up default OFF** (--startup-warmup off) so the service doesn't carry the slowdown.
2. **Make RADIANCE_SPEC_MULTI_DEPTH=1 the default** (depth 3 alone, depth 1 for 2+ users: +24 % per user at 2 users,
   flat at 3, texts identical) once 1a-1c pass.
3. **Decode-hot expert pinning** (note 191.3, built in image 246): H134 / H135 as staged; default at >= 1.5 %.
4. **Note 195 lever #2, the prefill all-reduce overlap** (the next half-chunk's GEMMs during the all-reduce; est. -0.3 to
   -0.4 s of the 7K): exact, behind a flag; B-runs in B135's order; default at >= 5 % off the 7K, or >= 1.5 % if exact.
   **Lever #3** (the protected experts' joint_matrix form, not bit-exact): build behind a flag, judge by note 149's bar;
   it stays opt-in either way and is reported.
5. **The 6-user batch variance** (0/6 texts match the 1-user text at 6 users on the default path, pre-existing): find
   which kernel's reduction order depends on the batch (desk + small checks), report the cause and the cost of a
   batch-invariant form. No default change without the owner.
- Fault rules unchanged (notes 175, run.sh). Builds one at a time. Report each item in WAITING.

## 198. Owner: diagnose and optimise the megakernel / hc merge -- "I believe we can do better" (2026-11-06 02:42 UTC)

The owner wants the fused forms pursued, not parked. Working hypothesis (run 190 / note 196): every block of a fused
kernel declares the largest phase's LDS (the hc merge: up to ~60 KB, ~4 groups a core) and register budget, plus the
ticket / fence protocol. That is a design choice to engineer around, not a hardware wall. Order: right after note 197
item 1 (the warm-up fix, a live service bug), ahead of the rest of note 197.
1. **Diagnose with counters first (~$15):** unitrace (decode-only steps; stop at the first stall) on M280's hc merge
   (hc_v2_ab_kernel<4>) and M269/M274's 1b, against the separate kernels (M270): achieved occupancy (groups / threads a
   Xe core vs the limit), SLM bytes a group, GRF mode and spills, stall reasons (barrier, atomics / the ticket counters,
   memory), and time per phase if the profiler can split it. Write the result into the megakernel brief as section 7,
   confirming or overturning the LDS-budget explanation with numbers.
2. **Then the optimisations, each exact (bit-identical texts and the CPU row test) and timed per connection against the
   separate kernels' 67.3 us (hc merge) / 42.7 us (1b's chain):**
   a. a smaller B-side (up-mix) tile, so the fused kernel's SLM falls from ~60 KB toward <= 16 KB;
   b. the up-mix staging through registers / sub-group shuffles or L2 instead of SLM;
   c. fusing only phases with similar SLM / GRF needs (e.g. down rows + gains, leaving up-mix separate);
   d. keep the ticket order (it prevents starvation); trim the fences to what the data dependencies need.
   Default only if a form beats the separate kernels per connection, then the M-pair (texts identical) and a 10-min
   rotation (>= 1.5 %).
3. **If the hc merge wins**, apply the same recipe to 1b, and only then re-estimate stage 2 (the dense chain between MoE
   layers) on the new numbers, for the owner.
- Budget ~$100-150 for 1-2; report after 1 and after 2. Fault rules and heat unchanged.

## 199. Owner: resume on safe terms after C30 -- no more 6-user drafting on the cards (2026-11-06 03:40 UTC)

The owner said yes (~03:40 UTC) to resuming. PAUSED lifted by the supervisor on his word; faults before now are not
judged again. Terms:
1. **The multi-user policy, capped at 3:** RADIANCE_SPEC_MULTI_DEPTH with depth 3 alone, depth 1 for 2-3 sequences, and
   **no drafting at 4+** (the existing gate's behaviour above 3). Measured with the warm-up on (C29): +17 % per user at 2,
   +6 % at 3, texts identical. Re-check that config once (2-3 users, and 4-6 users confirming drafting is off and texts as
   today), then make it the DEFAULT, and commit with the warm-up KV reset (247) once the CPU suite's open reruns pass.
2. **Continue note 198** (the megakernel / hc merge counters on ri-m3ut:4, then the exact optimisations), then **note 197
   item 4** (prefill lever #2; #3 opt-in).
3. **No 6-user drafting steps on the cards** until the owner says so. The second warm-up leftover (C15 / C24 / C30 at
   0x00008000fffe0000) is investigated at the desk: add the flag that separates the KV loan / weight-table side from the
   warm-up, reason from the code, and write up the hypothesis and the one card test that would confirm it, for the owner.
- The warm-up stays OFF by default until that leftover is understood. Fault rules unchanged.

## 200. Owner: "let's see if we can modify and do better for Intel" -- shape-matched fusion (2026-11-06 04:25 UTC)

Note 198 (1)'s static half (megakernel brief section 7) shows why the fused forms lost on the B70: the column phase
(down rows: 64 threads, 32 B SLM, 1,024 resident) is forced into the up-mix's shape (256 threads, ~58 KB, 128 resident)
-- 8x less residency -- and both fused bodies spill (hc merge 576 B, 1b 4,864 B) where no separate kernel does.
The Intel rule: **fuse only phases with matching shapes (threads, SLM, GRF), and never let a fused body spill.**
1. **The hc chain, shape-matched (note 198 option 2c), exact, behind flags:**
   a. norm + down rows -> one 64-thread, near-zero-SLM kernel (keeps ~1,024 resident);
   b. up mix + gains -> one 256-thread kernel within the ~54 KB budget the up mix already has;
   c. no spills in either (check the Level Zero spill bytes = 0 before timing).
   Timed per connection against the four separate kernels (67.3 us); then the M-pair (texts identical) and a 10-min
   rotation; default at >= 1.5 %.
2. **The stall counters (mutstep.sh, allowed)** before or beside 1, to confirm on the hardware what the static numbers say.
3. **If 1 wins, sweep the decode step for other shape-matched neighbours** (from the kernel scoreboard: e.g. the router
   with the small bf16 GEMVs that read the same rows, hc_write with hc_v2_norm, the small tails) and write a short
   "Intel fusion map" in the brief: which pairs share a shape, the expected saving each, in order; then build the top
   ones the same way (exact, behind flags, default by the usual rule).
4. **Megakernel stages** get re-planned on this rule (segments of like-shaped phases, not one budget for all) only after
   1 and 3 show the per-pair gains.
- Budget ~$150 for 1-3; report after each. Order: now, ahead of note 197 item 4 (prefill lever #2), which follows.

## 201. Owner: three single-kernel ideas from upstream 1.12-1.17 for decode, plus the Intel fusion map (2026-11-06 05:38 UTC)

Tonight's lesson: on the B70, fusions only win when the phases share a shape and do not wait on each other (GU: -7.5 % on
its pair, M290); ordinary single-kernel improvements are the other safe path. From ~/code/radiance-intel/upstream-1.17.0
(read the commits in log-1.11-1.17.txt), port these as single-kernel changes, each exact (bit-identical CPU row test +
M-pair texts), behind a flag, timed per kernel (-prof) and then a 10-min rotation; default at >= 1.5 %:
1. **hc_read's "dots off LDS round trips"** (1.14, the part of upstream's hc_read change that is NOT the merge) on our
   **hc_up_mix<4>** (31.5 us a launch, ~5.5x its floor) and hc_v2_down_rows if it applies.
2. **gemm_fp8a8's narrow form** (1.14: scales staged through LDS, the split finisher coalesced; 83.6 -> 68.3 us at 32 rows)
   on our E4M3-weight kernels at decode's 4 rows (the hc family's E4M3 W_up / W_down rows).
3. **The decode split law** (1.14-1.15: "size a three-wave verify by the step's depth", "bound the three-wave verify floor by
   workgroups") on our verify attention (attn_split at decode's 4 rows) -- check our split count against upstream's law.
4. **The Intel fusion map** (note 200 item 3), now that GU shows the rule works: sweep the decode step for shape-matched,
   no-wait neighbours (from the kernel scoreboard), write the map in the megakernel brief, build the top pairs the same way.
- Order: after H137's verdict on GU; 1 first (largest gap), then 2, 3, with 4 alongside as desk work. Budget ~$150.
  Report each in WAITING. Fault rules and heat unchanged.

## 202. Owner: the Intel software stack under our kernels -- the compiler first, then the runtime (2026-11-06 ~06:00 UTC)

The owner asked whether the Xe driver can be optimised. Of the three layers, the kernel driver (xe) is not where decode's
time goes (the cards are busy ~34 of ~37 ms a step) and stays untouched; xe direct stays closed (note 64); no root or
kernel-parameter change on intel-host (any need comes to the owner as a proposal). The owner's order: **A, the compiler (IGC),
then B, the runtime (Level Zero / NEO).** Test images only, never production. Each change exact where it can be
(bit-identical CPU row test + M-pair texts); a compiler change that is not bit-identical is judged as GL was (note 149:
relative to the non-bit-identical controls). Timed per kernel (-prof), then a 10-min rotation; default at >= 1.5 %.

**A. The compiler (~$60).** IGC 2.41.5 alone gave hc_up_mix -18 % and the expert down -14 % (note 146).
1. **A newer IGC:** check whether a release after 2.41.5 (and its matching NEO) exists; if so, rebuild the test image's
   kernels with it and compare per kernel against 2.41.5 (the decode step's top 10 by time), then the M-pair + rotation.
2. **GRF mode per hot kernel:** list each decode kernel's GRF mode, spill bytes and residency (IGC dumps / zebin). Try the
   large (256) GRF mode on every kernel that spills or is latency-bound, and the small one where residency is the limit.
   Flag per kernel; exact by construction.
3. **Read the machine code of the three biggest decode kernels** (format C gate_up / down, hc_up_mix<4>, the 1-row INT8
   trunk GEMV at ~53 % of bandwidth): are the weight loads 2D block loads / wide (16-byte) LSC loads with prefetch, as
   Intel's own kernels and our 530-573 GB/s probe use? SLM bank conflicts? Write the gaps in WAITING, then fix the
   largest in source (block-load builtins / intrinsics), exact.

**B. The runtime (~$50).** The step's device idle is ~2.1 ms (~1,790 kernel boundaries at ~0.8 us, run 176).
1. **The per-launch cost inside a recorded list:** measure it under NEO's documented settings that touch submission and
   replay (counter-based events / in-order list options, direct-submission settings, flush behaviour) -- environment
   variables only, one A/B each, on the M-pair; anything >= 0.3 ms a step goes to a rotation.
2. **A newer NEO than 26.35** (if A.1 finds one): runtime only, as note 146 did; release notes checked first for launch /
   appended-list changes.
- Order: note 201 item 1 (in progress) -> 202 A -> 202 B -> note 201 items 2-4. Report each in WAITING. Fault rules,
  heat rules, the 6-user rule (note 199) unchanged.
- **Order amended by the owner (~06:10 UTC):** fewer kernels and leaner commands come before any draft-side idea.
  New order: note 201 item 1 (in progress) -> 202 A (compiler) -> **202 B (leaner commands) and note 201 item 4 (the
  fusion map, then building its top pairs)** -> note 201 items 2-3. The draft-head ideas (a trimmed draft vocabulary,
  fewer draft experts) are parked and not queued.
- **Owner (~06:15 UTC): no to fewer experts in the main model (10 -> 8); not tested, not queued.** Answers stay exactly today's.
- **Supervisor addendum to 202 A (~06:20 UTC):** in A.3, also list which decode kernels still run through the HIP shim (xe_lanes.h: xe::wmma's shuffle emulation of wave32 WMMA, sub-group 32) instead of native DPAS / joint_matrix at sub-group 16, with their ms a step -- each is a candidate for a native rewrite.
- **Owner (~06:15 UTC): add A.4, the ESIMD probe (~$40-60).** After A.3's machine-code read, write the format C expert
  down kernel (<3,2,2,1>, 66 us a launch vs a ~14.7 us floor) in ESIMD (sycl::ext::intel::esimd: explicit vector width,
  2D block / wide LSC loads with prefetch, explicit SLM, register use; inline vISA only where A.3 shows a need) as a
  standalone probe on one expert at decode's rows. Exact: bit-identical outputs against today's kernel (keep the
  per-lane partial sums and their tree order). Go at >= 25 % off the launch; then the gate_up and hc_up_mix<4> the same
  way, behind flags, -prof, a 10-min rotation, default at >= 1.5 %. Hand-written machine code is NOT in scope; propose
  it for one kernel only if ESIMD stalls well above the floor and A.3 shows an IGC limit that cannot be worked around.
  Heat / fault rules unchanged (a probe that faults the card counts as a fault).
- **Owner goal (~06:17 UTC): single-user decode ABOVE 80 tok/s** (the M-pair, exact answers). Supervisor reorder toward
  it: the biggest lever goes first. After note 201 item 1: **A.3 for the format C expert kernels only, then A.4 (the
  ESIMD probe)**; then A.1 / A.2 / the rest of A.3; then B + the fusion map; then note 201 items 2-3. Report the M-pair
  tok/s after every adopted step as "tok/s vs the 80 goal".
- **Supervisor, after run's A.3 (~06:4x UTC):** GU + D16 measured, both exact and opt-in (M-pair 71.1; rotation +0.7-0.9 %).
  A.3: format C's machine code (IGC 2.28.4) = 194 scalar SLM gathers, 12 wide loads, no 2D block loads, no prefetch;
  SEL decode +33-35 % and 256 GRF +30-63 % (issue-bound, lives on thread parallelism). **Order for the next run: A.1
  first IF the owner has created /srv/bulk/ri/build-m1l0neo2640 (NEO 26.40 + IGC 2.42.6: format C and the hc kernels
  have never been timed under a newer IGC; note 69's 2.41.5 run was mixed, -18 % hc_up_mix / -14 % one expert kernel
  but +44 % on the old codebook kernel that format C replaced), timed per kernel against 2.28.4 -- and if mixed, propose
  a per-kernel compiler choice; then A.4 (ESIMD) built with whichever IGC wins. If the folder is not there, A.4 first.**
- **Owner YES (~13:25 UTC 11-06): T1c, the format C probe** (sub-group block loads on T1b's contiguous layout, wide scale
  loads, more loads in flight; exact; bar <= 75 % of a launch at decode, bit-identical; ~$40-60), down first, then gate_up.
  Order: finish pairs 2 / 4's current CPU tests and card checks -> **T1c** -> the group rotation (with T1c in it if it
  passes) -> A.1 once /srv/bulk/ri/build-m1l0neo2640 exists (the owner is creating it), T1c re-timed under the new IGC.
- **Owner (~13:35 UTC 11-06): decode stays first.** The prefill all-reduce overlap (note 197 item 4, ~-0.3-0.4 s on the 7K) stays after the decode items (pairs 2/4 -> T1c -> group rotation -> A.1 -> note 201 items 2-3 -> note 197 items 4-5).
- **Owner YES (~14:2x UTC 11-06): the five exact fusion-map flags become the DEFAULT** (RADIANCE_XE_HC_GU, RADIANCE_XE_HC_D16,
  RADIANCE_HC_WRITE_FUSE, RADIANCE_XE_GATE_QUANT, RADIANCE_XE_GATED_HAD; each =0 opts out). Evidence: H141 vs H139 median
  +1.7 % (91.7 vs 90.2), mean +1.4 %; CPU row tests bit-identical; every flag-on M-pair text identical. One commit, then an
  image and a flag-free M-pair (texts identical to M285, tok/s >= the flag-on figures). Intel test images only.
- **Supervisor (~14:4x UTC 11-06): A.1 measured** -- NEO 26.40 + IGC 2.42.6 base M-pair 72.17 vs 70.33 (+2.6 %), texts identical; hc_up_mix -21.5 %, qsa_select -17 %, attn_split -13 %, format C down -7.2 %; trunk gemv_i8a8 +7.3 %, gemv_bf16 +12 %. Becomes the test images' default if the combined image (261: five fusion defaults + A.1) clears >= 1.5 % vs H139 on the rotation with identical texts. Per-kernel compiler mix (GEMVs from 2.28.4, ~0.5-0.7 ms a step) = proposal after.

## 203. Owner: tune our kernels to the new compiler -- fix the regressions, then per-kernel compile settings (2026-11-06 ~14:50 UTC)

After the combined image (261) and its rotation, on the 26.40 / IGC 2.42.6 stack. Exact throughout (bit-identical CPU
row tests + M-pair texts); test images only; gates as note 201 (-prof per kernel, then a 10-min rotation, default at
>= 1.5 % for the set). Budget ~$70-100.
1. **The regressions under 2.42.6 (~$30-40):** gemv_i8a8 (trunk, 24.6 -> 26.4 us), gemv_bf16 (16.1 -> 18.1),
   gemv_bf16_pair (12.7 -> 13.7). Diff each kernel's machine code under 2.28.4 vs 2.42.6 (loads, unroll, spills, SIMD
   width, GRF), then change our SOURCE so 2.42.6 emits the good form (pragmas, explicit vector loads, loop shape). No
   second compiler in the build. Target: each back to or under its 2.28.4 time.
2. **Per-kernel compile settings (~$40-60):** for the decode top 10 by time, sweep a few EXACT settings each -- GRF mode
   (default / auto-large / 256), unroll factors, work-group size -- and keep the fastest per kernel behind a per-kernel
   table. Excluded: anything that can change the arithmetic or sum order (fast-math, contraction, a different
   sub-group size where it changes the reduction tree). Report a table: kernel, setting, us before / after.
- Order: 261 + its rotation first, then 1, then 2. The two-compiler build (A.1 item 2) is dropped unless 1 fails.
  Report "vs the 80 goal". Fault / heat rules and the Sat 03:00-03:25 UTC no-amd-host-build window unchanged.
- **Adopted (~15:4x UTC 11-06, by the note 202 bar): NEO 26.40 + IGC 2.42.6 = the Intel test images' stack**, with the five fusion defaults. H142 vs H139 median 95.2 vs 90.2 (+5.5 %), mean +5.6 %; M317 74.42 single-user, texts identical to M285; -fz- opt-out 72.30. Note 203 item 1 cause: 2.42.6 repacks dot_acc's int8 bytes before each dp4a (mov 396 vs 227) -> fix with __builtin_IB_dp4a_ss in the BMG AOT pass.

## 204. Owner: keep optimising for the Intel compiler -- round 2, the whole decode step (2026-11-06 ~16:30 UTC)

After note 203 (the dp4a / GEMV regressions, then per-kernel settings). Same rules: exact (bit-identical CPU row tests
+ M-pair texts), Intel test images only, -prof per kernel, then a 10-min rotation, default at >= 1.5 % for the set.
Budget ~$80-120. Controls: image 261 / M317 / H142 (or newer adopted).
1. **ISA audit of the decode top 30 under IGC 2.42.6** (m3/igc-dump.sh): per kernel, instruction counts by class (mov,
   shifts, predication, sends), spills, SIMD width, GRF mode, the address mode of each send (stateless A64 vs stateful /
   32-bit offsets). Flag every kernel with the dp4a-style bloat (byte repacks, redundant movs, address math in the inner
   loop) and fix it in SOURCE (builtins, explicit vector types, hoisted address math). A table in WAITING: kernel, us,
   share of the step, what the ISA shows, the fix, us after.
2. **Addressing:** where a kernel's buffers are < 4 GB and its sends are stateless 64-bit, test the exact compile-time
   options or source forms that let IGC use 32-bit offsets / stateful access; keep per kernel only where faster.
3. **Sub-group width on the SG32 kernels** (~62 % of decode time: gemv_i8a8, hc_v2_down_rows, ar, ...): an SG16 variant
   ONLY where today's per-lane partial sums and their reduction order can be reproduced exactly (explicit tree in the
   same order); bit-identical or dropped.
4. **A newer IGC than 2.42.6** if one exists (A.1's method: runtime + rebuilt kernels, per-kernel compare).
- Order: 1 (it feeds 2-3), 2, 3, 4. Report "vs the 80 goal" after each adopted set. Heat / fault rules unchanged.

## 205. Owner: probe IGC 2.43.x (unreleased release branch) built from source (2026-11-06 ~16:50 UTC)

The owner chose B: build intel-graphics-compiler releases/2.43.x from source now instead of waiting for the release.
Runs AFTER note 203 (before 204). Test images only; never production. Budget ~$30-50.
1. **Pin the source:** one commit of releases/2.43.x (record the hash and its date; the branch moves -- 11-04's
   "vISA: Revert merge scalar pass tuning" shows it is still settling) plus the dependency revisions its
   "Update release dependencies" commit names. Build in a container at nice 19 / ionice idle on amd-host, into a folder
   under /srv/bulk/ri (the owner creates it on request; ask the supervisor which path), never on / (disk floor).
2. **Pair it with NEO 26.40's ocloc / runtime** (no 2.43-matched NEO exists); if ocloc refuses the newer libigc, stop
   and report rather than patching NEO.
3. **Rebuild our kernels with it** (the 2640 tree's method, a separate tree), then per-kernel -prof against image 262
   (or the newest adopted) and the M-pair with texts identical to M285. Gates as note 201; adoption only by the
   rotation bar, and only as an opt-in test image unless the owner says otherwise (an unreleased compiler).
- Any new step script (an IGC build script) comes to the supervisor for review before use; new allow rules are the
  owner's. Heat / fault rules, the disk floor and the no-amd-host-build windows unchanged.
- **Supervisor (~16:58 UTC 11-06): run 207's B156 (prefill 7K on image 261) HUNG, card1 CAT (bcs, guc_id 6) at the stop -- fault 1 of the run.** The 26.40 / 2.42.6 stack stays the DECODE test base (M317, H142 clean) but is NOT cleared for prefill / long prompts until the cause is separated (next run: B on 261 -fz-, then on 260, then the NEO debug keys one at a time; each as the run's first engine step after a clean peer probe). Note 175: peer1.sh before any further card step.
- **Run 207 (~17:5x UTC 11-06): note 203 item 1 DONE, image 263 (db07670b5429, commit 7127b3a): dp4a builtin on the card pass + one 16-byte bf16 activation load. M322 single-user 80.75 tok/s (M317 74.42, +8.5 %), texts identical to M285 -- OVER THE OWNER'S 80 GOAL on the M-pair, exact.** gemv_i8a8 26.3->19.1, gemv_bf16 18.3->15.3, pair 14.6->11.6, format C gate_up 116->108, down 61.3->57.3; device total -7.6 %. H143 (rotation vs H142) decides 263 as the decode base.
- **Adopted (~18:1x UTC 11-06, note 203's bar): image 263 = the decode test base.** H143 vs H142: median 102.7 vs 95.2 (+7.9 %), mean +7.5 %, aggregate +7.6 %; M322 80.75 single-user, texts identical. Controls: 263 / M322 / H143. Prefill on the 26.40 stack still uncleared.
- **Owner (~18:55 UTC 11-06): keep the order; come back to prefill.** This run continues decode (IGC 2.43 probe, router GRF, note 204). **The NEXT run's first engine step (after a clean peer1.sh probe) is the prefill separation**: B on 263 -fz-, then on 260 (stack alone), then the NEO debug keys one at a time; then the 7K / 20K first-token times on the new standard vs B155 (4.62 / 9.67 s), logprobs compared.
- **For the owner (cleanup, no hurry):** /srv/bulk/ri/igc-243/build is a stale IGC build tree (clang built without opencl-clang's patches; the runner's rm was refused, so the rebuild uses /srv/bulk/ri/igc-243/build2). Remove when convenient: `rm -rf /srv/bulk/ri/igc-243/build`.

## 206. Owner: optimise for the Intel compiler further -- shape specialisation, cache hints, scheduler knobs (2026-11-06 ~19:25 UTC)

After note 205 and note 204 item 1 (the ISA audit picks the kernels). Exact throughout (bit-identical CPU row tests +
M-pair texts), Intel test images only, -prof per kernel, then a 10-min rotation, default at >= 1.5 % for the set.
Budget ~$80-120. Controls: the newest adopted image (263 today) / its M-pair / its rotation.
1. **Shape-specialised instances** of the hot decode kernels (format C gate_up / down, gemv_i8a8, gemv_bf16 / pair,
   hc_v2_gu, logits_i8): the model's fixed N / K / row counts as template constants, so IGC can unroll fully and fold
   address math; the runtime-shape instance stays as the fallback for any other shape. Same arithmetic and order.
2. **Cache-control hints on loads:** weights (read once a step) as streaming / L1-bypass loads, activations and tables
   cached -- via SYCL's annotated_ptr cache_control properties or the LSC load builtins with cache controls. Only
   where -prof shows a gain; values unchanged.
3. **IGC scheduling / unroll knobs per kernel at AOT** on the top 5 kernels (including 2.43's new loop-unroll
   controls if note 205 is adopted): instruction scheduling, send scheduling, unroll heuristics. Nothing that touches
   arithmetic (no fast-math, no contraction, no reassociation). A table: kernel, knob, us before / after.
- Order: 1, 2, 3. Any new step-script suffix to the supervisor first. Report "vs the 80 goal" (80.75 today).
- **(DONE 11-06 ~19:5x, owner removed it)** /srv/bulk/ri/igc-243/build was again a superseded tree (run 5, pre-translator re-pin); run 6 builds in build2. When 205 is done: `rm -rf /srv/bulk/ri/igc-243/build`.
- **Note 205 DONE (~21:05 UTC 11-06):** IGC 2.43.0-dev.0 (fda9face + deps as of 10-31) built, ocloc 26.40 accepted it; image 267 M332 81.82 single-user (+1.2 %), texts identical; H144 vs H143 median 103.4 vs 102.7 (+0.7 %) -> BELOW the bar, NOT adopted (opt-in probe). Lessons for 204/206: router -18 % at 128 GRF under 2.43; logits_i8_rows +9.8 %. Run 207 ended (budget); next run starts with the prefill separation.
- **PAUSED 21:02 UTC 11-06 (run.sh): card0 (0b:00.0) CAT:bcs at 16:51:44 = B156's hang (card1's line was reviewed, card0's was not), counted as a recurrence of c81bc4e's 10-25 bcs fix. Supervisor did NOT clear it (a safety pause is the owner's). Resume = owner adds the card0 reviewed-fault line + rm PAUSED; the next run's first engine work is the prefill separation that targets exactly this.**
- **Run 208 (22:13 UTC 11-06): B157 (prefill 7K on 263, the five fusion defaults OFF) HUNG like B156; CAT bcs on BOTH cards at the stop. => the fusion defaults and dp4a are NOT the cause; suspects: the NEO 26.40 runtime or the family's NEO debug keys (DisableIndirectAccess=1, OverrideCmdListCmdBufferSizeInKb=64). Card work stopped; run.sh will PAUSE at run end (owner). Desk this run: c81bc4e vs 26.40, and a discriminator image (263 kernels on the old 26.35 runtime) built but not run.**
- **Run 209 (00:19 UTC 11-07): PAUSED (note 175): the peer probe PR8 (image 188, old runtime, no engine) HUNG; card1 bcs0 (copy engine) timed-out job, GT reset, devcoredump 20261107T001934Z-devcd6.txt. Card1's copy engine looks wedged since B157 (22:13). Owner: intel-host reboot (root). After it: peer1 must pass, then the prefill separation restarts from B on 268 (B157 counts as unproven).**
- **00:32-00:38 UTC 11-07: owner rebooted intel-host; the soak quadlet ri-m3-flashnext auto-started at boot (both B70s) -> owner stopped it and moved ri-m3-flashnext.container to intel-host:~/.config/containers/systemd.disabled/. PAUSED removed (supervisor, owner OK). Next: peer1 must pass, then B on 268.**
- **Run 209 (00:39-00:59 UTC 11-07): B158 = prefill 7K on image 268 (263's kernels on the OLD 26.35 runtime) on a FRESHLY REBOOTED intel-host after a clean PR9: HUNG the same way.** => NOT the 26.40 runtime, NOT a wedged card. Remaining: the IGC 2.42.6 AOT compile or a code change since image 244 (B155 passed: 26.35 + 2.41.5 AOT + 2cb491f). Next run: B on today's HEAD built with IGC 2.41.5 AOT (or a surviving midpoint image 245-260). Decode unaffected.
- **Run 210 (~02:3x UTC 11-07): B159 = prefill on image 270 (HEAD built with IGC 2.28.4 AOT, 26.35 runtime) CLEAN: 7K 7.19 s, 20K 9.56 s, 7klp 5.03 s, logprobs identical to B155. => THE PREFILL HANG IS THE IGC 2.42.6 COMPILE of our code** (B158 = same code with 2.42.6 hung). Code and fusion defaults cleared. Next: X2 (268 with 270's libxe.so) to split libxe vs core device code; then per-unit compiler mixing. Image 270 = prefill test base; 263 = decode base. Side item: 7K first request +2.6 s vs B155.
- **Supervisor (03:56 UTC 11-07), for the next run if nothing is unblocked:** X2 still waits on the owner's `Bash(bash logs/bstepx.sh *)` rule, and the four-flag default (H146 +2.6 %) waits on the owner's yes. Until either lands, keep intel-host busy with DECODE-ONLY work (short prompts, no prefill on any 2.42.6 image): note 206 item 3 (IGC scheduling / unroll knobs per kernel at AOT, top 5 kernels), then note 201 items 2-3 (gemm_fp8a8 narrow form on the E4M3 hc kernels; the decode split law for verify attention). Measure against image 272 with the four opt-ins ON as the working base (M344 82.59), texts vs M285. X2 or the default flip jump the queue the moment the owner acts.
- **Owner YES (~04:13 UTC 11-07): the four-flag bundle becomes the DEFAULT** (RADIANCE_XE_TOPK_GRF, RADIANCE_XE_QSA_GRF, RADIANCE_XE_ROUTER_GRF, RADIANCE_XE_GEMV_WCC=5; each =0 opts out), default-on in libxe as the fusion defaults (1ab3146). Evidence: H146 vs H143 median 105.4 vs 102.7 (+2.6 %), M344 82.59 single-user, texts identical. One commit, image, a flag-free M-pair (texts = M285, tok/s >= M344) and a =0 opt-out M-pair. Intel test images only; prefill on 2.42.6 images stays uncleared.
- **Owner (~04:2x UTC 11-07): decode optimisation FIRST, then fix prefill.** Order: the -dr- down-shape test -> the four-flag default flip (commit, image, M-pairs) -> further decode work (a -dr- winner + the next bundle) -> THEN the prefill fix (X2 -> the libxe/core split -> the per-unit compiler mix -> the +2.6 s first-request cost -> the AR overlap). X2 no longer jumps the queue.
- **Default flip DONE (~05:0x UTC 11-07): commit df614e9, image 274 = 7c312931fb69. M348 flag-free 82.62 tok/s single-user (texts = M285); -nob4- 80.57. New decode controls: 274 / M348 / H146 (rotation 105.4).**
- **Supervisor (06:27 UTC 11-07), unblocked card work while the owner's two answers wait (the W2A8_R=16 default; the jm32 prefill test):** a MULTI-USER check of today's decode stack, short prompts only: on image 275 with -wr16- (the four defaults on), concurrency 1 / 2 / 3 users under the default policy (RADIANCE_SPEC_MAX_SEQS=3, MULTI_DEPTH=1), headroom >= 4 GiB, the usual coherence pass. NO 4+-user drafting phase (note 199). Compare per-user and aggregate tok/s against the last policy measurements (note 191/199 era) to show the new GRF / cache / wide-draft defaults don't regress 2-3 users. One step, then stop if nothing else is unblocked.
- **Owner YES x2 (~06:50 UTC 11-07):**
  1. **RADIANCE_XE_W2A8_R=16 becomes the DEFAULT** (=2 keeps the rows form, =0 the one-row form): H147 vs H146 median 108.7 vs 105.4 (+3.1 %), M354 85.67 single-user, texts identical; CPU byte-for-byte (3615305). One commit, image 276, a flag-free M-pair (texts = M285, >= M354) and an =2 opt-out M-pair.
  2. **The jm32 prefill test runs now**, as the run's first engine step after a clean peer1: B on image 268 (the 2.42.6 stack) with the GDN chunk scan at v block 32 instead of 64 (a new nomtpcappfmjm32tr config = nomtpcappfmjm64tr with RADIANCE_XE_GDN_CS_JM=32; the m3-prefill.sh diff to the supervisor first). Clean => the v64 JM kernel's 2.42.6 code (3-4x spills) is the cause; hang => X2 next run. "7k 20k 7klp" vs B155 / B159, logprobs compared.
  Order this run: the prefill diff -> peer1 -> jm32 B-step -> the W2A8 default flip (builds may overlap the card step; production is off on amd-host, RAM ample).
- **Run 218 (07:02 UTC 11-07): B160 = jm32 (GDN chunk scan v block 32, 78848 B LDS) on 268 HUNG like B158 => the v64 JM kernel's spills are NOT the cause.** Card work stopped. W2A8_R=16 default committed 4779ad3, image 276 = 0275d4cb8ae0 built and loaded; its M-pair checks (M355 flag-free, M356 -wr2-) wait for the next run. Next prefill step: X2 (268 with 270's libxe.so).

## 207. Owner: "do all of them" -- the remaining decode levers, compiler and process (2026-11-07 ~07:10 UTC)

Decode first (the owner's order), then the prefill fix (X2 ->). Exact throughout (bit-identical CPU row tests + M-pair
texts; anything not exact is a design note only), Intel test images only, -prof per kernel, a 10-min rotation, default
at >= 1.5 % for a set. Base: image 276 (W2A8_R=16 default) once M355 / M356 confirm it. Any new suffix / step script to
the supervisor first.
A. Compiler-side (note 204's open items):
  1. **32-bit addressing:** every global send in the top 30 is A64 stateless; where buffers are < 4 GB, the compile /
     source forms that give 32-bit offsets or stateful access, per kernel, kept only where faster.
  2. **logits_i8_rows' spills** (16 vars, 2 KB a thread; many WGs, so not a blind 256 GRF): a targeted source fix.
  3. **The two draft-head ideas** (un-parked): (a) a trimmed draft output vocabulary (~32K tokens, chosen from the
     rotation + teaching outputs so code / medical / tool-call tokens stay in; exact -- the verify catches misses);
     (b) fewer draft experts (4-6 of 10), judged by net tok/s after acceptance.
  4. **SG16 variants of the SG32 kernels**, only where the per-lane sums and reduction order are reproduced exactly.
  5. **A newer Intel LLVM (SYCL front end) than 7.2.0**, if released: release notes first, then the per-kernel compare.
B. Process-side:
  6. **Expert misses:** re-measure misses a step / residency on today's base (m3/stats_delta.py), then free VRAM for
     experts (headroom, tape memory) and refine residency -- exact.
  7. **Rank skew from the power caps** (card1 230 W, card0 275 W): measure each rank's wait at the all-reduces on today's
     base; report only (the owner's cap decision stands).
  8. **Tree drafting:** verify two candidates at the first draft position (where acceptance is lowest), behind a flag;
     exact by construction (verify decides); judged by tokens a step and tok/s.
  9. **Expert-parallel layout instead of TP2** (fewer all-reduces): a design note only (not exact), no build.
  10. **intel-host's CPU governor and PCIe ASPM:** the owner runs the root commands; the runner then A/Bs (M-pair +
     rotation) against the same image. Until the owner has, skip.
- Order: M355 / M356 (276's checks) -> A1 -> A3 -> A2 -> B6 -> B7 -> B8 -> A4 -> A5 -> B9; B10 when the owner's commands
  are in; then the prefill fix (X2 -> the libxe half-split -> the per-unit compiler mix -> the +2.6 s first request ->
  the AR overlap). Report "vs 90" from here (the owner's new aim).
- **Owner (~07:15 UTC 11-07): intel-host's CPU governor set to performance** (scaling_governor + EPP = performance; not persistent, resets on reboot). PCIe ASPM is already disabled on both B70 links (closed). So B10 is ACTIVE: M355 (276 flag-free) now runs under performance and doubles as the A/B vs M354 (275 -wr16-, the same decode config under powersave, 85.67); then H148 = bestr on 276 vs H147 (108.7). Report the governor's effect separately from the W2A8 default's checks. Note it in every step's precheck (governor read).
- **Owner OK (~07:25 UTC 11-07): note 207 item B11, the draft loop on the card.** Today each of the 3 draft passes
  needs the previous pass's sampled token, so the host reads it back and starts the next tape while the card idles
  (upstream's graphrate.hip names the same problem on AMD). (a) MEASURE first on today's base: the device-idle gaps
  between draft passes and before the verify (the tape profiler's edge table, RADIANCE_L0_NTAPE_EDGES, or the
  RADIANCE_DEBUG_STAGE host times), ms a step and % of the step. (b) Then a DESIGN note (no build without the
  supervisor): the draft token picked on the device (argmax / the existing rowtopk) and fed into the next draft pass's
  embedding inside one recorded list -- the draft chain as one tape, the host only at the verify. Exact by construction
  (the same tokens). Place B11a right after M355 / M356 (it is a measurement); B11b's design after A1.
- **Run 220 (~11:0x UTC 11-07): X2 (B162) CLEAN: image 268 (2.42.6 stack) with 270's 2.28.4 libxe.so -> 7K 7.96 s, 20K 9.63 s, 7klp 5.05 s, logprobs identical to B155 / B159. => the prefill hang is in LIBXE's IGC 2.42.6 compile (not core / engine device code).** Next: the libxe half-split; the mixing method must first be proven to keep the swapped units' 2.28.4 device code (RDC device link would recompile everything).
- **Supervisor (~11:25 UTC 11-07): the +2.6 s first-request cost = the startup warm-up being OFF today (B155 had it ON: its 4096-row chunk ran at startup). Proposal C36 (warm-up ON, 1-3 drafting, 4-6 not) is HELD: the warm-up is itself a prefill-sized step and would hang on any 2.42.6 libxe. Gate it after the prefill fix; the warm-up default stays the owner's call (note 197). No warm-up-on step on a 2.42.6-libxe image.**
- **Supervisor (~11:50 UTC 11-07): APPROVED logs/run221-step0-proposal.diff** (nomtpcappfmjm64tr2k = jm64tr with --max-num-batched-tokens 2048; apply it, then confirm the substitution actually happened in the step's serve.log args). Next run order: split-link (if the owner's rule is in) -> peer1 -> B163 = bstep.sh B163 4ce8ac5eacbf nomtpcappfmjm64tr2k B155 "7k 20k 7klp" on image 270 (2.28.4 libxe, WCC 0); 7klp logprobs vs B159, per-chunk times vs B159.
- **Owner (~15:55 UTC 11-07): the split-link allow rule is IN** (Bash(bash m3/split-link.sh *), verified). Next run: m3/split-link.sh split1 (the 16 GDN / attention / QSA objects on 2.42.6, the rest on 2.28.4) -> the duplicate-kernel-name check, the CPU suites on the mixed .so, the ISA proof -> peer1 -> B on 268 with split1's libxe.so via bstepx (the run's first prefill engine step). Hang = the culprit is in the GDN / attention / QSA half; clean = the other half. Then halve again.
- **Owner (~15:56 UTC 11-07): "get the prefill first".** The prefill split is this run's FIRST and ONLY priority, before any other work. Supervisor exception for it: m3/split-link.sh (each container capped at 4 CPUs / 10 GB) MAY run while 3a's A5 battery has sparse-run active, provided MemAvailable >= 45 GiB at the start of each container stage (production is up; the LoRA guard trips at 30). Then the proofs, peer1, and the split1 B-step on 268 via bstepx on intel-host (intel-host isn't shared with amd-host's tests). Report the split1 result to the supervisor at once.

## 208. Owner: maintenance after the prefill fix -- Level Zero loader 1.35.0, then kernel 7.3.x (2026-11-07 ~16:20 UTC)

ONLY after the prefill fix has landed (the split's culprit unit found and a working mixed libxe), one change at a time:
1. **Level Zero loader 1.29.0 -> 1.35.0** in the test images (the runtime images' libze1; spec 1.20; min GPU driver
   21.09, NEO 26.40 is fine). Exact by construction (the loader only dispatches). Check: peer1, a decode M-pair
   (texts = M285, tok/s within noise of the base), one prefill B-step on the fixed libxe. Adopt if clean.
2. **Kernel 7.2.6 -> 7.3.x on intel-host** (7.2 is EOL at 7.2.15; 7.3 has the Battlemage VRAM-size fix 818bebeb63dd).
   The owner installs (root, a reboot), keeping 7.2.6 bootable as the rollback. Then the full re-baseline: peer1, the
   M-pair, a rotation, a prefill B-step, the klog; compare with today's base. Note the GuC stays 70.58.0 (70.72.1
   broke the peer exchange on 10-29).
3. Revisit Linux 7.4 (the migration-queue / ULLS changes touching the copy engine) once it is stable.
Usage is near the weekly limit (98 %, resets Tue 11-10): these wait behind the prefill work.
- **Supervisor (16:35 UTC 11-07): APPROVED logs/run250-split-proposal.md.** (1) m3/split-link.sh.proposed replaces m3/split-link.sh (two shared objects: half B -> libxe_b.so linked in ri-m1buildneo:2 / IGC 2.42.6, half A -> libxe.so in ri-m1build:3 / IGC 2.28.4 with DT_NEEDED libxe_b.so + rpath /out; the object list from build.ninja's libxe link line, not find; fmha excluded -- X2 already cleared it). (2) logs/bstepx.sh gains the one line copying libxe_b.so beside libxe.so. Then: split-link split1 with the 15 objects -> readelf + dups.txt proofs (any kernel name in both halves: move its units together) -> peer1 -> B164 on 268 via bstepx (split1's libxe.so). Hang = half B (GDN / attention / QSA); clean (logprobs = B155) = half A. Prefill stays top priority.

## 209. Owner: speed up PREFILL the way decode was sped up (2026-11-07 ~16:45 UTC)

Starts right after the prefill hang fix (the split's culprit unit on 2.28.4, the rest on 2.42.6, a working mixed libxe);
note 208's maintenance waits behind it. Exact throughout (bit-identical CPU row tests + 7klp logprobs identical to the
base; a non-identical form only as a note under note 149's rule), test images only, one prefill engine step per run as
the first engine step after a clean peer1 while any prefill hang risk remains. Gate: a B-step (7K / 20K / 7klp) and
default at >= 5 % off the 7K or 20K (note 195's prefill bar). Base: the fixed mixed image, warm-up OFF (its default
is the owner's call, note 197 / C36).
1. **Per-kernel compiler choice for prefill:** -prof a 7K + 20K with every prefill unit under 2.28.4 vs 2.42.6 (the
   split's machinery); keep the faster compiler per unit when the result is identical. Table: kernel, ms / chunk, both.
2. **ISA audit of the prefill top kernels** (desk first, m3/igc-dump.sh + isa_hist.py): the expert DPAS kernel
   (~2.3 s of a 4,096-row chunk) FIRST, then the GDN chunk scan (JM, v64 / v32), prefill attention, hc_read PF_DPAS.
   Per kernel: instruction classes, spills / scratch, GRF, SIMD, sends (block2d vs scattered), DPAS issue density.
   Flag decode-style waste (byte repacks, narrow loads, address math, spills) and fix it in source, exact.
3. **Loads into the matrix unit:** where the audit shows LDS round trips or scattered loads feeding DPAS, the 2D
   block-load + VNNI path straight into DPAS operands (sycl-tla's structure; run 5x's lesson), behind a flag.
4. **Register mode per prefill kernel** where it spills, measured (decode's lesson: 256 GRF only where it spills and
   occupancy allows).
5. Afterwards (owner decisions, not builds): the warm-up default (C36 re-gate), PCIe 5, the exact AR overlap (needs the
   op that differs at a 2048-row boundary found first, B163).
Report "7K / 20K vs B159 (7.19 / 5.03 warm / 9.56 s)" after each adopted step. Usage is near the weekly limit: desk
steps first, card steps lean.
- **Supervisor (18:35 UTC 11-07): APPROVED logs/run255-nohcpf-proposal.md** (nomtpcappfmjm64trnohcpf = jm64tr + RADIANCE_XE_HC_PF_DPAS=0, on stock image 268). Run it BEFORE run 256's timer step (the timer question is secondary): it pins the hang inside r4d_hc_bf16. If clean: the fix = move the prefill hc PF_DPAS kernels into their OWN translation unit built by IGC 2.28.4 (the two-library arrangement, as libxe_b / fmha), so the decode hc kernels keep 2.42.6's gains (hc_v2_gu was -23 % under 2.42.6); measure PF_DPAS=0's prefill cost too, as the stopgap. If it hangs: ISA of the unit's other prefill-reachable kernels. intel-host disk at the floor: keep deleting each step's copies; the old SYCL cache prune is the owner's.
- **Supervisor (19:52 UTC 11-07): APPROVED logs/run257-grf-proposal.md** (nomtpcappfmjm64trgrf = jm64tr + RADIANCE_XE_I4I8_DPAS_GRF=1; split6 libxe; gate 7klp = B155 / B169, default at >= 5 % off 7K or 20K vs B169 6.49 / 8.46). ORDER for the next runs: (1) the nohcpf test approved at 18:35 (pins the hang kernel -- the fix that makes prefill work on the decode image comes first), (2) B172 GRF. The per-op timer is retired for prefill (hangs on its own).
- **Supervisor (21:20 UTC 11-07): APPROVED logs/run258-hcpf-unit-design.md** -- hc_pf_dpas_kernel<0/1> + its helpers MOVED unchanged into libxe/hcpf/xe_hcpf.cpp as MODULE libxe_hcpf.so built by IGC 2.28.4 (build-m1l0), dlopened by r4d_hc_bf16 like libxe_fmha.so, missing/refused -> per-row path (slower, never a hang); the shared helpers in one header; make-rt-neo2.sh copies that one file from build-m1l0 (the script edit is covered by this approval). Gates as written: hc CPU tests, the module's ISA = igcdump-hc228, a B-step (7klp = B155 bit for bit, 7K / 20K at B169's), a decode M-pair (texts = M285, tok/s at M365's). THIS IS THE TOP PRIORITY (the owner: prefill first). Process: when a proposal waits for the supervisor, ALSO send it by SendMessage to radiance-test-95 (ListAgents), not only in a file -- three reviews today waited unseen for 30-60 min.

## 210. Owner: the prefill expert kernel rewrite -- design note first (2026-11-07 ~21:35 UTC)

After the hc PF_DPAS fix (note 209's base image). DESK ONLY first; nothing built without the supervisor's review.
Context: format C's prefill expert GEMM (moe_i4i8_dpas_kernel) is ~2.3 s of a ~3.2 s 4,096-row chunk (~70 % of
prefill) at an estimated few % of the B70's int8 DPAS peak; 128 GRF spills 120 vars; 256 GRF removes the spills but
halves occupancy and is slower (B173); AMD production does 8K in ~1.4 s.
Write logs/run2xx-expert-gemm-design.md covering:
1. **Measure first (desk + one -prof if needed):** achieved TOPS per card, rows per expert at 7K / 20K (the routing
   histogram), time split inside the kernel (B loads + int4 decode / A staging / DPAS / epilogue), and the floor
   (bytes and DPAS ops) -- what a good kernel could reach.
2. **The design:** a grouped GEMM over experts -- rows gathered and sorted by expert into larger tiles (or a
   persistent kernel walking an expert work list); B (4-bit codes) by 2D block loads, decoded to s8 in registers
   (format C's level table) and fed straight into DPAS in VNNI layout; A (s8 activations) by block loads; no LDS
   round trips where avoidable; tile shape for occupancy at 128 GRF (B173's lesson). Reuse what exists:
   sycl-tla's structure, vllm-xpu-kernels' grouped_gemm tiling (note 169, ideas not code), the ESIMD expert-decode
   reference (Strata, note 169's reading).
3. **Exactness:** same per-element products and accumulation order as today (int32 accumulators, the same K order),
   or if a different order is unavoidable, say so -- that is the owner's call (note 149's rule for non-identical).
4. **A probe plan:** a standalone kernel on ONE expert's real shapes (fc_tile_probe style), bit-identical to today's,
   go at >= 1.5x on the expert kernel's time; then the full kernel behind a flag; then B-steps (7K / 20K / 7klp).
5. **Estimate:** expected kernel speed-up, 7K / 20K after, and the work (runs) needed.
Then the supervisor reviews and takes it to the owner before any build.

## 211. Owner: extra usage bought -- PREFILL ONLY until Tuesday (2026-11-07 ~22:15 UTC)

The owner bought extra usage and wants it spent purely on prefill. Until the weekly reset (Tue 11-10 12:00 UTC):
- **Do:** (1) finish the hc PF_DPAS module fix (logs/run258-hcpf-unit-design.md; the uncommitted xe_hcpf.cpp + edits
  from run 261 are the starting point) and its gates; (2) then note 210's expert-GEMM design note (desk) and, after the
  supervisor's review and the owner's word, its probe; (3) note 209's prefill items in between where they need no new
  design (per-kernel compiler choice for prefill units).
- **Hold until Tuesday:** all decode work, note 208's maintenance (loader / kernel), and anything not prefill.
- **Lean:** desk steps first, card steps only when they decide something; skip long write-ups; message the supervisor
  (SendMessage radiance-test-95) when a review is needed.
LoRA work on amd-host is also on hold until Tuesday (the owner).
- **Run 26x (~22:5x UTC 11-07): THE PREFILL HANG IS FIXED on a stock-built image.** Image ri-m3rtneo:283 = 50018225563c (2.42.6 libxe + the 2.28.4 libxe_hcpf.so module): B175 7K 6.37 s, 20K 8.41 s, 7klp 4.60 s, 7klp logprobs = B155 bit for bit, klog clean. Pending M367 (decode on 283) to make 283 the single prefill + decode base.
- **Image 283 = the single base for prefill AND decode (~23:1x UTC 11-07):** M367 decode 85.75 tok/s (M365 85.93, noise), texts = M285; B175 prefill 6.37 / 8.41 / 4.60 s, logprobs = B155. Commit ae0cfb9. Prefill vs B159: -11 % / -12 %.
- **Owner YES (~23:15 UTC 11-07): the startup warm-up ON, after its gate on image 283** (C36's shape: -warmon, 1-3 users with drafting, 4-6 users WITHOUT drafting under the cap -- no 4+-user drafting, note 199; headroom >= 4 GiB). Plus one B-step with -warmon (first 7K should be ~4.6 s; 7klp logprobs = B155). Clean on both -> flip the warm-up default ON for the Intel images (one commit / ENV, =off opts out), and report startup time added. A fault -> stop, review.
- **Owner (~23:30 UTC 11-07): "approved" -- stage 3 (note 210) is pre-approved:** once the supervisor has reviewed the expert-GEMM design note, the one-expert probe (bit-identical, go at >= 1.5x on the kernel) and, if it passes, the full kernel behind a flag may proceed WITHOUT another owner round-trip. The default flip still follows the gates (B-steps: 7klp = B155, >= 5 % off 7K or 20K; decode M-pair unchanged). Anything non-exact, or any change outside the expert GEMM, still comes back to the owner.
- **Supervisor (~00:0x UTC 11-08): logs/run262-expert-gemm-design.md APPROVED** (owner pre-approval, 23:30). Proceed: step 0 DPAS_ROWS=16 B-step -> probe (bit-identical, >= 1.5x) -> full kernel behind RADIANCE_XE_I4I8_DPAS_V2 -> gates. B177 measured: 42.9 % / 51.5 % padded rows at the 7K's chunks; kernel at 1.4-2.9 % of the int8 pipe; estimate kernel 2.0 -> 0.5-0.7 s a rank (7K 6.37 -> ~4.9-5.2 s no warm-up, 20K 8.41 -> ~6.3-6.8 s). The 56-token-tail DPAS cut-off (not exact) is recorded for the owner, not part of this.
- **Supervisor (~00:2x UTC 11-08): design correction ACCEPTED (section 2b):** the register-built B fragment is withdrawn (8x the decode); kept: tile list by run length, A from global, B via double-buffered SLM (one barrier a group), hoisted scales, persistent walk. Revised estimate: kernel ~2.0 -> ~1.0-1.2 s a rank; 7K warm-up-on 4.21 -> ~3.3-3.5 s, cold 6.37 -> ~5.4-5.6, 20K 8.41 -> ~6.9-7.3. Probe bar unchanged (>= 1.5x).
- **Owner YES (~00:3x UTC 11-08): read upstream's AMD prefill expert kernel's machine code (desk, ~1 h, part of note 210).** Compile upstream-1.17.0's prefill MoE GEMM (libr4d, the gfx1201 path the R9700 production uses) in ri-m1rocm with --save-temps / -Rpass to get the ISA; report its tile shape, LDS use (double-buffering? bytes), barriers a K step, load widths (buffer_load_dwordx4 etc.), WMMA density, register use / occupancy. Then list which of its techniques fit the B70 design (format C decode stays), and fold them into the probe before it is built. CPU-only, amd-host; one build at a time; no card.
- **Warm-up default DONE (~00:3x UTC 11-08): image 285, ENV RADIANCE_STARTUP_WARMUP=on (Intel images only), commit 79b609f. B178 (no flag): first 7K 4.18 s, 20K 8.39 s, 7klp = B155, +6.37 s startup. Base for prefill + decode: 285 (V2 image 286 = 285 + the V2 switch, off by default).**
- **Run 265 (00:26-00:45 UTC 11-08): expert-GEMM probe NO-GO.** V2 exact but slower: 6,603 us a launch at 128 GRF (1.40x slower), 5,373 at 256 GRF (1.14x slower) vs today's 4,708 (go bar <= 3,139). Cause: instruction stream -- today's loop 1,453 instructions per 64-K group (421 mov, 160 divergent blocks, 93 64-bit address ops, ~410 for the 4-bit decode at ~11 per code); AMD's 404 per 128-K pass = 7.2x less per MAC. **Supervisor (~00:5x): approved (a) V3 desk (clamped unconditional loads, compile-time switches, 32-bit offsets, 16-byte code loads; gate = loop-body count, then one B-step; est. +15-25 % kernel, ~6-10 % prefill), (b) commit V2 as an opt-in record, (c) ESIMD byte-LUT decode at the desk (ISA only).**
- **V3 desk gate met (~01:1x UTC 11-08):** the expert DPAS loop body 1,453 -> 726 instructions a 64-K group (-50 %), exact by construction. Root find: lvl()'s pk[4] array was compiled to PRIVATE-memory gathers (32 a group) inside branches; four scalars + three register selects fix it. B184 on image 288 next. **Lead for AFTER Tuesday: the decode format C run / dec kernels carry the same pk[] private-gather pattern (format C ~27 % of decode).**
- **V3 ADOPTED (~01:5x UTC 11-08): image 288 = 65ef03f8efa9, commit 4e1e25f (xe_moe_i4i8.cpp): B184 first 7K 3.73 s (-10.8 % vs 4.18), 20K 7.02 s (-16.3 % vs 8.39), 7klp = B155 bit for bit; warm 7K 3.55 s; startup warm-up 5.93 s. ESIMD decode: real (~1.5 instr/code) but invoke_simd crashes ocloc 26.40 -> a whole-kernel ESIMD rewrite only, recorded for the owner, not started.**
- **Owner YES (~02:1x UTC 11-08): V3b on the trunk GEMMs gemm_i8a8_dpas / gemm_bf16_dpas** (prefill-only; unrolled staging, i8a8 fold from wi_data, scales once a group, clamped loads; exact). B185 after V3: expert 2,404 us (-49 %), device total -18.8 %; trunk GEMMs 20.7 %. Desk gate, then one B-step vs B184 (3.73 / 7.02).

Owner YES (~03:1x UTC 11-08, via coordinator radiance-test-95, after B187): (1) lever #3 = the protected experts' moe_gemm_bf16 on the matrix units (joint_matrix bf16, sorted rows + block map): NOT bit-exact, so OPT-IN behind a flag, default only if it passes note 149's KL-replay bar; numbers to the owner before any default flip. Order: after expert round 3. (2) lever #2 = prefill all-reduce overlapped with the next half-chunk's GEMMs: DESIGN NOTE ONLY now (which ops split, why each stays bit-identical given B163, expected gain); coordinator review, then the owner; no code until approved. Still prefill only until Tue 11-10 12:00 UTC (note 211).

Owner (~04:xx UTC 11-08, via coordinator radiance-test-95, after the run267 design notes): lever #3's design approved by the coordinator (conditions sent to the runner: flag-off bits = B155, unit test vs the old kernel incl. edge shapes, fallback outside tested shapes, desk gate <= 687 / no private memory / spills <= 37, no default flip before the KL numbers reach the owner). Lever #2: CARD TEST ONLY -- the step-0 two-queue concurrency probe on card1 (bounded spin + host timeout, fault rules apply, measure the memory-bound kernel's own slowdown), after lever #3's B-step; script to the coordinator for review first. No lever #2 build; decide after the probe result.

Coordinator radiance-test-95 (~03:29 UTC 11-08; the owner raised the usage limit after the 03:27 spend-limit stop): MB2 + B188 read and PASSED. MB2: no differing output farther from fp64 than the old order by > 1 ulp (matrix form closer on 77 of 108), every > 2 ulp outlier's cancellation ratio >= 270k, C2 max 1 ulp -> the 6 ulp in shape C is re-association, not a defect. B188 (290, flag off): 7klp token logprobs EQUAL to B155 (compared the .http files), 7K 3.48 / 20K 6.72 (= B186 within noise). GO: B189 (nomtpcappfmjm64trmbd, times; 7klp recorded, not gated) then B190 (nomtpcappfmjm64trmbdprof, us a launch vs 5,625) -- two engine steps a run, fault rules as always. Commit lever #3's four files as one revertable commit (flag default off) after B189 runs clean. No default flip: the KL-replay plan comes to the coordinator first. The mbstep.sh / tqstep.sh allow rules are not needed (timeout prefix, note 41).

## 212. Owner answers to run 270's WAITING items (owner via coordinator radiance-test-95, ~14:1x UTC 11-10)

1. **Lever #3 DEFAULT ON.** RADIANCE_XE_MOE_BF16_DPAS defaults to 1 (=0 keeps the WMMA form as the opt-out). Passed note
   149's bar on both corpora (KMB1 0.0027 / 98.9 %, KMB2 0.0021 / 98.65 %). One revertable commit. A new exact
   reference replaces B155 for later 7klp gates: one plain B-step on the default-on image (7k 20k 7klp), recorded in
   STATE.md as the new reference (B155 stays the reference for the flag-off path). Decode stays bit-exact (T > 256
   class only) -- confirm with the next M-pair / rotation as usual.
2. **The scalarized format C decode hunk (image 292): commit it as an exact tidy-up** (one revertable commit, its own),
   recorded as a null result (M368 86.22, H150 108.0).
3. **Start now, in this order, with the usual review (designs / scripts to the coordinator before any card time):**
   a. **Note 208, Level Zero loader 1.29.0 -> 1.35.0**: the coordinator fetches libze1 1.35.0 (note 144); the runner
      builds the test image and runs PR + B + M-pair against today's; kernel 7.3.x stays the owner's, later.
   b. **The hc prefill module's 2.42.6 hang: DESK ONLY** -- the ISA comparison 2.42.6 vs 2.28.4 of hc_pf_dpas_kernel;
      no card test without the owner's OK.
   c. **Lever #2's precondition: the light poll** in xe_allreduce.cpp (one thread a block polls, the rest wait at the
      barrier; exact) -- design to the coordinator first; then PR + B (7klp exact) + M-pair (decode uses the same
      all-reduce). The overlap itself stays a later owner decision.
4. **LoRA stays on hold** (not lifted in this answer). Note 211's prefill-only rule is over (the weekly reset).
   Note 212 item 3a: the coordinator fetched it -- m3/l0-1.35.0/libze1_1.35.0+u24.04_amd64.deb (GitHub release v1.35.0,
   1,131,026 bytes = the release asset size, sha256 8420f4fd...dfcf3 in m3/l0-1.35.0/SHA256, .sig beside it). The images
   take libze1.deb from ri-m1build:3's /debs: put this one in the test image's runtime stage only (a new Containerfile,
   to the coordinator for review), never the build image.

## 213. Stretch the Claude usage: hand cheap jobs to the local Flash model (owner, ~14:2x UTC 11-10, via coordinator)

The weekly limit is the constraint now. For cheap, read-only jobs use the local Qwen3.8-Flash-Next instead of reading
big files yourself: `timeout 300 bash ~/code/radiance-test/flash.sh "question" file [file ...]` (summarise a serve.log
or build log, pull the numbers out of a results file, list the errors in a klog). It answers in seconds; input is
capped at 48k characters. Keep with Claude: designs, reviews, exactness calls, anything that edits code or decides
what runs next -- and spot-check a number it gives you before you put it in STATE.md or a gate. If it fails, do the
job yourself. It shares production's cards, so one request at a time and never in a loop.

## 214. Order after note 212: decode is back on the list (owner YES via coordinator radiance-test-95, ~14:2x UTC 11-10)

All items below are exact; the usual review applies (designs / scripts to the coordinator before card time; anything
non-exact or a default change goes to the owner).
1. **Finish note 212 in its order** (lever #3 default + new 7klp reference, the format C tidy-up commit, 3a loader
   1.35.0, 3b hc hang desk only, 3c light-poll design to the coordinator). The T forms (B196/B197): note 209's >= 5 %
   gate stands -- opt-in unless a fix of their own overhead reaches it.
2. **Then a decode profile on the new base** (lever #3 default-on image, MTP as production runs it): one profiled
   M-step, and refresh briefs/kernel-scoreboard.md with each kernel's share of the step and its distance to its floor.
   Report the top 3 exact decode items with estimated gains (candidates already in the notes: 172 LDS bank conflicts
   then attention, 167 format C to floor, 165 AR fusion, plus whatever the light poll does for decode's all-reduce).
3. **Then the best exact decode item**: design to the coordinator first, then PR + M-pair + rotation (aim: 90 tok/s
   single-user; today 86.2 / rotation 108.8).
flash.sh: if it prints nothing or fails, do the job yourself (it was fixed 11-10 to report an empty answer as a failure).

## 215. The three exact decode fusions DEFAULT ON (owner, ~21:3x UTC 11-10, via coordinator radiance-test-95)

Owner answer: **Default ON.** RADIANCE_XE_GATED_QUANT_PAIR, RADIANCE_XE_QK_PROLOGUE and RADIANCE_XE_ROUTER_QUANT_PAIR
default to 1 in the Intel builds (=0 opts out each). Evidence: same image 301, M388 87.14 vs M387 85.55 (+1.9 %),
rotation H152 vs H153 +1.6-1.8 % (every prompt length up), byte-identical on the B70 (QK1, RQ1), M389's kernel table.
- One revertable commit for the flip, with the evidence in the message. Production (the service) is not changed (note 96).
- After the flip: one plain M-step on the new image with NO flags (texts = M285; tok/s ~87) and its -prof twin
  showing the three fused kernels by default -- the new decode reference. Then pairs 4 / 5 measure against it.
- Pairs 4 / 5 (RADIANCE_XE_GEMV_MULTI) stay opt-in until their own numbers reach the owner.

## 216. Coordinator answers to runs 275-278's two open items (coordinator radiance-test-9e, ~21:4x UTC 11-10)

The coordinator session restarted and run 275's messages were lost; these are the answers. From now on, if a coordinator
ask has had no answer for one run, write it here as a question and stop -- the owner reads this file.
1. **H154 OK** (the four-flag rotation, staged): `timeout 3000 bash logs/hstep.sh H154 d9f5eafd6d92 bestrxgrf3w16p4 10`.
   Reviewed: BX 28 = BX 27 + RADIANCE_XE_GEMV_MULTI=1, the name is in the config arm (line 153) and BX 28 in the bundle
   list. Compare with H152 (three flags, image 301; 301 -> 302 adds only opt-in code); every prompt length reported.
   Also state the standalone gemv_i8a8<1,4,1,0,5> / gemv_bf16<0,0,4,0> us a launch in M391 vs M389 (my ±2 % check).
2. **Then note 215's flip** (the three fusions default on, owner YES) + the flagless M-step and its -prof twin as the
   new decode reference. Pairs 4 / 5 (GEMV_MULTI) stay opt-in; their default goes to the owner with M392 + H154.
3. **Router redesign (logs/run275-router-design.md): GO, bench first.** RB4 (sort: list in registers, fixed-64 unrolled
   loop) and RB5 (selection: per-lane sorting network then head compares) on card1 via router-bench.sh, no engine,
   ~$1.5 each, keys / ids / weights / sorted / offsets / counts byte for byte vs today's at M 1 / 4 / 8. Build the
   kernel only if RB4 + RB5 together show >= 3 us a launch saved at 4 rows; else record and stop. Then the usual:
   row test, card row check, profile first (the router kernel's us a launch), texts, M-pair. Opt-in flag.

## 217. Pairs 4 / 5 (RADIANCE_XE_GEMV_MULTI) DEFAULT ON (owner, ~22:3x UTC 11-10, via coordinator radiance-test-9e)

Owner answer: **Default ON.** Evidence: M392 88.50 vs M393 87.18 (+1.5 %), H154 vs H152 +1.0-1.2 % at every prompt
length, GM1 byte-identical on the B70, M391's kernel table, M394 = M389 for the standalone GEMVs. One revertable commit
(=0 opts out), after RB4 or at a clean point; then a flagless M-step + its -prof twin on the new image = the new decode
reference (expected ~88.5-89). Add GEMV_MULTI=0 to `-nopairs-` so that control stays "no fusions". Service untouched.

## 218. Owner: DO THE REWRITE -- whole-kernel ESIMD for the 4-bit expert kernels (owner, ~22:4x UTC 11-10, via coordinator radiance-test-9e)

Owner: "do the rewrite" (the ESIMD option, ~$200 class). Target: the format C decode expert kernels -- gate_up
`moe_i4i8_run_kernel<5,16,..>` (~107 us a launch x 51 = 5.5 ms a step, floor ~29 us = gap ~3.6-3.9x) first, then down
`<3,2,2,..>` (2.9 ms a step). Background: run 265's ESIMD decode was real (~1.5 instr / 4-bit code vs ~11 today) but
invoke_simd crashed ocloc 26.40, so it must be a WHOLE kernel in ESIMD (no SPMD/ESIMD mix). After note 217's reference.
Staged, each stage a stop gate; designs and step scripts to the coordinator before card time:
0. **Desk, ~$10:** confirm the kernel compiles as pure ESIMD under the image's IGC 2.42.6 / ocloc (a minimal ESIMD
   kernel with the format C load + decode + DP4A loop, AOT for the B70); if it does not, STOP and report. Reconcile the
   scoreboard's gate_up floor/gap line (the 29.4 us floor at the verify's distinct experts) so the target is one number.
1. **Design note:** register layout, block loads (2D / lsc), the int4 -> int8 decode, the DP4A / dot order. State
   whether it is BIT-EXACT vs today's kernel (same per-row accumulation order: preferred) or not. A non-exact form stays
   opt-in and goes through note 149's KL replay to the owner before any default.
2. **Standalone kernel + row test + bench on card1** (decode shapes: 1 / 4 rows, the verify's expert counts): go only
   if >= 1.5x on the kernel (gate_up <= ~70 us a launch) and bytes equal (or the KL path, if declared non-exact).
3. **Engine:** opt-in flag, profile first (the ESIMD kernel present, the old one gone), texts / M-pair, rotation.
   Then the down kernel the same way. The default goes to the owner with the numbers.
Budget guide: stop and report at any stage that exceeds ~$80 without passing its gate. The service is untouched.
Prefill's expert DPAS kernel (same 4-bit decode, ~410 instr of decode a 64-K group) is the natural phase 2 -- not yet.

## 219. Owner: ALSO the new expert weight layout and the new all-reduce (owner, ~22:5x UTC 11-10, via coordinator radiance-test-9e)

Owner: "new all reduce and new expert weight layout" -- both GO, with note 218's staging and gates.
A. **Expert weight layout = part of note 218, not a second rewrite.** briefs/2026-11-05-expert-layout.md targets the
   same format C kernels. The ESIMD design (218 stage 1) chooses the load-time layout its block loads want (the brief's
   stage 1 probe on one expert of each shape is the measurement), so there is ONE kernel rewrite and ONE layout. Rules:
   the relayout is a permutation of the same bytes (exact by construction; prove it with a round-trip test); done at
   load time if it fits the load budget, else by rad-convert -- and any rad-convert / quantize runs only under the
   night-timer rule (no builds while production has users). Pick the layout before the ESIMD kernel is finalised.
B. **New all-reduce (decode LL form 5 = 2.06 ms a step at 19.5 us x 106; prefill's bandwidth form 6).**
   0. Measure first, ~$30, card1+card0, no engine: device timestamps inside the LL kernel's phases (launch -> payload
      store -> poll -> emit / hc write -> arrival counter) on the real decode message sizes (2-20 KB). Report where the
      ~17-19 us goes vs a ~10 us two-GPU floor. STOP and report if no phase holds >= 5 us that a new form could remove.
   1. Design to the coordinator: candidates from AO1 / the run-271 analysis -- fence once after the match instead of per
      poll iteration, the payload by the copy engine or fewer / wider stores, fused emit. The sums must stay bit-exact
      (the same two-operand bf16 adds); the protocol must keep the tear / sequence protection that forms 0-3 lacked.
   2. Gates: allreduce_test incl. XE_AR_TEST_TEAR, a long soak (>= 1 M launches both ranks, 0 mismatches, 0 stalls),
      the peer probe, then profile first, texts / 7klp = B199, M-pair, rotation. Opt-in; the default goes to the owner.
Order on the cards: note 217's reference -> 218 stage 0 (desk) -> B.0 (cheap card measurement while 218's design is in
review) -> 218 stages 2-3 with A's layout -> B.1-2. One card step at a time; the ~$80-per-stage stop rule applies to each.

## 220. Owner: dig further into the Intel compiler (owner, ~23:0x UTC 11-10, via coordinator radiance-test-9e)

Facts: we are on the newest releases (compute-runtime 26.40.40196.8 of 11-03, IGC 2.42.6 of 10-26; checked 11-10).
So this is about getting more out of IGC 2.42.6, exact only. Slot it right after note 217's reference, before 218
stage 2 (its findings feed the ESIMD build flags). Budget ~$40; stop and report if nothing passes.
1. **Per-kernel compile options on the top decode kernels** (format C gate_up / down, gemv_i8a8<1,4,1,0,5>, the paired
   GEMVs, hc_v2_gu / down_rows, the LL all-reduce): GRF mode 128 / 256 / auto, -ze-opt-large-register-file, the
   relevant IGC options for loop unroll / scheduling / SIMD width where the kernel allows. Desk first: the ISA dumps
   (instructions, spills, send count) per variant; then one bench per kernel on card1. A variant counts only if the
   kernel's output is BYTE-IDENTICAL (row test on the card -- compiler options can change FP contraction) and it is
   >= 5 % faster a launch. Winners go behind per-kernel switches, then profile first + M-pair; defaults to the owner.
2. **The hc prefill module on 2.42.6** (note 212 3b, desk only so far: no structural ISA difference found): one more
   desk pass with IGC's own debug output (the hang's kernel under 2.42.6 vs 2.28.4: barriers, SLM fences, named
   barriers, send ordering). A card test still needs the owner's OK -- ask with the evidence. Worth ~2-4 % of prefill.
3. **Watch list:** the next NEO / IGC release (about monthly) gets the loader-style A/B (image, PR, B + M-pair, exact).

## 221. Owner: the hc prefill module's 2.42.6 hang -- CARD TEST OK (owner, ~23:1x UTC 11-10, via coordinator radiance-test-9e)

Owner: "card test is ok" for note 220 item 2 (the hc prefill module built with IGC 2.42.6, which hung the card in
run 262). Rules, because a hang can need a GT reset:
1. Desk pass first (note 220 item 2) and a written hypothesis + the exact build to test, to the coordinator.
2. **Smallest reproducer first:** the hc_pf unit / row test alone on card1 (no engine, no model), bounded spin, host
   timeout, guard --hot 80, the kernel log before and after. Only if it runs clean and byte-identical to the 2.28.4
   module: one engine B-step (7k 20k 7klp, 7klp = B199) as the run's FIRST engine step after the peer probe.
3. A hang or fault = STOP at once: no retry in that run, the peer probe and the reset procedure as in notes 173 / m3
   fault 6 (CAT:bcs -> stop the next load), report with the klog. The owner is told before any second attempt.
4. Adoption (the module on 2.42.6 by default) is exact-gated (7klp bit for bit) and still goes to the owner with the
   prefill numbers (expected ~2-4 % of a first token).

## 222. Coordinator: B.0 design (logs/run279-ar-b0-design.md) APPROVED for the next run (coordinator radiance-test-9e, ~23:3x UTC 11-10)

Build the stamps + xe_ar_stamps query + the bench table as designed (XE_AR_STAMPS=1 opt-in; off = today's kernel but
for the one uniform branch), CPU allreduce test green, then send me logs/arstep.sh + m3/intel-host/ar-stamps.sh before the
step (one both-card step, no engine). Additions:
1. Report the bench's launch wall with stamps OFF and ON at each size, so the measurement's own cost is visible.
2. If xe_allreduce_test can run the fused form (HW = the hc_write emit, which decode actually runs), measure it too;
   if not, say so and note how far the unfused bench is from the engine's 19.2 us.
3. Cross-card stamps are not comparable (each card's own clock); read T2 - T1 per card as "wait for the peer" only.
4. Gate as note 219 B.0: a phase >= 5 us that a new form could remove -> B.1 design to me; else stop and report.
hc 2.42.6 module: HELD (no runs) pending the owner; the reproducer folder logs/hcpf-igc-report/ is the IGC report.

## 223. Owner: REPAIR the hc 2.42.6 bug -- a source workaround, not an IGC patch (owner, ~23:4x UTC 11-10, via coordinator radiance-test-9e)

Owner: "hc compiler bug can u repair it?" -> yes, by making the kernel avoid the code IGC 2.42.6 gets wrong.
Evidence: xe_hcpf.cpp already computes every index in `long` (no source overflow); 2.42.6's ISA has +64 mul (the
64-bit integer multiply emulation, mul / mach / macl -- Xe2 has no native 64-bit int multiply); HC1 = fine at a 1-block
grid, a wild address (page fault) at 64 blocks -> a wrong high half in an emulated 64-bit offset product.
**The fix (exact by construction):** compute the offsets in 32-bit and widen once when adding to the pointer --
`A + (size_t)(uint32_t)(m * K + k0 + kc)` style, or per-block base pointers (`A_blk = A + (size_t)m0 * K` once, then
32-bit offsets inside) -- for every global access in hc_pf_dpas_kernel<0> / <1> (A, W, injw, dout, inj, nbuf ...).
Assert at the host side that every product fits (rows x K, rows x nw, rows x hc x n < 2^31; 4,096 x 8,192 = 33.5 M);
fall back to today's form if not. The same addresses -> the same bytes; V3 used this on the expert kernel on 2.42.6.
Steps:
1. Desk: patch, build the module with BOTH compilers; the gate before any card time = the 2.42.6 ISA's 64-bit multiply
   emulation gone from the address paths (mul / mach / macl count near 2.28.4's), 0 spills. Send me the ISA diff.
2. Card (owner OK'd card tests, note 221): HC2 = hcpf-probe.sh as HC1 (64 rows, then 4,096) with the PATCHED 2.42.6
   module vs today's 2.28.4 module and the per-row reference; bytes pairwise. A fault = stop, peer probe (peer1.sh),
   report; at most TWO faulting attempts in total, then it goes back to the owner.
3. If HC2 is clean and byte-equal: an image with the patched 2.42.6 module, the B-step (7k 20k 7klp; 7klp = B199 bit for
   bit) as the run's first engine step after a peer probe, then a prefill profile. Expected ~2-4 % of a first token.
   Also confirm the patched 2.28.4 build is byte-equal (the images' module today). Default = owner, with numbers.
Keep logs/hcpf-igc-report/ as is -- the bug still goes to Intel. Slot: before B.0 (it is smaller and fully desk-gated).
- **Addendum to 223 (coordinator, ~23:5x UTC 11-10):** IGC releases/2.43.x (tags to v2.43.9, no published packages; NEO
  releases/26.44 branch exists, no release) carries "Fix LSC address scale with emulated offset" (6718b73462, 10-23,
  wrong addresses when a large immediate offset is emulated) -- NOT in 2.42.6. But our faulting kernel's 2.42.6 ISA has
  80 a64 global sends with no immediate offset and no address scale, so that fix does not visibly match; whether 2.43
  fixes HC1 is unproven. Note 223's 32-bit-offset workaround goes ahead unchanged. Watch list (note 220 item 3): when
  NEO 26.44 + IGC 2.43 are released, run the 2-second HC1 reproducer with the module built by 2.43 (one bounded step).
- **Owner decision on 223 (~23:55 UTC 11-10, via coordinator):** do the recommendation -- the 32-bit-offset workaround
  first. **IF it does not pass** (the desk ISA gate, or HC2 faults / bytes differ), build **IGC v2.43.9 from source
  TONIGHT** in the night window: start no earlier than 03:00 UTC (19:00 local time), nice 19 / ionice idle, <= 16 threads,
  under `timeout`, in /srv/bulk/ri/igc-2.43.9/ (or /srv/bulk/archive/ if /srv/bulk/ri is not writable); stop it by
  12:00 UTC if unfinished. Then compile ONLY the hc module with it (ocloc 26.40 + the 2.43.9 libigc, the module's AOT
  flags unchanged) and do the desk ISA comparison first (address math vs 2.28.4); one bounded HC3 reproducer step only
  if the ISA looks fixed, under note 221's rules. If the workaround PASSES, no IGC build (2.43 stays on the watch list).
  Booking: the coordinator adds the conditional line to ~/code/SCHEDULE.md.

## 224. Owner: "do all the recommendations" -- the rest of the decode program (owner, ~00:4x UTC 11-11, via coordinator radiance-test-9e)

Order for the runner after note 223 (HC2 / B201) and AR1 -- one card step at a time, designs to the coordinator first,
the ~$80-per-stage stop rule, defaults to the owner:
1. **New all-reduce** (note 219 B): B.1 design if AR1 shows a removable phase >= 5 us; else record and stop.
2. **GDN recurrent update redesign** (run 271 item 1 corrected): the convolution staging reworked (two q|k channels a
   thread) so VS 8 / 16 become reachable, the anchor form's shift marker per slice; target 31.6 -> <= ~12 us a call
   (36 a step). Exactness gate: hashes equal at 1 / 2 / 3 / 4 tokens AND on a rollback case, then the card row check,
   profile first, texts, M-pair, rotation. Read STATE's run 113 record (e35db12) before designing.
3. **The small exact items:** the play composer's barrier for the PLE gather (cuthost, run 272: <= 0.18 ms), and the
   step-boundary host work (221 us: profile accept / plan / enqueue first; a design only if a part >= 50 us is
   removable without a busy wait).
4. **MTP draft head v2 -- DATA PART on intel-host (the runner's):** the 11-05 head failed on (a) a domain gap (teaching /
   tools corpus vs broad traffic) and (b) depth 2-3 drift. v2 data: Flash-Next's OWN MTP-on generations from the Intel
   engine (no production traffic, no user data) on a broad public prompt mix (chat, code, tools, long-form, multilingual),
   dumping per position the 4 hc streams + the full model's next-token top-32 at depths 1-3 (the 11-04 dump format,
   extended to depth 3), ~1 M positions on /srv/bulk/ri/mtp-distill/flashnext-v2/. Design + prompt list to the
   coordinator first; card time on intel-host only (no production effect). The TRAINING part belongs to a LoRA session
   (amd-host cards 2,3, a night window, qwen27b / textsearch / vlembed paused ~30 min, restore23 after) -- the coordinator
   arranges it with the owner; the LoRA hold is lifted for this item only.
- **Addendum to 224 (owner, ~00:4x UTC 11-11): "lift the LoRA" -- the LoRA hold is LIFTED entirely** (not only for MTP v2).
  Production rules stand: each production-off block needs the owner's OK for its window. intel-host stays the runner's.

## 225. AMD-HOST BACK -- resume (coordinator radiance-test-9e, 01:58 UTC 11-11)

Maintenance done: amd-host rebooted 01:43:42 UTC with all FOUR R9700 at PCIe 5.0 x16 (03 / 23 / c3 / f3; GPU3 c3:00.0 is
back), both new T705 SSDs at PCIe 5.0 x4, bulk ONLINE; Flash-Next /health 200, qwen27b / textsearch / vlembed active and
200, assistant + database up. PAUSED removed. Resume from STATE's NEXT: item 0 first (build the kept xe_allreduce.cpp stamps
edit + the GDN bench changes; `git checkout -- src/libxe/xe_allreduce.cpp` if that build fails), then the IGC 2.43.9 build
NOT BEFORE 03:00 UTC as approved, then the rest of NEXT. The owner may create the /srv/train SSD pool meanwhile (no effect
on the runner; LoRA scratch only).

## 226. /srv/train exists -- amd-host's training scratch (owner + coordinator radiance-test-9e, ~02:0x UTC 11-11)

ZFS pool `train` = stripe of the two Crucial T705 4 TB (PCIe 5.0 x4), 7.2 T at /srv/train (lz4, recordsize 1M,
autotrim). SCRATCH ONLY: a stripe loses everything if one drive fails -- recreatable data and active job files; finished
results go to /srv/models or /srv/bulk. For note 224 item 4: copy the MTP v2 dumps from intel-host to
**amd-host:/srv/train/mtp-distill/flashnext-v2/** (not /srv/bulk); the trainer's outputs go to /srv/train/checkpoints/ and the
kept head to /srv/bulk. Nothing of the runner's own builds moves.

## 227. Owner: RUN HC3 -- the hc module built with IGC v2.43.9 (owner, ~04:0x UTC 11-11, via coordinator radiance-test-9e)

Owner YES (fault attempt 2 of 2 under note 223). `timeout 1900 bash logs/hcstep2.sh HC3 02e462b5b74c 120
/srv/bulk/ri/build-m1l0neo2640 /srv/bulk/ri/build-m1l0neo243/radiance_home/kernels/xe/libxe_hcpf.so` after KPAR1c (one card
step at a time). Rules as HC2: 64 rows then 4,096 only if clean and byte-equal; stop at once on a non-zero rc; a fault =
STOP for good on this line (no attempt 3), peer probe (peer1.sh), report with the klog. If HC3 is CLEAN and byte-equal at
both sizes: one image with the 2.43.9 module, a peer probe, then the B-step (7k 20k 7klp; 7klp = B199 bit for bit; times vs
B201's 3.14 / 6.41) -- and a proposal (desk, to the coordinator) for an A/B of the WHOLE engine built with IGC 2.43.9 vs
2.42.6 (the loader-test pattern: image, PR, B + M-pair, exact gates).

## 228. Owner: MTP v2 is THE priority ("give me the 10 %"); VS 2 split + L1 to default if the rotation agrees; VS 8 stays OPEN (owner, ~04:2x UTC 11-11, via coordinator radiance-test-9e)

1. **MTP draft head v2 first on the cards.** Order: KPAR2 + the copy -> the LoRA side's CPU unroll check against
   parity2.bin (coordinator arranges) -> the generation client tested on 3 documents (design / script to the coordinator)
   -> bulk generation in parts -> the dumps in parts to /srv/train/mtp-distill/flashnext-v2/. Training on amd-host follows
   (the coordinator books the window with the owner). Then rad-convert --reuse + the engine test on intel-host (>= +1.5 %
   tokens/step-driven tok/s, texts as usual for MTP: the head changes drafts, not the target model's outputs).
2. **VS 2 split + L1 (exact) -> DEFAULT if a same-image rotation shows a gain:** fold L1 (done), then one image, a 10-min
   rotation pair on it (the split + L1 via -gs2- vs plain, H-step pattern), texts unchanged. Gain on median / mean /
   aggregate -> default ON (RADIANCE_XE_GDN_SPLIT=2 by default in the Intel builds; =0 opts out), one revertable commit.
   Slot it where the cards are free between MTP v2 steps (it must not delay generation).
3. **VS 8 slice-group: NOT closed.** The coordinator writes a VS 8 v2 design (briefs/2026-11-11-gdn-vs8-v2.md): the
   tail replaced by a co-resident per-key-head barrier + parallel factor writes, the per-token stores batched, and
   stage 2's shared q/k norm. It is built and benched AFTER the MTP v2 data is dumped (lower priority than item 1).

## 229. Coordinator: next run's first items (coordinator radiance-test-9e, ~04:4x UTC 11-11)

1. **UPDATE (coordinator, ~04:5x): run 283 added a `-gs2` SUFFIX to m2-heat.sh instead (line 33 strips it before the
   exact-name case; line 329 adds RADIANCE_XE_GDN_SPLIT=2 to the engine env) -- reviewed: `bash -n` clean, placement
   correct. So the pair is `timeout 3000 bash logs/hstep.sh H155 e6fd2ffe7f01 bestrxgrf3w16p4 10` vs `... H156
   e6fd2ffe7f01 bestrxgrf3w16p4-gs2 10`; confirm in H156's serve.log that GDN_SPLIT=2 is set (the stripped name is
   logged as the plain config). The BX 29 form below is NOT needed.**
   (superseded:) The rotation pair for note 228 item 2 (VS 2 split + L1 -> default if it agrees), pre-approved in this form: add the
   name `bestrxgrf3w16p4gs2` to m2-heat.sh's config arm (line ~153) AND its mapping beside BX 28's (BX 29 = BX 28 + the
   environment `RADIANCE_XE_GDN_SPLIT=2`) AND BX 29 in the known-bundle list -- the three places H151 taught us; `bash -n`;
   then `timeout 3000 bash logs/hstep.sh H155 e6fd2ffe7f01 bestrxgrf3w16p4 10` and `... H156 e6fd2ffe7f01
   bestrxgrf3w16p4gs2 10` (same image 308, back to back). Gain on median / mean / aggregate and texts unchanged -> the
   default flip (RADIANCE_XE_GDN_SPLIT=2 by default in the Intel builds, =0 opts out), one revertable commit; else record.
   Fit it in a card gap; MTP v2 generation goes first if both are ready.
2. **MTP v2:** the generation scripts (sample / gen in-container at 3 users / render + role.bin / genstep parts) -> to the
   coordinator -> GEN0 (3 documents). KPARD (the parity corpus's hidden-state dump) ONLY if the LoRA side's CPU check asks
   for it -- the coordinator writes that here when it does.

## 230. Owner: try the VS 4 HYBRID slice-group before VS 8 v2 (owner, ~05:1x UTC 11-11, via coordinator radiance-test-9e)

The slice-group form (ru_sg_update_kernel: one workgroup = a key head's 3 value heads) at **VS 4**: 4 slices of 32 rows
per value head -> 8 key heads x 4 = **32 workgroups** (one per Xe-core), 3 x 32 rows x KT 8 = 768 threads a workgroup;
the q|k conv runs 4x per key head (VS 2 split: 6x; VS 8: 8x).
**Why it can win where VS 8 lost (coordinator):** the factor block's fixed 32-row groups make every byte belong to exactly
ONE workgroup at VS <= 4 (libr4d's own rule, r4d_gdn_recurrent_update...hip l.136: "a head's v rows may be split over up to
four workgroups ... without a cross-workgroup barrier"). VS 8 broke it (two slices per group) -> the last-arriver tail
(7 us of KG17d2's excess). At VS 4 each slice replays from and rewrites ONLY its own group's k copy, v rows and (g, bt):
write right after its own loop, no shared-factor tail. The last arriver keeps only the marker stamp and the q|k conv-state
rewrite (cheap). Also batch the per-token stores (VS 8 v2 item B).
Risks (experiment, not a promise): one 768-thread workgroup per core (occupancy), SLM per workgroup, registers (HK 16
as VS 8: no change per thread). Check zeinfo: GRF 256, 0 spills, 0 private memory.
Gates as always: hashes = SPLIT 0 at T 1 / T 4, both forms, anchor-seq x20 (+ x500 stress, since writes move); bench at
5000 launches one T a process vs the VS 2 split + L1 (20.5 us, T 4) and the fused kernel; >= 10 % better than the split
-> engine M-pair + prof -> rotation -> owner. Order: AFTER MTP v2's generation + dumps are under way (card gaps), BEFORE
VS 8 v2. Lift the `VS == 8 || VS == 16` static_assert for VS 4 in the slice-group kernel only (the split keeps VS 2 / 4).

## 231. Owner YES: one microbenchmark of two new format C decode forms (owner, ~05:3x UTC 11-11, via coordinator radiance-test-9e)

In m3/... xe_fc_tile_probe (the E-form harness: today's kernel through libxe and the new form on the same inputs, every
bf16 output + canaries compared, then timed), gate_up first, at 4 tokens / 27 experts and 1 token / 10 experts:
- **E4 = select-tree decode (option A):** the 16 int8 levels held in registers; each 4-bit code -> its level by 4
  rounds of compare / select on the code's bits (no gather, no SLM, no a0-indirect move); ordinary SYCL (not ESIMD);
  today's dp4a order and partial-sum order kept -> bit-exact by construction. Report instructions a dp4a (dump) too.
- **E5 = expanded int8 weights (option B):** the same experts' codes pre-expanded to their int8 levels (2x the bytes),
  read straight into dp4a in today's order -> bit-exact. Report the achieved GB/s: to TIE today it needs ~410 GB/s
  (44.8 MB in 109 us), to win by 15 % ~480 GB/s -- our best measured int8 streaming is ~343-370. VRAM: E5 for hot
  experts only would cost residency (misses stream over intel-host's PCIe 4.0 x8); say what that would mean before any
  engine design.
Gate: bytes equal to today's kernel AND >= 1.3x faster on gate_up at 4 tokens -> a design note to the coordinator;
otherwise record and stop (C: keep today's kernel). Desk ~$15-20, one ~5-min card step (estep.sh form, card1, no engine).
ORDER: after MTP v2 generation + dumps and the VS 4 hybrid; card gaps only.
- **Addendum to 231 (owner-shared review + coordinator, ~06:0x UTC 11-11):** E4 is a DECODER-ONLY swap inside TODAY'S
  gate_up kernel (same work-group size, tile, loads, dp4a and accumulation order; ESIMD's E3 changed the work split AND
  the decoder at once, so its loss can't be attributed). Two decoder forms, both exact (same int8 values):
  E4a = select tree: the 16 levels as uniform registers, 15 vector selects driven by the code's 4 bits (one select
  instruction serves every lane, so ~1 instruction a code at SIMD16+). E4b = packed shift: the 16 int8 levels packed into
  4 dwords; per lane pick the dword by code>>2 (3 selects), shift right by (code&3)*8 (per-lane shift), sign-extend the
  byte (~5-6 vector instructions per 16+ codes). Interleave 2-4 independent decode / dp4a chains. Desk gate before any
  card step: the ISA dump shows NO r[a0] indirect access and no new SLM gather; report instructions per dp4a and
  spills. Bench at 5000 launches (steady clock, as KG17d2 taught), 4 tok / 27 experts and 1 tok / 10. Report against
  today 73.7 us: any repeatable gain below it is worth recording; >= 1.3x -> design note; Amdahl at 19 % of the step:
  1.5x -> ~+6.8 % decode, 2x -> ~+10.5 %.

## 232. Coordinator (owner-shared hc review): a DESK-ONLY hc fusion audit, last in the queue (~06:1x UTC 11-11)

For each decode hc producer -> consumer pair (hc_v2_gu, hc_v2_down_rows, hc_v2_write_norm, hc_quant_rows and their
neighbours in M373's edge table): what the consumer needs from the producer (elementwise / a whole-row reduction / a
GEMV input), whether that dependency can be satisfied inside the producing workgroup (no cross-workgroup hand-off -- the
N1 lesson: a fence + last-arriver fold cost +10 %), and a saving estimate from the measured launch counts and us a launch.
First candidate to look at: hc_quant_rows (0.36 ms a step, 3.7 us x 97) folded into its producer if its max / scale can
be formed within one workgroup. Output: a short table in briefs/; no build, no card time; ~$5-10. Only pairs that fit in
one workgroup become build candidates (each exact, with the usual gates). Order: after notes 228-231.
- **Addendum to 232 (owner-shared GEMV review, ~06:3x UTC 11-11):** extend the same desk audit to the remaining ~145
  standalone gemv_i8a8 launches a step (M398): for each, list its input tensor, its pass/layer position and its consumer;
  flag groups that read the SAME activation at the same point with no dependency between them (pairs 4/5's pattern:
  gemv_i8a8_x3 / gemv_i8a8_bf16_pair). Any group found = a build candidate with pairs 4/5's gates (row test bytes,
  card row check, profile first, M-pair, rotation). Note for the record: the engine's tapes already are the "command
  graph" (recorded Level Zero lists replayed each step), so a SYCL Graph port is not proposed.

## 233. MTP v2 end to end WITHOUT the coordinator session (owner: "you can run everything"; coordinator, ~06:2x UTC 11-11)

The coordinator session may disappear (the owner's laptop sleeping ends it). Everything below must run from systemd /
the runner alone. Generation of parts 2-5 runs in user units mtpv2-gen-a / mtpv2-gen-b (amd-host production Flash-Next) ->
~/code/radiance-intel/logs/mtp-v2/amd-host/docs-partN.jsonl; `parts-done.txt` there gets "part N done" lines.
1. **Runner, per part (1-5):** when its docs file is complete (part 1 = your GENP1; parts 2-5 = a "part N done" line),
   render -> dump on image 306 -> dump_check -> copy to /srv/train/mtp-distill/flashnext-v2/partN/, then write the
   marker file /srv/train/mtp-distill/flashnext-v2/partN/DONE (one line: positions, generated share, check PASS). Report
   the generated share per part in STATE.
2. **Training starts by itself** (a systemd watcher the coordinator installs; if the coordinator vanished before installing
   it, there will be no file ~/code/lora-stack/mtp/WATCHER-INSTALLED -- then write "QUESTION FOR COORDINATOR/OWNER:
   MTP v2 training not started" at the top of WAITING and go on with your other work). It stops qwen27b / textsearch /
   vlembed, trains variants A and B on amd-host cards 2 and 3, restores the services, and writes
   /srv/train/checkpoints/mtp-v2-TRAINED (the two checkpoint paths + logs). NOT your job to start or stop those services.
3. **Runner, after mtp-v2-TRAINED exists:** run the offline eval (~/code/lora-stack/mtp/eval_mtp_v2.py, CPU, its
   README_v2) on both variants against the current head; pick the better at depths 2-3 with depth 1 not worse. If NEITHER
   beats the current head offline: stop, record, WAITING. Else: rad-convert --reuse of the production .rad with the
   winner's mtp.* tensors (the 11-05 recipe: mtp_distill_copy / the merge tools kept from the closed MTP distill),
   copy to intel-host, then an M-pair (new head vs today's, texts = M285 -- MTP never changes the target text) and the
   10-min rotation pair. REPORT per-position acceptance (depth 1 / 2 / 3), tokens per step, AND tool-call requests
   separately (v1 lost there: 0.761 -> 0.641). Adopt line: >= +1.5 % tok/s with no depth's acceptance worse; adoption
   on the Intel images is an owner default (WAITING); production AMD is a separate owner decision. If the winner fails
   in the engine, test the other variant once before concluding.
4. Card gaps between dumps as already agreed (M402 / M403, KG19, E4 / E5). Dumps first.

## 234. Owner: GEMV groups A + B DEFAULT ON; coordinator answers to runs 285-286 (~11:0x UTC 11-11, coordinator radiance-test-9e)

1. **Owner: both groups default ON.** RADIANCE_XE_GEMV_MULTI2 defaults to 1 (A + B) in the Intel builds; =0 opts out,
   =a / =b keep the single groups. Evidence: GSH1 byte-identical (M 1/4/8/12, both groups), M404 / M405 91.06 / 91.01 vs
   M402 88.99 (+2.3 %), M406 profile (gemv_i8a8 144.5 -> 96.3 a step, device 28.2 -> 27.5 ms), H157 vs H158 +2.0 % on
   median / mean / aggregate, texts 11 / 11. One revertable commit; then a flagless M-step + -prof twin on the new image
   = the new decode reference (expected ~91). Service untouched.
2. **ar + write + norm one-launch design (briefs/2026-11-11-ar-write-norm-design.md): SHELVED.** Net ~0.2-0.4 ms
   (0.7-1.5 %) for ~$120 and the all-reduce's correctness surface (re-blocking the LL form); recorded as an option for the
   owner, not built or benched.
3. **The 221 us step boundary (run 286): accepted as device / driver-side** (the next pass is already issued ahead; no host
   part >= 50 us). The ~$20 span diagnostic is NOT run (prize <= 0.8 % and probably not fixable from our side). Recorded.
4. MTP v2: closed with "no winner" (KL down, top-1 flat; depth 4 already no gain on 11-03). Nothing further queued.
After item 1 the decode queue is EMPTY of exact items with an estimate >= 1 %. Next run: item 1, then idle-skip until a
new owner note (no busy work).

## 235. Owner: queue the GDN half-width STATE accuracy test (non-exact; owner, ~11:1x UTC 11-11, via coordinator radiance-test-9e)

The GDN token loop is bound by the fp32 state rows' traffic (KG20). The kernel already supports an fp16 state cache
(SDT=1 in r4d_gdn_recurrent_update...: storage fp16, all arithmetic fp32; the libr4d comment: at 8 sequences it saves
about a third of state traffic). Use THAT (fp16), not bf16 -- no new kernel.
1. Desk: find how the engine chooses the state cache dtype; if there is no switch, add an opt-in one
   (RADIANCE_GDN_STATE_F16=1) that allocates the recurrent state cache in fp16 for BOTH the prefill chunk scan
   (r4d_gdn_chunk_scan: check its h0 / ht support f16, or convert at its boundary) and the decode update. Check the
   state's value range on a real dump first (fp16 max 65504; report max |h| per layer) -- if anything is near the range,
   stop and report (that would be the bf16 question, a new kernel).
2. Accuracy FIRST (it's cheap): note 149's KL replay on both corpora (KMB1 agreement + KMB2 long) against today's fp32
   default, the bar as lever #3's (mean KL well inside 0.026-0.042 / top-1 >= 96.2 %, ideally lever #3's 0.002 / 98.7 %),
   plus long-context drift: KL by position bucket (the state accumulates error with length -- report 0-2K / 2-8K /
   8-20K separately). Owner decision before any speed work if the KL is outside lever #3's class.
3. Speed only after (2) passes: the GDN bench at 5000 launches (T 1 / T 4) and an M-pair vs the flagless reference, then
   the rotation. Expected: GDN ~1/3 faster at best, ~+1 % decode. Default = owner, with the KL numbers.
Order: after note 234 item 1 (the GEMV default + new reference). Budget ~$30-40 for steps 1-2.

## 236. Owner YES: fix the multi-user + tool-call STALL (reliability; owner, ~12:5x UTC 11-11, via coordinator radiance-test-9e)

GEN0 (run 284): 3 users in flight, one of them a tool-call request (the grammar path), headroom 2048 -> serve.log:
"l0 tape 50: appending 1 list(s) failed out of device memory (0x70000003); draining ... retrying (attempt 1 of 3; owner
note 58)" on device 0, the same for tape 49 on device 1, then "rank barrier: a thread has waited 10 s at the step barrier
(generation 375, 1 of 3 arrived)" and a play wait on device 1 -- the engine hung, no fault, and did not exit on the kill.
hr4g (headroom 4096) avoids it (GEN1), but the default must not hang. After note 235's accuracy step (one card step at
a time):
1. **Reproducer first** (one bounded step, the step rules, a host timeout that KILLS the container): the GEN0 setting at
   headroom 2048, 3 in flight, the Glaive document + two others, repeated until it stalls or 10 tries pass; capture the
   serve.log around the first append failure on BOTH ranks.
2. **Desk diagnosis:** why the two ranks leave step after the out-of-memory retry (one rank retries / re-records while the
   other proceeds? the drain path skipping the barrier? a tape slot left half-appended?). Also why the grammar request
   needs the extra VRAM (its mask buffers, a separate tape shape).
3. **The fix, exact:** the out-of-memory path must keep the ranks in lock-step and degrade, never hang -- e.g. both ranks
   agree to drop to an untaped (eager) submission for that step, or the request fails cleanly with an error to the client;
   and the stall should end with an error, not an indefinite wait (bounded barrier wait -> abort the step). Prefer
   preventing the OOM (reserve the grammar path's buffers up front) if the diagnosis shows that is the cause.
4. Gates: the reproducer passes (>= 10 runs, no hang) at headroom 2048; texts = M285 on the M-pair; one rotation with a
   tool mix shows no regression; klog clean. Design (step 3) to the coordinator before building. Budget ~$40.
