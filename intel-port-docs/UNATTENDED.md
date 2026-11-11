# UNATTENDED.md: how to run PROMPT.md with nobody watching (milestone 3)

You are Claude Code on **amd-host** (user `user`), started by `run.sh` from the user timer
`radiance-intel.timer` (every 10 min, gated on subscription auth, the budget and daily cap, amd-host's
load, memory, disk and production health). The owner is away for days. Nobody will answer a
question. Read this file with `PROMPT.md` at the start of every run. Where the two disagree, this
file wins.

**`OWNER-NOTES.md` comes first.** Read it at the start of every run, before this file and PROMPT.md; the
owner adds standing notes there between runs, and it overrides both. You cannot edit it.

Earlier work, read-only for you:
- `round1/` (the probe) and `round2/` (measure and tune): their `PROMPT.md`, `STATE.md`, `runs/`,
  `results/`, run logs. **Round 2's `results2/` moved to `round2/results2/`** (so PROMPT.md's
  `results2/m7b/` is `round2/results2/m7b/`). Paths in `round2/STATE.md` given as `results2/…` are
  under `round2/results2/…`.
- `results2/heat/` stays where it is: a read-only sampler of production writes it until 2026-10-25
  06:45 UTC, then `results2/heat/report.md` appears. Never touch that folder or its timer.
- Working files still in place to reuse: `probe/`, `probe2/`, `ref/`, `images/`, `logs/`,
  `build-cpu/`, `out/`, `out2/`, `src/`.

## How a run works

- A run can be stopped at any moment: the 12 h cap, SIGTERM, a usage limit, its own **$80 budget**
  (`--max-budget-usd 80`), or the daily cap at the start of a cycle. So work in **small steps that
  can be restarted**: one build, one op, one kernel, one test set. Write results to files under
  `~/code/radiance-intel/` as you go. **Commit in `src/` after every step that passes** (see Git).
- After **every** step, update `~/code/radiance-intel/STATE.md`:
  - `Session ID: <uuid>` (given in the first prompt; keep this line exactly);
  - the PROMPT.md step and sub-step;
  - done, with the file and commit for each result;
  - next: the exact next step, specific enough to resume cold;
  - the op table (op, lines, CPU pass, card1 pass), test results, line counts;
  - open problems, failure counts per check, canary results, GPU faults.
- **The block at the top of STATE.md between `<!-- runner:begin` and `<!-- runner:end -->` belongs
  to `run.sh`** (usage against the budget and today's cap, GPU faults). Leave it as it is.
- Later runs get only "continue from STATE.md". Trust STATE.md, the files and `git log` over your
  memory of the conversation. If a step was in progress, check its outputs and redo it if incomplete.
- End a run cleanly at a waiting point or after about 2.5 h. Update STATE.md first.

## Budget and daily cap

- Usage is the CLI's estimate at API list prices (on the owner's Max plan it stands in for plan usage,
  not a bill): for each session the largest `total_cost_usd`, summed. The runner block shows it.
- **Milestone budget: $1,000.** At $1,000 `run.sh` pauses. **Daily cap: lifted by the owner (daily-cap files) for this milestone; the default is $80 per calendar day in
  America/New_York**: once a day reaches it, `run.sh` skips cycles until midnight ET (a wait, not a
  pause). A run already going can overshoot the cap by up to its $80.
- **By $700**, judge whether the budget reaches milestone 3's targets (PROMPT.md). If not, say so in STATE.md
  and in a notification, and finish the step you are on cleanly. Prefer a correct, committed, tested smaller
  result over a large untested one.
- A plan usage limit is not a failure; `run.sh` retries.

## Git (in `src/` only)

- `src/` is the clone the audit read (`main` @ `9eaf10d`, Codeberg `origin`). Create the local branch
  `intel-l0` from `9eaf10d` and work only there.
- Use `git -C src …` (or `git -C ~/code/radiance-intel/src …`); `cd src && git …` is refused without
  approval, and git anywhere else is not allowed.
- **Never push, never add, rename or change remotes, never change git config's remote or URL
  settings** (all denied). Nothing leaves amd-host.
- Commit after every step that passes its tests; the message says what passed and how it was tested.
  **Never commit a red build.** Never rewrite history that is committed.
- `~/code/radiance-src` is earlier work (fix-98): read it if useful, never write to it.
- Don't change AMD code paths beyond the minimum build-system and backend-table hooks. The CPU build
  (`./build.sh --no-hip`, 86 of 88 tests; the 2 failures are FFmpeg-only) must stay as it is. Check it
  after any change outside your new files. A HIP build needs ROCm, which amd-host's host lacks: build it in
  a container (round 1's `localhost/ri-hostbuild:1` has no ROCm; a ROCm image is `docker.io/rocm/dev-ubuntu-24.04`
  by digest, already on amd-host) and record what you checked.
- License: Radiance's files keep their Apache-2.0 notices. New files get the same Apache-2.0 header
  plus one line saying they are part of the Intel port. Nothing is published.

## Resources, and staying out of everyone's way

- amd-host runs production (`radiance`, Flash-Next on GPUs 0,1), qwen27b, text search and vlembed (GPUs
  2,3), the another OCR project / other-project work and many timers. **You use none of the GPUs on amd-host.**
- Stay inside the service's limits: 16 CPU threads, 32 GiB RAM, `nice -n 19`, `-j16` or less. Containers
  and builds escape the service's cgroup, so cap them yourself:
  - every local `podman run`: `--rm --name ri-... --cpus 16 --memory 28g`, `timeout` inside;
  - every `podman build`: `--cpu-quota 800000 --memory 20g --label ri-probe=1`, `-j8` or less.
  (`--cpuset-cpus` does not work on amd-host.) Keep `MemAvailable` above 40 GiB.
- Wrap anything long in `timeout`, locally and on intel-host. You cannot kill processes.
- **Off-limits:** `~/code/other-project/`, the another OCR project, `/srv/other*`, the `pipeline` database;
  production and its units (`radiance`, `flashnext`, `qwen27b`, `~/.config/containers/systemd`,
  `~/code/radiance-test/prod`, `engine.sh`, the watchdog, `app-ctl`, `/etc/notify.conf`);
  `/etc`; other sessions and memory files; `run.sh`, `intel-host-check.sh`, `prod-api.sh`, the unit files,
  `.claude/settings.json`, `runs/`, `budget.txt`, `intel-host-engine-guard.sh` (all denied; you may run prod-api.sh and the guard).
- Commands run headless against `.claude/settings.json`; anything not allowed is denied with no prompt.
  Round 1–2 lessons: no `dri` substring in a local `podman run`; plain `cp` (no flags) or tar pipes;
  copy files back with `scp intel-host:/mnt/fast-storage/models/test/radiance-intel/… …`; never put the name
  of intel-host's card0 service in a command.
- Writes outside `~/code/radiance-intel/` are allowed only for the report files
  `/srv/models/test/reports/<date>-radiance-intel-m3.md` and `.json`.
- The current `app_ai_system_reference.md` is `~/apps/mission-control/app_ai_system_reference.md`
  (2026-10-16). Read it; never edit it. Rounds 1–2 already listed contradictions; add only new ones.

## Images and disk (`~/code` has ~83 GB free; the floor is 50 GB)

- Build environment: a container with **Intel LLVM v7.2.0** and the Level Zero headers, pinned by
  digest (round 2's `localhost/ri-probe:6` / `ri-tune:*` images and `images/` show how). Record
  compiler versions. **AOT-compile for the B70** (`-fsycl-targets=intel_gpu_bmg_g31`) so nothing
  JIT-compiles on intel-host.
- Two stages for runtime images; `LABEL ri-probe=1` right after every `FROM`; tags
  `localhost/ri-<name>:<n>`. Clean up as soon as a runtime image passes on amd-host:
  `podman rmi localhost/ri-<tag>` (one per command), `podman image prune -f --filter label=ri-probe=1`,
  `podman rmi` on Intel base images you pulled. Never `-f`, never a bare prune.
- **Never write an image tar to disk:** `podman save localhost/ri-<name>:<n> | ssh intel-host podman load`.
  Check `df -B1 ~/code` before each pull or build; pause if it would go under 50 GB.

## The production `.rad` copy (PROMPT.md step 7, only after step 6 passes)

- Source: `/srv/models/llm/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad`,
  121,969,901,568 bytes. Its sha256 is in `SHA256SUMS` beside it:
  `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`. **Do not hash amd-host's file**
  (that is a second full read); hash only the copy on intel-host and compare.
- Read it once, gently: `ionice -c3 nice -n 19 rsync --bwlimit=50000 --partial --inplace <src>
  intel-host:/mnt/fast-storage/models/test/radiance-intel/` (50 MB/s; the link measured 67 MB/s over
  Tailscale, about 40 min at this cap). `--partial` lets a stopped run resume the copy. Run it under
  `timeout` in pieces if needed; rsync picks up where it stopped.
- While it runs, check every few minutes: production's `/health` OK, amd-host's load, and intel-host's
  **Tctl: pause the copy (let `timeout` end it) above 80 °C**; the 85 °C pause rule still applies.
- Then `sha256sum` on intel-host (one read of the copy) and compare with the value above.
- `/mnt/fast-storage` had 506 GB free; the copy needs 114 GiB.

## Locked memory on intel-host (needed for step 7's real-model load)

- As of 2026-10-24 intel-host gives SSH sessions and rootless containers **8 MiB of locked memory**
  (`ulimit -l` 8192; `--ulimit memlock=-1:-1` cannot raise it). One card's run pins about 90 GiB.
- The owner has the root fix. Before the real-model load, check
  `ssh intel-host 'ulimit -Hl'`. If it is not `unlimited`, do everything in step 7 that does not need the
  load (the copy, the hash, the config), then create `PAUSED` ("locked memory on intel-host is 8 MiB") with
  these commands for the owner, and end the run:

      echo 'user - memlock unlimited' | sudo tee /etc/security/limits.d/90-user-memlock.conf
      # then, from amd-host (a new SSH login picks it up; nothing restarts):
      ssh intel-host 'ulimit -Hl; ulimit -Sl'     # both: unlimited
- When it is unlimited, run containers with `--ulimit memlock=-1:-1`.

## Memory guard on intel-host (every engine load; owner 2026-10-24)

While it still runs, chatterbox-vllm shares intel-host's RAM (121 GiB; MemAvailable was 110 GiB on
2026-10-24). Your engine must never squeeze it. (Job B retires it during this milestone; the guard's
8 GiB floor stays either way.)

- **Name every engine container `ri-m3-<something>`** (e.g. `ri-m3-engine`). Only `ri-m1-*`, `ri-m2-*` and
  `ri-m3-*` containers can be stopped by the tools below; you cannot stop anything else.
- **Before loading the real model:** work out the pinned host memory your config plans, from the
  engine's own plan (`--debug-placement`, or the startup budget lines: host pool for the experts that do
  not fit VRAM, the embedding, the n-gram table if `--ngram-placement ram`, the snapshot slots, the KV
  host tier `--prefix-cache-mib`, the arena). Prefer **`--ngram-placement disk`**, as
  `flashnext-tp1.yaml` does: it saves the 47.7 GiB pinned copy of the n-gram table. Write the figure in
  STATE.md. Then:

      ./intel-host-engine-guard.sh --precheck <planned GiB>

  It passes only if intel-host's MemAvailable exceeds the planned figure **plus 16 GiB**. If it refuses,
  shrink the plan (smaller host tier, fewer snapshot slots, disk n-gram) or end the run and try later.
  Never load without a passing precheck in the same run.
- **During the load and the whole run:** start the guard **before** the engine, in the background, and
  keep it running until the engine container is gone:

      timeout 4h ./intel-host-engine-guard.sh ri-m3-engine 600 > logs/guard-<step>.txt 2>&1 &

  It waits up to 600 s for the container, then reads intel-host's MemAvailable every second. **Under 8 GiB it
  stops the container at once** (podman stop, 2 s grace), logs it to `logs/memguard.log` and notifies.
  If the guard reports `STOPPED`, record it in STATE.md (what was loading, the lowest MemAvailable),
  don't restart with the same plan, shrink it first, and count it like a failed correctness check (twice
  → pause).
- **To stop your engine yourself at any moment** (e.g. a wrong config or a hang):

      ./intel-host-engine-guard.sh --stop ri-m3-engine

- Run the engine container with `--rm`, a `timeout`, `--ulimit memlock=-1:-1` (once intel-host's limit is
  unlimited), the `--cpus` cap from OWNER-NOTES (notes 3, 4, 7: 0.3 for engine steps, 1 or 2 with the guard at `--hot 75`
  where a note allows it), the card's nodes only (card1; both cards only after the card0 gate), and the
  1 Hz Tctl sampler beside it.

## Production's API (agreement checks only)

- **Only through `./prod-api.sh REQUEST.json OUT.json`.** You cannot read the key, and direct `curl`
  to amd-host's port 8000 (other than `/health`) is denied. The helper enforces the milestone's rules: at
  most **30 requests** in total, temperature 0, no streaming, `max_tokens` ≤ 256, and amd-host's 5-minute
  load under 16 (otherwise it refuses; try a later cycle). REQUEST.json is
  `{"endpoint": "chat", "messages": [...], "max_tokens": 256}` or
  `{"endpoint": "completions", "prompt": "...", "max_tokens": 256}` (optional: `seed`, `logprobs`,
  `top_logprobs`, `stop`, `tools`, `tool_choice`). Plan the 30 before sending any; a refused request is
  not counted, a sent one is counted whatever the answer.

## intel-host: card1 pre-approved; card0 only after the gate

Each card1 step may start only when **all** of these hold, checked in this run:

1. `intel-host-precheck.txt` is **less than 30 minutes old** (`time_epoch` against `date +%s`) and shows
   `card1_pci: 0000:0f:00.0` and `card1_clients: 0`. The service line is `active` until job B retires
   the card0 service; after that (OWNER-NOTES says when) `inactive` is the expected state, not a reason
   to stop. Otherwise end the run.
2. Your own read-only check agrees (round 1's `ri-check.sh` in the intel-host folder): no process holds
   card1, and no container maps card1's nodes except, while it still runs, the card0 service's
   (owner-accepted, `ACCEPT-card0-maps-all-dri`), holding only `0000:0b:00.0` handles.
3. intel-host's 5-minute load is under 4 (the `.rad` copy and model loading excepted, under the Tctl rule),
   and intel-host has the memory and disk the step needs.
4. Every GPU fault the runner has logged since your last card1 step ("GPU fault on intel-host" in
   `logs/run.log`) is listed under "Reviewed GPU faults" in STATE.md (see "GPU faults and canaries").

If card1 is busy, write `waiting for card1: <what you saw>` in STATE.md and end the run.

**The card0 gate (stage 2's two-card steps).** A step that touches card0 (`0000:0b:00.0`: its nodes
`/dev/dri/card0`, `/dev/dri/renderD128`, peer exchanges, a TP2 load) may start only when the same fresh
`intel-host-precheck.txt` also shows `card0_clients: 0` **and** `dri_containers: none` (no running container
maps `/dev/dri`). Until then, write `waiting for card0: <the card0_clients and dri_containers lines>` in
STATE.md and go on with the single-card steps; never stop or touch whatever holds card0.

On intel-host:
- Every file in `/mnt/fast-storage/models/test/radiance-intel/`; write nowhere else.
- **No builds and no CPU-heavy work** (heatsink pending). The `.rad` copy and model loading are the
  only sustained work, both under the Tctl rule.
- **card1, and card0 only after the gate above.** Before the gate: nothing on card0, no buffers, no
  peer exchanges. Never start, stop or change any service, timer, cron job or container you did not
  start, including the ones the owner turned off or retired.
- Containers: `podman run --rm --name ri-... --cpus 2 --runtime crun --group-add keep-groups`, images
  by digest, `timeout` inside, card1's nodes only (`/dev/dri/card1`, `/dev/dri/renderD129`; one Level
  Zero device, index 0 in the container; check it is `0000:0f:00.0` by PCI/UUID). The real-model
  server may need more CPUs while loading: `--cpus 4` at most, and say so in STATE.md.
- Keep round 1's 1 Hz sampler beside long card1 runs (the load, the decode and replay runs), and pause
  above 85 °C as before.

## GPU faults and canaries

- Every output buffer of every new kernel gets canary padding on both sides (≥ 4 KiB, a known
  pattern), checked after every run. **A broken canary counts as a GPU fault.**
- New non-matrix kernels run on amd-host's CPU SYCL device first, small; every new kernel runs on card1
  small before full size.
- `run.sh` checks intel-host's kernel log after every run (card1 `0000:0f:00.0`, card0 `0000:0b:00.0`).
  It keys each fault on its dump id (or the time of its first kernel-log line) and PCI address, logs it
  as "GPU fault on intel-host ... key K, signature S", notifies every new one, and never counts a fault
  twice or one listed under "Reviewed GPU faults". The runner block shows the number of distinct
  card1 faults.
- You check too after every card1 step: `ssh intel-host journalctl -k --since @<step start>` (page fault,
  CAT error, VM worker error, engine reset, GT reset, timed-out job, wedged, devcoredump) plus your
  canaries.
- **On a card1 fault or a broken canary, don't pause.** Work through it:
  1. Stop that step. Record the kernel lines and the dump: intel-host's udev rule saves every card1
     devcoredump to `/mnt/fast-storage/models/test/radiance-intel/coredumps/<UTC>-devcdN.txt`.
  2. Diagnose from the dump (copy it to amd-host; it gives the faulting address and the engine) and the
     lines. Then add the fault under "## Reviewed GPU faults" in STATE.md, one line each, using the
     runner's key and signature from `logs/run.log`:
     `reviewed-fault: <key> | <signature> | <one-line diagnosis>`
  3. Fix it and verify the fix on amd-host's CPU device. When you commit a fix, add:
     `fix-claimed: <signature> | <commit> | <UTC time, e.g. 2026-10-26T09:00:00Z>`
  4. Retest on card1 at a small size first (canaries, kernel log), then full size.
- **Pause only when** (both cards follow the same rules since 2026-10-26; the card0 service is retired):
  - a card reports wedged;
  - a card hasn't recovered 10 minutes after a fault: the device is missing, or the kernel log isn't
    clean;
  - the same fault signature comes back after a `fix-claimed:` line for it;
  - two faults happen in one run.
  `run.sh` checks the same four conditions after every run and pauses on them itself.
- **When the owner removes `PAUSED`,** whether between runs or during one, don't create it again for
  anything that happened before the removal.

## Decide instead of asking

**When you would ask the owner to choose, choose yourself:** take the cheapest path that still
produces a report or a verified fix. Write the choice and the alternative in STATE.md,
`decision: <what you chose> | alternative: <what you didn't> | why`, then go on.
This covers PROMPT.md's "stop and ask", an assumption in PROMPT.md that turns out wrong, and a
correctness check that fails twice: record it, pick the cheapest path, continue.

**Pause only for what can't be undone, or what you can't do:** write `PAUSED` (first line: the reason
in one sentence; then details, and the exact commands if root is needed), update STATE.md, notify, end
the run. The owner resumes with `rm PAUSED`. Pause when:
- a step would touch production, the OCR work, card0 or its service, or the services the owner
  turned off;
- root is needed;
- you'd delete data you didn't create in this milestone;
- the work would spend past the budget;
- the GPU fault conditions above apply (a card wedged or not recovered, a recurrence after a claimed
  fix, two faults in one run on either card);
- `intel-host-postcheck.txt` shows the card0 service `active` at your run's precheck and not `active` after it
  (once job B has retired it, `inactive` both times is normal);
- free disk is under 50 GB on amd-host or intel-host;
- a permission is denied and the work can't go on without it.

## Notifications

`notify "radiance-intel: <message>"`: one short line, no secrets. `run.sh` sends: milestone
start, every GPU fault, a short line after every run, pauses for auth, budget, faults and 3 failed
runs, the daily cap, plan usage limits, 6 h of skipped cycles, and DONE. **You** send: stage 1 start; each kernel that cuts the
step by ≥ 5 ms; stage 1 done, with tok/s; the M2b result; the first two-card output, with tok/s; any
fault; a budget warning by $350 if stage 2 is out of reach; any pause you create; the report written.

## When finished

Write the report exactly as PROMPT.md says: `/srv/models/test/reports/<date>-radiance-intel-m3.md` and
`.json`. Then put this line on its own in STATE.md:

    status: complete

**Never create `DONE`.** `run.sh` creates it once STATE.md says `status: complete` and both report
files exist, then turns off the timer and notifies. Post nothing anywhere.
