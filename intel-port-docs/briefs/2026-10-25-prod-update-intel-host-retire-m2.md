# Three jobs: production to Radiance 1.11.0, retire Chatterbox and Qwen-Image on intel-host, start milestone 2

(Owner's brief, pasted 2026-10-25 ~20:00 UTC; saved verbatim. Job C Part 2 is also this folder's PROMPT.md.)

Paste into Claude Code on **amd-host** (VS Code Remote-SSH, user `user`), in `~/code/radiance-intel`. Nothing can run before the plan limit resets on **2026-10-27 12:00 UTC**. Do the interactive parts (A, B and C's Part 1) with the owner present; C's Part 2 is the unattended milestone.

Owner's decisions, 2026-10-25:
- Production moves to Radiance **1.11.0**, through the usual test-window protocol.
- **Chatterbox (`chatterbox-vllm`) and Qwen-Image 2.1 (`comfyui`) are retired from intel-host.** Their settings, secrets, custom images and data are saved to disk so they can be rebuilt on amd-host later. Both B70s are then free for the port.
- **Milestone 2** starts: kernel tuning first (card1 only, can start right after the reset), then two cards.

Rules as always: production is never edited outside a test window; root commands are handed to the owner; nothing posted anywhere; the vLLM rollback is never touched.

---

# A — Production to Radiance 1.11.0

Work in `~/code/radiance-test` (the existing production and test tooling) and `~/code/radiance-src`.

## A1. Plan, no downtime

1. **What production runs today.** Confirm from the Quadlet and `engine.sh`: image `radiance:1.9.0-keyenv` (local build, revision `73caca3`), flags, the `.rad` path, and **how the API key reaches the engine** (the `keyenv` patch, or native). Find the patch diff and where it lives.
2. **1.11.0 source.** The port's tree at `~/code/radiance-intel/src` is `main` @ `9eaf10d` = 1.11.0. Use the same commit for production, from a fresh clone in `~/code/radiance-test/src-1.11.0`, untouched by the `intel-l0` branch. Record the commit and the upstream release tag.
3. **Forward-port the keyenv patch** if 1.11.0 still lacks native env-var key reading. Keep it as one small diff in `~/code/radiance-test/patches/`. If 1.11.0 reads the key from the environment natively, drop the patch and say so.
4. **Changelog 1.9.0 → 1.11.0.** Read the commits between `73caca3` and `9eaf10d`. List everything that touches: the mover and expert streaming (`104c81f`), the prefix cache and snapshots, sampling and MTP, the `.rad` loader, the API and the chat template, tensor-parallel. For each: expected effect on production, and which of our checks covers it.
5. **Build** `radiance:1.11.0-keyenv` for gfx1201 on amd-host's CPU (the same `docker/build.sh` or `podman build` route as 1.9.0; `nice -n 19`, ≤ 8 threads, `MemAvailable` > 40 GiB). Save it as an OCI archive under `/srv/models/test/radiance/`, pinned by digest. Run the CPU test suite with the build and record pass/fail.
6. **Window plan.** The same protocol as 1.7.0 on 2026-10-16: feature checks (17), crash replay (9), the night-2 soak (seed 20261016, 27 min) with the production bar, and the labeler eval if the soak passes. Production will be down for about 60–90 minutes. Pick the night window the owner already uses (02:00–06:00 ET), avoiding `app-autogen` (01:30 ET) and anything on `sparse-night-lora.timer`; show both calendars and propose a start time.

**STOP A1:** the patch status, the changelog table, the build digest and test result, and the proposed window. The owner approves the window.

## A2. The window

Under `~/code/radiance-test`'s test harness, with the production guard and the 3-hour cap:

1. Switch to the 1.11.0 image with **exactly production's flags** (TP2, `--tp-wire exact`, 64 slots, `keep-all`, the reasoning-effort map, `HIP_VISIBLE_DEVICES=0,1`, the key from the environment).
2. Feature checks, crash replay, the soak. Compare with the 2026-10-16 1.7.0 numbers and the night-2 production bar.
3. **Prompt 10 from milestone 1:** send the card's exact prompt at temperature 0 and record the first token and its log-probability. (Production on 1.9.0 said "The" −0.05; the card on 1.11.0 said "We" −0.32.)
4. **Heat-engine counters before and after:** residency, misses, link MB per step over the soak, so the `104c81f` mover fix's effect is measured.
5. **Promote only if** the bar holds (first token p50/p95, per-turn p50/p95 within the 1.7.0 tolerances, 0 errors, all tool calls valid, cache hits ≥ 90%). Otherwise roll back with `engine.sh` and report. Either way, production is healthy before the window ends.

**STOP A2:** the report `/srv/models/test/reports/<date>-amd-host-radiance-1.11.0.md` + `.json`, with the prompt-10 answer stated plainly: version difference or not.

---

# B — Retire Chatterbox and Qwen-Image on intel-host (owner present, no unattended part)

Voice goes down when `chatterbox-vllm` stops, until a amd-host version exists. Say so before stopping anything.

## B1. Save everything first

Into `/mnt/fast-storage/models/test/retired-2026-10/` on intel-host, then mirrored to the 4 TB mirror (the owner says where):
- **Quadlets and units:** `chatterbox-vllm.container`, `comfyui.container`, the backup `chatterbox.container` in `~/quadlet-backup/`, any drop-ins, timers and env files.
- **Secrets:** `podman secret inspect --showsecret` for `tts_api_key` and any other secret these units use, written to a root-only file (`chmod 600`), with the secret names. Say which secrets exist on intel-host so none are missed.
- **Images:** `podman save` the custom builds `localhost/chatterbox-vllm-xpu:latest`, `localhost/chatterbox-xpu:latest`, `localhost/comfyui-xpu:latest`, `localhost/xpu-base:*`, each as an OCI archive with its digest and the Containerfile or build notes that produced it (`~/chatterbox-vllm-xpu/`, the ComfyUI build folder). These are the only copies.
- **Data:** ComfyUI's workflows, custom nodes list, settings and `~/comfyui-data/` (minus `output/warmup/`); Chatterbox's voice files, prompt and config; the model folders they use under `/mnt/fast-storage/models/` stay where they are (read-only mount).
- **A `RESTORE.md`** that says how to bring each back on intel-host in one page: units, secrets, images, data paths, ports (8004, 8188), keys, and which callers use them (assistant voice via `tts.openai.base_url`, the study GUI's `CHATTERBOX_CONCURRENCY=3`, the image service and `assistant-service`).
- sha256 of every archive; total size.

## B2. Callers

Before stopping: list every caller of `intel-host:8004` and `intel-host:8188` / the image service, from the reference doc and the live configs on amd-host (assistant containers' `tts.openai.base_url`, `assistant-service`, the image service's own units, the study GUI env, Mission control's collector). Propose for each: leave pointing at intel-host (it fails cleanly) or switch assistant voice back to its default Edge TTS until Chatterbox exists on amd-host. The owner decides; apply only what's approved.

## B3. Stop and disable

`systemctl --user disable --now chatterbox-vllm comfyui` on intel-host; `image-service` and `qwen36` are already off. Confirm with `podman ps` that no container maps `/dev/dri`, both cards show no clients, and the mirror copy's hashes match. Don't delete the images or the Quadlet files from intel-host; leave them in place, disabled.

**STOP B:** the archive listing with sizes and hashes, `RESTORE.md`, the caller table and what was switched, and "both cards idle".

Moving them to amd-host is a separate job: both images are Intel XPU builds and need ROCm rebuilds, and amd-host's cards 2,3 carry the 27B, `textsearch` and `vlembed`. Write a one-page sizing note (VRAM each needs, what's free on cards 2,3, which base images exist for ROCm) into the report, and stop there.

---

# C — Milestone 2

## Part 1 — setup (interactive)

Same runner as milestone 1. Archive milestone 1 into `m1/` (keep `results2/heat/`), empty `STATE.md`, no session ID.

- `budget.txt` = **500**; daily cap **$80**; `--max-budget-usd 20` per run.
- `DONE` pattern `*-radiance-intel-m2`.
- Apply the pause-rule changes already agreed (fault counting by dump id; the agent diagnoses and fixes faults itself; pause only for card0/Chatterbox, an unrecovered card, a repeated signature, or two faults in one run; decision rule for choices that can be undone).
- **card0 gate:** Part 2's two-card steps may run only when `intel-host-precheck.txt` shows **both** cards with no clients and no container mapping `/dev/dri`. Until B3 is done, those steps wait; the runner writes "waiting for card0" and the single-card steps continue.
- Production API: `prod-api.sh` stays the only route, now **30 requests** for the milestone.
- `OWNER-NOTES.md` carries over, plus: "Production will move to 1.11.0 during milestone 2 (job A). After it does, agreement checks compare like with like; before it, mark any divergence as possibly version-related."

Dry run, STOP, "enable".

## Part 2 — PROMPT.md (unattended)

(See this folder's PROMPT.md: the same text, verbatim.)
