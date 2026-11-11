# PROMPT.md — milestone 3 (2026-10-26)

Owner, 2026-10-26 08:10 UTC: "continue build it, the result needs improvement." Milestone 2's report
(`m2/2026-10-26-radiance-intel-m2.md`, §8 is the plan this milestone executes), `m2/STATE.md`, the
`intel-l0` branch at `a4ab300`, images `ri-m2rt:23` (hybrid play) and `:24` (+ the native-launch probe)
on intel-host. Goal: **the two B70s serving Flash-Next at a speed and a first-token time a user would accept**,
with milestone 2's open items closed. Stay on `intel-l0`; commit per passing step; nothing pushed.

**Targets** (every speed claim measured two ways: the single repeated prompt, warm, AND the median of the
11-request rotation used by the heat replay — milestone 2's 36 tok/s headline hid a 22 tok/s median):
- decode, one user: ≥ 60 tok/s single prompt, ≥ 40 tok/s rotation median (from 36 / 22);
- aggregate at 6 users: ≥ 120 tok/s (from 75.6);
- first token: ≤ 5 s at 7K prompt tokens, ≤ 15 s at 20K (from 53 / 94 s; production 0.5 / 2.2);
- MTP on a net gain over MTP off, with replay;
- the recorded-pass pool ≤ 1 GB a card, in VRAM or host;
- the 20-minute heat replay with moves on: no fault, misses ≤ 4 a step on the rotation (from 8.3).

**Stage A — the step (decode speed).**
1. **Tapes on direct Level Zero command lists** (report §7, owner note 28; 2 runs): the recorder in
   `launch_as` (tuple argument, group size, local memory) and in the core's `rad_l0k` launches and
   memcpys; ze events for the records and the cross-stream waits; one regular list per stretch between
   host decision points, played by appending to the stream's immediate list; per-device state and the
   deferred free (the fault-4 rule). Acceptance as note 28: tape/device/all-reduce tests PASS on the CPU
   device and the cards; the tiny model's recorded and plain runs give identical tokens; two-card tok/s
   with replay no worse than plain; the pool measured before and after (`drm-total-vram0`,
   `drm-total-gtt`); kernel logs clean. If not passing after 2 runs: leave the hybrid SYCL-graph tapes,
   record where it stopped, go on.
2. **MTP draft-pass correctness under recording** (§5.3; 1 run): dump the draft passes' arguments under
   recording and plain, find the frozen by-value argument or the host-written buffer, key it or make it a
   device read; then MTP on with replay, both measurements. Expected ~40 tok/s if acceptance returns to 46 %.
3. **The persistent-kernel exchange** (owner note 25, M2b's 24 µs form; 1–2 runs): replace the one-shot
   collective launches (two a layer) with the persistent kernel; measure the step and the launch count.
4. **Stage 1 kernels 4–6** (§3.1; 1 run each, oracle after every change, card1 small before full size):
   the GDN recurrent update (296 µs a layer → target ≤ 100 µs: 4–8 threads a state row, the 32 KB spill),
   hc_read fusion (3 kernels → 1), moe_gemm bf16. Re-measure tok/s after each kernel that moves the step
   ≥ 2 ms. Stop stage A when the targets hold or these are done.

**Stage B — prefill.**
5. **Matrix-unit kernels** (§5.4; 2–3 runs): the trunk INT8 GEMM with the per-128 rescale (round 2 M4)
   and the expert GEMM at batch ≥ 64 (W4 planes × INT8 activations) on the XMX units; oracle against
   `libref`; first token at 7K and 20K against the targets. `--ngram-placement ram` if intel-host can pin the
   47.7 GiB and still leave > 16 GiB free with both pinned pools (measure first; if not, say so).

**Stage C — the mover policy** (§5.6; 1 run): a minimum residency age / hysteresis / frequency-weighted
score in the heat engine so a unit demoted seconds ago is not brought back; the 20-minute replay on the
rotation before and after: misses a step, median tok/s, readmits. Keep the change behind a flag; the
default stays production's unless the replay is better on both counts.

**Stage D — the service, prepared, not switched on for anyone.** A Quadlet unit for intel-host (image by
digest, `--ulimit memlock=-1:-1`, both `/dev/dri` pairs, the sleeping wait, graphs per stage A's result,
the API key from the environment as production does), a watchdog on `/health` + the kernel log, and a
**24-hour soak** under the heat guard with the rotation at 1–3 users: faults, heat stops, tok/s drift,
agreement on the 11 requests at the end. **Nothing on amd-host or in any caller points at it**; the owner
decides promotion from the soak's numbers. Write the unit files under
`/mnt/fast-storage/models/test/radiance-intel/m3/service/`, not into `~/.config/containers/systemd/`.

**Rules:** all of milestone 2's (production untouched — job A's 1.11.0 window on 2026-10-27 02:00–03:30 ET
is the owner's script, not yours: **no amd-host builds or amd-host CPU load between 05:45 and 07:45 UTC on
2026-10-27**; amd-host 16 threads otherwise; intel-host heat guard 80 °C, agent `--hot 75`, CPUs by the note-16
ramp; files under `/mnt/fast-storage/models/test/radiance-intel/`; `--rm`, pinned images; nothing posted;
`prod-api.sh` the only route to production, **20 requests** this milestone; the fault rules of note 11;
containers `ri-m3-*`). Either card for single-card steps (note 19). Budget $1,000; by $700 judge whether
the targets are reachable and say so in STATE.md. Notify at: milestone start, each stage done with its
numbers, any fault, the soak's start and end, done.

**Report** `/srv/models/test/reports/<date>-radiance-intel-m3.md` + `.json`: the targets table (before /
after / target, both measurements); the pool before and after; MTP acceptance and tok/s; the exchange's
step and launch count; the per-kernel table after kernels 4–6; prefill first-token times; the mover
policy's replay before and after; the soak's 24 hours (faults, heat, drift, agreement); the service unit
and a promotion recommendation with criteria; what remains, sized. Then `status: complete`.

**Not in scope:** promoting the service to any caller; vision; anything on amd-host's GPUs; publishing.
