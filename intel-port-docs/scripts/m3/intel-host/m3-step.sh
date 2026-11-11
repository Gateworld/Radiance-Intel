#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3). m3-step.sh STEP TLIMIT IMAGE CMD... -- one step on
# intel-host's card1. Milestone 1's m1-step.sh with the milestone-2 names (owner note 13: containers
# ri-m2-<step>; the guard stops ri-m1-* and ri-m2-*) and D/m2/STEP mounted at /out. Round 2's
# ri-step2.sh gate re-run, round 1's 1 Hz sampler beside the container, --rm, crun, keep-groups,
# card1's nodes only, one Level Zero device = index 0, a timeout inside.
# Output in D: out-STEP.txt, cpu-STEP.csv/.times, gate-STEP.txt, start-STEP.txt (for ri-klog.sh).
set -u
D=/mnt/fast-storage/models/test/radiance-intel
step=$1; tl=$2; img=$3; shift 3
ctr=ri-m3-$step; flag=$D/flag-$step
mkdir -p "$D/m3/$step" "$D/sycl-cache"   # owner note 21: a persistent SYCL kernel cache for the card containers
chk=$(bash "$D/ri-check.sh"); echo "$chk" > "$D/gate-$step.txt"
ok=1
echo "$chk" | grep -q '^card1_holders: none$' || ok=0
read -r _ l5 _ < /proc/loadavg
[ "${l5%.*}" -lt 4 ] || ok=0
allowed=" $(cat "$D/allowed-mappers.txt" 2>/dev/null) "
for m in $(echo "$chk" | sed -n 's/^card1_mapped_in_containers://p' | grep -o '[0-9]*(' | tr -d '('); do
    case "$allowed" in *" $m "*) ;; *) ok=0 ;; esac
done
echo "$chk" | grep '^card1_mapped_in_containers' | grep -q 'holds:[^ ]*0000:0f:00.0' && ok=0
# STEP_CARD (owner notes 18, 19): 1 (default) = card1 only; 0 = card0 only; both = the two-card step. Card0
# has no ri-check line, so its holders are read from fdinfo as round 2's m2b-run.sh does.
card=${STEP_CARD:-1}
if [ "$card" != 1 ]; then
  busy=""
  for pp in /proc/[0-9]*; do
    for f in "$pp"/fdinfo/*; do
      if grep -qsE 'drm-pdev:[[:space:]]*0000:0b:00.0' "$f"; then busy="$busy ${pp#/proc/}"; break; fi
    done
  done
  [ -z "$busy" ] || { echo "step $step: processes hold card0 (pids:$busy)"; ok=0; }
fi
# no other ri container
rc_list=$(echo "$chk" | sed -n 's/^ri_containers: *//p' | tr ' ' '\n' | grep -v -E '^$')
[ -z "$rc_list" ] || ok=0
if [ $ok = 0 ]; then echo "step $step GATE FAILED"; cat "$D/gate-$step.txt"; exit 3; fi
# STEP_DEADLINE (epoch s): the precheck's expiry (30 min after its time_epoch). A gate that hung (podman
# unresponsive, run 1's P1b) must not start the container after it.
if [ -n "${STEP_DEADLINE:-}" ] && [ "$(date +%s)" -ge "$STEP_DEADLINE" ]; then
  echo "step $step: past STEP_DEADLINE $STEP_DEADLINE at $(date +%s); not started"; exit 4
fi
# STEP_CPUS: the CPU cap. Owner notes 3, 4, 13: engine steps at 0.3; never above 1, except the
# note-7 / note-13 exception (2 CPUs, one run each, the guard started with --hot 75 from amd-host),
# named by STEP_NOTE7_CPUS2=1.
cpus=${STEP_CPUS:-0.3}
if [ "${STEP_NOTE7_CPUS2:-0}" = 1 ] && { [ "$cpus" = 2 ] || [ "$cpus" = 4 ] || [ "$cpus" = 8 ]; }; then
  echo "step $step: --cpus $cpus under owner note 16 (CPUs by heat: 2, 4 or 8; guard --hot 75)"
else
  awk -v c="$cpus" 'BEGIN{exit !(c > 0 && c <= 1)}' || { echo "step $step: STEP_CPUS=$cpus refused (0 < cpus <= 1)"; exit 3; }
fi
# STEP_ENGINE=1: the real-model engine -- locked memory unlimited and the intel-host folder read-only
# at /d for the .rad.
extra=()
[ "${STEP_ENGINE:-0}" = 1 ] && extra=(--ulimit memlock=-1:-1 -v "$D:/d:ro")
# STEP_SECCOMP=<file> (owner note 45, run 43): a seccomp profile for this container only (the io_uring test:
# podman's default profile plus io_uring_setup, io_uring_enter and io_uring_register)
[ -n "${STEP_SECCOMP:-}" ] && extra+=(--security-opt "seccomp=$STEP_SECCOMP")
# STEP_ARCHIVE=1 (run 189, the supervisor: the trained draft head's model lives on intel-host's archive disk, owner-approved):
# /mnt/archive/radiance-intel read-only at /a, beside /d
[ "${STEP_ARCHIVE:-0}" = 1 ] && extra+=(-v /mnt/archive/radiance-intel:/a:ro)
date +%s > "$D/start-$step.txt"
touch "$flag"
timeout $((tl + 90)) bash "$D/ri-sample.sh" "$step" "$ctr" "$flag" &
sp=$!
# STEP_VRAM=1 (owner note 24): the per-card VRAM sampler beside the container (ri-vram.sh, read-only fdinfo)
vp=""
if [ "${STEP_VRAM:-0}" = 1 ]; then timeout $((tl + 90)) bash "$D/ri-vram.sh" "$step" "$flag" & vp=$!; fi
case "$card" in
  1)    nodes=(--device /dev/dri/card1 --device /dev/dri/renderD129); sel=(-e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZE_AFFINITY_MASK=0 -e RADIANCE_L0_PCI=0000:0f:00.0) ;;
  0)    nodes=(--device /dev/dri/card0 --device /dev/dri/renderD128); sel=(-e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZE_AFFINITY_MASK=0 -e RADIANCE_L0_PCI=0000:0b:00.0) ;;
  both) nodes=(--device /dev/dri/card0 --device /dev/dri/renderD128 --device /dev/dri/card1 --device /dev/dri/renderD129)
        sel=(-e 'ONEAPI_DEVICE_SELECTOR=level_zero:*') ;;
  *)    echo "step $step: STEP_CARD=$card refused (1, 0 or both)"; exit 3 ;;
esac
echo "step $step: card $card"
timeout --kill-after=20 $((tl + 45)) podman run --rm --name "$ctr" --cpus "$cpus" --runtime crun --group-add keep-groups \
    "${nodes[@]}" -v "$D/m3/$step:/out" "${extra[@]}" \
    "${sel[@]}" -e ZES_ENABLE_SYSMAN=1 -e UR_LOADER_USE_LEVEL_ZERO_V2=${STEP_UR_V2:-0} -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl-cache -v "$D/sycl-cache:/sycl-cache" \
    "$img" timeout "$tl" "$@" > "$D/out-$step.txt" 2>&1
rc=$?
echo "rc=$rc" >> "$D/out-$step.txt"
rm -f "$flag"
wait "$sp"
[ -n "$vp" ] && wait "$vp"
echo "step $step rc=$rc"
