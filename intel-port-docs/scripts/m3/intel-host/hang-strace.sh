#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 87, owner note 82): the first-step hang's ioctl. Run 87's
# barrier diagnostic plus NEO's symbols put the stuck step loop in NEO::Drm::ioctl <- IoctlHelperXe::xeVmBind <-
# makeResidentWithinOsContext <- the immediate list's flush of the mover's memcpy (R7, R8), a VM_BIND that NEO
# reinvokes. This says with which errno and how often. Read-only: /proc reads and strace of failed ioctls for 5 s.
# Run beside `m3-step.sh STEP ...` (container ri-m3-STEP, serve.log under m3/STEP/CFG/).
#   bash hang-strace.sh STEP CFG
set -u
D=/mnt/fast-storage/models/test/radiance-intel
step=$1; cfg=$2
out=$D/m3/$step/hang-strace.txt
log=$D/m3/$step/$cfg/serve.log
: > "$out"
st_secs=5
# run 95 (H63: one CPU at 100 % right after the load, the heat stop 9 s later): `hang-strace.sh STEP CFG load` --
# 3 s after the log's checkpoint-slots line with no arena line yet, 3 s of strace (the heat allows no more)
if [ "${3:-}" = load ]; then
  st_secs=3
  for i in $(seq 1 300); do grep -q 'checkpoint slots' "$log" 2>/dev/null && break; sleep 1; done
  sleep 3
  # the rank's "I[0] arena:", not "activation arena:"
  if grep -q -F '] arena:' "$log" 2>/dev/null; then echo "hang-strace $step: past the arena, no spin at the load's end" >> "$out"; exit 0; fi
  echo "hang-strace $step: no arena line 3 s after the checkpoint slots" >> "$out"
fi
# the barrier's warning, or (a hang before the first step, e.g. a bind at allocation) 200 s with the container up
for i in $(seq 1 300); do
  [ "${3:-}" = load ] && break
  grep -q 'a thread has waited 10 s at the step barrier' "$log" 2>/dev/null && break
  [ "$i" -ge 200 ] && podman container exists "ri-m3-$step" 2>/dev/null && ! grep -q 'tok/s\|data: \[DONE\]' "$log" 2>/dev/null && break
  sleep 1
done
grep -q 'a thread has waited 10 s at the step barrier' "$log" 2>/dev/null || echo "hang-strace $step: no barrier warning after $i s" >> "$out"
podman container exists "ri-m3-$step" 2>/dev/null || { echo "hang-strace $step: the container is gone" >> "$out"; exit 0; }
cpid=$(podman inspect --format '{{.State.Pid}}' "ri-m3-$step" 2>/dev/null)
[ -n "$cpid" ] && [ "$cpid" != 0 ] || { echo "hang-strace $step: no container pid" >> "$out"; exit 1; }
pid=$(pgrep -n --ns "$cpid" -x radiance)
[ -n "$pid" ] || { echo "hang-strace $step: no radiance in the container's namespaces" >> "$out"; exit 1; }
echo "hang-strace $step: $(date -u +%T) radiance pid $pid (host), container pid $cpid" >> "$out"
for t in /proc/$pid/task/*; do
  st=$(awk '{print $3}' "$t/stat" 2>/dev/null)
  [ "$st" = R ] || continue
  echo "tid ${t##*/} state R syscall: $(cat "$t/syscall" 2>&1)" >> "$out"
done
ls -l /proc/$pid/fd 2>/dev/null | grep -E 'dri|xe' | awk '{print "fd", $9, $10, $11}' >> "$out"
# run 87 (R10: EBUSY every ~80 ms, in the kernel): each card's memory as the driver counts it for this process --
# total above resident is what was evicted
for fd in 3 4; do
  echo "fdinfo $fd:" >> "$out"
  grep -E '^drm-(pdev|total|resident|shared|active|purgeable|evicted)' /proc/$pid/fdinfo/$fd >> "$out" 2>&1
done
timeout $st_secs strace -f -tt -T -Z -e trace=ioctl -o "$out.raw" -p "$pid" 2>> "$out"
echo "failed ioctls in $st_secs s: $(wc -l < "$out.raw")" >> "$out"
sed -E 's/^[0-9]+ +[0-9:.]+ //; s/0x[0-9a-f]+/P/g; s/ <[0-9.]+>$//' "$out.raw" | sort | uniq -c | sort -rn | head -20 >> "$out"
head -5 "$out.raw" >> "$out"
for t in /proc/$pid/task/*; do
  st=$(awk '{print $3}' "$t/stat" 2>/dev/null)
  [ "$st" = R ] || continue
  echo "after: tid ${t##*/} state R syscall: $(cat "$t/syscall" 2>&1)" >> "$out"
done
echo "hang-strace $step: done $(date -u +%T)" >> "$out"
