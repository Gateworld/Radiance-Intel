#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, owner note 59 phase 0). Attaches strace to a probe in the step's
# container from the host side (read-only: ioctl calls only, timestamps), so the brief can count the submissions'
# system calls: run beside `m3-step.sh STEP ... /out/PROGRAM ...`.
#   bash x-strace.sh STEP [PROGRAM=xe_list_pool_probe]
set -u
D=/mnt/fast-storage/models/test/radiance-intel
step=$1
prog=${2:-xe_list_pool_probe}
out=$D/m3/$step/strace.txt
for i in $(seq 1 600); do
  pid=$(pgrep -n -f "^/out/$prog" || true)
  [ -n "$pid" ] && break
  sleep 0.1
done
[ -n "${pid:-}" ] || { echo "x-strace $step: no $prog process"; exit 1; }
echo "x-strace $step: pid $pid"
timeout 120 strace -f -tt -e trace=ioctl -o "$out" -p "$pid"
echo "x-strace $step: rc $? lines $(wc -l < "$out")"
