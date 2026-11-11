#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 23/24). Read-only: every ${2:-5} s for ${1:-900} s, intel-host's
# MemAvailable and the kernel's accounting that a GPU driver's host memory shows up in (Unevictable, Mlocked,
# Shmem, SUnreclaim, PageTables, in MiB), and for every running radiance process its VmRSS, VmSwap and thread
# count, so a host memory fall can be split between the engine's own (pageable) memory, pinned/driver memory
# and the rest:
#   <epoch> MemAvailable_MiB pid VmRSS_MiB VmSwap_MiB threads Unevictable Mlocked Shmem SUnreclaim PageTables
end=$(( $(date +%s) + ${1:-900} ))
while [ "$(date +%s)" -lt "$end" ]; do
  read -r ma ue ml sh su pt < <(awk '/^MemAvailable/ {a=$2} /^Unevictable/ {u=$2} /^Mlocked/ {m=$2} /^Shmem:/ {s=$2} /^SUnreclaim/ {r=$2} /^PageTables/ {p=$2} END {printf "%d %d %d %d %d %d", a/1024, u/1024, m/1024, s/1024, r/1024, p/1024}' /proc/meminfo)
  pids=$(pgrep -x radiance)
  if [ -n "$pids" ]; then
    for pid in $pids; do
      rss=$(awk '/VmRSS/ {printf "%d", $2/1024}' /proc/$pid/status 2>/dev/null)
      swp=$(awk '/VmSwap/ {printf "%d", $2/1024}' /proc/$pid/status 2>/dev/null)
      thr=$(awk '/Threads/ {print $2}' /proc/$pid/status 2>/dev/null)
      echo "$(date +%s) $ma $pid ${rss:-0} ${swp:-0} ${thr:-0} $ue $ml $sh $su $pt"
    done
  else
    echo "$(date +%s) $ma - 0 0 0 $ue $ml $sh $su $pt"
  fi
  sleep "${2:-5}"
done
