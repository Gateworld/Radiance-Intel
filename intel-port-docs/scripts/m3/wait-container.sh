#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 132). Waits while a container whose name starts
# with PREFIX exists on intel-host (another session's probe, e.g. ri-m3-TP1), checking every 30 s, at most MAX seconds.
#   bash m3/wait-container.sh PREFIX [MAX=2700]
p=${1:?prefix}; max=${2:-2700}; t=0
while ssh intel-host podman ps -a --format "{{.Names}}" | grep -q "^$p"; do
  [ "$t" -ge "$max" ] && { echo "wait-container: $p still there after $t s"; exit 1; }
  sleep 30; t=$((t + 30))
done
echo "wait-container: no $p after $t s"
