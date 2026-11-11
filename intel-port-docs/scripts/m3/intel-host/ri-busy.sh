#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 20). Read-only: for every DRM client of the running
# radiance server, the xe fdinfo cycle counters (drm-cycles-* = the client's busy cycles on that engine,
# drm-total-cycles-* = the engine's elapsed cycles), sampled twice ${1:-2} s apart. A client whose
# drm-cycles-ccs grows as fast as drm-total-cycles-ccs keeps the compute engine busy (kernels spinning);
# one whose cycles stand still waits on an event.
for round in 1 2; do
  date -u +%H:%M:%S
  for pid in $(pgrep -x radiance); do
    for f in /proc/$pid/fdinfo/*; do
      grep -q drm-client-id "$f" 2>/dev/null || continue
      echo "pid $pid fd ${f##*/}: $(grep -h -e drm-pdev -e drm-cycles-ccs -e drm-total-cycles-ccs -e drm-cycles-bcs "$f" | tr '\n' ' ')"
    done
  done
  [ $round = 1 ] && sleep "${1:-2}"
done
