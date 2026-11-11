#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 26). Read-only: ${2:-150} s after it starts, the thread
# states of container ri-m3-$1 (podman top: tid, state, kernel wait point, CPU, name) and the last 25 lines of
# the step's server log ($3), to D/top-$1.txt -- a stalled step's picture before the heat builds.
D=/mnt/fast-storage/models/test/radiance-intel
sleep "${2:-150}"
{
  date -u
  podman top "ri-m3-$1" -eLo tid,state,wchan:30,pcpu,comm 2>&1
  echo "--- serve.log tail"
  tail -n 25 "${3:-$D/m3/$1/nomtp/serve.log}" 2>&1 | cut -c1-200
} > "$D/top-$1.txt" 2>&1
