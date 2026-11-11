#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 283; the coordinator's ruling on builds beside the IGC build): watch amd-host's MemAvailable
# while a constrained libxe build (podman container ri-m3-buildneo) runs beside the 12-job IGC build; under FLOOR GiB
# the libxe build is stopped first (podman stop), as ruled. Exits when the build container is gone or at the limit.
#   bash logs/memwatch.sh [FLOOR_GIB=30] [LIMIT_S=7200]
set -u
FLOOR=${1:-30}; LIMIT=${2:-7200}
t0=$(date +%s)
while :; do
  avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
  if ! podman container exists ri-m3-buildneo 2>/dev/null; then echo "$(date -u +%H:%M:%S) build container gone (MemAvailable ${avail} GiB)"; exit 0; fi
  if (( avail < FLOOR )); then
    echo "$(date -u +%H:%M:%S) MemAvailable ${avail} GiB < ${FLOOR}: stopping ri-m3-buildneo"
    podman stop -t 10 ri-m3-buildneo
    exit 3
  fi
  (( $(date +%s) - t0 > LIMIT )) && { echo "memwatch limit reached (MemAvailable ${avail} GiB)"; exit 0; }
  sleep 30
done
