#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3). The Level Zero build and its tests inside
# localhost/ri-m1build:3 (Intel LLVM v7.2.0, AOT for intel_gpu_bmg_g31 + spir64), on amd-host's CPU within
# UNATTENDED's limits (owner note 17: --cpus 16 --memory 28g, nice 19, -j16; one build at a time, MemAvailable > 40 GiB). Run from amd-host:
#   m2/build-l0.sh l0b            build only (build-m1l0)
#   m2/build-l0.sh l0t 'xe_moe'   build, then ctest -R 'xe_moe' (the CPU SYCL device)
#   m2/build-l0.sh l0             build + the whole no-GPU tier
# Logs to logs/build-l0-<mode>-<UTC>.txt.
set -u
MODE=${1:-l0b}; PAT=${2:-}
W=$HOME/code/radiance-intel
L=$W/logs/build-l0-$MODE-$(date -u +%Y%m%dT%H%M%S).txt
# Constrained mode (owner 2026-11-03, note 170): while a LoRA / Clover window shares amd-host (run.sh writes
# AMD-HOST-CONSTRAINED), build at 4 jobs in a hard 10 GB container, and only with > 20 GiB MemAvailable at the start.
CPUS=16; MEM=28g; JOBS=16
if [[ -e $W/AMD-HOST-CONSTRAINED ]]; then
    CPUS=4; MEM=10g; JOBS=4
    avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
    (( avail > 20 )) || { echo "build refused: AMD-HOST-CONSTRAINED and MemAvailable ${avail} GiB <= 20 -- retry after the window"; exit 75; }
    echo "build in constrained mode: --cpus 4 --memory 10g -j4 (MemAvailable ${avail} GiB)"
fi
# run 220: /srv/bulk/ri mounted as itself, as build-l0-neo.sh -- build-m1cpu2 (and other trees) are symlinks there
# (owner note 145), dangling without the mount ("Unable to (re)create the private pkgRedirects directory")
timeout 3h podman run --rm --name ri-m3-build --cpus $CPUS --memory $MEM -v "$W:/w" -v /srv/bulk/ri:/srv/bulk/ri -e J=$JOBS \
    -e OMP_NUM_THREADS=8 -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/w/sycl-cache \
    localhost/ri-m1build:3 nice -n 19 bash /w/m1/build.sh "$MODE" $PAT > "$L" 2>&1
rc=$?
echo "build $MODE rc=$rc log $L"
tail -5 "$L"
exit $rc
