#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 207, owner note 205): IGC releases/2.43.x built from source at
# pinned revisions, in a container at nice 19 / ionice idle, into /srv/bulk/ri/igc-243 only (sources, build tree,
# packages), packaged as the intel-igc-core-2 / intel-igc-opencl-2 debs images/m1buildneo243 installs over 2.42.6.
# A test compiler only. The in-container part is m3/igc-build-inner.sh (mounted read-only).
#   bash m3/build-igc.sh [JOBS=12] [BUILD=build] [IGC_REV]   BUILD: the build tree under /srv/bulk/ri/igc-243 (run 207:
#   build2); IGC_REV (run 280, owner note 223's fallback): the IGC tag or commit to build, e.g. v2.43.9 (default: run
#   207's probe pin fda9face), fetched into the existing clone and rebuilt incrementally in BUILD.
# Supervisor's review (run 207): 24 GB, 12 jobs (amd-host's MemAvailable ~57 GiB with production on; the LoRA guard trips
# at 30); start only with sparse-run inactive and production up (or A4 training with production off).
set -u
JOBS=${1:-12}
BUILD=${2:-build}
IGC_REV=${3:-fda9face0b56672ceaa31f46c5ab29065bef5aab}
case "$BUILD" in *[!A-Za-z0-9_-]*) echo "bad build dir name"; exit 1 ;; esac
case "$IGC_REV" in *[!A-Za-z0-9_.-]*) echo "bad IGC revision"; exit 1 ;; esac
W=$HOME/code/radiance-intel
O=/srv/bulk/ri/igc-243
[ -d "$O" ] || { echo "no $O (the owner creates it)"; exit 1; }
df -B1 "$O" | tail -1
L=$W/logs/build-igc-$(date -u +%Y%m%dT%H%M%S).txt
# the toolchain (gcc, make, bison, flex, python3-mako, zlib): images/igcbuild/Containerfile
timeout 30m podman build --label ri-probe=1 -t localhost/ri-igcbuild:1 "$W/images/igcbuild" > "$L" 2>&1 \
  || { echo "toolchain image failed ($L)"; exit 1; }
timeout 8h nice -n 19 ionice -c 3 podman run --rm --name ri-igc-build --cpus "$JOBS" --memory 24g \
  -e JOBS="$JOBS" -e BUILD="/o/$BUILD" -e IGC_REV="$IGC_REV" -v "$O:/o" -v "$W/m3/igc-build-inner.sh:/inner.sh:ro" localhost/ri-igcbuild:1 \
  nice -n 19 ionice -c 3 bash /inner.sh >> "$L" 2>&1
rc=$?; echo "build-igc rc=$rc ($L)"; tail -5 "$L"; cat "$O/REVISIONS" 2>/dev/null; ls "$O/pkg" 2>/dev/null
exit $rc
