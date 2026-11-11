#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 113, owner notes 69 / 144 item 2): m3/build-l0.sh's build
# with Intel's 26.35 AOT compiler (localhost/ri-m1buildneo:1: ocloc 26.35 + IGC 2.41.5) into build-m1l0neo, so
# build-m1l0 (the current compiler's) stays the reference. Same limits as build-l0.sh.
#   m3/build-l0-neo.sh l0b | l0 | l0t PATTERN [IMAGE TREE [EXTRA]]
# run 207 (owner note 202 A.1): IMAGE (default localhost/ri-m1buildneo:1) and TREE (default /w/build-m1l0neo) select
# another compiler's build: e.g. localhost/ri-m1buildneo:2 (26.40 / IGC 2.42.6) into /srv/bulk/ri/build-m1l0neo2640
# (mounted as itself), with its own SYCL cache inside the tree. PATTERN "-" for none.
# run 284: EXTRA, one configure flag for m1/build.sh's cmake (L0_CMAKE_EXTRA), e.g. -DXE_GDN_SG_DIAG=ON for the GDN
# sweep's bench build (cached by the tree: configure with -DXE_GDN_SG_DIAG=OFF before an image from it).
set -u
MODE=${1:-l0b}; PAT=${2:-}; IMG=${3:-localhost/ri-m1buildneo:1}; TREE=${4:-/w/build-m1l0neo}; EXTRA=${5:-}; CONSTR=${6:-}
case "$EXTRA" in *[!A-Za-z0-9_=.-]*) echo "bad EXTRA flag"; exit 1 ;; esac
# run 283 (the coordinator's ruling, 03:1x 11-11): a 6th argument `constrained` asks for note 170's form on purpose --
# a libxe build beside the 12-job IGC build (one at a time; logs/memwatch.sh stops it first under 30 GiB MemAvailable)
[ "$PAT" = - ] && PAT=
CACHE=/w/sycl-cache-neo
[ "$TREE" != /w/build-m1l0neo ] && CACHE=$TREE/sycl-cache
W=$HOME/code/radiance-intel
L=$W/logs/build-l0neo-$MODE-$(date -u +%Y%m%dT%H%M%S).txt
# run 114 (owner note 145): build-m1l0neo is a symlink into /srv/bulk/ri (the supervisor moved it), mounted as itself
# Constrained mode (owner 2026-11-03, note 170): while a LoRA / Clover window shares amd-host (run.sh writes
# AMD-HOST-CONSTRAINED), build at 4 jobs in a hard 10 GB container, and only with > 20 GiB MemAvailable at the start.
CPUS=16; MEM=28g; JOBS=16
if [[ -e $W/AMD-HOST-CONSTRAINED || "$CONSTR" = constrained ]]; then
    CPUS=4; MEM=10g; JOBS=4
    avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
    (( avail > 20 )) || { echo "build refused: AMD-HOST-CONSTRAINED and MemAvailable ${avail} GiB <= 20 -- retry after the window"; exit 75; }
    echo "build in constrained mode: --cpus 4 --memory 10g -j4 (MemAvailable ${avail} GiB)"
fi
timeout 3h podman run --rm --name ri-m3-buildneo --cpus $CPUS --memory $MEM -v "$W:/w" -v /srv/bulk/ri:/srv/bulk/ri -e J=$JOBS \
    -e OMP_NUM_THREADS=8 -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=$CACHE -e L0_BUILD_DIR=$TREE \
    ${EXTRA:+-e L0_CMAKE_EXTRA=$EXTRA} \
    $IMG nice -n 19 ionice -c 3 bash /w/m1/build.sh "$MODE" $PAT > "$L" 2>&1
rc=$?
echo "build $MODE ($IMG into $TREE) rc=$rc log $L"
tail -5 "$L"
exit $rc
