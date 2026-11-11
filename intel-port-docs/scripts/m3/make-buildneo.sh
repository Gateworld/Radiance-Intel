#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 113, owner notes 69 / 144 item 2): the test build image
# localhost/ri-m1buildneo:N (images/m1buildneo/Containerfile: ri-m1build:3 with ocloc 26.35 + IGC 2.41.5).
#   m3/make-buildneo.sh N [CONTEXT]      run 205 (owner note 202 A.1): CONTEXT = images/m1buildneo2640 for 26.40 / IGC 2.42.6
set -u
N=${1:?tag number}
W=$HOME/code/radiance-intel
CTX=${2:-images/m1buildneo}
L=$W/logs/build-m1buildneo-$N.txt
df -B1 $W | tail -1
timeout 30m podman build --cpu-quota 800000 --memory 20g --label ri-probe=1 -t localhost/ri-m1buildneo:$N \
  $W/$CTX > "$L" 2>&1
rc=$?; echo "build image rc=$rc ($L)"; tail -8 "$L"
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m1buildneo:$N"
exit $rc
