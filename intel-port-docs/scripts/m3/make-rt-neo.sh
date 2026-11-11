#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 113, owner notes 69 / 144 item 1). The runtime test image
# localhost/ri-m3rtneo:N: images/m1rt/ctx exactly as the last m3/make-rt.sh left it (the same binaries, kernels and
# plugins as that ri-m3rt image), built with images/m1rt/Containerfile.neo2635 -- NEO 26.35.39758.10, gmmlib
# 22.10.0 and IGC 2.41.5's runtime libraries in place of NEO 26.05 / IGC 2.28.4. Then, with `load`,
# `podman save localhost/ri-m3rtneo:N | ssh intel-host podman load`.
#   m3/make-rt-neo.sh N [load]
set -u
N=${1:?tag number}; LOAD=${2:-}
W=$HOME/code/radiance-intel
L=$W/logs/build-rtneo-m3-$N.txt
df -B1 $W | tail -1
timeout 1h podman build --cpu-quota 800000 --memory 20g --label ri-probe=1 -f $W/images/m1rt/Containerfile.neo2635 \
  -t localhost/ri-m3rtneo:$N $W/images/m1rt > "$L" 2>&1
rc=$?; echo "image build rc=$rc ($L)"; [ $rc = 0 ] || { tail -20 "$L"; exit $rc; }
grep -E "^(libze1|libze-intel-gpu1|libigdgmm12|intel-igc-core-2|intel-igc-opencl-2)\s" "$L" | head -5
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m3rtneo:$N"
if [ "$LOAD" = load ]; then
  timeout 40m podman save localhost/ri-m3rtneo:$N | ssh intel-host podman load 2>&1 | tail -2
fi
