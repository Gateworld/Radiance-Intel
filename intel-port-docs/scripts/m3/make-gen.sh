#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 284, owner note 228 item 1): the generation step image
# localhost/ri-m3gen:N = a runtime image BASE plus python3 (images/gen/Containerfile), for the MTP v2 client inside the
# step container; with `load`, saved straight into intel-host's podman (never an image tar on disk). Light caps: an apt layer.
#   m3/make-gen.sh N BASE_IMAGE [load]      e.g. m3/make-gen.sh 1 localhost/ri-m3rtneo:308 load
set -u
N=${1:?tag number}; BASE=${2:?base image}; LOAD=${3:-}
W=$HOME/code/radiance-intel
L=$W/logs/build-gen-m3-$N.txt
df -B1 $W | tail -1
timeout 30m podman build --cpu-quota 400000 --memory 10g --label ri-probe=1 --build-arg BASE=$BASE \
  -t localhost/ri-m3gen:$N $W/images/gen > "$L" 2>&1
rc=$?; echo "image build rc=$rc ($L)"; [ $rc = 0 ] || { tail -20 "$L"; exit $rc; }
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m3gen:$N"
if [ "$LOAD" = load ]; then
  timeout 40m podman save localhost/ri-m3gen:$N | ssh intel-host podman load 2>&1 | tail -2
fi
