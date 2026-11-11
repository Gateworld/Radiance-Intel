#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 163, owner note 169 item 3). The unitrace test image
# localhost/ri-m3ut:N = the runtime image BASE plus unitrace's install (build-unitrace/install, from m1/build.sh
# `unitrace`), built with images/unitrace/Containerfile; with `load`, saved straight into intel-host's podman.
#   m3/make-ut.sh N BASE_IMAGE [load]      e.g. m3/make-ut.sh 1 localhost/ri-m3rtneo:185 load
set -u
N=${1:?tag number}; BASE=${2:?base image}; LOAD=${3:-}
W=$HOME/code/radiance-intel
L=$W/logs/build-ut-m3-$N.txt
df -B1 $W | tail -1
mkdir -p $W/images/unitrace/ctx $W/images/unitrace/ctx/metrics/lib
tar -C $W/build-unitrace -cf - install | tar -C $W/images/unitrace/ctx -xf -
# the metrics libraries (run 164): build-metrics/install's lib plus the extracted libdrm2 .so, when built
if [ -d $W/build-metrics/install/lib ]; then
  tar -C $W/build-metrics/install -cf - lib | tar -C $W/images/unitrace/ctx/metrics -xf -
  tar -C $W/build-metrics/sysroot/usr/lib/x86_64-linux-gnu -cf - . | tar -C $W/images/unitrace/ctx/metrics/lib -xf -
fi
ls $W/images/unitrace/ctx/metrics/lib | head
timeout 1h podman build --cpu-quota 800000 --memory 20g --label ri-probe=1 --build-arg BASE=$BASE \
  -f $W/images/unitrace/Containerfile -t localhost/ri-m3ut:$N $W/images/unitrace > "$L" 2>&1
rc=$?; echo "image build rc=$rc ($L)"; [ $rc = 0 ] || { tail -20 "$L"; exit $rc; }
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m3ut:$N"
if [ "$LOAD" = load ]; then
  timeout 40m podman save localhost/ri-m3ut:$N | ssh intel-host podman load 2>&1 | tail -2
fi
