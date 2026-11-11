#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 224 item 4 / note 226; the coordinator's parity2 instruction): a parity2
# step's parity2.bin (kld-gl.sh mtpparity3 at today's served numerics) from intel-host's step folder to amd-host's
# /srv/train/mtp-distill/NAME/ (the LoRA side's reference: training must match the served numerics), through a
# container with the pool mounted as logs/mtp_distill_copy.sh does (this session writes only under ~/code/radiance-intel),
# with the parity corpus, the record's layout note and an INFO.txt; then the listing.
#   bash logs/parity2_copy.sh STEP NAME "INFO LINE" [CORPUS.jsonl=logs/mtp-corpus-parity200.jsonl]
set -u
STEP=$1; NAME=$2; INFO=$3; CORPUS=${4:-$HOME/code/radiance-intel/logs/mtp-corpus-parity200.jsonl}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
DEST_ROOT=${MTP_DEST_ROOT:-/srv/train}
case "$NAME" in *[!A-Za-z0-9_.-]*) echo "bad NAME"; exit 1 ;; esac
run_cp() { podman run -i --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b localhost/ri-m1build:3 sh -c "$1"; }
ssh intel-host test -f "$D/m3/$STEP/mtpdump/parity2.bin" || { echo "no parity2.bin in $STEP"; exit 1; }
ssh intel-host cat "$D/m3/$STEP/mtpdump/parity2.bin" | run_cp "mkdir -p /b/mtp-distill/$NAME && cat > /b/mtp-distill/$NAME/parity2.bin" || { echo "copy of parity2.bin failed"; exit 1; }
echo "copied parity2.bin"
run_cp "cat > /b/mtp-distill/$NAME/corpus.jsonl" < "$CORPUS" || exit 1
# the layout: the checker's header comment is the record's description
head -12 "$W/m3/tools/parity2_check.py" | run_cp "cat > /b/mtp-distill/$NAME/LAYOUT.txt" || exit 1
echo "$INFO" | run_cp "cat > /b/mtp-distill/$NAME/INFO.txt" || exit 1
podman run --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b:ro localhost/ri-m1build:3 \
    sh -c "ls -la /b/mtp-distill/$NAME && sha256sum /b/mtp-distill/$NAME/parity2.bin"
