#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 283, owner note 224 item 4; the coordinator's parity2 gate): a parity2 step's
# parity2.bin fetched from intel-host's step folder to results-m3/card01/STEP/, the 11-04 parity.bin beside it (from the
# reference step's folder on intel-host, default MD5c), then m3/tools/parity2_check.py on the pair: 200 records,
# rounds == 3 everywhere, the fed tokens following the previous round's top-1, and round 1 against the 11-04 rows
# (byte-identical on the B155 prefill path, KPAR1; the matrix-unit path, KPAR2, may differ past prompt row 256).
#   bash logs/parity2_fetch.sh STEP [REF_STEP=MD5c]
set -u
STEP=$1; REF=${2:-MD5c}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01/$STEP
mkdir -p "$R" || exit 1
ssh intel-host cat "$D/m3/$STEP/mtpdump/parity2.bin" > "$R/parity2.bin" || { echo "no parity2.bin in $STEP"; exit 1; }
[ -s "$R/parity2.bin" ] || { echo "parity2.bin is empty"; exit 1; }
ssh intel-host cat "$D/m3/$REF/mtpdump/parity.bin" > "$R/parity-1104.bin" || { echo "no parity.bin in $REF"; exit 1; }
ls -la "$R/parity2.bin" "$R/parity-1104.bin"
sha256sum "$R/parity2.bin" "$R/parity-1104.bin"
python3 -I "$W/m3/tools/parity2_check.py" "$R/parity2.bin" "$R/parity-1104.bin"
