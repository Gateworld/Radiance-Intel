#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 228 item 1): a KL-mode dump's small files (meta.bin, top32.bin, the listing
# with every size) from intel-host's step folder to results-m3/card01/STEP/mtpdump/ for m3/tools/dump_check.py; hidden.bin
# (20,480 B a position) stays on intel-host -- its size is in sizes.txt, which dump_check.py reads when the file is absent.
# With `hidden`, hidden.bin is fetched too (a small part only).
#   bash logs/dump_fetch.sh STEP [hidden]
set -u
STEP=$1; HID=${2:-}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01/$STEP/mtpdump
mkdir -p "$R" || exit 1
ssh intel-host ls -l "$D/m3/$STEP/mtpdump/" > "$R/sizes.txt" || exit 1
ssh intel-host cat "$D/m3/$STEP/mtpdump/meta.bin" > "$R/meta.bin" || exit 1
ssh intel-host cat "$D/m3/$STEP/mtpdump/top32.bin" > "$R/top32.bin" || exit 1
[ "$HID" = hidden ] && { ssh intel-host cat "$D/m3/$STEP/mtpdump/hidden.bin" > "$R/hidden.bin" || exit 1; }
cat "$R/sizes.txt"
