#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284): a snapshot of a build tree's GDN bench products -- tests/xe_gdn_split_bench,
# radiance_home/kernels/libxe.so and the CMakeCache.txt (the diag-build check) -- into a tree-shaped folder under
# results-m3, so a step (logs/gdndiag.sh / gdnstep.sh TREE=...) runs the build it was reviewed on while the source
# tree is rebuilt for the next one.
#   bash logs/snap-tree.sh SRC_TREE DEST_DIR      e.g. bash logs/snap-tree.sh /srv/bulk/ri/build-m1l0neo2640 /srv/bulk/ri/results-m3/kg17d-tree
set -u
SRC=${1:?source tree}; DST=${2:?destination}
case "$DST" in /srv/bulk/ri/results-m3/*) ;; *) echo "DEST must be under /srv/bulk/ri/results-m3"; exit 1 ;; esac
mkdir -p "$DST/tests" "$DST/radiance_home/kernels" || exit 1
cp "$SRC/tests/xe_gdn_split_bench" "$DST/tests/xe_gdn_split_bench" || exit 1
cp "$SRC/radiance_home/kernels/libxe.so" "$DST/radiance_home/kernels/libxe.so" || exit 1
cp "$SRC/CMakeCache.txt" "$DST/CMakeCache.txt" || exit 1
sha256sum "$DST/radiance_home/kernels/libxe.so" "$DST/tests/xe_gdn_split_bench"
grep "^XE_GDN_SG_DIAG:BOOL=" "$DST/CMakeCache.txt"
ls -la "$DST/radiance_home/kernels/libxe.so"
