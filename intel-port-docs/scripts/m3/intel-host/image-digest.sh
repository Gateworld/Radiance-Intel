#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 90). Read-only: the image ID and digest of runtime images on
# intel-host, for moving the stage D unit's Image= line (an ssh command line cannot carry podman's format braces intact).
#   bash image-digest.sh 70 80 ...   (no tags: every ri-m3rt image with its digest)
#   RI_REPO=ri-m3rtneo bash image-digest.sh 221   (run 177: the NEO-runtime test images, the ones the steps run)
REPO=localhost/${RI_REPO:-ri-m3rt}
[ $# -eq 0 ] && exec podman images --digests --format '{{.Tag}} {{.ID}} {{.Digest}}' "$REPO"
#   bash image-digest.sh check sha256:...   (does the unit's form REPO@<digest> resolve here, and to which ID)
if [ "$1" = check ]; then
  ref="$REPO@$2"
  if podman image exists "$ref"; then echo "$ref resolves: $(podman image inspect --format '{{.Id}}' "$ref")"; else echo "$ref does NOT resolve"; fi
  exit 0
fi
for n in "$@"; do
  echo "${REPO#localhost/}:$n $(podman image inspect --format '{{.Id}} {{.Digest}}' "$REPO:$n" 2>&1)"
done
