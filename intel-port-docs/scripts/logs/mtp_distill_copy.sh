#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 179, owner note 181 phase 1): a finished MTP dump from intel-host's step folder to amd-host's
# DEST_ROOT/mtp-distill/NAME/ (the owner's destination), through a container with that dataset mounted (this
# session writes only under ~/code/radiance-intel), with the README, the corpus and an INFO.txt; then the sizes.
#   bash logs/mtp_distill_copy.sh STEP NAME "INFO LINE" [RECORDS CORPUS.jsonl [RENDER_PREFIX]]
# run 284 (the MTP v2 parts): RENDER_PREFIX names the renderer's outputs on amd-host (m3/tools/mtp_v2_render.py's
# OUT_PREFIX): PREFIX.role.bin -> role.bin, PREFIX.ids.bin -> ids.bin, PREFIX.roles.json -> roles.json beside the dump.
# RECORDS: copy only the first RECORDS positions of each file (a dump cut off mid-run: m3/tools/dump_prefix.py finds
# the prefix of whole documents) and CORPUS in place of the whole corpus; hidden rows are 10,240 bf16 (20,480 bytes).
# run 283 (owner note 226; the coordinator's OK): the destination root is /srv/train (amd-host's training scratch pool),
# where the v2 dumps go; MTP_DEST_ROOT=/srv/bulk/ri restores phase 1's place.
set -u
STEP=$1; NAME=$2; INFO=$3; RECS=${4:-}; CORPUS=${5:-$HOME/code/radiance-intel/logs/mtp-corpus-1m.jsonl}; RPFX=${6:-}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
DEST_ROOT=${MTP_DEST_ROOT:-/srv/train}
put() {   # put SRC_ON_INTEL-HOST DEST_NAME [BYTES]
  if [ -n "${3:-}" ]; then rd=(head -c "$3"); else rd=(cat); fi
  ssh intel-host "${rd[@]}" "$1" | podman run -i --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b \
      localhost/ri-m1build:3 sh -c "mkdir -p /b/mtp-distill/$NAME && cat > /b/mtp-distill/$NAME/$2"
}
declare -A per=([meta.bin]=12 [top32.bin]=256 [hidden.bin]=20480)
for f in meta.bin top32.bin hidden.bin; do
  n=""; [ -n "$RECS" ] && n=$(( RECS * ${per[$f]} ))
  put "$D/m3/$STEP/mtpdump/$f" "$f" $n || { echo "copy of $f failed"; exit 1; }
  echo "copied $f"
done
# run 187: the parity file (kld-gl.sh mtpparity), when the step wrote one; whole (268 bytes a document)
if ssh intel-host test -f "$D/m3/$STEP/mtpdump/parity.bin"; then
  put "$D/m3/$STEP/mtpdump/parity.bin" parity.bin || { echo "copy of parity.bin failed"; exit 1; }
  echo "copied parity.bin"
fi
extra=()
[ -n "$RPFX" ] && extra=("$RPFX.role.bin:role.bin" "$RPFX.ids.bin:ids.bin" "$RPFX.roles.json:roles.json")
for f in "$W/m3/mtpdump-README.md:README.md" "$CORPUS:corpus.jsonl" "${extra[@]}"; do
  src=${f%%:*}; dst=${f##*:}
  podman run -i --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b localhost/ri-m1build:3 \
      sh -c "cat > /b/mtp-distill/$NAME/$dst" < "$src" || exit 1
done
echo "$INFO" | podman run -i --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b localhost/ri-m1build:3 \
    sh -c "cat > /b/mtp-distill/$NAME/INFO.txt" || exit 1
podman run --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v "$DEST_ROOT":/b:ro localhost/ri-m1build:3 \
    sh -c "ls -la /b/mtp-distill/$NAME"
ssh intel-host ls -la "$D/m3/$STEP/mtpdump"
