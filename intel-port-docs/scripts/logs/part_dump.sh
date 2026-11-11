#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 228 item 1): one MTP v2 part from its generated documents to its dump in
# amd-host's training pool, in redoable stages:
#   render -- m3/tools/mtp_v2_render.py with Flash-Next's tokenizer: logs/mtp-v2/partN.jsonl (the KL corpus) + .role.bin /
#             .ids.bin / .roles.json; prints the generated share of positions (roles 1+2+3 over all);
#   dump   -- the KL-mode dump on intel-host (kdstep.sh GENP<N>D, mode mtpdump, image 306 = the parity engine, both cards);
#   check  -- logs/dump_fetch.sh + m3/tools/dump_check.py (ids position by position against meta.bin);
#   copy   -- logs/mtp_distill_copy.sh to /srv/train/mtp-distill/flashnext-v2/partN/ with role.bin / ids.bin / roles.json;
#   all    -- the four in order, stopping at the first failure.
#   bash logs/part_dump.sh N DOCS.jsonl render|dump|check|copy|done|all [DUMP_LIMIT_S=6600] [STEP_SUFFIX]
# STEP_SUFFIX (e.g. b) names a re-dump GENP<N>D<suffix> (part 4's first dump held a 9,627-token document the replay split).
# The dump mode is mtpdumpbig (no reference written: ~0.5 MB a position saved; the long limit) -- a part is ~300,000
# positions (part 2: 292,910), ~6 GB of hidden.bin on intel-host, ~30-40 min of record pass.
set -u
N=$1; DOCS=$2; MODE=$3; TL=${4:-6600}; SFX=${5:-}
W=$HOME/code/radiance-intel
TOK=/srv/models/llm/qwen3.8-flash-next-mxfp4-fp8-gptq/tokenizer.json
PFX=$W/logs/mtp-v2/part$N
STEP=GENP${N}D$SFX
IMG=02e462b5b74c
render() {
  timeout 1800 python3 -I "$W/m3/tools/mtp_v2_render.py" "$TOK" "$DOCS" "$PFX" || return 1
  python3 -I - "$PFX.roles.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); c = j['role_counts']; n = j['tokens']
print('RESULT part render: %d documents, %d tokens; generated share (roles 1+2+3) = %d / %d = %.1f %%' % (len(j['documents']), n, c[1] + c[2] + c[3], n, 100.0 * (c[1] + c[2] + c[3]) / max(n, 1)))
PY
}
dump() { timeout $(( TL + 2400 )) bash "$W/logs/kdstep.sh" "$STEP" "$IMG" mtpdumpbig KDP2 "$PFX.jsonl" "$TL"; true; }
check() {
  timeout 900 bash "$W/logs/dump_fetch.sh" "$STEP" || return 1
  timeout 300 python3 -I "$W/m3/tools/dump_check.py" "$W/results-m3/card01/$STEP/mtpdump" "$PFX"
}
copy() {
  local eng; eng=$(grep -o '"engine": "[^"]*"' "$DOCS" | head -1 | cut -d'"' -f4)
  timeout 1800 bash "$W/logs/mtp_distill_copy.sh" "$STEP" "flashnext-v2/part$N" \
    "MTP v2 part $N: documents from $(basename "$DOCS") (engine: ${eng:-?}; the client m3/tools/mtp_v2_gen.py, temperature 1.0 / top_p 0.95 / top_k 20 / seed = id, 3 users), rendered by m3/tools/mtp_v2_render.py (Qwen3 template; Flash-Next's tokenizer.json sha 0997f410...), dumped by kld-gl.sh mtpdump on image ri-m3rtneo:306 (today's served numerics; the parity2 engine 128f06c) as step $STEP; role.bin / ids.bin / roles.json beside meta / hidden / top32 (dump_check PASS); $(date -u +%Y-%m-%dT%H:%MZ)." \
    "" "$PFX.jsonl" "$PFX"
}
done_marker() {   # owner note 233 item 1: the part's DONE marker (positions, generated share, the check) in the pool
  local line
  line=$(python3 -I - "$PFX.roles.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); c = j['role_counts']; n = j['tokens']
print('positions %d, generated share %.1f %%, dump_check PASS, %d documents' % (n, 100.0 * (c[1] + c[2] + c[3]) / max(n, 1), len(j['documents'])))
PY
)
  echo "$line" | podman run -i --rm --name ri-m3-distillcp --cpus 2 --memory 2g -v /srv/train:/b localhost/ri-m1build:3 \
      sh -c "cat > /b/mtp-distill/flashnext-v2/part$N/DONE" || return 1
  echo "RESULT part $N DONE: $line"
  # intel-host's step disk: the copied dump's hidden.bin (20 KB a position) goes once the pool holds it (07:5x: the disk was
  # full with 120 GB of staged libxe.so copies + 46 GB of delivered hidden.bin files)
  ssh intel-host rm -f "$D/m3/$STEP/mtpdump/hidden.bin" && echo "intel-host: $STEP/mtpdump/hidden.bin removed (copied)"
}
D=/mnt/fast-storage/models/test/radiance-intel
case "$MODE" in
  render) render ;;
  dump) dump ;;
  check) check ;;
  copy) copy ;;
  done) done_marker ;;
  all) render && dump && check && copy && done_marker ;;
  *) echo "mode?"; exit 2 ;;
esac
