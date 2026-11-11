#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 233): is an MTP v2 part's docs file complete? The part's id range (1-300,
# 301-600, 601-900, 901-1200, 1201-1420) against the ids present (one record a document, errors counted apart); exit 0
# when every id is there, 1 otherwise. parts-done.txt's "part N done" lines are shown too (an rc other than 0 = a
# terminated attempt, not a completion).
#   bash logs/part_ready.sh N DOCS.jsonl
set -u
N=$1; DOCS=$2
W=$HOME/code/radiance-intel
case "$N" in 1) LO=1; HI=300 ;; 2) LO=301; HI=600 ;; 3) LO=601; HI=900 ;; 4) LO=901; HI=1200 ;; 5) LO=1201; HI=1420 ;; *) echo "part?"; exit 2 ;; esac
grep "part $N done" "$W/logs/mtp-v2/amd-host/parts-done.txt" 2>/dev/null
python3 -I - "$DOCS" "$LO" "$HI" <<'PY'
import json, sys
docs, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids, errs = set(), 0
try:
    for line in open(docs, encoding='utf-8'):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if 'error' in d: errs += 1
        else: ids.add(d['id'])
except FileNotFoundError:
    print(f'{docs}: missing'); sys.exit(1)
want = set(range(lo, hi + 1)); missing = sorted(want - ids)
print(f'{docs}: {len(ids & want)} of {len(want)} documents present, {errs} error records, {len(missing)} missing' + (f' (first {missing[:5]})' if missing else ''))
sys.exit(0 if not missing else 1)
PY
