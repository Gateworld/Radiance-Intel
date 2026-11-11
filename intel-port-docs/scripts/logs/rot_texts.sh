#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 228 item 2): a rotation pair's texts compared -- the first rotation's
# responses (the request counter k = 0 .. N-1, the same bodies in the same order in both halves) fetched from intel-host's
# step folders and compared by m3/cmp_choices.py (text, tokens, every logprob, usage; ids and times ignored).
#   bash logs/rot_texts.sh STEP_A STEP_B [N=11]
set -u
A=$1; B=$2; N=${3:-11}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
mkdir -p "$R/$A/resp" "$R/$B/resp" || exit 1
ssh intel-host ls "$D/m3/$A/resp" > "$R/$A/resp/list.txt" || exit 1
ssh intel-host ls "$D/m3/$B/resp" > "$R/$B/resp/list.txt" || exit 1
same=0; diff=0; miss=0
for k in $(seq 0 $(( N - 1 ))); do
  fa=$(grep -- "-$k\.http$" "$R/$A/resp/list.txt" | head -1); fb=$(grep -- "-$k\.http$" "$R/$B/resp/list.txt" | head -1)
  if [ -z "$fa" ] || [ -z "$fb" ]; then miss=$((miss + 1)); echo "k=$k: missing ($fa / $fb)"; continue; fi
  ssh intel-host cat "$D/m3/$A/resp/$fa" > "$R/$A/resp/$fa" || exit 1
  ssh intel-host cat "$D/m3/$B/resp/$fb" > "$R/$B/resp/$fb" || exit 1
  if python3 -I "$W/m3/cmp_choices.py" "$R/$A/resp/$fa" "$R/$B/resp/$fb" > "$R/$B/resp/cmp-$k.txt" 2>&1; then
    same=$((same + 1))
  else
    diff=$((diff + 1)); echo "k=$k ($fa vs $fb): DIFFER -- $(head -3 "$R/$B/resp/cmp-$k.txt" | tr '\n' ' ' | cut -c1-200)"
  fi
done
echo "RESULT rot_texts $A vs $B: $same of $N requests identical (text, tokens, logprobs, usage), $diff differ, $miss missing"
