#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 60; owner note 49): the 11 agreement requests to production,
# through ./prod-api.sh only (<= 30 in the milestone; 16 were used before this run, and the owner allowed at most
# 11 here): m3/prod-agree/NN-name.json (m3/make_agree_bodies.py: the rotation's 11 greedy requests with the
# logprobs asked for) -> results-m3/prod/NN-name.json. A request already answered (its .json present and
# non-empty) is not sent again, so a rerun never spends a request twice.
W=~/code/radiance-intel
O=$W/results-m3/prod; mkdir -p "$O"
n=0
for f in "$W"/m3/prod-agree/[0-9][0-9]-*.json; do
  name=$(basename "$f" .json)
  if [ -s "$O/$name.json" ]; then echo "$name: already answered"; continue; fi
  if "$W/prod-api.sh" "$f" "$O/$name.json"; then n=$((n + 1)); echo "$name: ok ($(wc -c < "$O/$name.json") bytes)"
  else echo "$name: FAILED (rc $?)"; rm -f "$O/$name.json"; fi
  sleep 2
done
echo "sent $n request(s); last log line: $(tail -1 "$W/logs/prod-api.log")"
