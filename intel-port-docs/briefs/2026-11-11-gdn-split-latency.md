# GDN decode update: cut the fixed load chain (coordinator, 2026-11-11 ~03:0x UTC; owner: "don't give up on the GDN idea")

For the runner. Exact changes only (same arithmetic, same order, same state bytes). Base = the VS 2 split (e35db12:
engine 22.2 us + ru_split_norm 5.3 us = 27.5 us a launch vs the fused 33.0; 36 launches a step, all T 4) -- the stronger
base per KG16 / M401. The same changes apply to the VS 8 slice-group kernel (91083cc) and are measured there too.

## Why: the fixed part is a dependent load chain

Every thread, before any token work (r4d_gdn_recurrent_update_kernel ~l.394-450, ru_sg_update_kernel ~l.887-938):
1. `sidx[n][0..5]` (independent loads, one round trip)
2. `fb[RU_MARK_OFF]` -- needs sf from step 1 -> round trip 2
3. `si = sidx[n][acol]` -- needs the marker -> round trip 3. **REDUNDANT:** acol is 1..3 and columns 1-3 are already
   in registers from step 1 (sa1, sa2, sf0 -- check the column map); select instead of loading.
4. the state row `h` at slot si -> round trip 4
5. the replay: `for t < nacc` loads kk / vv / eg / bt from `fr + t*RU_FSTR` inside the loop -> up to nacc more round
   trips if the compiler does not hoist them (runtime trip count). **They do not depend on h:** issue all of them
   (nacc <= 4: unrolled to 4 with a live predicate) before the h load returns; the math after keeps today's order.
At Xe2's ~0.7-1 us a DRAM round trip that is ~5-7 us of pure latency a launch, the same in both kernels -- the T 1 time
of the VS 2 split (~15 us in KG16) is mostly this plus launch.

## The changes (each its own switch for the A/B, then folded)

- **L1 (select, not reload):** si from the registers of step 1 (exact: the same integer). -1 round trip.
- **L2 (replay loads up front):** the nacc factor rows loaded into registers right after fr is known (4 x (HK kk +
  vv + eg + bt) floats = ~76 registers at HK 16: check GRF 256 / no spill), then the replay math unchanged. -(nacc-1)
  round trips.
- **L3 (marker side copy):** the last arriver / fused tail ALSO writes the marker into a small per-(sequence, head)
  side array (allocated beside cv.ctr); the kernel reads it together with sidx in round trip 1, so the marker hop
  overlaps step 1. The state's own marker byte is still written as today -> state hashes unchanged. Engine change
  (allocation + init at sequence start = today's initial marker) -- do it only if L1 + L2 show the chain matters.
- **N1 (fold the norm):** ru_split_norm's body as a device function called by the split's last arriver per head
  (the draft's C11), removing the second launch + its boundary (~5.3 us + ~0.5 us). Exact by construction if the
  device function is the same code; the norm kernel stays for the fallback.

## Gate and order

1. Bench (KG17, after the diagnostic sweep): VS 2 split + L1, + L1 L2, + N1, each at T 1 / T 4 per process, anchor
   nacc 1-3, hashes = SPLIT 0 everywhere (incl. anchor-seq x 20). Expect T 4: 27.5 -> ~20 us (engine scale).
2. If >= 15 % per launch at T 4: engine M-pair -gs2- with the new switches vs plain on the same image + -prof
   (fused gone, the split's us a launch). Adoption line: >= +1 % same image -> rotation -> owner (default).
3. L3 only if 1-2 show the remaining chain is still >= 3 us at T 1.
4. Feed the diagnostic sweep's findings for the VS 8 kernel back in: if a single culprit (V1-V6) explains its +19 us
   and +8 us/token, fix it there too and re-bench against the improved VS 2 split -- the better of the two goes on.
Stop rule: a hash difference that survives two fixes, or < 10 % per launch at T 4 after L1 + L2 + N1.

## Results (runs 283-284; steady clock, 5,000 launches a process -- 200-launch timings were clock-confounded)

- KG17 (run 283): L1 / L2 / N1 all EXACT (hashes = SPLIT 0 per call, anchor-seq x20, N1 stress x500 3,000 / 3,000);
  only L1 gains (-1.3 us a launch, -7 %), L2 and N1 are slower -> L1 folded (463485e; KG18 gate equal), N1 kept as an
  opt-in record (RADIANCE_XE_GDN_LAT=7). L3 (an engine change) not built: step 2's >= 15 % line was not met.
- Engine (owner note 228): the VS 2 split + L1 made the default (7fa51f2, image ri-m3rtneo:309) on a same-image rotation
  of +0.9 % tok/s, texts 11 / 11 identical, the profile with the fused kernel gone (M402 / M403).
- KG19 (run 284, note 230): the VS 4 HYBRID slice-group (k / g / beta parked to shared, no per-token global stores) is
  exact but ties the split at T 4 -> opt-in record (RADIANCE_XE_GDN_HYBRID=4, 549d012). VS 8 v2 (restaging) parked.
- KG20 (run 284): brief C (the per-token q / k scales once a workgroup, RADIANCE_XE_GDN_LAT=8) is exact (3,000 / 3,000
  stressed calls) and ties at T 4 (anchor 17.5 -> 17.2 us, per_candidate 19.7 -> 19.7; T 1 anchor 11.3 -> 10.3,
  per_candidate 8.8 -> 9.1; noise +-0.4) -> opt-in record. The T 4 token loop is bound by the state rows' loads /
  stores, not by the per-row scale math.
- Stop rule reached: nothing in this brief is left to bench.
