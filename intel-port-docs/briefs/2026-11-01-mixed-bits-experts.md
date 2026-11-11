# Fewer bytes for the experts: mixed bits by how often each expert is routed

2026-11-01, supervisor (owner's request). Not an owner note: nothing here is queued until the owner approves it.

## Why
- Format C's expert kernels run within 1.3-1.5x of their weight-streaming floor (run 117): at the verify, gate_up
  112 us against ~86 us to stream ~34 MB, and down 65 us against ~43 us. More expert speed now needs fewer bytes,
  not cheaper math.
- Expert residency is the other lever: ~26 misses a step on the rotation, and each one is read over PCIe 4.0 x8.

## The idea
Give each expert a bit width by how often the router picks it:
- hot experts keep today's 4-bit w4nl codebook (4.125 bits);
- rarely routed experts get a 3-bit codebook (~3.1-3.4 bits), with its levels also rounded to int8 at load as in
  format C, so the same DP4A path serves both.

Two effects:
1. **Streaming:** bytes read a step fall only by the cold experts' share of routed traffic. That share is small,
   so most of the streaming gain would need hot experts at fewer bits too, which costs quality where it matters.
2. **Residency:** VRAM holds more experts when cold ones are smaller. That gives fewer misses, mostly on varied
   traffic. This is likely the larger effect.

## A cheap first step (no cards, no convert)
1. **The popularity curve:** routed rows per expert and layer are in each per-expert down Gram's header (RADGRAM2:
   u64 K, u64 rows, u64 domain), summed over r0+r2 / r1+r3. Find what share of routed rows the least-used 25 / 50 /
   75 % of experts carry, layer by layer. The supervisor started this on amd-host (results below when done).
2. **The error at 3 bits:** on formatc/ (42 sampled experts, 7 layers), docs/MOE-W4.md's held-out output error
   for a 3-bit codebook (Lloyd-fitted like w4nl), alongside w4nl and format C. Done in the runner's CPU check
   harness from note 146 item 4.
3. **The VRAM model:** with the slab's measured sizes (~20,800 slots of 992 KiB a card), how many more experts fit
   at a given mix, and the miss rate H85-style traffic would see. Use the heat engine's own stats if they have
   per-expert counts.
Then decide whether a convert (rad-convert per-expert rules, note 140's Grams) is worth it.

## Quality caveat
Radiance measured its formats against the bf16 model's logits (docs/MOE-W4.md). We can't run the 360 GB bf16 model
on the B70s, so on our side the check is relative to today's format (as format C's FCK1: KL 0.036). An absolute
check would need a bf16 reference run somewhere else (amd-host's CPU, slowly).

## Results of step 1

### Step 1 measured (supervisor, 2026-11-02 18:00 UTC): routed rows per expert, 1,001,221 calibration tokens, all 48 layers
- the least-used 25 % of experts carry **6.7 %** of a layer's routed rows on average (range 1.0-10.5 %)
- the least-used 50 % of experts carry **22.2 %** of a layer's routed rows on average (range 7.0-29.7 %)
- the least-used 75 % of experts carry **47.0 %** of a layer's routed rows on average (range 26.8-58.0 %)
- the most-used 25 % carry 53.0 % (range 42.0-73.2 %)
- the most-used 50 % carry 77.8 % (range 70.3-93.0 %)
- experts with < 2,560 routed rows (their own down Gram is too thin): 30.3 a layer on average, at most 141
- Reading: if the least-used half carry only a small share of the traffic, giving them 3 bits barely changes the bytes streamed a step, but it shrinks half the experts by ~25 %, so ~12 % more of them fit in VRAM: the gain is fewer misses, not faster kernels.

## Results of steps 2 and 3 (agent, run 160 and run 175, owner note 177 item 7)

### Step 2 (run 160, logs/formatc-3bit-study.txt, a3a0fde)
- A 3-bit codebook (four Lloyd magnitudes 0.094 / 0.292 / 0.521 / 0.833 of the group amax, and their negatives),
  quantised by the served rule (GPTQ, a searched E4M3 scale per 64, cd 3), scored by the held-out output error
  under the calibration Grams on formatc/'s 84 matrices: **gate_up 0.0970 against w4nl's 0.0485, down 0.1153
  against 0.0577 -- 2.00x the error**; its levels rounded to int8 add 0.2 %.

### Step 3 (run 175): the VRAM and miss model, from the rotation's own measurements
- Bits a weight: w4nl 4 + 8/64 (the E4M3 scale) = 4.125; the 3-bit form 3.125: **a cold expert at 0.76 of its size.**
- The slab today (H119, headroom 1280): 21,662 VRAM units of 24,576 (48 layers x 512 experts): 88 % resident, the
  ~2,900 coldest units outside; **28.1 misses a step** (1.9 % of 1,460 routed).
- The slab's sensitivity, measured today: shrinking it 6.8 % (H120: 20,196 units, the decode-once scratch) raised
  the misses 73 % (48.7). Misses sit on the steep part of the curve.
- **The coldest half at 3 bits:** 12,288 + 12,288 x 0.76 = 21,627 units' worth -- every expert fits in today's
  slab: **misses to ~0 (predicted -90 % or more)**, for 2x the output error on 22 % of the routed rows (about +22 %
  of the average expert error).
- **The coldest quarter at 3 bits:** 18,432 + 6,144 x 0.76 = 23,101 units' worth against 21,662: the units outside
  fall from ~2,900 to ~1,440; by the measured sensitivity, **misses roughly -50 to -60 %**, for 2x the error on
  6.7 % of the rows (about +7 % of the average error).
- **The quality side, the open part:** the only calibration of output error against note 149's KL bar is format
  C -- +0.8 % output error replayed at mean KL 0.036 (FCK1), the same as any last-bit change (GL 0.035, ATTN_MX
  0.042 = the bar). The bar is therefore close to "a last-bit change"; +7 % to +22 % of expert error is 9-27x
  format C's increment. Format B (+23 % on every expert) was never replayed. **The model predicts the misses gate
  is met easily and cannot predict the bar; by the size of the error it is more likely outside.** The only way to
  know is a convert of the cold quarter and a KL replay -- the owner's call (note 177: report before any convert).
