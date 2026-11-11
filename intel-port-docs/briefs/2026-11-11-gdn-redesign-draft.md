# GDN decode update redesign: VS 8 / 16 with the convolution staged once per slice-group (owner note 224 item 2) -- desk design

2026-11-11, desk draft (agent, read-only on code; nothing built or run). For the runner (to build after its current
steps) and the coordinator (to book). Labels: **M** = measured (source given), **E** = my estimate from the code.

Source: `src/libxe/ports/r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.cpp` (HEAD 5b37bc2). Line numbers below
are from that file. Prior work: STATE run 113 (e35db12, VS 2 / 4 anchor split, exact, -12 % a call, no end-to-end
gain), the run-271 correction (STATE ~3610), owner notes 138 and 224.

## 0. The decision in one paragraph

Today the per-call cost is not state bandwidth (floor 5.4 us). It is a **long per-thread chain repeated by every
thread**: each of the 1,024 lanes of a head redoes the q / k l2 norms, six transcendentals and four work-group
barriers per token, and the anchor replay pays one global-load latency per accepted token. Splitting alone (VS 2 / 4)
hardly helped, and VS 4 was slower than VS 2 (KG13: 20.6 vs 21.8 us), because the total sub-group count is
fixed at 768 per call whatever VS is. Only the chain per thread can shrink. The design therefore does three
things:
1. **New grid (N, Hg, VS).** One work-group is a *slice-group*: the G = H / Hg = 3 value heads that share a key head,
   x NR = 128 / VS rows each, x 8 lanes a row. The key head's q|k convolution and normalisation are staged **once per
   slice-group** in SLM (1 channel a thread at VS 8, 2 adjacent channels a thread at VS 16). Today it is redone in
   G x VS work-groups.
2. **The factor parts that slices share (each 32-row group's k copy and its decay / beta pair) are written only by
   the key head's last-arriving work-group, after its token loop.** This removes the run-271 race. The shift marker
   keeps run 113's last-arriver stamp.
3. **Per-token math reduced to the irreducible core** (one dot, the rank-1 update and the output dot). Norms,
   gates and replay factors are prepared once in the staging phase, in exactly today's operation order.

Expected: ~31.6 -> **~11 us a call** with the separate norm kernel (stage 2), **~8 us** with the norm folded in
(stage 3). That is -0.74 to -0.85 ms a rank-step, about +2-2.5 % tok/s. Confidence: medium-low (section 7).

## 1. Today's kernel anatomy and where the 31.6 us go

The production instantiation (MTP on = `state_form anchor`, gnt0): `r4d_gdn_recurrent_update_kernel<1, NM=2, VS=1,
NT=0, ANC=1, STG=0, SDT=0, CV=1, KT=8>`, launched by `ru_conv_recurrent_launch` (l. 1040), GRF256. Grid (N=1, H=24,
1); 1,024 threads a work-group = 32 sub-groups. One work-group fills one Xe-core at GRF256, so 24 of 32 cores are
used. A call is one layer's verify at T = 4 (MTP depth 3) with nacc ~ 2.9 accepted tokens replayed. There are 35.9
calls a step (scoreboard P87 / P110: 31.6 / 31.7 us, 1.14 ms a step, 3.9 % of device time).

Phases, in program order:

| phase | lines | what it does | est. us |
|---|---|---|---|
| prologue | 290-445 | cu, cidx, naccept, sidx row; conv inputs issued (x, w, conv history); **dependent chain** sidx -> fb[RU_MARK_OFF] (the marker) -> sidx[acol] -> 64 B a lane of state | ~3 (3 dependent global hops) |
| anchor replay | 444-485 | dynamic loop over nacc tokens. Each token loads its factors (k copy 16 floats a lane, v, g, beta) from global, then h *= e^g, dot, 3-step xor, u, h += u k | ~3 (one exposed load latency a token, ~2.9 tokens) |
| publish + barrier | 461-485 | anchor stored at column 2; barrier (32 sub-groups); stamp (VS 1) | ~0.5 |
| conv | 510-560 | one q|k channel + one v channel a thread, 4 taps, cp_silu, bf16 into SLM; barrier; key-head counter arrival | ~1 |
| token loop x T | 593-750 | **per token per lane:** 32 bf16->f32 from SLM, two 16-term sum-of-squares + 2 x 3 xor + 2 rsqrt, 32 scaling muls, softplus / sigmoid / exp (6 transcendentals), h *= e^g + 16-term dot + 3 xor, the factor stores (group rows), h += u k with a 16-term output dot + 3 xor, **fused norm** (5 xor, SLM, barrier, 32-term serial sum, rsqrt, gate, barrier), **fused int8 quantiser** (5 xor, SLM, barrier, 32-term max, barrier), stores | **~5.4 a token -> ~21.5** |
| tail | 750-775 | the last of G arrivers rewrites the key head's q|k conv state; drain | ~1 |

The per-token slope is derived (M-derived). KG13's anchor fused figure of 23.5 us is the mean of T = 1 and T = 4
calls at nacc 2. With T = 4 at 31.6 us (production), that gives T = 1 ~ 15.4, so **~5.4 us a token plus ~10 us
fixed**. GB4 (per_candidate, H 16) agrees: 13.2 / 30.3 us at T 1 / 4. FLA's fused_recurrent at our shape takes 6.0 /
9.5 us at T 1 / 4 (note 138, M), without the conv, the norm or the anchor replay. The state's byte floor is 5.4 us.

The split's own record (KG13, M, T 1 / 4 mixed): VS 2 16.7 + 3.9 (ru_split_norm_kernel), VS 4 17.8 + 4.0. Removing
the per-token norm barriers paid; adding work-groups did not. Every work-group still runs the whole chain, and
768 sub-groups over 256 XVEs (~3 a XVE) is already the density from VS 2 on.

## 2. New thread / work-group mapping (VS 8 and 16)

New kernel function, **not** a new instantiation of the shipped template: `ru_sg_update_kernel<VS, NT, ANC, SDT,
STAGE>` in the same file, so the fused VS 1 kernel (the hash reference and production default) keeps its ISA
byte-for-byte. Launch branch: in `ru_conv_recurrent_launch` beside RC_SPLIT. `RADIANCE_XE_GDN_SPLIT` is parsed with
atoi and accepts {2, 4, 8, 16}. Today's `e[0] == '2' || '4'` test would read "16" as off. 8 / 16 go to the new
kernel, 2 / 4 to the existing one. Extra arguments go in a new struct, not in RuConv, so the fused kernel's argument
block is unchanged. The tape audit compares argument bytes (the M104 lesson).

| | VS 8 | VS 16 |
|---|---|---|
| grid | (N, Hg = 8, 8) = 64 WGs | (N, 8, 16) = 128 WGs |
| WG | G 3 heads x 16 rows x 8 lanes = 384 thr (12 sub-groups) | 3 x 8 x 8 = 192 thr (6 sub-groups) |
| thread -> (head, row, half) | `half = tid & 7`, `rl = (tid >> 3) % NR`, `gi = (tid >> 3) / NR`, `hv = hk * G + gi`, `row = s * NR + rl` | same |
| a sub-group holds | 4 rows of one head (no head straddles a sub-group) | 4 rows of one head |
| 32-row group | `grp = row >> 5`, uniform a sub-group; a slice lies in one group, **shared by 2 slices** | shared by **4 slices** |
| q\|k conv | threads 0-255: one channel each (today's code) | threads 0-127: **channels 2p, 2p+1** (adjacent: one 32-bit x load a token, one 16 B weight load, history pairs) |
| v conv | threads 256-303: the WG's 48 v channels (own rows) | threads 128-151: the 24 v channels |
| key-head counter arrivals | VS = 8 (today G x VS) | 16 |

KT stays **8** and HK stays 16 (KT 16 would change the dot's association: 8 in-lane terms + 4 xor steps). Sub-group
count is 768 for both, as today, but spread over all 32 cores (2 or 4 WGs a core). G is a runtime value: blockDim =
G x NR x 8, at most 1,024 (refuse otherwise and fall back to fused). N x Hg <= kMaxConvCtr, as today.

## 3. Convolution and per-token operands, staged once per slice-group

Staging phase. Its inputs are issued at the top, so they are in flight during the prologue chain and the replay:
1. **Conv** (stage 1 and on): today's `conv` lambda, run per channel. At VS 16 it runs twice per thread (channels 2p,
   2p+1); bf16 pairs are packed in one dword until use. Output: bf16 `cvq` / `cvk` [t][128] and `cvv` [t][own rows]
   in SLM, as today. The q|k history is parked in `cvh` indexed **by channel** (not tid) for the last arriver's
   rewrite. Barrier.
2. **Norms and scaling** (stage 2): 16 teams of 8 consecutive lanes (team = (token, q or k), lane & 7 = half). Each
   lane sums its 16 channels `16h .. 16h+15` serially from j = 0, then `ru_row_sum<8>`, then
   `__frsqrt_rn(s + 1e-6f) * scale` (q) or `__frsqrt_rn(s + 1e-6f)` (k). Each lane writes its 16 scaled values as
   fp32: `qh[t][128]`, `kh[t][128]`. T <= 8 needs 128 lanes, which fits both WG sizes.
3. **Gates** (stage 2): lanes `< G x T` compute `g = -alog * ru_softplus(ra + dtb, thr)`, `bt = 1 / (1 + __expf(-rb))`
   and `eg = __expf(g)` per (head, token) into SLM, from today's `sab` staging.
4. **Replay factors** (stage 2): issued right after the sidx row arrives. With RU_ANC_DUP = 1, `fr = fb` (column 4)
   does not depend on the marker, so this is one hop and runs in parallel with the marker -> state chain. Copied
   into SLM: for each head of the WG, its slice's group k copy (128), its rows' v and its group's (g, beta), for
   t < min(nacc, 4). Tokens 4-7 (not reachable at MTP depth 3) replay from global as today.
   Barrier (merged with 1's barrier where the order allows).

Token loop per lane, no barriers: read `eg`, `bt` (broadcast), `kh[t][k0 .. k0+15]`, then `h *= eg; dot += h * kh`,
`ru_row_sum<8>`, `u = (vraw - dot) * bt`, then loop A `h[j] += u * kh[j]`, then loop B `on += h[j] * qh[j]`,
`ru_row_sum<8>`, then the `onf` store by `half == 0` (stage 2) or the stage-3 path. Each slice writes **its own
rows'** v into the factor block (all dup_f copies). Nothing else is written per token.

### Exactness, change by change (the gate is still the hashes; this is why they should match)

| change | why every floating-point result is unchanged |
|---|---|
| C1 grid / mapping | No arithmetic moves. Each (row, half) lane does today's operations on today's operands. The 8 lanes of a row are still consecutive in one sub-group, so `ru_row_sum<8>` pairs the same halves in the same xor 1 -> 2 -> 4 order. |
| C2 two conv channels a thread | Each channel's `acc = bs; acc += w[i] * win[i]` (i = 0..3), cp_silu and f2bf are unchanged; channels are just processed one after another. Packing bf16 in a dword moves bits only. |
| C3 last arriver writes the shared factor parts | Stored values are bitwise today's: `(ST)(bf2f(k) * ks)`, `(ST)g`, `(ST)bt` are the same in every group and computed identically by every row. Each address is written once a launch, so write order cannot change the bytes. |
| C4 marker | Same values (fb0 = 1, fb = 2, fb2 = 3), same order, stamped by the last arriver as in run 113 (`stamp_group`, l. 488). Only the arrival count changes (VS instead of G x VS). |
| C5 norms once per team | Same 16-term serial in-lane sum (j = 0..15, mul then add; libxe builds with `-ffp-contract=off`, CMakeLists l. 78), same butterfly. After today's butterfly every lane of a row holds the same bits (fp add is commutative), so one team's result is each row's result. Same rsqrt / scale expression. |
| C6 fp32 qh / kh | The same single multiply `bf2f(x) * s` that today's lanes do in registers, stored as exact fp32. |
| C7 gates once per (head, token) | Same expressions on the same inputs, so the same instructions (`__expf`, `__logf` are deterministic). |
| C8 replay factors via SLM | A copy only: ST -> float as `ru_load8` converts (fp16 -> fp32 is exact). `eg = __expf((float)g_stored)` is today's expression. |
| C9 the update and output loops split | Each `h[j]` gets the same single update. `on` still sums `h[j] * q[j]` for j = 0..15 in order after h[j] is final, which is what the fused loop computes. |
| C10 GRF 128 vs 256 | Register allocation only. |
| C11 (stage 3) norm in the last arriver | The body of `ru_split_norm_kernel` is moved to a device function and called by both: s2 = on * on; `(s2[4i] + s2[4i+2]) + (s2[4i+1] + s2[4i+3])`; a serial 32-term sum from 0; `on * ((sc * nw) * act)`; f2bf; amax; codes. That is run 113's KG13-verified order. |

## 4. Shift marker, factor block and the run-271 race, per slice

**The race (run 271).** At VS 8 rows 32g..32g+15 and 32g+16..32g+31 are different work-groups. Both replay from
group g's k copy and (g, beta) pair at the top of the kernel. Today's code has the group's row 32g (`row & 31 == 0`,
l. 643) overwrite them with this step's token-0 factors inside its token loop. Work-groups are neither ordered nor
co-scheduled, so a late slice can replay this step's k instead of the previous step's: the state is wrong and the
error grows from there. At VS 16 four slices share a group.

**The fix.** No slice writes group-shared data in its token loop. The shared parts (all 4 group k copies and all 4
(g, beta) pairs, for every token t < T, for each of the G heads, in fb and in fb0 / fb2 when `dup_f`) are written by
**the last arriver on the key head's counter** after its own token loop. It already does the q|k conv rewrite and
the stamp there. It holds everything needed in SLM: `kh` (per key head, so the same for all G heads) and g / beta
for all G heads (the slice-group covers them; option B's main advantage). That is ~3 x 4 x 520 floats x T = ~25 KB
of coalesced stores at T 4, and ~1/3 of that without dup_f (the engine's linear pool).

Why it is race-free. Each WG arrives after its replay (arrival follows the conv barrier, which follows the replay and
the publish barrier). Its replay loads have completed by then, since their values were consumed and published
before the barrier. The arriver that sees `VS - 1` therefore knows every slice of every head of the key head has
finished reading the previous step's factors. Its writes are control-dependent on that atomic result. This is the
same read-before-rewrite argument that already guards the conv-state rewrite (relaxed atomic plus barrier), and it is
hash-gated in production. v rows are per row: written only by their own slice, read in replay only by their own
lane, and ordered by that WG's barrier.

**Marker semantics per slice:**
- Who reads: every WG reads each of its heads' `fb[RU_MARK_OFF]` at the top (l. 418) and derives `mrk` and
  `acol = 3 - mrk`.
- Who writes: only the key head's last arriver, after all VS arrivals (`stamp_group`, all G heads, the 1 / 2 / 3
  order).
- Why it equals the unsplit form: no WG can read a marker stamped by its own launch. So every slice sees the
  previous launch's stamp: the same `acol`, the same anchor slot `si`, the same anchor bytes. The VS 1 kernel gets
  the same from its read-then-barrier-then-stamp order.
- Shift algebra: unchanged (read at column 4, publish at column 2, never read column 0; vLLM's precopy invariant,
  l. 370-395 and 160-176, untouched). Step s+1 at page shift d reads stamp `2 + d`, so `acol = 2 - d`, which is step s's column 2.
- Per-slice marker *storage* is rejected. It would add bytes past RU_MARK_OFF, so the state hash would differ from
  today's kernel, and libref / libavx's AnchorLayout and docs/OPS.md would need changing.

**At every MTP depth, including a rollback.** Step s (T = 4: x0, d1, d2, d3) writes factors for t = 0..3. If step
s+1 has nacc = 2 (d2, d3 rejected), each slice replays t = 0, 1 onto the anchor read at `acol` and ignores t = 2, 3.
The VS 1 kernel does exactly this per row, so per row the state is the same sequence of operations from the same
anchor bytes. The conv side rolls back through `off = naccept - 1` (the history read offset). Every WG reads the same
offset, and the last arriver rewrites from that offset's history, as VS 1 does. Fewer drafts than the previous step
gives shift -1. That changes only which slot each column names, not who reads or writes.

**Uniform early exits** (barriers and the counter need them). `cslot < 0`, any anchor / factor slot `< 0`, and
`si < 0` are per-sequence. Make them WG-uniform before the first barrier. `si` is per head; under consistent markers
all heads agree. If they ever disagreed, today's code would leave a counter un-reset; the new kernel should treat
"any head's si < 0" as padding for the whole slice-group. T == 0 (fold-only) arrives after the publish, and the last
arriver stamps there (l. 496-508), with no factor writes.

## 5. Register / SLM budget (SIMD32; 128 GRF = 64 fp32 vectors a lane, ~56 usable)

| phase | live a lane (fp32 / dword vectors) | VS 8 / 16 |
|---|---|---|
| prologue | h 16 (state in flight) + q\|k conv inputs packed (x 8 + w 4 + hist 3 + bias 1 = 16) + v conv packed (~9) + replay-staging share (<= 9 at nacc <= 4) + addresses ~8 | **~58: spill risk** |
| replay (SLM factors) | h 16 + k 16 + ~4 | ~36 |
| staging compute | conv regs ~25 + h 16 | ~45 |
| token loop A | h 16 + kh 16 + ~8 | ~40 |
| token loop B | h 16 + qh 16 + ~8 | ~40 |

Ordering rule for the prologue: issue the replay-staging loads first and retire them to SLM as soon as they return,
before the conv inputs are consumed. Stage 1 keeps today's per-lane q / k / h (~58 in the token loop), so **build
stage 1 at GRF256** (no occupancy cost at N = 1: 768 sub-groups fit GRF256's 1,024 slots). Try 128 GRF from stage 2.
The rule: zero scratch in zeinfo (xe_kernel_props, K1's method), otherwise stay at 256. 128 GRF matters for N >= 2.

SLM a WG (static; RU_CVT 8; replay staged for 4 tokens):

| buffer | bytes |
|---|---|
| replay factors, 3 heads x (128 + NR + 2) x 4 tok x 4 B (aliased afterwards by cvq / cvk bf16, 4 KB) | 7.0 / 6.6 KB |
| qh / kh fp32, 2 x 8 x 128 x 4 B | 8 KB |
| cvh (q\|k history park, 10 x 256 x 2 B) | 5 KB |
| cvv, gates (g, beta, eg), sab | ~1.3 KB |
| **total** | **~21 KB**: VS 8 -> 2 WGs a core = 42 KB; VS 16 -> 4 a core = 84 KB |

Have the bench print `local_mem_size`. If 4 x 21 KB does not fit, the driver lowers co-residency and the profile
shows it. Then drop the SLM replay staging for a one-ahead register prefetch (+~1 us).

## 6. Expected time a call (E, T = 4, nacc ~3, N = 1)

| part | today | new (stage 2) | why |
|---|---|---|---|
| prologue chain | ~3 | ~2 | same 3 hops; factor staging and conv inputs overlap them |
| replay | ~3 | ~0.8 | one hidden load; ~0.25 us a token of compute |
| staging + barriers | ~1.5 | ~0.8 | 2-3 barriers over 6-12 sub-groups instead of 32 |
| token loop | ~21.5 | ~2 | ~110 instead of ~300 instructions a lane-token, 0 instead of 4 barriers, 2 serial 16-term chains |
| tail | ~1 | ~1 | last arriver: shared factors + conv rewrite + stamps |
| **update kernel** | **31.6** | **~7 (range 6-9)** | |
| ru_split_norm_kernel | -- | ~4 (M at VS 2 / 4: 3.9-4.4) + the recorded pass's dependent-launch gap | |
| **a call** | 31.6 | **~11 (stage 2)**, **~8 (stage 3: norm in a second last-arriver phase, ~1.5 us tail)** | |

Step: (31.6 - 11) x 35.9 = **-0.74 ms a rank-step** (-2.5 % of P110's 29.3 ms device), or -0.85 ms at stage 3,
about **+2-2.5 % tok/s** (~86 -> ~88 single-user). That clears note 138's 1.5 % rotation bar on paper only. Run
113's lesson applies: its bench gain vanished end to end, partly to the norm launch, which is why stage 3 exists.

## 7. Build stages, tests, profile rule

Each stage is its own commit, opt-in (`RADIANCE_XE_GDN_SPLIT=8|16`), and has a ~$80 cap (note 224).

**Stage 1: structure only (no math moved).** New kernel, option-B grid, conv at 1 / 2 channels a thread,
last-arriver shared-factor writes, token loop as today minus NORM (writes `onf`), the existing norm kernel. GRF256.
**Stage 2: math staging.** C5-C9 and the 128 GRF attempt.
**Stage 3 (only if stage 2's M-pair shows the norm launch).** A second counter block (or the upper half of ru_conv_ctr
with N x Hg x 2 <= 4096): a "done" arrival after the token loop, `atomic_fence(release, device)` before it and
`acquire` in the last arriver, which norms and quantises the G x T rows (C11). The release / acquire pair is new
here; today's counters rely on read-before-write only.

Tests, in order, each gating the next:
1. **Desk ISA (amd-host, ocloc for bmg_g31):** zeinfo shows zero scratch at the chosen GRF mode; record the instruction
   count of the token loop (stage 2 should be < ~half of today's).
2. **CPU row test (amd-host, new `xe_gdn_split_cpu_test`, CPU device):**
   - the new kernel at KT 8 (WGs of 192-384 threads; the CPU JIT's crash was at 1,024) against the run-113 kernel at
     VS 4 / KT 8 (256 threads), both instantiated in the test only (the CPU module's production KT 2 is untouched):
     o, codes, scales, state (anchor + factors + markers) and conv state byte-equal at T = 1 / 2 / 3 / 4, both forms;
   - the rollback schedule (item 3) byte-equal after every call;
   - libref's anchor form (`ref_gdn.cpp`) within tolerance, as a semantic check of the column algebra.
3. **Card bench (intel-host card1, one bounded step, `m3/intel-host/gdn-split.sh` extended to SPLIT 0 2 4 8 16):**
   `gdn_split_bench` gains:
   - T 1 / 2 / 3 / 4, each with its own timing line;
   - nacc 1-4;
   - **`anchor-seq`**, the rollback case. One state, six calls, hashed after each. Calls: (T 4, nacc 1, zeroed
     scratch = marker 0); (T 4, nacc 2: two drafts rejected, slide 0); (T 2, nacc 4, slide +1); (T 4, nacc 1: the
     drafts all rejected, slide -1 after the shorter window); (T 0 fold-only); (T 3, nacc 2). These are ref_test's
     acceptance 1, 2, 3, 1 and slide 0, +1, 0, -1 pattern plus the fold-only and fresh-block edges;
   - N = 2 with one padding sequence (cidx -1), to prove the counter still resets;
   - the six-call schedule repeated 200 times with a per-call hash, to show up any intermittent ordering race
     (stage 3 especially).

   Gate: **every hash equal to SPLIT 0 (today's fused kernel)** at every T, after every call, in both forms. Timing
   gate: stage 1 at T 4 (update + norm) <= ~0.85 x the VS 2 split at T 4; stage 2 <= 14 us; stage 3 <= 10 us.
4. **Card row check / profile first (one M-step, `svcmtpx-...-gs16-prof` on the current image + the step's
   libxe.so):**
   - must **disappear**: `r4d_gdn_recurrent_update_kernel<1,2,1,0,1,0,0,1,8>` (the fused int8 form, 35.9 a step);
   - must **appear**: `ru_sg_update_kernel<16,...>` at 35.9 a step, plus `ru_split_norm_kernel<2>` at 35.9 a step
     (stages 1-2; absent at stage 3);
   - must **not** appear: `gdn_conv_update` / `r4d_gdn_conv_*` (the fold must hold), new memsets or copies;
   - GDN rows 1.14 -> <= 0.45 ms a step; every other row within noise.
5. Then texts (the M-pair texts identical to the current reference), the M-pair, the rotation (note 138: >= 1.5 %
   median to propose a default). **The default is the owner's call.** MTP off (per_candidate) rides along in the
   bench (ANC = 0 works unchanged), but routing it is a separate decision.

## 8. Risks and stop rule

- **R1 (main): the fixed chain is the floor.** If the dependent prologue (sidx -> marker -> state) and kernel
  ramp / drain are ~5 us rather than ~3, stage 2 lands at ~13-15 us, not 11. The VS 2 -> 4 regression says latency,
  not throughput, rules this kernel. Check: the bench's T 1 line minus the token-loop slope gives the fixed part.
- **R2: exactness from moved math (C5-C7).** IGC could lower `__frsqrt_rn` / `__expf` differently in a new context.
  Stage 1 moves no math, so a stage-2 hash diff isolates C5-C9. Bisect by moving one item back into the loop at a
  time.
- **R3: spills at 128 GRF** in the prologue (~58 vectors live): stay at GRF256 for N = 1. The gain does not depend
  on 128 GRF.
- **R4: SLM at VS 16** (84 KB a core): fall back as in section 5.
- **R5: stage 3's release / acquire** on Xe2 (L1 not coherent): the 200x schedule replay is the stress. If it is
  ever flaky, stay at stage 2.
- **R6: tail on one WG a key head** (~25 KB of stores + conv rewrite): if the profile shows it (bench T 4 minus the
  model), spread the shared-factor writes with a per-(n, hv) counter, so each head's last slice writes its own
  head's parts.

**Stop rule:**
- **Exactness:** a hash diff that survives two fix attempts in a stage -> revert that stage, record it, stop.
- **Stage 1:** if it misses its timing gate, record and stop. Mapping plus conv staging alone does not pay.
- **Stage 2:** above 14 us at T 4 -> record and stop, no M-steps.
- **End to end:** an M-pair gain under +1 % after stage 2 (or 3) -> no rotation; record as not adopted, as e35db12.
- **Budget:** each stage stops at ~$80 whatever its state.
