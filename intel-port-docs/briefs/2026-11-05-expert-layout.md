# A load-time layout for the format C experts at decode: a design brief (owner note 188 item 2)

2026-11-05, run 190 (agent). Part of the Intel port. **Desk work only: nothing is built and no card time was used.**
The goal it serves is 1-user decode toward ~80 tok/s. The numbers come from profiles already on record:
- P87: the decode step by kernel;
- P88: the down kernel's byte floor;
- P91-P95: the run kernel's ablations;
- P96-P98: the exact rewrites that did not move it.

## 1. The cost today

The format C expert GEMM at decode is `moe_i4i8_run_kernel<.., FC>` (src/libxe/xe_moe_i4i8.cpp). Each layer runs
two launches, gate_up then down, each covering every expert routed to the card in that step (about 27 at P87).

| per layer launch (P87) | time | floor at 608 GB/s | over the floor |
|---|---|---|---|
| gate_up | 114.7 us | ~29 us | ~3.9x |
| down | 67.3 us | ~14.7 us (P88) | ~4.6x |
| both, x 48 layers | **~8.7 ms a step** | ~2.1 ms | **~6.6 ms** |

The whole decode device step is ~36 ms a rank (P87). The format C pair is its largest item: the kernel
scoreboard's decode row 1.

## 2. Why it sits at ~4x (item a)

### 2.1 The ablations

Each ablation removed one stream of work and kept the rest. Times are per launch, on image 216, which carries
4-9 us of the ablation switch itself.

| removed | gate_up | down | what it is |
|---|---|---|---|
| nothing (control) | 123.1 | 70.2 | |
| the activation staging (abl1) | −35.7 | −20.4 | each work-group copies its rows' int8 activations into LDS, split into even and odd lanes |
| the weight loads (abl3) | −28.0 | −22.4 | the code bytes themselves |
| the level lookups (abl2) | −10.9 | −10.1 | 16 LDS gathers per 16 bytes of codes, a row |
| the scale loads (abl4) | −4.0 | −1.2 | one E4M3 byte per (row, 64 of K), each lane's own gather |

**Two thirds of the time is not the weight bytes.** With the weight loads gone, down still takes 47.8 us, against a
14.7 us floor. The weight stream itself is near its floor for gate_up (28 against ~29) and ~1.5x for down (22
against 14.7).

### 2.2 Where the rest goes, and why the layout causes it

1. **The activation staging (the largest item, ~29 % of the launch).**
   - The code bytes hold elements 2i and 2i+1 in the low and high nibble of byte i. A byte looked up in the
     256-entry table yields two int8 levels, one even element and one odd.
   - To meet them with DP4A, the activations must be split into even and odd words (`a_even` / `a_odd`). The kernel
     does that once per work-group, into LDS, behind a barrier.
   - A run's columns are split over several work-groups (down: 5 column blocks). Each one stages the same rows
     again and rebuilds the same table.
   - **The de-interleave is forced by the nibble order.** With any order that puts consecutive elements in
     consecutive lanes of a word, the activations could be read straight from global memory. They are contiguous
     int8 rows, shared by every work-group of the launch and L1/L3-resident, so no staging and no barrier would be
     needed.
2. **The load pattern (down especially).**
   - Codes are row-major `[N, K/2]`. A team of LPR lanes reads consecutive 16-byte chunks of *one* row.
   - Down's instance (LPR = 2, R = 2, two chunks a unit) has a 16-lane sub-group reading 16 different rows per load
     instruction. That is 16 cache lines touched for 512 bytes, half of each 64-byte line per request; the other
     half is fetched by a later unit.
   - Gate_up's instance (LPR = 16) reads 256 contiguous bytes per row and is near its floor, as abl3 shows.
3. **The level lookups (~10 us).** These are 16 LDS gathers per 16 bytes, per row, in every work-group.
   - Run 176's register select chain, which keeps no table, was much slower: 167 / 166 us. The gather is the cheaper
     form of the lookup, and it stays.
   - The table *build* (two 16-level amax loops and 256 rints per work-group) can go: the table is a function of the
     entry alone.
4. **Fixed per-work-group work.** The binary search for the run, the routing reads, the table build and the
   end-of-row shuffle tree, all spread over a decode run of 1-4 rows.

**Why the exact rewrites didn't move it (runs 175-176).**
- Issuing the loads earlier, dropping the unused sums, and the select chain all kept the layout. So they kept the
  staging, the de-interleave and the load pattern.
- The ablations' savings come from removing whole request streams.
- Note 167 concluded "only fewer bytes or another layout would". Fewer bytes (mixed bits) failed the bar. This
  brief is the other.

## 3. The layout (item b)

A **load-time repack of each expert's format C planes**. It runs through libxe's relayout hook (RadRelayoutFn), the
way format C's joined form is already built at load. The values are unchanged, and nothing is converted in the
file.

| what | today | the tile layout |
|---|---|---|
| codes | row-major `[N, K/2]`; byte i = elements (2i, 2i+1) | **tiles of 16 rows x 64 K (one scale group): 512 contiguous bytes**, row r's 32 bytes at 32r; inside each 4-byte word, the low nibbles hold elements k..k+3 and the high nibbles k+4..k+7 |
| scales | E4M3 `[N, K/64]`, a byte per (row, group) | the tile's 16 E4M3 bytes right after its 512 code bytes (one 16-byte read a tile) |
| the level table | 16 f32 levels an entry; every work-group builds the 256-entry int8 pair table | the same 16 levels, plus the **512-byte pair table precomputed at load** (the same `rint(level x 127 / amax)` in f32), copied to LDS by a plain 512-byte load |
| activations | staged into LDS, even and odd split | **read directly from global**: a group's 64 bytes a token, one uniform (broadcast) load a sub-group |

**The kernel this enables (decode, 1-4 rows).**
- One 16-lane sub-group owns one tile column (16 rows). Each lane owns a row and walks its row's groups.
- Each group step does:
  - one 32-byte code load a lane, so a sub-group reads 512 contiguous bytes, 8 whole lines;
  - one 16-byte scale load a sub-group;
  - 16 LDS gathers a lane against the per-entry table;
  - one broadcast activation load per token.
- There is no staging, no barrier and no per-work-group table build.
- Per card the experts' intermediate is split over the two cards. Gate_up is N 768 or 512 by K 2,560, for the two
  parity classes. Down is N 2,560 by K 384 or 256. These are the instances P91 timed: gate_up `<UPL 5, LPR 16, R 1>`
  and down `<UPL 3, LPR 2, R 2, GL>`.
- Gate_up has 32-48 sub-groups an expert, and down 160. At about 27 experts a launch the card stays full at 1-4
  rows.

**What stage 1 built first (run 190, commit 37411b7): form T1.** It keeps today's lane mapping and tree line for
line and changes only the data sources:
- the nibble order (byte b = elements b and b + 4 of each 8-element block);
- the activations read from global memory in their natural order, with no LDS staging;
- the pair table and scale factor precomputed per entry.

The codes stay row-major. So T1 isolates the largest ablation item (the staging, abl1) and the table build. The
16 x 64 tiles and the one-lane-a-row mapping are the next form if T1's gain is real but down's loads remain. T1 is
bit-identical on the CPU device in all four cases (gate_up and down, at 4 tokens and at 1).

**Exactness: the same bytes out, by construction, if one rule is kept.**
- The integer part is order-free: each (row, group) sum is the same exact int32 sum of the same int8 products.
- The float part is not. Today a row's `acc += (fs * asc) * sbr` runs per lane over that lane's groups, in u order,
  and the lanes are then joined by a fixed xor-shuffle tree (LPR lanes). A kernel that walks a row's groups in one
  lane, in ascending order, sums the same terms in another order. That changes low bits.
- **The rule:** the new kernel keeps the old virtual-lane partial sums and the same tree. It does so with registers
  per virtual lane, or with a short in-lane tree that reproduces the old one. Only where the bytes come from
  changes. This is the stage-1a technique (same values, same order, different owner).
- If that proves too costly, the fallback is "exact to libref's order". That would be a new reference for the
  M-pair, the owner's call, and it is not proposed here.

## 4. The expected gain, the cost, and the risks (item c)

**The gain, reasoned from the ablations.**
- The staging goes: −35.7 / −20.4 us a launch.
- The direct activation reads cost back an estimated +3-5 us: L1 hits, one broadcast load per token per group.
- Down's loads move toward their floor: −5 of abl3's 7.7 us excess.
- The table build goes: −2-4 us, part of the fixed cost.
- The lookups stay.

**Estimate: gate_up ~114 → ~75-80 us; down ~67 → ~38-42 us; about −60 us a layer, −2.9 ms a step (48 layers).**
- At one user (M262 run 2: 70.2 tok/s, ~44 ms a step), that is **~+6-7 %, ~74-75 tok/s**.
- It is not 80 on its own. The rest of the way is mostly outside these kernels: the ~10 ms a step between device time
  and the step wall (note 177 item 6's launch path), and the hc family (megakernel 1b).
- If the tile kernel also reaches its floor, the full 6.6 ms is ~+15 %. That would take the lookups going too, and
  nothing on record says they can, exactly.

**The cost.**
- **VRAM: none for codes and scales** (a permutation in place). Plus 512 bytes an entry for the pair table: 512
  experts x 2 matrices x 48 layers ≈ 25 MB a model, ~12 MB a card.
- **Load time:** the repack runs in the relayout hook as the experts are copied out of the file (into the host pool
  and VRAM), ~62 GB of expert codes. A threaded host permute at several GB/s adds an estimated 10-30 s to today's
  ~115 s start. Stage 2 measures it.
- **The prefill path:** the matrix-unit expert kernel (`moe_i4i8_dpas_kernel`) reads the same planes. It must read
  the tiles too: its B tile staging is the place, and the change is small there, since it already stages through the
  level table.

**The risks.**
1. **One stored weight, every reader.** libref and libxe share the canonical joined form under one tag (rad.planes),
   so a weight stored once serves both libraries' bands.
   - The tile layout needs its own tag. Then **every band of the expert op must be libxe's**: decode, prefill and the
     KL mode. Otherwise the layout hook must decline the weight and leave it canonical.
   - The plan below makes that a startup check that refuses the layout, never a silent fallback.
2. **Moves and stages are byte copies of whole entries** (the mover, the prefill stager, the host pool). They don't
   care about layout. The tail skip reads only the entry's address. To be confirmed in stage 2 with the M-pair
   under a full rotation (the expert moves exercised).
3. **The exactness rule's cost.** Reproducing the old tree may need more registers per lane, and occupancy then
   decides. Stage 1 measures both forms: tree-faithful, and plain ascending-order for an upper bound.
4. **The estimate rests on the direct activation reads being cheap.** If L1 broadcast loads cost what the LDS staging
   cost, the gain shrinks to the load-pattern and table-build part: ~1 ms a step. Stage 1 answers this first and
   cheaply.

## 5. The staged plan

| stage | what | measure | cost | owner |
|---|---|---|---|---|
| 0 | this brief | | done | reads |
| **1** | **a standalone timing on one expert of each shape, no engine change:** a probe (in the manner of run 111's i4i8_probe.cpp) that holds one format C expert at the per-card shapes (gate_up 768 x 2,560, down 2,560 x 384) in both layouts and times today's run kernel against the tile kernel at 1-4 rows and ~27 experts a launch. Device timestamps on card1, 100 reps, the output bytes compared. | per-launch us both ways, bit-identical output (tree-faithful form), the ascending-order form's time as the upper bound | ~$40-60, ~15 min of card1 | **go / no-go: ≥ 25 % off the launch, bit-identical** |
| 2 | the relayout hook (its own tag, the startup refusal when another library holds a band), the run kernel on tiles behind `RADIANCE_XE_FC_TILE=1`, the prefill DPAS kernel's B staging on tiles | the M-pair texts identical, the KL 24 stable documents identical, the profiled step's per-launch times, prefill 7K / 20K unchanged, the load time | ~$80-120 | reports after |
| 3 | the rotation (H-run) against today's default; the default at ≥ 1.5 % on the median | median, tokens a step, misses | ~$10 | owner decides the default |

**Stage 1 decides cheaply whether the staging-free read is real** (risk 4), before anything touches the engine,
the load path or the plane tags.
