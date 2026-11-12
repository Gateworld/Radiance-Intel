# The v1 soak's arena-recall stop: the desk (owner notes 255 / 256)

2026-11-12, run 338 (agent). Part of the Intel port. The reproducer is staged, not yet run (it waits for the coordinator
to stop v1).

## 1. What happened

The 2 h soak (image 323, 3 users, the service's args) stopped twice, at 04:31:59 and 05:21:22, with no GPU fault:
- `E mover: recalled the activation arena loan to 1063264256 and a slot at 1072189440 is still held` (then 1096568832);
- `E rank 0: taking the arena back from the expert slab for a 456-token step: called in the wrong phase` (then 464 tokens);
- the step loop fails and the engine exits 1. systemd restarts it in ~155 s.

`Engine::arena_step` picks an arena level per step. A 456-token step needs a larger level than the decode steps around
it, so the slab must give back part of the loan: `Mover::reclaim(kLendArena, allow)`. Its last check
(`loan_held(lender) > loan_`, mover.cpp ~1500) found a held slot past the new loan and returned RAD_E_STATE. The
engine treats that as fatal.

## 2. How a recall works, and who can hold a slot

`reclaim()`:
1. `set_loan` lowers the arena segment's target, so every slot whose `need` exceeds the loan becomes uncovered.
2. It drains: `flush_moves`, sync the mover's stream, `retire`, sync compute, `retire`.
3. It frees every unit in an uncovered slot. A unit that kept its host copy changes address; a donor is copied into a
   fixed slot or the pool.
4. It sweeps the move records naming an uncovered slot: it frees the slot if `frees_slab()`, then clears the record.
5. It checks that nothing is held past the loan.

| who takes a slab slot | where | how it comes back |
|---|---|---|
| prefetch stage | `prefetch_to_op` → `start_stage` (Stage record) | `release_slot` (Release record) → quarantine |
| swap promote | `issue` → `promote_to` (Promote record) | the unit holds it after `retire`; freed with the unit |
| fill (lent slots) | `fill` → `promote_to` | the same |
| migrate (donor) | `migrate` (Migrate record, `to_slot`) | the old slot through the quarantine |
| recall donor | `reclaim` itself (a fixed slot) | -- (the old one goes to `freed`) |
| demote / drop / release | `issue` / `drop` / `release_slot` | the quarantine frees it (`frees_slab()` true) |

Every path frees its slot on its own terms, so reading alone finds no leak. Two cases do not fit the recall's
assumptions:
- **(i) A Promote record not retired when the recall sweeps.** Its slot is held, but the unit does not point at it
  yet. `frees_slab()` is false for Promote, so step 4 clears the record without freeing the slot. That is exactly a
  held slot past the loan with no owner. It needs the record's event to read not-done after the drains: an event
  recorded somewhere the syncs do not cover, or a record whose event was never recorded.
- **(ii) A unit promoted twice before the first promotion retired.** `fill()` filters on the heat engine's `busy`
  flag but not on `unit_in_flight()`. If anything clears `busy` while a promotion is in flight, a second promotion
  can be issued. The second `retire` overwrites `us.slab_slot`, and the first slot is orphaned: held, owned by no unit,
  and named by a record that is already cleared.
- (Stage records have the same blind spot as (i), but the soak's mover reported 0 staged.)

## 3. What only the soak had

- **The rotations post one request at a time** (m2-heat.sh). The soak ran 3 users against `--max-num-seqs 2`, so
  sequences join and leave decode batches and a third request waits in the queue.
- **The service args add `--busy-prefill-chunk 1024 --prefill-interleave 8`.** Prefill chunks interleave with decode,
  and the arena level changes with them.
- Either explains why a 450-token step recalls the loan mid-decode with promotions in flight; the rotations never
  produced that mix.

## 4a. What the trace found (V1R1, V1R2 -- run 340)

- **V1R1** (image 325, 3 users, the service's args, trace on) stopped after ~31 min, on a 350-token step: a new
  request's 348-token prefill admitted mid-decode.
  - The dump listed 12 held slots past the loan (the cap), class 7 slots 9836-9847.
  - **Every one was last taken by a swap promote of the current dispatch, and no unit and no record named it.** No
    double-promotion line appeared anywhere. So it is case (i): promotions still `Copying` when the recall's record
    sweep ran.
- **V1R2** (image 326, the trace extended to list the records still `Copying` right after the drains) ran its full
  90 min without a stop (2,674 requests). But **at every recall it listed records still in flight after the drains:**
  36 at the first, with events not done and nothing unflushed. They were promotions and demotions in budgeted slots,
  which the check ignores.
- **So `rad_stream_sync(mover_)` followed by `retire()` does not guarantee that the mover's copies have landed.** The
  stop happens when one of those in-flight promotions targets a lent slot past the loan. The same gap means such a copy
  can still be writing into memory the arena has just taken back.
- **Why the sync is short is open.** It is a Level Zero backend question about the transfer list under
  `RADIANCE_L0_GLUE_ALL=2`; `settle()` and `n_slot_wait()` look sound on reading. It is a separate item from this fix.

## 4b. The fix (v1.0.1 candidate, exact)

`Mover::reclaim`, after its drains:
- waits until no `Copying` record's copy touches an uncovered lent slot. It sleeps 50 µs, doubling to 1 ms, and calls
  `retire()` each round;
- is bounded at 10 s, after which it fails exactly as before.

Then every promotion into a lent slot is published, so the unit loop frees its slot. No copy into the returned memory
is in flight, and the record sweep has nothing to orphan. It only waits, so no served token changes. New counters,
`recall_waits` / `recall_wait_us`, appear in the mover's report.

The separate item, read at the desk (run 340, while V1R3 ran):
- The recall runs on the scheduler thread with every rank thread parked (`Engine::prepare_release` → `arena_step`,
  engine.cpp ~386). That thread never records a tape, so `z_stream_sync` → `enter()` does `settle()` the mover
  stream's pending slot. The transfer list (`ximm`) is created in order (l0.cpp ~1119), so that slot covers every
  glued copy before it, and `wait_and_throw` covers the runtime-queue copies and the barrier records.
- So reading does not show the gap. Not checked: whether a barrier event's SYCL status lags its queue's
  `wait_and_throw` on the v2 adapter, and whether anything other than the parked rank thread issues on the mover
  stream.
- Next instrument (when the item is picked up): in the after-drain dump, show for each record still `Copying` whether
  its event is a ring slot (`zdev/zslot/zserial`) or a runtime event, next to the stream's pending slot at the sync.

Gates (note 256 item 3):
- the reproducer twice, clean at its 90-min cap (V1R3, V1R4, image 327);
- the CPU tests;
- texts identical to M285;
- then v1.0.1 (tag and image), installed by the coordinator, and a fresh 2 h soak.

## 4. The instrument and the plan

- **`RADIANCE_MOVER_TRACE=1`** (commit ad9553c, image 325):
  - each slab take is tagged with its site, unit and dispatch;
  - a double promotion (case ii) is logged where it happens;
  - the failing check dumps each held slot past the loan with every unit and record that names it (kind, state,
    event done) and its last take. "NO unit and NO record names it" with a promote / fill tag points to (i) or (ii).
- **Reproducer V1R1:** m2-heat `v1svc-u3`, i.e. the v1 environment with the service's args, 3 concurrent users and the
  soak's 11 bodies; 90-minute cap. Then the A/B `v1svc-noint-u3`.
- **The fix, exact (no served token changes):**
  - for (i): the recall's sweep frees a Promote / Stage record's slot when the unit does not hold it, after the drains;
  - for (ii): `fill()` and swap promotions skip a unit with a copy in flight.
  - Then v1.0.1 and a fresh 2 h soak (note 256 item 3).
