You are reviewing the whole Radiance-on-Intel effort for its owner, who is away. Write the review to
~/code/radiance-intel/briefs/2026-10-27-system-review.md (Markdown, at most 2,500 words), then send a
summary of at most 1,200 characters with: /usr/local/sbin/notify "radiance-intel: system review done -- <summary>".
If writing the file is refused, print the entire review as your final message instead (it is captured), then notify.
Read-only on everything else: no builds, no containers, nothing on the cards, no edits to any file but the review, no
git write commands, no commands on intel-host except `./intel-host-check.sh pre`. Light CPU only; the unattended runner shares this
machine and must not be slowed. Do not read logs/live.log (too large); use logs/run.log and logs/memguard.log.

Read: m2/2026-10-26-radiance-intel-m2.md (the last report); STATE.md (Step, run log, card steps, faults, decisions,
blockers, Next); OWNER-NOTES.md (notes 28-43 matter most); PROMPT.md (targets); progress.tsv; run.sh,
intel-host-engine-guard.sh, UNATTENDED.md; src/ on branch intel-l0: core/device/l0.cpp, core/runtime/ctx.cpp,
libxe/xe_allreduce.cpp, core/sched/scheduler.cpp (RADIANCE_SPEC_MAX_SEQS), libxe/test/.

Facts already gathered (use them; do not recompute): 21 commits, +2,174/-196 lines since milestone 3 began (4,211
since milestone 1 ended); 17 test programs in libxe/test, 27 cmake targets; milestone 3 so far: 39 runner ticks, 3
failed ticks (a model quota), 3 pauses (2 kill artifacts of heat stops, 1 device OOM under concurrency), 3 GPU fault
lines, 13 heat stops with a maximum of 84 C on a host-reference oracle; $234 of the $1,000 budget. Targets now:
decode 1-prompt 35.7 of 60 tok/s (MTP off); rotation median 37.5 of 40 (MTP on); aggregate at 6 users 72.4 of 120;
MTP single prompt 42.0 (beats the 36.6 bar); first token 53 s against 5 s (stage B not started); recorded-pass pool
2.5 of 1 GB a card; replay misses 8.3 of 4 a step (stage C not started). Production moved to 1.11.0 today; prompt 10's
first-token divergence is the port's (note 40). Open bugs: the C6 stall (note 42 lists four recommendations), the
1.6 GB VRAM part of the pool, MTP losing from 3 users unless gated. Calibration: a 14 GB dense 27B serves at 58-60
tok/s on one B70 via vLLM+EXL3 (note 33); amd-host's production pair does 218 tok/s on this model.

Sections, in this order:
1. Executive summary: what works, what does not, where the numbers stand (10 lines).
2. Code and backend: robustness for a 24-hour multi-user soak; the five most important risks, each with file:line,
   a concrete failure scenario and a specific fix.
3. Measurement and method: what is sound (oracle gates, both measurements, overlapped profiles), what is missing;
   three gaps with the cheapest closure each.
4. Runner and supervision: run.sh, the fault judge, the heat guard, the owner notes; the pause and heat policies; the
   permission friction (refused commands); what to change.
5. Platform limits on intel-host: the failing heatsink, PCIe Gen4 x8, the DRAM-less NVMe and the n-gram table, RAM, no FP8
   on Xe2 -- which bind today, what each costs in measured terms, and which hardware change would buy what.
6. Plan: is the remaining order (stage A kernels, B prefill, C mover, D service + soak, report) right; what to cut,
   reorder or add; a realistic end state (single prompt, rotation median, aggregate at 6, first token at 7K) with
   the reasoning, stated as ranges.
7. Strategy for the owner: three options -- finish the port as planned; stop at a documented port and serve a
   B70-shaped model such as the 27B profile on these cards; buy AMD -- with what each buys and costs; your
   recommendation and the measurements that should decide it.
8. The top ten recommendations, ranked, one line each, naming who acts (agent / owner / supervisor).
Be concrete and honest; cite file:line, run and step ids, and numbers; no generic advice; do not restate the brief.
