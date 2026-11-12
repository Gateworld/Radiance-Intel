# radiance — Intel Arc Pro B70 port

This repository is a fork of **[radiance](https://codeberg.org/StillDeadcode/radiance)**, the LLM inference engine by
**StillDeadcode and the radiance contributors**. All credit for the engine, its design and its AMD backend goes to them.
It is licensed under the Apache License 2.0, like the original (see `LICENSE` and `NOTICE`). The original README is kept as [README-radiance.md](README-radiance.md).

I only modded it for Intel: this fork adds a backend for two Intel Arc Pro B70 GPUs (Xe2, SYCL / Level Zero), in
`libxe/` and related changes, tested with a Qwen3-Next-style model in tensor-parallel decode and prefill.

Most of the Intel port's code, tests and notes were written with **Claude (Anthropic)**, running as Claude Code
sessions, with me directing the work and making the decisions.

## What is here

- The engine source with its full history, including the upstream commits. My own commits are on top.
- `intel-port-docs/`: the working notes from the port, kept as they were written:
  - `OWNER-NOTES.md`: the decisions and instructions, numbered;
  - `STATE.md`, `WAITING-FOR-OWNER.md`: the running log of measurements and open questions;
  - `briefs/`: design notes and the kernel scoreboard;
  - `scripts/`: the build, image and test-step scripts used on the two machines.

Host names, paths and the names of unrelated services were replaced with neutral placeholders
(`amd-host`, `intel-host`, `~`, `assistant`, `app`, `database`). Old commit hashes mentioned in the notes are mapped
to this repository's hashes in `intel-port-docs/COMMIT-MAP.txt`.

## Model

All numbers here are for **Qwen3.8-Flash-Next**, from the checkpoint
[tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ](https://huggingface.co/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ)
(model type `qwen4_exp`, under the Qwen Community License 1.0; the weights are not part of this repository):

- 48 layers, hidden size 2,560, vocabulary 248,320;
- hybrid attention: Gated DeltaNet (linear attention, 16 key / 48 value heads) in three of every four layers, and full
  gated attention every fourth layer (24 query heads, 2 KV heads, head dim 256);
- mixture of experts: 512 routed experts, 10 active per token (expert width 640), plus a shared expert;
- 4 hyper-connection residual streams;
- one multi-token-prediction (MTP) layer, used as the draft head for 3-token speculative decoding.

On the B70s the engine converts it to its own `.rad` format: routed experts as 4-bit non-uniform codes with int8
activations ("format C"), the dense trunk as int8 weights × int8 activations, the draft head's projection at 2 bits,
split tensor-parallel across the two cards.

## Status (November 2026)

All figures are on 2× Intel Arc Pro B70, single user (one request at a time), with 3-token MTP speculative decoding:

| Measurement | Result |
|---|---|
| Decode, fixed test prompt | about 91–93 tokens/s |
| Decode, 11-prompt mixed set, greedy (temperature 0) | median about 112 tokens/s |
| Decode, mixed set, sampled (temperature 1.0, top_k 20, top_p 0.95), MTP coupling on | median 110.5, mean 108.0 tokens/s |
| Prefill, 5,604-token prompt | about 3.1 s to the first token |
| Prefill, 20K-token prompt | about 6.4 s |

The notes record what worked, what did not, and why.

### Recent changes (11 November 2026)

- **MTP coupling on sampled traffic.** Upstream's coupled speculative sampling (`RADIANCE_MTP_COUPLED=1`) accepts a draft
  token whenever the target's own sampled draw matches it under a maximal coupling. That is more often than requiring the
  draft to be the target's top token. On the sampled mixed set it raised tokens per step from 2.81 to 3.00 (+6.4 %) and
  mean tokens/s by +4.9 %. Greedy requests are unchanged by design (measured +0.1 %, identical texts). Two Intel-side
  fixes made the gain show up in tokens/s:
  - the coupled state is read back with one dense copy; a 2D copy made the host wait about 0.3 ms per step;
  - an opt-in cooperative verify (`RADIANCE_XE_MTP_COOP=1`) spreads its FP64 `exp` work over a work-group and keeps the
    sums in the serial order, so its tokens are bit-identical to the serial form.

  Both are being made the default. With coupling on, the sampled distribution is exact, but a request with a fixed seed
  no longer reproduces the tokens it gave without coupling.
- **Multi-user stall fix.** Under memory pressure, a failed command-list append on one card could hang the two-card step
  forever. Three changes fix it:
  - the engine now sets 64 KiB command buffers; the driver's default 1.13 MiB per list cost about 3 GiB of VRAM in
    recorded lists;
  - the drain after a failed append is bounded;
  - a failed step aborts on both cards together, and the requests get an error within about 6 s.

  Known limitation: after such a failure, the engine's exit is ended by a 300 s watchdog and can log GPU read faults
  while it tears down.
- **Tried and not adopted:**
  - an fp16 GDN state cache: it fails the accuracy bar on long contexts;
  - three retrained MTP draft heads: no gain on held-out text;
  - draft depth 2 or 4: depth 3 stays.

**Hardware note:** both cards run at **PCIe 4.0 x8** (about 14 GB/s measured for a card-to-card copy), the limit of the
board's slots. The B70 itself supports PCIe 5.0 x16, so these numbers are link-limited in two places: prefill streams
the experts that don't fit in VRAM over this link, and every decode step's all-reduce between the two cards crosses it.
Results on a PCIe 5.0 x16 platform should be better, prefill especially.
