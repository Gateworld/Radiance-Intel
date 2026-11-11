# radiance — Intel Arc Pro B70 port

This repository is a fork of **[radiance](https://codeberg.org/StillDeadcode/radiance)**, the LLM inference engine by
**StillDeadcode and the radiance contributors**. All credit for the engine, its design and its AMD backend goes to them.
It is licensed under the Apache License 2.0, like the original (see `LICENSE` and `NOTICE`). The original README is kept as [README-radiance.md](README-radiance.md).

I only modded it for Intel: this fork adds a backend for two Intel Arc Pro B70 GPUs (Xe2, SYCL / Level Zero), in
`src/libxe/` and related changes, tested with a Qwen3-Next-style model in tensor-parallel decode and prefill.

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

On 2× Intel Arc Pro B70, single-user decode with MTP: about 91 tokens/s; 7K-token prompt prefill: about 3.1 s.
The notes record what worked, what did not, and why.

**Hardware note:** both cards run at **PCIe 4.0 x8** (about 14 GB/s measured for a card-to-card copy), the limit of the
board's slots. The B70 itself supports PCIe 5.0 x16, so these numbers are link-limited in two places: prefill streams
the experts that don't fit in VRAM over this link, and every decode step's all-reduce between the two cards crosses it.
Results on a PCIe 5.0 x16 platform should be better, prefill especially.
