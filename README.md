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

## Status (November 2026)

On 2× Intel Arc Pro B70, single-user decode with MTP: about 91 tokens/s; 7K-token prompt prefill: about 3.1 s.
The notes record what worked, what did not, and why.
