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

## Releases

Release tags carry an `-intel` suffix, because upstream radiance already uses `v1.0.0` through `v1.12.0`.

- **`v1.0.3-intel` (13 November 2026): the current release.** It runs as the production service. It adds two things on
  top of `v1.0.2-intel`, and text output is bit-identical to `v1.0.2-intel`:
  - **An encoder-output cache for repeated images (opt-in).** On this hybrid model, a prefix-cache hit only reaches the
    last linear-state checkpoint, so a ~1,400-token image prompt has no reusable snapshot, and the same image was encoded
    again on every request (about 22 s per 1024×1024 image). The engine now keeps the vision encoder's output, keyed by
    the image's content key and its geometry, least recently used within a byte cap. It is set with
    `RADIANCE_MM_ENCODER_CACHE_MIB`: the default is 0 (off) in the engine, and the service script sets 512.
    - **Replay of 122 image requests about the same image:** 1 encode and 121 cache hits (under 1 ms each). All 122
      were answered in 35 minutes; before, 76 were answered in 50. The median time to the first token fell from 104 s
      to 32 s; what is left is queueing behind the client's own concurrent requests, not the engine. The answers scored at
      least as well as the AMD service's.
  - **The step barrier's 10 s warning is quiet while the engine is idle.** At idle the rank threads wait at the barrier
    for work, so the warning printed about once a minute without anything being stalled. A step that really stalls still
    warns.
  - **Validation:** the fixed test prompt gives texts identical to `v1.0.2-intel` (93.0 tokens/s), prefill log-probs
    are bit for bit the same, a 2-hour soak with long prompts in the mix ran clean (2,278 requests, 90 of them 18.7K
    tokens, no errors and no stalls), and a 30-minute run on the exact service settings ran clean (581 requests).
- **`v1.0.2-intel`:** it adds two things on top of `v1.0.1-intel`:
  - **A fix for a hang under long prompts.** The Level Zero backend reuses a ring of event slots. A command's
    device-side wait on an earlier slot was not counted as a user of that slot, so the host could reset the slot and
    reuse it while the wait was still pending. The wait then waited on a later command. During long prefills
    interleaved with decode, this could form a cycle and hang both cards with no fault. Every such wait is now
    registered, and a slot is only reused once its waiters have run (`RADIANCE_L0_RING_GUARD`, on by default).
    - **Before the fix:** a 2-user mix with an 18.7K-token prompt every 25th request hung after 35 minutes.
    - **After the fix:** the same mix ran 90 minutes clean (1,703 requests, 68 of them long prefills), with no speed
      cost.
  - **Image input.** The Intel build now includes the FFmpeg media decoder (FFmpeg 6.1 and 7 both build). The vision
    encoder's operations match the reference implementation (601 of 601 kernel checks on the real weights), and a
    labelling evaluation matched the AMD service's answers on every item it reached. Encoding a 1024×1024 image takes
    about 22 s; that is the next thing to speed up.
- **`v1.0.1-intel`:** it fixes a rare engine stop under concurrent load. When a
  step took back the activation-arena memory lent to the expert cache, an expert copy could still be in flight into a
  lent slot. The slot was then left marked as held, and the engine stopped. The take-back now waits, bounded at 10 s,
  until every such copy has landed. Served tokens are unchanged.
- **`v1.0-intel`:** the first release. It made MTP coupling and the cooperative verify the defaults, and set 64 KiB
  command buffers. The API key is read from `RADIANCE_API_KEY`, never from the command line.

How `v1.0.1-intel` was validated, running as a service on 2× B70:

- **2-hour soak with 3 concurrent users:** 4,128 requests, all answered (median 5.2 s, p90 7.4 s). One engine process ran
  throughout with no restarts and no GPU faults; the GPUs peaked at 68 °C. `v1.0-intel` had stopped at 19.5 minutes on
  the same soak.
- **Feature checks:**

| Feature | Result |
|---|---|
| Tool calls | pass |
| Reasoning on / off (reasoning returned in its own field) | pass |
| 18K-token prompt, fact retrieval | pass (25.9 s) |
| 64K-token prompt, fact retrieval | pass (95.1 s) |
| Image input | **from `v1.0.2-intel`**: works and matches the AMD service; about 22 s of encoding per 1024×1024 image (a repeated image is served from the encoder cache from `v1.0.3-intel`) |

## Status (November 2026)

All figures are on 2× Intel Arc Pro B70, single user (one request at a time), with 3-token MTP speculative decoding:

| Measurement | Result |
|---|---|
| Decode, fixed test prompt | about 91–93 tokens/s |
| Decode, 11-prompt mixed set, greedy (temperature 0) | median about 112 tokens/s |
| Decode, mixed set, sampled (temperature 1.0, top_k 20, top_p 0.95), MTP coupling on | median about 111, mean 108.7 tokens/s |
| Prefill, 5,604-token prompt | about 3.1 s to the first token |
| Prefill, 20K-token prompt | about 6.4 s |

The notes record what worked, what did not, and why.

### Recent changes (11–12 November 2026)

- **Shorter coupled verify.** The verify now stops after each row's last live candidate instead of walking the full
  1,024-slot plane. Results are bit-identical, and sampled decode is about 0.5 % faster.
- **Opt-in, measured but not adopted:**
  - an all-reduce block-count floor (`XE_AR_MIN_BLOCKS`): faster in isolation, no gain inside the engine;
  - the gate_up epilogue fold (`RADIANCE_XE_GU_EPI=1`), which folds SwiGLU and int8 quantisation into the expert kernel:
    its card tests are pending.

- **MTP coupling on sampled traffic.** Upstream's coupled speculative sampling (`RADIANCE_MTP_COUPLED=1`) accepts a draft
  token whenever the target's own sampled draw matches it under a maximal coupling. That is more often than requiring the
  draft to be the target's top token. On the sampled mixed set it raised tokens per step from 2.81 to 3.00 (+6.4 %) and
  mean tokens/s by +4.9 %. Greedy requests are unchanged by design (measured +0.1 %, identical texts). Two Intel-side
  fixes made the gain show up in tokens/s:
  - the coupled state is read back with one dense copy; a 2D copy made the host wait about 0.3 ms per step;
  - an opt-in cooperative verify (`RADIANCE_XE_MTP_COOP=1`) spreads its FP64 `exp` work over a work-group and keeps the
    sums in the serial order, so its tokens are bit-identical to the serial form.

  Both are now the default (`v1.0-intel`). With coupling on, the sampled distribution is exact, but a request with a fixed seed
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
