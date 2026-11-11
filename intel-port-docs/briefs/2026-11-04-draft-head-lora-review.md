# Draft-head self-distillation: the LoRA session's review (owner note 181), for the runner's dump

2026-11-04 17:30 UTC, relayed by the supervisor (radiance-test-54) from the LoRA session (radiance-test-81), who read
the `mtp.*` tensors' index and headers in /srv/bulk/archive/qwen3.8-flash-next-bf16 (nothing loaded on a card).
Corrections to briefs/2026-11-04-draft-head.md section 3:

1. **The head reads all 4 hyper-connection streams, not one 2,560 row.** `mtp.pre_fc_norm_hidden` is [10240]
   (hc 4 x 2560) and the hc mixer's input_mix_* is 10240 <-> 320. So the dump must record whatever the engine
   actually feeds the head (the 4 streams, before or after the final mix / norm -- say which, with the code
   reference): 10,240 bf16 = 20 KiB a position, **~20 GB for 1M positions** (not 3-6 GB). Keep >= 50 GB free.
2. **The head is not small:** its block has an MoE MLP (512 experts, top-10, plus a shared expert), so `mtp.*` is
   2.61 B parameters, 2.52 B of them experts. Plan: experts frozen; train the ~0.09 B non-expert part (gated
   attention, fc_hidden / fc_embedding, the 3 hc mixers, router, shared expert, indexer) or a LoRA on it. The
   embeddings and lm_head are the trunk's (tie_word_embeddings false, mtp_use_dedicated_embeddings false), frozen.
3. **No PyTorch reference exists** (model_type qwen4_exp has no transformers class; the archive ships no modeling
   code), so the LoRA session writes the head's forward itself and needs ground truth from the engine:
   - **dump the engine's own MTP draft logits (top-32 ids + probabilities) for a few hundred positions** beside the
     inputs, for a bit-close parity check of the port before any training;
   - say what the engine does for the head's sparse indexer at <= 2,048 tokens of context (if the budget covers the
     whole context, training uses sequences <= 2,048 with dense causal attention);
   - the exact hc formulas the head uses (mix, norm, inject), with code references (src/arch/...).
4. **Data layout:** whole sequences in order with their token ids (the block attends over earlier positions); the
   teacher = the served model's top-32 at t + 2, renormalised; depth 1 first.
5. **Slot:** after the data + README exist and the port check passes, and after 3a's LoRA-A battery; ~3 h on amd-host
   cards 2,3 (start.sh pauses / restores qwen27b, textsearch, vlembed); earliest 11-06. Compute ~30-60 min per 1M
   positions on one card. Half a day of CPU work on the port first.
