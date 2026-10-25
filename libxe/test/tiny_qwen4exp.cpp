/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * tiny_qwen4exp.cpp -- a small random Qwen3.8-Flash-Next (qwen4exp) checkpoint in the layout
 * rad-convert reads, written the way tests/tiny_model.h writes its tiny models (write_qwen35 is the
 * template; tensor names and config keys from arch/qwen4exp_fp8, see m1/qwen4exp-tiny-notes.md).
 * Converted with data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe it is a .rad in production's formats
 * (w4nl64a8h experts, int8 trunk, E4M3 hyper-connections), at production's per-head geometry
 * (head_dim 256, 24 q / 2 kv heads, GDN heads 128 x 128, conv 4, QSA indexer 4 x 128), small in width,
 * depth, experts and vocabulary.
 *
 *   xe_tiny_qwen4exp DIR [mtp=0] [seed=1]
 */
#include "tiny_model.h"

#include <cstdio>
#include <string>

namespace {

struct Q4 {
    int64_t layers = 4, interval = 4;                 /* layers 0-2 GDN, 3 full (QSA) */
    /* hidden % 512: libr4d's e4m3 hc_read row (ported to libxe) needs n % 512; production's 2560 is */
    int64_t hidden = 512, hc = 4, lowrank = 320;      /* lowrank % 80 (hc_up group 80) and % 64 (hc_read_e4m3) */
    int64_t heads = 24, kv_heads = 2, head_dim = 256;
    double  rotary = 0.25;
    int64_t lin_k_heads = 2, lin_v_heads = 6, lin_k_dim = 128, lin_v_dim = 128, conv = 4;
    int64_t experts = 8, top_k = 2, ffn = 128, shared = 128;
    int64_t idx_heads = 4, idx_dim = 128, idx_budget = 2048, idx_ratio = 4;
    int64_t vocab = 512, ctx = 4096, mtp = 0;
    uint64_t seed = 1;
    /* lm_head's scale on top of its usual gain (owner note 5: x10 makes greedy decoding decisive);
     * the same draws either way, so every other tensor is unchanged */
    float head = 1.0f;
};

using tiny::Tensor;

/* the two hyper-connections a layer has, or the mixer (no inject) */
void hc_block(std::vector<Tensor>& ts, const std::string& p, const Q4& s, tiny::Rng& r, bool inject) {
    const int64_t HN = s.hc * s.hidden;
    ts.push_back(tiny::noise(p + "hc_norm.weight", { HN }, r, 0.1f, 1.0f));
    ts.push_back(tiny::linear(p + "input_mix_weight_down.weight", s.lowrank, HN, r));
    ts.push_back(tiny::linear(p + "input_mix_weight_up.weight", HN, s.lowrank, r, 0.5f));
    if (inject) ts.push_back(tiny::linear(p + "block_inject_weight.weight", s.hc, HN, r, 0.5f));
}

void attention(std::vector<Tensor>& ts, const std::string& p, const Q4& s, tiny::Rng& r) {
    const int64_t H = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    ts.push_back(tiny::linear(p + "self_attn.q_proj.weight", 2 * q, H, r));   /* query | gate per head */
    ts.push_back(tiny::linear(p + "self_attn.k_proj.weight", kv, H, r));
    ts.push_back(tiny::linear(p + "self_attn.v_proj.weight", kv, H, r));
    ts.push_back(tiny::linear(p + "self_attn.o_proj.weight", H, q, r, 0.5f));
    ts.push_back(tiny::noise(p + "self_attn.q_norm.weight", { s.head_dim }, r, 0.1f, 1.0f));
    ts.push_back(tiny::noise(p + "self_attn.k_norm.weight", { s.head_dim }, r, 0.1f, 1.0f));
    if (s.idx_heads > 0) {
        ts.push_back(tiny::linear(p + "self_attn.indexer.index_qk_proj.weight", (s.idx_heads + 1) * s.idx_dim, H, r));
        ts.push_back(tiny::noise(p + "self_attn.indexer.q_layernorm.weight", { s.idx_dim }, r, 0.1f, 1.0f));
        ts.push_back(tiny::noise(p + "self_attn.indexer.k_layernorm.weight", { s.idx_dim }, r, 0.1f, 1.0f));
    }
}

void moe(std::vector<Tensor>& ts, const std::string& p, const Q4& s, tiny::Rng& r) {
    const int64_t H = s.hidden, E = s.experts, F = s.ffn;
    ts.push_back(tiny::linear(p + "mlp.gate.weight", E, H, r));
    /* stacked experts, no .weight suffix; gate rows before up rows */
    ts.push_back(tiny::noise(p + "mlp.experts.gate_up_proj", { E, 2 * F, H }, r, 1.0f / std::sqrt((float)H)));
    ts.push_back(tiny::noise(p + "mlp.experts.down_proj", { E, H, F }, r, 0.5f / std::sqrt((float)F)));
    ts.push_back(tiny::linear(p + "mlp.shared_expert.gate_proj.weight", s.shared, H, r));
    ts.push_back(tiny::linear(p + "mlp.shared_expert.up_proj.weight", s.shared, H, r));
    ts.push_back(tiny::linear(p + "mlp.shared_expert.down_proj.weight", H, s.shared, r, 0.5f));
    ts.push_back(tiny::linear(p + "mlp.shared_expert_gate.weight", 1, H, r));
}

bool write_qwen4exp(const std::string& dir, const Q4& s) {
    tiny::Rng r(s.seed);
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (int64_t l = 0; l < s.layers; ++l)
        types.push_back((l + 1) % s.interval == 0 ? "qwen_sparse_attention" : "linear_attention");
    nlohmann::ordered_json c;
    c["architectures"] = { "Qwen4ExpForCausalLM" };
    c["model_type"] = "qwen4_exp";
    c["vocab_size"] = s.vocab;
    c["hidden_size"] = s.hidden;
    c["moe_intermediate_size"] = s.ffn;
    c["shared_expert_intermediate_size"] = s.shared;
    c["num_experts"] = s.experts;
    c["num_experts_per_tok"] = s.top_k;
    c["num_hidden_layers"] = s.layers;
    c["num_attention_heads"] = s.heads;
    c["num_key_value_heads"] = s.kv_heads;
    c["head_dim"] = s.head_dim;
    c["max_position_embeddings"] = s.ctx;
    c["rms_norm_eps"] = 1e-6;
    c["rope_parameters"] = { { "rope_type", "default" }, { "rope_theta", 10000000.0 },
                             { "partial_rotary_factor", s.rotary } };
    c["full_attention_interval"] = s.interval;
    c["layer_types"] = types;
    c["linear_num_key_heads"] = s.lin_k_heads;
    c["linear_num_value_heads"] = s.lin_v_heads;
    c["linear_key_head_dim"] = s.lin_k_dim;
    c["linear_value_head_dim"] = s.lin_v_dim;
    c["linear_conv_kernel_dim"] = s.conv;
    c["output_gate_type"] = "sigmoid";
    c["hc_count"] = s.hc;
    c["hc_lowrank"] = s.lowrank;
    c["indexer_n_heads"] = s.idx_heads;
    c["indexer_head_dim"] = s.idx_dim;
    c["indexer_budget"] = s.idx_budget;
    c["indexer_compress_ratio"] = s.idx_ratio;
    c["mtp_num_hidden_layers"] = s.mtp;
    c["tie_word_embeddings"] = false;
    c["eos_token_id"] = (int)tiny::kEndOfText;

    const int64_t H = s.hidden, HN = s.hc * s.hidden;
    const int64_t kdim = s.lin_k_heads * s.lin_k_dim, vdim = s.lin_v_heads * s.lin_v_dim;
    const int64_t conv_dim = 2 * kdim + vdim;
    const std::string m = "model.language_model.";
    std::vector<Tensor> ts;
    ts.push_back(tiny::noise(m + "embed_tokens.weight", { s.vocab, H }, r, 1.0f));
    hc_block(ts, m + "hyper_connection_mixer.", s, r, false);
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        hc_block(ts, p + "attn_hyper_connection.", s, r, true);
        hc_block(ts, p + "mlp_hyper_connection.", s, r, true);
        if ((l + 1) % s.interval == 0) {
            attention(ts, p, s, r);
        } else {
            ts.push_back(tiny::linear(p + "linear_attn.in_proj_qkv.weight", conv_dim, H, r));
            ts.push_back(tiny::linear(p + "linear_attn.in_proj_z.weight", vdim, H, r));
            ts.push_back(tiny::linear(p + "linear_attn.in_proj_a.weight", s.lin_v_heads, H, r));
            ts.push_back(tiny::linear(p + "linear_attn.in_proj_b.weight", s.lin_v_heads, H, r));
            ts.push_back(tiny::noise(p + "linear_attn.conv1d.weight", { conv_dim, 1, s.conv }, r, 0.5f));
            ts.push_back(tiny::noise(p + "linear_attn.A_log", { s.lin_v_heads }, r, 1.0f, 1.0f, "F32"));
            ts.push_back(tiny::noise(p + "linear_attn.dt_bias", { s.lin_v_heads }, r, 0.5f, 1.0f, "F32"));
            ts.push_back(tiny::noise(p + "linear_attn.norm.weight", { s.lin_v_dim }, r, 0.1f, 1.0f));
            ts.push_back(tiny::linear(p + "linear_attn.out_proj.weight", H, vdim, r, 0.5f));
        }
        moe(ts, p, s, r);
    }
    ts.push_back(tiny::linear("lm_head.weight", s.vocab, H, r, 2.0f * s.head));
    if (s.mtp > 0) {
        ts.push_back(tiny::linear("mtp.fc_hidden.weight", H, H, r));
        ts.push_back(tiny::linear("mtp.fc_embedding.weight", H, H, r));
        ts.push_back(tiny::noise("mtp.pre_fc_norm_hidden.weight", { HN }, r, 0.1f, 1.0f));
        ts.push_back(tiny::noise("mtp.pre_fc_norm_embedding.weight", { H }, r, 0.1f, 1.0f));
        hc_block(ts, "mtp.hyper_connection_mixer.", s, r, false);
        hc_block(ts, "mtp.layers.0.attn_hyper_connection.", s, r, true);
        hc_block(ts, "mtp.layers.0.mlp_hyper_connection.", s, r, true);
        attention(ts, "mtp.layers.0.", s, r);
        moe(ts, "mtp.layers.0.", s, r);
    }
    return tiny::write_checkpoint(dir, c, ts);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xe_tiny_qwen4exp DIR [mtp=0] [seed=1] [head=1]\n"); return 2; }
    Q4 s;
    if (argc > 2) s.mtp = std::atoll(argv[2]);
    if (argc > 3) s.seed = (uint64_t)std::atoll(argv[3]);
    if (argc > 4) s.head = (float)std::atof(argv[4]);
    /* Owner note 4's relayout probe: production's per-weight shapes with few experts
     * (RI_TINY_HIDDEN=2560 RI_TINY_FFN=640 RI_TINY_EXPERTS=16 ...); unset keeps the tiny model. */
    auto env = [](const char* k, int64_t& v) { if (const char* e = std::getenv(k)) v = std::atoll(e); };
    env("RI_TINY_HIDDEN", s.hidden);
    env("RI_TINY_FFN", s.ffn);
    env("RI_TINY_SHARED", s.shared);
    env("RI_TINY_EXPERTS", s.experts);
    env("RI_TINY_VOCAB", s.vocab);
    if (!write_qwen4exp(argv[1], s)) { std::fprintf(stderr, "xe_tiny_qwen4exp: write failed\n"); return 1; }
    std::printf("xe_tiny_qwen4exp: wrote %s (mtp %lld, seed %llu, lm_head x%g)\n", argv[1], (long long)s.mtp,
                (unsigned long long)s.seed, (double)s.head);
    return 0;
}
