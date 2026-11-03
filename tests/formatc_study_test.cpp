/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 116, owner note 146 item 4: format C, the codebook at int8).
 *
 * formatc_study_test.cpp -- the CPU check before any kernel: today's w4nl codebook served against int8 activations, its
 * 16 levels rounded to int8 at load (round(t * 127 / amax) * amax / 127: the owner's {6, 19, 32, 46, 63, 81, 104,
 * 127} at amax 22), against the exact levels and against format B (w4g64i8h), on a sample of real experts.
 *
 * Each expert matrix is quantised by libquant through the converter's own loop (quant_fixture.h), with the served
 * recipe's expert rule (`gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* scale2_value=2^-13
 * transform=fwht128 rule=search cd=3`) and with format B's (`gptq codes=i4 group=64 scale=bf16 transform=fwht128
 * cd=3`), against the sample's Grams (calib=DIR: the layer's pooled gate_up Gram; an expert's own down Gram where it
 * holds >= 4 K rows, else the pooled one -- libquant's own rule). The served codes are then decoded twice: as stored,
 * and with the table plane's 16 levels rounded to int8.
 *
 * The measure is docs/MOE-W4.md's: sqrt(tr(dW H dW^T) / tr(W H W^T)), H the same Gram. The files are recorded on the
 * input rotated by the width-128 Hadamard (domain 128), so W and dW are rotated the same way first (libquant's
 * unnormalised butterflies; the scale cancels in the ratio). The Grams are the calibration's own (in-sample), as the
 * three forms are all quantised against them: a comparison between forms, not a held-out score.
 *
 *   RI_FORMATC_RAW=DIR RI_FORMATC_GRAMS=DIR [RI_FORMATC_MAX=84] formatc_study_test
 * (the raw directory from m3/tools/formatc-unpack.py; $RADIANCE_HOME for libquant). Without them the case skips, so
 * the unit tier passes over it. */
#include "quant_fixture.h"
#include "rad_planes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

using namespace qfix;

namespace {

std::vector<float> read_f32(const std::string& p, size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f || std::fread(v.data(), 4, n, f) != n) v.clear();
    if (f) std::fclose(f);
    return v;
}

/* One Gram file's K x K f32 added into `h`; its row count, or -1. */
long long add_gram(const std::string& p, int64_t k, std::vector<double>& h) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return -1;
    char hdr[32];
    uint32_t kk = 0;
    uint64_t rows = 0;
    bool ok = std::fread(hdr, 1, 32, f) == 32 && !std::memcmp(hdr, "RADGRAM2", 8);
    std::memcpy(&kk, hdr + 8, 4);
    std::memcpy(&rows, hdr + 16, 8);
    std::vector<float> g((size_t)(k * k));
    ok = ok && (int64_t)kk == k && std::fread(g.data(), 4, g.size(), f) == g.size();
    std::fclose(f);
    if (!ok) return -1;
    for (size_t i = 0; i < g.size(); ++i) h[i] += g[i];
    return (long long)rows;
}

/* The Gram libquant reads for this weight: gate_up the layer's pooled (r0-r3); down the expert's own (its rank
 * files) at >= 4 K rows, else the pooled. */
std::vector<float> gram_for(const std::string& dir, int l, int e, bool down, int64_t k, std::string* which) {
    std::vector<double> h((size_t)(k * k), 0.0);
    const char* base = down ? "ffn_down_exps" : "ffn_gate_up_exps";
    long long own = 0;
    if (down) {
        for (int r = 0; r < 4; ++r) {
            char p[512];
            std::snprintf(p, sizeof p, "%s/gram.r%d.blk.%d.%s.%d.bin", dir.c_str(), r, l, base, e);
            const long long n = add_gram(p, k, h);
            if (n > 0) own += n;
        }
        if (own >= 4 * k) { *which = "own " + std::to_string(own); }
        else std::fill(h.begin(), h.end(), 0.0);
    }
    if (!down || own < 4 * k) {
        long long rows = 0;
        for (int r = 0; r < 4; ++r) {
            char p[512];
            std::snprintf(p, sizeof p, "%s/gram.r%d.blk.%d.%s.bin", dir.c_str(), r, l, base);
            const long long n = add_gram(p, k, h);
            if (n > 0) rows += n;
        }
        *which = "pooled " + std::to_string(rows);
    }
    std::vector<float> hf(h.size());
    for (size_t i = 0; i < h.size(); ++i) hf[i] = (float)h[i];
    return hf;
}

/* libquant's width-128 Hadamard on each 128-block of every row, unnormalised (quant_algo_test's fwht_rows). */
void fwht_rows(std::vector<float>& m, int64_t rows, int64_t k) {
    for (int64_t r = 0; r < rows; ++r) {
        float* v = m.data() + r * k;
        for (int64_t b0 = 0; b0 < k; b0 += 128)
            for (int64_t s = 1; s < 128; s *= 2)
                for (int64_t i = b0; i < b0 + 128; i += 2 * s)
                    for (int64_t j = i; j < i + s; ++j) {
                        const float x = v[j], y = v[j + s];
                        v[j] = x + y;
                        v[j + s] = x - y;
                    }
    }
}

/* sum_r m_r H m_r^T over 8 threads: t = H m_r by rows of H (an axpy the compiler vectorises), then m_r . t. */
double quad(const std::vector<float>& m, int64_t rows, int64_t k, const std::vector<float>& h) {
    const int nt = 8;
    std::vector<double> part((size_t)nt, 0.0);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            std::vector<float> acc((size_t)k);
            for (int64_t r = t; r < rows; r += nt) {
                const float* d = m.data() + r * k;
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int64_t i = 0; i < k; ++i) {
                    const float di = d[i];
                    if (di == 0.0f) continue;
                    const float* hr = h.data() + i * k;
                    for (int64_t j = 0; j < k; ++j) acc[(size_t)j] += di * hr[j];
                }
                double s = 0.0;
                for (int64_t j = 0; j < k; ++j) s += (double)d[j] * acc[(size_t)j];
                part[(size_t)t] += s;
            }
        });
    for (auto& x : th) x.join();
    double s = 0.0;
    for (double p : part) s += p;
    return s;
}

double rel_out_err(const std::vector<float>& wrot, double den, const std::vector<float>& w,
                   const std::vector<float>& dq, int64_t rows, int64_t k, const std::vector<float>& h) {
    (void)wrot;
    std::vector<float> d(w.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = dq[i] - w[i];
    fwht_rows(d, rows, k);
    return std::sqrt(quad(d, rows, k, h) / den);
}

/* The served codes with the table plane's levels rounded to int8 at the levels' own amax. */
Quantised int8_levels(const Quantised& q, std::string* lv) {
    Quantised r = q;
    const RadEncPlane* tp = rad_enc_plane(&r.enc, "table");
    if (!tp) return r;
    const int ti = (int)(tp - r.enc.plane);
    std::vector<uint8_t>& p = r.planes[(size_t)ti];
    const size_t n = p.size() / 4;
    float* t = (float*)p.data();
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(t[i]));
    std::ostringstream os;
    for (size_t i = 0; i < n; ++i) {
        const float c = std::nearbyint(t[i] * 127.0f / amax);
        if (lv && i < 16) os << t[i] << "->" << (int)c << (i + 1 < n ? " " : "");
        t[i] = c * amax / 127.0f;
    }
    if (lv) *lv = os.str();
    return r;
}

}  // namespace

TEST(format_c_int8_levels_against_the_exact_codebook_and_format_b) {
    const char* rawe = std::getenv("RI_FORMATC_RAW");
    const char* gde = std::getenv("RI_FORMATC_GRAMS");
    if (!rawe || !gde) SKIP("RI_FORMATC_RAW / RI_FORMATC_GRAMS not set (a study run by hand)");
    if (std::getenv("RI_FORMATC_STUDY")) SKIP("RI_FORMATC_STUDY selects another study");
    NEED_LIBQUANT();
    const std::string raw = rawe, gdir = gde;
    const int maxn = std::getenv("RI_FORMATC_MAX") ? std::atoi(std::getenv("RI_FORMATC_MAX")) : 84;
    std::ifstream idx(raw + "/index.txt");
    REQUIRE(idx.good());
    int l, e;
    std::string which;
    int64_t rows, cols;
    int done = 0;
    double sum[2][3] = { { 0 } };
    int cnt[2] = { 0, 0 };
    bool said = false;
    std::printf("layer expert matrix gram | served(w4nl) int8-levels formatB | int8/served B/served\n");
    while (done < maxn && idx >> l >> e >> which >> rows >> cols) {
        const bool down = which == "down";
        char path[512];
        std::snprintf(path, sizeof path, "%s/%d.%d.%s.f32", raw.c_str(), l, e, which.c_str());
        const std::vector<float> w = read_f32(path, (size_t)(rows * cols));
        if (w.empty()) std::fprintf(stderr, "cannot read %s\n", path);
        REQUIRE(!w.empty());
        std::string gw;
        const std::vector<float> h = gram_for(gdir, l, e, down, cols, &gw);
        std::vector<float> wr = w;
        fwht_rows(wr, rows, cols);
        const double den = quad(wr, rows, cols, h);
        char name[128];
        std::snprintf(name, sizeof name, "blk.%d.%s.%d.weight", l, down ? "ffn_down_exps" : "ffn_gate_up_exps", e);

        Opts os;
        os.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*")
          .f("scale2_value", 0.0001220703125).s("transform", "fwht128").s("rule", "search").i("cd", 3)
          .s("calib", gdir.c_str());
        const Quantised qs = quantise(quantizer("gptq"), os, name, rows, cols, w);
        Opts ob;
        ob.s("codes", "i4").i("group", 64).s("scale", "bf16").s("transform", "fwht128").i("cd", 3).s("calib", gdir.c_str());
        const Quantised qb = quantise(quantizer("gptq"), ob, name, rows, cols, w);
        REQUIRE_EQ(qs.status, RAD_OK);
        REQUIRE_EQ(qb.status, RAD_OK);
        std::string lv;
        const Quantised qc = int8_levels(qs, said ? nullptr : &lv);
        if (!said) {
            std::printf("# served encoding %s; format B %s\n# table levels -> int8: %s\n", rad::enc_name(qs.enc).c_str(),
                        rad::enc_name(qb.enc).c_str(), lv.c_str());
            said = true;
        }
        const double es = rel_out_err(wr, den, w, decode(qs, rows, cols), rows, cols, h);
        const double ec = rel_out_err(wr, den, w, decode(qc, rows, cols), rows, cols, h);
        const double eb = rel_out_err(wr, den, w, decode(qb, rows, cols), rows, cols, h);
        std::printf("%2d %3d %-7s %-14s | %.5f %.5f %.5f | %.4f %.4f\n", l, e, which.c_str(), gw.c_str(), es, ec, eb,
                    ec / es, eb / es);
        std::fflush(stdout);
        const int d = down ? 1 : 0;
        sum[d][0] += es; sum[d][1] += ec; sum[d][2] += eb; ++cnt[d];
        ++done;
    }
    for (int d = 0; d < 2; ++d)
        if (cnt[d])
            std::printf("MEAN %-7s over %2d | %.5f %.5f %.5f | %.4f %.4f\n", d ? "down" : "gate_up", cnt[d],
                        sum[d][0] / cnt[d], sum[d][1] / cnt[d], sum[d][2] / cnt[d], sum[d][1] / sum[d][0],
                        sum[d][2] / sum[d][0]);
}

/* Owner note 162 item 4 (track 2; briefs/2026-11-01-mixed-bits-experts.md step 2, run 160): a 3-bit codebook -- four
 * magnitudes and their negatives, Lloyd-fitted on the sample's rotated, group-amax-normalised weights the way w4nl's
 * sixteen were -- quantised by the served rule (a searched E4M3 scale per 64, GPTQ, cd 3) and scored by the same
 * measure beside the served w4nl, with and without format C's int8 rounding of the levels. RI_FORMATC_STUDY=3bit
 * runs this study alone (m1/build.sh fc3bit). */
TEST(format_c_three_bit_codebook_against_w4nl) {
    const char* rawe = std::getenv("RI_FORMATC_RAW");
    const char* gde = std::getenv("RI_FORMATC_GRAMS");
    const char* st = std::getenv("RI_FORMATC_STUDY");
    if (!rawe || !gde || !st || std::strcmp(st, "3bit"))
        SKIP("RI_FORMATC_STUDY=3bit with RI_FORMATC_RAW / RI_FORMATC_GRAMS (a study run by hand)");
    NEED_LIBQUANT();
    const std::string raw = rawe, gdir = gde;
    const int maxn = std::getenv("RI_FORMATC_MAX") ? std::atoi(std::getenv("RI_FORMATC_MAX")) : 84;
    int l, e;
    std::string which;
    int64_t rows, cols;

    /* pass 1, the fit: every eighth rotated weight of every group of 64, as |w| / the group's amax */
    std::vector<float> samp;
    {
        std::ifstream idx(raw + "/index.txt");
        REQUIRE(idx.good());
        int done = 0;
        while (done < maxn && idx >> l >> e >> which >> rows >> cols) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/%d.%d.%s.f32", raw.c_str(), l, e, which.c_str());
            std::vector<float> wr = read_f32(path, (size_t)(rows * cols));
            REQUIRE(!wr.empty());
            fwht_rows(wr, rows, cols);
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t g0 = 0; g0 + 64 <= cols; g0 += 64) {
                    const float* p = wr.data() + r * cols + g0;
                    float amax = 0.0f;
                    for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(p[i]));
                    if (amax <= 0.0f) continue;
                    for (int i = (int)(r & 7); i < 64; i += 8) samp.push_back(std::fabs(p[i]) / amax);
                }
            ++done;
        }
    }
    REQUIRE(!samp.empty());
    double c[4] = { 0.08, 0.25, 0.5, 1.0 };
    for (int it = 0; it < 30; ++it) {
        double s[4] = { 0, 0, 0, 0 }, n[4] = { 0, 0, 0, 0 };
        for (float v : samp) {
            int b = 0;
            double bd = 1e30;
            for (int k = 0; k < 4; ++k) {
                const double d = std::fabs((double)v - c[k]);
                if (d < bd) { bd = d; b = k; }
            }
            s[b] += v;
            n[b] += 1.0;
        }
        for (int k = 0; k < 4; ++k) if (n[k] > 0.0) c[k] = s[k] / n[k];
    }
    /* the table at w4nl's scale (its top level 22), so the searched E4M3 scales sit in the served range */
    const double top = c[3] > 0.0 ? c[3] : 1.0;
    char tbl[256];
    std::snprintf(tbl, sizeof tbl, "%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f", c[0] * 22.0 / top, c[1] * 22.0 / top,
                  c[2] * 22.0 / top, 22.0, -c[0] * 22.0 / top, -c[1] * 22.0 / top, -c[2] * 22.0 / top, -22.0);
    std::printf("# 3-bit Lloyd magnitudes (amax-normalised) %.4f %.4f %.4f %.4f over %zu samples; table=%s\n",
                c[0], c[1], c[2], c[3], samp.size(), tbl);
    std::vector<float>().swap(samp);

    /* pass 2: the error, served w4nl against the 3-bit codebook exact and with its levels rounded to int8 */
    std::ifstream idx(raw + "/index.txt");
    REQUIRE(idx.good());
    int done = 0;
    double sum[2][3] = { { 0 } };
    int cnt[2] = { 0, 0 };
    bool said = false;
    std::printf("layer expert matrix gram | served(w4nl) w3nl w3nl-int8 | w3nl/served w3nl-int8/served\n");
    while (done < maxn && idx >> l >> e >> which >> rows >> cols) {
        const bool down = which == "down";
        char path[512];
        std::snprintf(path, sizeof path, "%s/%d.%d.%s.f32", raw.c_str(), l, e, which.c_str());
        const std::vector<float> w = read_f32(path, (size_t)(rows * cols));
        REQUIRE(!w.empty());
        std::string gw;
        const std::vector<float> h = gram_for(gdir, l, e, down, cols, &gw);
        std::vector<float> wr = w;
        fwht_rows(wr, rows, cols);
        const double den = quad(wr, rows, cols, h);
        char name[128];
        std::snprintf(name, sizeof name, "blk.%d.%s.%d.weight", l, down ? "ffn_down_exps" : "ffn_gate_up_exps", e);
        Opts os;
        os.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*")
          .f("scale2_value", 0.0001220703125).s("transform", "fwht128").s("rule", "search").i("cd", 3)
          .s("calib", gdir.c_str());
        const Quantised qs = quantise(quantizer("gptq"), os, name, rows, cols, w);
        Opts o3;
        o3.s("table", tbl).i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*")
          .f("scale2_value", 0.0001220703125).s("transform", "fwht128").s("rule", "search").i("cd", 3)
          .s("calib", gdir.c_str());
        const Quantised q3 = quantise(quantizer("gptq"), o3, name, rows, cols, w);
        REQUIRE_EQ(qs.status, RAD_OK);
        REQUIRE_EQ(q3.status, RAD_OK);
        std::string lv;
        const Quantised q3i = int8_levels(q3, said ? nullptr : &lv);
        if (!said) {
            std::printf("# 3-bit encoding %s\n# its levels -> int8: %s\n", rad::enc_name(q3.enc).c_str(), lv.c_str());
            said = true;
        }
        const double es = rel_out_err(wr, den, w, decode(qs, rows, cols), rows, cols, h);
        const double e3 = rel_out_err(wr, den, w, decode(q3, rows, cols), rows, cols, h);
        const double e3i = rel_out_err(wr, den, w, decode(q3i, rows, cols), rows, cols, h);
        std::printf("%2d %3d %-7s %-14s | %.5f %.5f %.5f | %.4f %.4f\n", l, e, which.c_str(), gw.c_str(), es, e3, e3i,
                    e3 / es, e3i / es);
        std::fflush(stdout);
        const int d = down ? 1 : 0;
        sum[d][0] += es; sum[d][1] += e3; sum[d][2] += e3i; ++cnt[d];
        ++done;
    }
    for (int d = 0; d < 2; ++d)
        if (cnt[d])
            std::printf("MEAN %-7s over %2d | %.5f %.5f %.5f | %.4f %.4f\n", d ? "down" : "gate_up", cnt[d],
                        sum[d][0] / cnt[d], sum[d][1] / cnt[d], sum[d][2] / cnt[d], sum[d][1] / sum[d][0],
                        sum[d][2] / sum[d][0]);
}

/* Owner note 147's real-weight check, its host half (run 116): four real experts of one layer (the sample's first
 * four in index.txt order) quantised by the served rule, as above, and written as the loader's joined entries --
 * rad_planes_relayout of (codes, table) and of (scale, scale.1), the order xe_moe_planes.cpp's view_ok accepts -- to
 * RI_FORMATC_JOINED/<n>.<gate_up|down>.{c,s}.bin, with meta.txt (n, matrix, rows, cols). libxe's moe_i4i8_test reads
 * them (RI_FORMATC_JOINED, RI_LIBREF) and compares xe_rad_moe_gemm_w4nl_i8 on amd-host's CPU device with libref.
 *   RI_FORMATC_RAW=DIR RI_FORMATC_GRAMS=DIR RI_FORMATC_JOINED=OUT formatc_study_test */
TEST(format_c_real_experts_written_as_joined_entries) {
    const char* rawe = std::getenv("RI_FORMATC_RAW");
    const char* gde = std::getenv("RI_FORMATC_GRAMS");
    const char* oute = std::getenv("RI_FORMATC_JOINED");
    if (!rawe || !gde || !oute) SKIP("RI_FORMATC_RAW / RI_FORMATC_GRAMS / RI_FORMATC_JOINED not set (by hand)");
    NEED_LIBQUANT();
    const std::string raw = rawe, gdir = gde, out = oute;
    std::ifstream idx(raw + "/index.txt");
    REQUIRE(idx.good());
    std::ofstream meta(out + "/meta.txt");
    REQUIRE(meta.good());
    int l, e, l0 = -1;
    std::string which;
    int64_t rows, cols;
    int nexp[2] = { 0, 0 };
    while (idx >> l >> e >> which >> rows >> cols) {
        if (l0 < 0) l0 = l;
        const int d = which == "down" ? 1 : 0;
        if (l != l0 || nexp[d] >= 4) continue;
        char path[512];
        std::snprintf(path, sizeof path, "%s/%d.%d.%s.f32", raw.c_str(), l, e, which.c_str());
        const std::vector<float> w = read_f32(path, (size_t)(rows * cols));
        REQUIRE(!w.empty());
        char name[128];
        std::snprintf(name, sizeof name, "blk.%d.%s.%d.weight", l, d ? "ffn_down_exps" : "ffn_gate_up_exps", e);
        Opts os;
        os.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*")
          .f("scale2_value", 0.0001220703125).s("transform", "fwht128").s("rule", "search").i("cd", 3)
          .s("calib", gdir.c_str());
        const Quantised qs = quantise(quantizer("gptq"), os, name, rows, cols, w);
        REQUIRE_EQ(qs.status, RAD_OK);
        const char* roles[2][2] = { { "codes", "table" }, { "scale", "scale.1" } };
        for (int v = 0; v < 2; ++v) {
            int sel[2];
            RadTensor pl[2]{};
            for (int x = 0; x < 2; ++x) {
                const RadEncPlane* p = rad_enc_plane(&qs.enc, roles[v][x]);
                REQUIRE(p != nullptr);
                sel[x] = (int)(p - qs.enc.plane);
                pl[x].dtype = p->dtype;
                pl[x].rank = 2;
                pl[x].data = (void*)qs.planes[(size_t)sel[x]].data();
                if (p->kind == RAD_PLANE_TABLE) { pl[x].shape[0] = p->extent[0]; pl[x].shape[1] = p->extent[1]; }
                else if (p->block[0] == 0) { pl[x].shape[0] = 1; pl[x].shape[1] = 1; }   /* one value (scale.1) */
                else { pl[x].shape[0] = rows / p->block[0]; pl[x].shape[1] = cols / p->block[1]; }
                REQUIRE_EQ(rad_dtype_bytes(pl[x].dtype, pl[x].shape[0] * pl[x].shape[1]), (int64_t)qs.planes[(size_t)sel[x]].size());
            }
            RadPlanesEntry pe[2];
            std::vector<uint8_t> j((size_t)rad_planes_plan(&qs.enc, sel, pl, 2, pe), 0);
            REQUIRE_EQ(rad_planes_relayout(&qs.enc, sel, pl, 2, j.data(), (int64_t)j.size()), RAD_OK);
            std::snprintf(path, sizeof path, "%s/%d.%s.%c.bin", out.c_str(), nexp[d], which.c_str(), v ? 's' : 'c');
            FILE* f = std::fopen(path, "wb");
            REQUIRE(f != nullptr);
            REQUIRE_EQ(std::fwrite(j.data(), 1, j.size(), f), j.size());
            std::fclose(f);
        }
        meta << nexp[d] << " " << which << " " << rows << " " << cols << " " << l << " " << e << "\n";
        std::printf("joined %d %s (layer %d expert %d, %lld x %lld): %s\n", nexp[d], which.c_str(), l, e,
                    (long long)rows, (long long)cols, rad::enc_name(qs.enc).c_str());
        std::fflush(stdout);
        ++nexp[d];
    }
    REQUIRE_EQ(nexp[0], 4);
    REQUIRE_EQ(nexp[1], 4);
}

RAD_TEST_MAIN()
