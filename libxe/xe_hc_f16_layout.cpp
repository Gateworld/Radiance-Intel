// SPDX-License-Identifier: Apache-2.0
// Part of the Intel port of radiance (milestone 3, owner note 56).
//
// hc_read's mixing matrices stored with every E4M3 code expanded to FP16, once, when the model loads
// (RADIANCE_XE_HC_F16=1). libr4d's hooks (r4d_fp8_layout.cpp) store a row as its K E4M3 codes, then its K / G
// f32 group scales, padded to 16 bytes; this form keeps the scales and the padding rule and widens each code to
// its FP16 value. Every E4M3 value (the NaN included) is exact in FP16, so the relayout is lossless, the
// unrelayout recovers the codes bit for bit, and the f32 a kernel decodes from a code is the same in both forms
// (r4d_hc_bf16.cpp, the W16 dots). With the flag off these hooks are libr4d's.
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" int r4d_rad_layout_hc_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                      const int* sel, const RadTensor* pl, int n, RadLayout* out);
extern "C" int r4d_rad_relayout_hc_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                        const int* sel, const RadTensor* pl, int n, void* dst, int64_t dst_bytes);
extern "C" int r4d_rad_unrelayout_hc_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                          const int* sel, const void* src, int64_t src_bytes, const RadTensor* pl,
                                          int n);
bool xe_hc_f16_on();                                      // r4d_hc_bf16.cpp
bool xe_hc_i8_on();                                       // r4d_hc_bf16.cpp (owner note 158, run 125)
long long xe_hc16_row_bytes(long long K, long long G);    // r4d_hc_bf16.cpp

namespace {

struct Geo { long long rows = 0, K = 0, ng = 0, stride = 0; };

/* libr4d's checks first (encoding, planes, extents), then the FP16 row's geometry from the planes. */
int geo(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel, const RadTensor* pl,
        int n, RadLayout* out, Geo* g) {
    const int rc = r4d_rad_layout_hc_e4m3(p, n_p, operand, enc, sel, pl, n, out);
    if (rc != RAD_OK) return rc;
    if (n < 2 || pl[0].rank != 2 || pl[1].rank != 2) return RAD_E_SHAPE;
    g->rows = pl[0].shape[0];
    g->K = pl[0].shape[1];
    g->ng = pl[1].shape[1];
    if (g->rows <= 0 || g->K <= 0 || g->ng <= 0 || g->K % g->ng) return RAD_E_SHAPE;
    g->stride = xe_hc16_row_bytes(g->K, g->K / g->ng);
    out->tag = std::strcmp(out->tag, "r4d.hc.e4m3.g128") == 0 ? "xe.hc.f16.g128" : "xe.hc.f16.q4";
    out->shape[1] = g->stride;
    out->bytes = g->rows * g->stride;
    return RAD_OK;
}

uint16_t f16_of(float f) {
    const _Float16 h = (_Float16)f;   // exact for every E4M3 value
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float f32_of(uint16_t b) {
    _Float16 h;
    std::memcpy(&h, &b, 2);
    return (float)h;
}

/* Owner note 158 (run 125, RADIANCE_XE_HC_I8=1): libr4d's E4M3 rows rewritten in place as int8. Each group of G codes
 * is rounded to int8 at its own max -- q = nearbyint(v * 127 / max), v the code's E4M3 value -- and its f32 scale
 * multiplied by max / 127, so q x scale' approximates v x scale; the same bytes and row layout (K codes, then the
 * K / G scales, padded), under its own tag so a cached E4M3 layout is never read as int8. Lossy: no unrelayout. */
int i8_rows(uint8_t* d, long long rows, long long K, long long ng, long long stride) {
    const long long G = K / ng;
    for (long long r = 0; r < rows; ++r) {
        uint8_t* row = d + r * stride;
        for (long long g = 0; g < ng; ++g) {
            float amax = 0.0f;
            for (long long k = g * G; k < (g + 1) * G; ++k) {
                const float v = rad_fp8e4m3_to_f32(row[k]);
                if (v != v) return RAD_E_DTYPE;   /* a NaN code: quantised weights never hold one */
                amax = std::max(amax, std::fabs(v));
            }
            float sc;
            std::memcpy(&sc, row + K + 4 * g, 4);
            for (long long k = g * G; k < (g + 1) * G; ++k) {
                const float v = rad_fp8e4m3_to_f32(row[k]);
                const float q = amax > 0.0f ? std::nearbyint(v * 127.0f / amax) : 0.0f;
                row[k] = (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
            if (amax > 0.0f) sc = sc * (amax / 127.0f);
            std::memcpy(row + K + 4 * g, &sc, 4);
        }
    }
    return RAD_OK;
}

}  // namespace

extern "C" int xe_hc_layout_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                 const RadTensor* pl, int n, RadLayout* out) {
    if (xe_hc_i8_on()) {
        const int rc = r4d_rad_layout_hc_e4m3(p, n_p, operand, enc, sel, pl, n, out);
        if (rc == RAD_OK) out->tag = std::strcmp(out->tag, "r4d.hc.e4m3.g128") == 0 ? "xe.hc.i8.g128" : "xe.hc.i8.q4";
        return rc;
    }
    if (!xe_hc_f16_on()) return r4d_rad_layout_hc_e4m3(p, n_p, operand, enc, sel, pl, n, out);
    Geo g;
    return geo(p, n_p, operand, enc, sel, pl, n, out, &g);
}

extern "C" int xe_hc_relayout_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    if (xe_hc_i8_on()) {
        int rc = r4d_rad_relayout_hc_e4m3(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
        if (rc != RAD_OK) return rc;
        if (n < 2 || pl[0].rank != 2 || pl[1].rank != 2) return RAD_E_SHAPE;
        const long long rows = pl[0].shape[0], K = pl[0].shape[1], ng = pl[1].shape[1];
        if (rows <= 0 || K <= 0 || ng <= 0 || K % ng || dst_bytes % rows) return RAD_E_SHAPE;
        return i8_rows((uint8_t*)dst, rows, K, ng, dst_bytes / rows);
    }
    if (!xe_hc_f16_on()) return r4d_rad_relayout_hc_e4m3(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
    RadLayout L{};
    Geo g;
    const int rc = geo(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.rows; ++r) {
        uint16_t* row = (uint16_t*)(d + r * g.stride);
        for (long long k = 0; k < g.K; ++k) row[k] = f16_of(rad_fp8e4m3_to_f32(codes[r * g.K + k]));
        std::memcpy(d + r * g.stride + 2 * g.K, sc + r * g.ng, (size_t)(4 * g.ng));
    }
    return RAD_OK;
}

extern "C" int xe_hc_unrelayout_e4m3(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                     const int* sel, const void* src, int64_t src_bytes, const RadTensor* pl,
                                     int n) {
    if (xe_hc_i8_on()) return RAD_E_UNSUPPORTED;   /* the int8 rows are lossy: the codes cannot be recovered */
    if (!xe_hc_f16_on()) return r4d_rad_unrelayout_hc_e4m3(p, n_p, operand, enc, sel, src, src_bytes, pl, n);
    RadLayout L{};
    Geo g;
    const int rc = geo(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    for (long long r = 0; r < g.rows; ++r) {
        const uint16_t* row = (const uint16_t*)(s + r * g.stride);
        for (long long k = 0; k < g.K; ++k)
            ((uint8_t*)pl[0].data)[r * g.K + k] = rad_f32_to_fp8e4m3(f32_of(row[k]));
        std::memcpy((float*)pl[1].data + r * g.ng, s + r * g.stride + 2 * g.K, (size_t)(4 * g.ng));
    }
    return RAD_OK;
}
