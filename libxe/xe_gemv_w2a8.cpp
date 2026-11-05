/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * xe_gemv_w2a8.cpp -- gemm_nt_q at dtype w2a8 (the MTP draft head: u2 codes, an f16 scale and an
 * integer zero per (row, 128 K)) as a GEMV over the CANONICAL planes, written for Xe2 rather than
 * ported: libr4d's gemm_w2a8_nt is an iu8-WMMA kernel on a fragment-order weight, which the load
 * would have to build (owner note 4: the load copies, it does not rearrange).
 *
 *   y[m, n] = sum over 128-K groups g, ascending:
 *             ((isum(m, n, g) - zero[n, g] * a_sum[g][m]) * a_scale[m]) * scale[n, g]
 *
 * isum is the exact int32 sum of 128 products int8 x u2, so a group's term is libref's (ref_gemm.cpp
 * gemm_zp, ZP_I8_ROW) bit for bit; only the order the group terms are added in differs (four groups
 * at a time across the sub-group, then a butterfly).
 *
 * Layout: b [N][K] u2 rows as stored (K/4 bytes, element k at bits 2k), b_scale the joined planes
 * (abi/rad_planes.h) "scale" (f16 [N][K/128]) then "zero" (u8 or u4 [N][K/128]) -- the kernel reads
 * the zero's dtype and both offsets from the entries, so libref reads the same bytes; a [M][K] int8
 * in quant_act_i8's step order (bits 2 and 3 of the column swapped inside each 16-column step),
 * a_scale [M] f32, a_sum [K/128][Mpad] f32, y [M][N] bf16.
 *
 * Mapping: a sub-group computes one output row n; lane l takes 16-column chunk (l & 7) of group
 * 4*j + (l >> 3): 4 code bytes and 16 activation bytes. Chunk position p holds column
 * i8_step_pos(p), so activation quad q meets code byte sq(q) = q with bits 0 and 1 swapped.
 */
#include "xe_lanes.h"
#include "r4d_args.h"
#include "rad_planes.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kRowsPerWG = 8;
constexpr int kMaxM = 4;   /* activation rows a weight pass serves; more rows loop the launch */

template <int MB>
void gemv_w2a8_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as,
                      const float* __restrict__ asum, long mpad, const uint8_t* __restrict__ b,
                      const uint8_t* __restrict__ bs, uint16_t* __restrict__ y, long y_ld, int N, int K, int M,
                      uint32_t dt_u8) {
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5);
    const int G = K / 128;
    const int chunk = lane & 7, gsub = lane >> 3;
    /* the joined planes' entries: scale is plane 0, zero plane 1 (the layout hook checked the roles) */
    const RadPlanesEntry* e = (const RadPlanesEntry*)(bs + sizeof(RadPlanesHeader));
    const uint8_t* sp = bs + e[0].offset;
    const long s_rb = (long)e[0].row_bytes;
    const uint8_t* zp = bs + e[1].offset;
    const long z_rb = (long)e[1].row_bytes;
    const bool z8 = e[1].dtype == dt_u8;
    float acc[MB];
#pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
    const bool live = n < N;
    const uint8_t* brow = b + (size_t)(live ? n : 0) * (size_t)(K / 4);
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int g = g0 + gsub;
        const bool gl = live && g < G;
        uint32_t w = 0;
        if (gl) w = *(const uint32_t*)(brow + (size_t)g * 32 + chunk * 4);
        float sb = 0.0f, z = 0.0f;
        if (gl) {
            sb = (float)*(const __fp16*)(sp + (size_t)n * s_rb + (size_t)g * 2);
            const uint8_t zb = zp[(size_t)n * z_rb + (size_t)(z8 ? g : g >> 1)];
            z = (float)(z8 ? zb : (uint8_t)((zb >> (4 * (g & 1))) & 15));
        }
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            int s = 0;
            if (gl && m < M) {
                const uint4 x = *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16);
                const unsigned xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int sq = (q == 1) ? 2 : (q == 2 ? 1 : q);
                    const unsigned cb = (w >> (8 * sq)) & 0xffu;
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        s += (int)(int8_t)(xv[q] >> (8 * k)) * (int)((cb >> (2 * k)) & 3u);
                }
            }
            /* the group's exact int32 sum over its 8 lanes */
            s += __builtin_amdgcn_ds_bpermute((lane ^ 1) << 2, s);
            s += __builtin_amdgcn_ds_bpermute((lane ^ 2) << 2, s);
            s += __builtin_amdgcn_ds_bpermute((lane ^ 4) << 2, s);
            if (gl && m < M && chunk == 0) {
                const float sub = (float)s - z * asum[(size_t)g * mpad + m];
                acc[m] += (sub * as[m]) * sb;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        float v = acc[m];
        v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 16);
        if (live && lane == 0 && m < M) {
            const uint32_t u = __float_as_uint(v);   /* round to nearest even, NaN kept quiet */
            y[(size_t)m * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                 : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
        }
    }
}

/* The rows form (owner note 151 C, run 123): R output rows a sub-group. Per group a lane read 4 bytes of codes but 16
 * bytes of activation for every row and every output row -- the same activation for all of them, a lane's (group,
 * chunk) being the same for every row. Here a sub-group takes R output rows and loads each group's activation chunk
 * once (MB rows), then for each of its rows does exactly the one-row-a-sub-group form's work: the same exact int32
 * sum, the same butterfly, the same float expression, the groups in the same order, so the same bytes. */
/* The sum of the four signed byte products plus acc: DP4A's arithmetic, in the form IGC recognises (xe_moe_i4i8.cpp's
 * dp4; sycl's dot_acc is out of line and cannot be linked twice). */
inline int32_t w2_dp4(int32_t a, int32_t b, int32_t acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}
/* A code byte's four u2 codes (element k at bits 2k) spread to the four bytes of a word, byte k = code k: the codes
 * are 0..3, so as signed bytes they are the same values. */
inline int32_t w2_spread(uint32_t cb) {
    return (int32_t)((cb & 3u) | ((cb & 0xCu) << 6) | ((cb & 0x30u) << 12) | ((cb & 0xC0u) << 18));
}

/* kD4 (owner note 183 item 1, RADIANCE_XE_W2A8_DP4=1): each quad's four products by one DP4A on the spread codes,
 * the spread done once a (row, group) for every activation row, in place of sixteen extracts and multiply-adds a chunk.
 * The group's int32 sum is exact either way (|128 x 127 x 3| is far inside int32), so the same bytes. */
template <int MB, int R, bool kD4 = false>
void gemv_w2a8_rows_kernel(const int8_t* __restrict__ a, long a_ld, const float* __restrict__ as,
                           const float* __restrict__ asum, long mpad, const uint8_t* __restrict__ b,
                           const uint8_t* __restrict__ bs, uint16_t* __restrict__ y, long y_ld, int N, int K, int M,
                           uint32_t dt_u8) {
    const int lane = (int)threadIdx.x & 31;
    const int n0 = ((int)blockIdx.x * kRowsPerWG + ((int)threadIdx.x >> 5)) * R;
    const int G = K / 128;
    const int chunk = lane & 7, gsub = lane >> 3;
    const RadPlanesEntry* e = (const RadPlanesEntry*)(bs + sizeof(RadPlanesHeader));
    const uint8_t* sp = bs + e[0].offset;
    const long s_rb = (long)e[0].row_bytes;
    const uint8_t* zp = bs + e[1].offset;
    const long z_rb = (long)e[1].row_bytes;
    const bool z8 = e[1].dtype == dt_u8;
    float acc[R][MB];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int g = g0 + gsub;
        const bool gg = g < G;
        uint4 xq[MB];
#pragma unroll
        for (int m = 0; m < MB; ++m)
            xq[m] = gg && m < M ? *(const uint4*)(a + (size_t)m * a_ld + (size_t)g * 128 + chunk * 16) : uint4{0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r;
            if (n >= N) break;   /* uniform over the sub-group */
            const bool gl = gg;
            const uint8_t* brow = b + (size_t)n * (size_t)(K / 4);
            uint32_t w = 0;
            if (gl) w = *(const uint32_t*)(brow + (size_t)g * 32 + chunk * 4);
            float sb = 0.0f, z = 0.0f;
            if (gl) {
                sb = (float)*(const __fp16*)(sp + (size_t)n * s_rb + (size_t)g * 2);
                const uint8_t zb = zp[(size_t)n * z_rb + (size_t)(z8 ? g : g >> 1)];
                z = (float)(z8 ? zb : (uint8_t)((zb >> (4 * (g & 1))) & 15));
            }
            int32_t cw[4] = {0, 0, 0, 0};   /* kD4: activation quad q's codes, spread (code byte sq(q)) */
            if constexpr (kD4) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int sq = (q == 1) ? 2 : (q == 2 ? 1 : q);
                    cw[q] = w2_spread((w >> (8 * sq)) & 0xffu);
                }
            }
#pragma unroll
            for (int m = 0; m < MB; ++m) {
                int s = 0;
                if (gl && m < M) {
                    const unsigned xv[4] = {xq[m].x, xq[m].y, xq[m].z, xq[m].w};
                    if constexpr (kD4) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) s = w2_dp4((int32_t)xv[q], cw[q], s);
                    } else {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int sq = (q == 1) ? 2 : (q == 2 ? 1 : q);
                            const unsigned cb = (w >> (8 * sq)) & 0xffu;
#pragma unroll
                            for (int k = 0; k < 4; ++k)
                                s += (int)(int8_t)(xv[q] >> (8 * k)) * (int)((cb >> (2 * k)) & 3u);
                        }
                    }
                }
                s += __builtin_amdgcn_ds_bpermute((lane ^ 1) << 2, s);
                s += __builtin_amdgcn_ds_bpermute((lane ^ 2) << 2, s);
                s += __builtin_amdgcn_ds_bpermute((lane ^ 4) << 2, s);
                if (gl && m < M && chunk == 0) {
                    const float sub = (float)s - z * asum[(size_t)g * mpad + m];
                    acc[r][m] += (sub * as[m]) * sb;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            float v = acc[r][m];
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 16);
            if (n < N && lane == 0 && m < M) {
                const uint32_t u = __float_as_uint(v);
                y[(size_t)m * y_ld + n] = (uint16_t)(((u & 0x7fffffffu) > 0x7f800000u) ? ((u >> 16) | 0x40u)
                                                                                     : ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16));
            }
        }
    }
}

/* RADIANCE_XE_W2A8_R: the rows form (owner note 151 C, run 123) -- the default R=2 since owner note 155
 * (bit-identical); 4 the wider form, 0 one row a sub-group. Read at every call (a few a step), so a test can compare
 * the forms in one process. */
int w2a8_rows() {
    const char* v = std::getenv("RADIANCE_XE_W2A8_R");
    const int r = v ? std::atoi(v) : 2;
    return r == 2 || r == 4 ? r : 0;
}
/* RADIANCE_XE_W2A8_DP4=1 (owner note 183 item 1, opt-in until timed): the rows form's quads by DP4A (kD4 above). Read
 * at every call, as RADIANCE_XE_W2A8_R, so the test compares both in one process. */
bool w2a8_dp4() {
    const char* v = std::getenv("RADIANCE_XE_W2A8_DP4");
    return v && v[0] == '1';
}

/* The w2a8 grid's b_scale: the encoding's "scale" (f16) and "zero" (u8 or u4), one each per
 * (row, 128 K), over u2 codes -- libref's ref_layout_gemm_q check, at group 128. */
bool w2_view_ok(const RadParam* p, int n_p, const RadEncoding* enc, const int* sel, int n) {
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    if (!dt || std::strcmp(dt, "w2a8") || rad_param_geti(p, n_p, "group", 0) != 128) return false;
    if (!enc || !rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes < 3 || !sel || n != 2) return false;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return false;
    const RadEncPlane& c = enc->plane[0];
    const RadEncPlane& s = enc->plane[sel[0]];
    const RadEncPlane& z = enc->plane[sel[1]];
    if (std::strncmp(s.role, "scale", RAD_ENC_STR) || std::strncmp(z.role, "zero", RAD_ENC_STR)) return false;
    if (std::strncmp(c.role, "codes", RAD_ENC_STR) || c.dtype != RAD_U2 || c.block[0] != 1 || c.block[1] != 1)
        return false;
    return s.kind == RAD_PLANE_TILED && s.dtype == RAD_F16 && s.block[0] == 1 && s.block[1] == 128 &&
           z.kind == RAD_PLANE_TILED && (z.dtype == RAD_U8 || z.dtype == RAD_U4) && z.block[0] == 1 &&
           z.block[1] == 128;
}

}  // namespace

/* Operand 3 (b_scale) joined; the codes (operand 2) are one plane and stored as they are. */
extern "C" int xe_w2a8_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                              const RadTensor* pl, int n, RadLayout* out) {
    if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
    if (!w2_view_ok(p, n_p, enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, pl, n, out);
}
extern "C" int xe_w2a8_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
    if (!w2_view_ok(p, n_p, enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_relayout(enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int xe_w2a8_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                  const void* src, int64_t src_bytes, const RadTensor* pl, int n) {
    if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
    if (!w2_view_ok(p, n_p, enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, pl, n);
}

/* gemm_nt_q operands: a, a_scale, b, b_scale, y, a_sum [, b_ref]. */
extern "C" int xe_rad_gemm_w2a8_gemv(const RadArgs* args, RadStream s) {
    const RadTensor* A = r4d_opt(args, 0);
    const RadTensor* AS = r4d_opt(args, 1);
    const RadTensor* B = r4d_opt(args, 2);
    const RadTensor* BS = r4d_opt(args, 3);
    const RadTensor* Y = r4d_opt(args, 4);
    const RadTensor* SUM = r4d_opt(args, 5);
    if (!A || !AS || !B || !BS || !Y || !SUM || !A->data || !AS->data || !B->data || !BS->data || !Y->data ||
        !SUM->data)
        return R4D_NO(RAD_E_INVAL);
    if (r4d_opt(args, 6)) return R4D_NO(RAD_E_UNSUPPORTED);   /* no e8m0 reference */
    const char* dt = r4d_gets(args, "dtype");
    if (!dt || std::strcmp(dt, "w2a8")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(args, "pb1", 0) || r4d_geti_or(args, "pb2", 0) || r4d_geti_or(args, "split_n", 0))
        return R4D_NO(RAD_E_UNSUPPORTED);
    if (A->dtype != RAD_I8 || B->dtype != RAD_U2 || AS->dtype != RAD_F32 || SUM->dtype != RAD_F32 ||
        Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long long K = r4d_geti_or(args, "K", r4d_dim(B, (int)B->rank - 1));
    const long long N = r4d_geti_or(args, "N", r4d_dim(B, (int)B->rank - 2));
    if (K <= 0 || N <= 0 || K % 128 || r4d_geti_or(args, "group", 128) != 128) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(B) || rad_tensor_numel(B) != N * K) return R4D_NO(RAD_E_STRIDE);
    long long M = 0, Ma = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &M);
    const long long a_ld = r4d_rows_pitch(A, K, &Ma);
    if (y_ld < 0 || a_ld < 0 || !r4d_contig(AS) || !r4d_contig(SUM)) return R4D_NO(RAD_E_STRIDE);
    const long long mpad = (M + 15) / 16 * 16;
    if (M <= 0 || Ma < M || rad_tensor_numel(AS) < M || rad_tensor_numel(SUM) < (K / 128) * mpad)
        return R4D_NO(RAD_E_SHAPE);
    /* 16-byte activation loads, 4-byte code loads (a row is K/4 bytes, a multiple of 32), the joined
     * entries' int64 fields (the planes inside are 64-byte aligned to the base) */
    if ((uintptr_t)A->data % 16 || a_ld % 16 || (uintptr_t)B->data % 4 || (uintptr_t)BS->data % 8)
        return R4D_NO(RAD_E_STRIDE);
    try {
        sycl::queue* q = (sycl::queue*)s;
        const dim3 grid((unsigned)((N + kRowsPerWG - 1) / kRowsPerWG)), block(32 * kRowsPerWG);
        const int rr = w2a8_rows();
        const bool d4 = w2a8_dp4();
        for (long long m0 = 0; m0 < M; m0 += kMaxM) {
            const int mb = (int)(M - m0 < kMaxM ? M - m0 : kMaxM);
            const int8_t* am = (const int8_t*)A->data + (size_t)m0 * a_ld;
            const float* asm_ = (const float*)AS->data + m0;
            const float* sm = (const float*)SUM->data + m0;
            uint16_t* ym = (uint16_t*)Y->data + (size_t)m0 * y_ld;
            if (rr > 0) {
                const dim3 gr((unsigned)((N + kRowsPerWG * rr - 1) / (kRowsPerWG * rr)));
#define XE_W2R(MB_, R_, D_) XE_LAUNCH(q, (gemv_w2a8_rows_kernel<MB_, R_, D_>), gr, block, 0, am, (long)a_ld, asm_, \
                                      sm, (long)mpad, (const uint8_t*)B->data, (const uint8_t*)BS->data, ym,     \
                                      (long)y_ld, (int)N, (int)K, mb, (uint32_t)RAD_U8)
                if (d4) {
                    if (mb == 1) { if (rr == 2) XE_W2R(1, 2, true); else XE_W2R(1, 4, true); }
                    else { if (rr == 2) XE_W2R(kMaxM, 2, true); else XE_W2R(kMaxM, 4, true); }
                } else {
                    if (mb == 1) { if (rr == 2) XE_W2R(1, 2, false); else XE_W2R(1, 4, false); }
                    else { if (rr == 2) XE_W2R(kMaxM, 2, false); else XE_W2R(kMaxM, 4, false); }
                }
#undef XE_W2R
                continue;
            }
            if (mb == 1)
                XE_LAUNCH(q, (gemv_w2a8_kernel<1>), grid, block, 0, am, (long)a_ld, asm_, sm, (long)mpad,
                          (const uint8_t*)B->data, (const uint8_t*)BS->data, ym, (long)y_ld, (int)N, (int)K, mb,
                          (uint32_t)RAD_U8);
            else
                XE_LAUNCH(q, (gemv_w2a8_kernel<kMaxM>), grid, block, 0, am, (long)a_ld, asm_, sm, (long)mpad,
                          (const uint8_t*)B->data, (const uint8_t*)BS->data, ym, (long)y_ld, (int)N, (int)K, mb,
                          (uint32_t)RAD_U8);
        }
    } catch (const sycl::exception&) {
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}
