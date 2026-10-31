/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, owner note 109 item 5).
 *
 * The GDN chunk scan on the matrix units (joint_matrix at sub-group 16), included by
 * r4d_gdn_chunk_scan_k128_v128_c64_bf16.cpp after its emulated-WMMA kernel; the same algebra, the
 * same operands and the same bf16 roundings (W, sA, V', the state's bf16 copy, gs*K), so only the
 * order of the fp32 sums inside a 16-deep product differs. Opt-in (owner note 96):
 * RADIANCE_XE_GDN_CS_JM=64 | 32 | 16 picks the v block (the emulated form otherwise). GPU only.
 *
 * Work-group: 8 sub-groups of 16, one 8-token row tile each (BT = 64), and a block of BVJ value
 * columns (grid V/BVJ x H x N, so 16 and 32 trade repeated U / P products for more work-groups).
 * Per chunk, with every operand row-major in LDS (joint_matrix A [8][16], B [16][16]):
 *   (1) staged: q, k [t][k]; kT [k][t]; V -> D [t][v]; A [t][j]; the gates.
 *   U = K@S, Oq = Q@S, P = Q@K^T  (one k loop; A rows are this sub-group's own tokens)
 *   W = beta*(V - ge*U) -> s.W;  Oq -> scratch over this sub-group's own K rows (dead by then)
 *   (2) V' = A@W; sA = scale*e^{g_i-g_j}*tril(P) over the own A rows; V' -> D; kT *= gs
 *   (3) O = scale*ge*Oq + sA@V' -> global;  S = e^{gl}*S + (gs*K)^T@V' (accumulators; the decay
 *       through the scratch); S -> s.S
 *   (4)
 * q, k, A and sA rows are read only by the sub-group that owns them; W, D, kT and S cross
 * sub-groups, hence the four barriers. Accumulator elements are reached through a per-sub-group
 * 8 x 16 fp32 scratch (store, then lane = column), as xe_moe_planes.cpp's matrix form does. */
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace cs_jm {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kSG = 16;
constexpr int kNW = BT / 8;                 // one 8-token row tile a sub-group
constexpr int kNT = kNW * kSG;              // 128 work-items

using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;
using MA  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major>;
using MB  = jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::row_major>;

template <int BVJ> struct Smem {
  unsigned short q[BT][KD];                 // A: Q
  unsigned short k[BT][KD];                 // A: K; then each sub-group's Oq (8 x BVJ fp32) over its rows
  unsigned short kT[KD][BT];                // B: K^T for P; then A: (gs*K)^T for the state
  unsigned short S[KD][BVJ];                // B: the state's bf16 copy, [k][v]
  unsigned short A[BT][BT];                 // A: FLA's solve_tril output; then sA, row tile by row tile
  unsigned short W[BT][BVJ];                // B: W
  unsigned short D[BT][BVJ];                // V; then V' (B for O and for the state)
  float c[kNW][8 * 16];                     // a sub-group's accumulator scratch
  float ge[BT], gv[BT], gg[BT], bt[BT];     // gv = gs, gg = raw g
};
static_assert(8 * 64 * sizeof(float) <= 8 * KD * sizeof(unsigned short), "Oq must fit over 8 K rows");

template <class T> inline auto lp(T* p) {
  return sycl::address_space_cast<sycl::access::address_space::local_space,
                                  sycl::access::decorated::no>(p);
}

template <int BVJ, int SDT, bool H0, bool HT>
void kernel(const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
            const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
            const float* __restrict__ gcs, const float* __restrict__ beta,
            const typename cs_state<SDT>::T* __restrict__ h0, unsigned short* __restrict__ o,
            typename cs_state<SDT>::T* __restrict__ ht, const int* __restrict__ cu,
            const int* __restrict__ sidx, long sidx_stride, long st_slot, long st_head,
            int H, int Hg, float scale)
{
  XE_SHARED_VAR(Smem<BVJ>, s);
  constexpr int NC = BVJ / 16;              // 16-column tiles of the v block
  constexpr int NJ = BT / 16;               // 16-column tiles of a chunk's tokens
  const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
  const int tid = threadIdx.x, w = tid / kSG, lane = tid % kSG;
  const int hq = hv / (H / Hg), v0 = bv * BVJ;
  const int tb = cu[nq], te = cu[nq + 1];
  const long si = sidx ? (long)sidx[(size_t)nq * sidx_stride] : (long)nq;
  if (si < 0) return;                       // a padded row owns no state
  const size_t st_base = (size_t)si * st_slot + (size_t)hv * st_head;
  const int nchunk = (te - tb + BT - 1) / BT;
  const int r0 = 8 * w;                     // this sub-group's tokens
  const int kr0 = 16 * w;                   // its state rows (k): two 8-row tiles
  sycl::sub_group sg = xe::sg();
  float* const cw = s.c[w];
  float* const oq = (float*)&s.k[r0][0];

  // The state: St[rt][c] holds S[kr0 + 8 rt .. +8][16 c .. +16] in fp32, [k][v]. h0 is [v][k].
  Acc St[2][NC];
#pragma unroll
  for (int rt = 0; rt < 2; ++rt)
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int kk = kr0 + 8 * rt + e, vv = c * 16 + lane;
        float x = 0.f;
        if constexpr (H0) x = (float)h0[st_base + (size_t)(v0 + vv) * KD + kk];
        cw[e * 16 + lane] = x;
        s.S[kk][vv] = f2bf(x);
      }
      xe::wave_barrier();
      jm::joint_matrix_load(sg, St[rt][c], lp(cw), 16, jm::layout::row_major);
      xe::wave_barrier();
    }

  const float lgs = __logf(scale);
  for (int ci = 0; ci < nchunk; ++ci) {
    const int c0 = tb + ci * BT, nval = min(BT, te - c0), lim = nval - 1;
    const float gl = gcs[(size_t)(c0 + lim) * H + hv];

    // ---- staging; rows past the chunk end are zero (the emulated form's rule, kept)
    for (int i = tid; i < BT * KD / 8; i += kNT) {
      const int r = i / (KD / 8), c = (i % (KD / 8)) * 8;
      uint4 x = PFZERO, y = PFZERO;
      if (r < nval) {
        const size_t g = ((size_t)(c0 + r) * Hg + hq) * KD + c;
        x = *(const uint4*)(q + g);
        y = *(const uint4*)(k + g);
      }
      *(uint4*)&s.q[r][c] = x;
      *(uint4*)&s.k[r][c] = y;
      unsigned short ys[8];
      __builtin_memcpy(ys, &y, sizeof ys);
#pragma unroll
      for (int e = 0; e < 8; ++e) s.kT[c + e][r] = ys[e];
    }
    for (int i = tid; i < BT * BVJ / 8; i += kNT) {
      const int r = i / (BVJ / 8), c = (i % (BVJ / 8)) * 8;
      uint4 x = PFZERO;
      if (r < nval) x = *(const uint4*)(v + ((size_t)(c0 + r) * H + hv) * VD + v0 + c);
      *(uint4*)&s.D[r][c] = x;
    }
    for (int i = tid; i < BT * BT / 8; i += kNT) {
      const int r = i / (BT / 8), c = (i % (BT / 8)) * 8;
      uint4 x = PFZERO;
      if (r < nval) x = *(const uint4*)(Amat + ((size_t)(c0 + r) * H + hv) * BT + c);
      *(uint4*)&s.A[r][c] = x;
    }
    if (tid < BT) {
      // As in the emulated form: a padding row gets gl for g (its scores stay finite), 0 for gs.
      const bool ok = tid < nval;
      const float ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
      s.bt[tid] = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
      s.ge[tid] = __expf(ggr);
      s.gg[tid] = ok ? ggr : gl;
      s.gv[tid] = ok ? __expf(gl - ggr) : 0.f;
    }
    xe::syncthreads();                                       // (1)

    // ---- U = K@S, Oq = Q@S, P = Q@K^T
    Acc U[NC], O[NC], P[NJ];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      jm::joint_matrix_fill(sg, U[c], 0.0f);
      jm::joint_matrix_fill(sg, O[c], 0.0f);
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) jm::joint_matrix_fill(sg, P[j], 0.0f);
    {
      MA ak, aq;
      MB b;
#pragma unroll
      for (int ks = 0; ks < KD / 16; ++ks) {
        jm::joint_matrix_load(sg, ak, lp((bf16*)&s.k[r0][ks * 16]), KD);
        jm::joint_matrix_load(sg, aq, lp((bf16*)&s.q[r0][ks * 16]), KD);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          jm::joint_matrix_load(sg, b, lp((bf16*)&s.S[ks * 16][c * 16]), BVJ);
          jm::joint_matrix_mad(sg, U[c], ak, b, U[c]);
          jm::joint_matrix_mad(sg, O[c], aq, b, O[c]);
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          jm::joint_matrix_load(sg, b, lp((bf16*)&s.kT[ks * 16][j * 16]), BT);
          jm::joint_matrix_mad(sg, P[j], aq, b, P[j]);
        }
      }
    }
    xe::wave_barrier();                     // this sub-group's K rows are read: Oq goes over them
#pragma unroll
    for (int c = 0; c < NC; ++c)
      jm::joint_matrix_store(sg, O[c], lp(oq + c * 16), BVJ, jm::layout::row_major);
    // W = beta * (V - ge*U), this sub-group's rows
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      jm::joint_matrix_store(sg, U[c], lp(cw), 16, jm::layout::row_major);
      xe::wave_barrier();
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int t = r0 + e, vv = c * 16 + lane;
        s.W[t][vv] = f2bf(s.bt[t] * (bf2f(s.D[t][vv]) - s.ge[t] * cw[e * 16 + lane]));
      }
      xe::wave_barrier();
    }
    xe::syncthreads();                                       // (2)

    // ---- V' = A@W; then sA over the own A rows; V' -> D; kT *= gs
    {
      Acc d[NC];
#pragma unroll
      for (int c = 0; c < NC; ++c) jm::joint_matrix_fill(sg, d[c], 0.0f);
      MA a;
      MB b;
#pragma unroll
      for (int ks = 0; ks < BT / 16; ++ks) {
        jm::joint_matrix_load(sg, a, lp((bf16*)&s.A[r0][ks * 16]), BT);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          jm::joint_matrix_load(sg, b, lp((bf16*)&s.W[ks * 16][c * 16]), BVJ);
          jm::joint_matrix_mad(sg, d[c], a, b, d[c]);
        }
      }
      xe::wave_barrier();                   // the own A rows are read: sA goes over them
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        jm::joint_matrix_store(sg, P[j], lp(cw), 16, jm::layout::row_major);
        xe::wave_barrier();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = r0 + e, jj = j * 16 + lane;
          // where kept, g_i - g_j <= 0: the exp cannot overflow (log(scale) is in the exponent)
          s.A[i][jj] = jj <= i ? f2bf(cw[e * 16 + lane] * __expf(s.gg[i] + lgs - s.gg[jj]))
                               : (unsigned short)0;
        }
        xe::wave_barrier();
      }
      // The own D rows held V, read only by this sub-group's W above.
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        jm::joint_matrix_store(sg, d[c], lp(cw), 16, jm::layout::row_major);
        xe::wave_barrier();
#pragma unroll
        for (int e = 0; e < 8; ++e) s.D[r0 + e][c * 16 + lane] = f2bf(cw[e * 16 + lane]);
        xe::wave_barrier();
      }
    }
    // K^T <- K^T * gs per token (every P read of kT retired before (2)), rounded as the emulated form
    for (int i = tid; i < KD * BT; i += kNT) {
      const int kk = i / BT, t = i % BT;
      s.kT[kk][t] = f2bf(bf2f(s.kT[kk][t]) * s.gv[t]);
    }
    xe::syncthreads();                                       // (3)

    // ---- O = scale*ge*Oq + sA@V' -> global
    {
      Acc x[NC];
#pragma unroll
      for (int c = 0; c < NC; ++c) jm::joint_matrix_fill(sg, x[c], 0.0f);
      MA a;
      MB b;
#pragma unroll
      for (int ks = 0; ks < BT / 16; ++ks) {
        jm::joint_matrix_load(sg, a, lp((bf16*)&s.A[r0][ks * 16]), BT);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          jm::joint_matrix_load(sg, b, lp((bf16*)&s.D[ks * 16][c * 16]), BVJ);
          jm::joint_matrix_mad(sg, x[c], a, b, x[c]);
        }
      }
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        jm::joint_matrix_store(sg, x[c], lp(cw), 16, jm::layout::row_major);
        xe::wave_barrier();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int t = r0 + e, vv = c * 16 + lane;
          if (t < nval) {
            const float g = scale * s.ge[t];
            o[((size_t)(c0 + t) * H + hv) * VD + v0 + vv] = f2bf(oq[e * BVJ + vv] * g + cw[e * 16 + lane]);
          }
        }
        xe::wave_barrier();
      }
    }

    // ---- S = e^{gl} * S + (gs*K)^T @ V'   (gl <= 0; the decay is already on K)
    {
      // The decay through the scratch: joint_matrix_apply on these array-held accumulators fails
      // the backend ("__spirv_AccessChain call 1st argument must be pointer to target extension type").
      const float ec = __expf(gl);
#pragma unroll
      for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          jm::joint_matrix_store(sg, St[rt][c], lp(cw), 16, jm::layout::row_major);
          xe::wave_barrier();
#pragma unroll
          for (int e = 0; e < 8; ++e) cw[e * 16 + lane] *= ec;
          xe::wave_barrier();
          jm::joint_matrix_load(sg, St[rt][c], lp(cw), 16, jm::layout::row_major);
          xe::wave_barrier();
        }
      MA a;
      MB b[NC];
#pragma unroll
      for (int ks = 0; ks < BT / 16; ++ks) {
#pragma unroll
        for (int c = 0; c < NC; ++c)
          jm::joint_matrix_load(sg, b[c], lp((bf16*)&s.D[ks * 16][c * 16]), BVJ);
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
          jm::joint_matrix_load(sg, a, lp((bf16*)&s.kT[kr0 + 8 * rt][ks * 16]), BT);
#pragma unroll
          for (int c = 0; c < NC; ++c) jm::joint_matrix_mad(sg, St[rt][c], a, b[c], St[rt][c]);
        }
      }
      // S's bf16 copy: every read of s.S this chunk retired before (2)
#pragma unroll
      for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          jm::joint_matrix_store(sg, St[rt][c], lp(cw), 16, jm::layout::row_major);
          xe::wave_barrier();
#pragma unroll
          for (int e = 0; e < 8; ++e) s.S[kr0 + 8 * rt + e][c * 16 + lane] = f2bf(cw[e * 16 + lane]);
          xe::wave_barrier();
        }
    }
    xe::syncthreads();                                       // (4)
  }

  if constexpr (!HT) return;
  typename cs_state<SDT>::T* p = ht + st_base;
#pragma unroll
  for (int rt = 0; rt < 2; ++rt)
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      jm::joint_matrix_store(sg, St[rt][c], lp(cw), 16, jm::layout::row_major);
      xe::wave_barrier();
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int kk = kr0 + 8 * rt + e, vv = c * 16 + lane;
        p[(size_t)(v0 + vv) * KD + kk] = (typename cs_state<SDT>::T)cw[e * 16 + lane];
      }
      xe::wave_barrier();
    }
}

template <int BVJ> size_t lds_need() { return sizeof(Smem<BVJ>); }

// The v block RADIANCE_XE_GDN_CS_JM asks for (64, 32 or 16), or 0: unset, another value, not a GPU,
// or a work-group's LDS too small for the block (said once on stderr).
inline int pick(sycl::queue* qq) {
  static const int bvj = [qq] {
    const char* e = std::getenv("RADIANCE_XE_GDN_CS_JM");
    const int want = e ? std::atoi(e) : 0;
    if (want != 64 && want != 32 && want != 16) return 0;
    const sycl::device d = qq->get_device();
    if (!d.is_gpu()) return 0;
    const size_t need = want == 64 ? lds_need<64>() : want == 32 ? lds_need<32>() : lds_need<16>();
    const size_t have = d.get_info<sycl::info::device::local_mem_size>();
    if (need > have) {
      std::fprintf(stderr, "W gdn chunk scan: RADIANCE_XE_GDN_CS_JM=%d needs %zu B of LDS, the device has %zu;"
                           " the emulated form runs\n", want, need, have);
      return 0;
    }
    std::fprintf(stderr, "I gdn chunk scan: the matrix-unit form, v block %d (%zu B LDS)\n", want, need);
    return want;
  }();
  return bvj;
}

template <int BVJ, int SDT, bool H0, bool HT>
void launch1(sycl::queue& qq, const void* q, const void* k, const void* v, const void* A, const void* g,
             const void* beta, const void* h0, void* o, void* ht, const void* cu, const void* sidx,
             long sidx_stride, long st_slot, long st_head, int N, int H, int Hg, float scale) {
  using T = typename cs_state<SDT>::T;
  constexpr auto K = kernel<BVJ, SDT, H0, HT>;
  xe::launch_as_grf256<K, kSG>(K, qq, dim3(VD / BVJ, H, N), dim3(kNT),
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
      (const unsigned short*)A, (const float*)g, (const float*)beta, (const T*)h0, (unsigned short*)o,
      (T*)ht, (const int*)cu, (const int*)sidx, sidx_stride, st_slot, st_head, H, Hg, scale);
}

template <int BVJ, int SDT>
void launch2(sycl::queue& qq, const void* q, const void* k, const void* v, const void* A, const void* g,
             const void* beta, const void* h0, void* o, void* ht, const void* cu, const void* sidx,
             long sidx_stride, long st_slot, long st_head, int N, int H, int Hg, float scale) {
#define CSJ(H0C, HTC) launch1<BVJ, SDT, H0C, HTC>(qq, q, k, v, A, g, beta, h0, o, ht, cu, sidx, sidx_stride, \
                                                  st_slot, st_head, N, H, Hg, scale)
  if (h0 && ht) CSJ(true, true);
  else if (h0)  CSJ(true, false);
  else if (ht)  CSJ(false, true);
  else          CSJ(false, false);
#undef CSJ
}

inline void run(int bvj, sycl::queue& qq, const void* q, const void* k, const void* v, const void* A,
                const void* g, const void* beta, const void* h0, void* o, void* ht, const void* cu,
                const void* sidx, long sidx_stride, long st_slot, long st_head, int N, int H, int Hg,
                float scale, int state_fp16) {
#define CSB(B) do { if (state_fp16) launch2<B, 1>(qq, q, k, v, A, g, beta, h0, o, ht, cu, sidx, sidx_stride, \
                                                      st_slot, st_head, N, H, Hg, scale);                  \
                    else            launch2<B, 0>(qq, q, k, v, A, g, beta, h0, o, ht, cu, sidx, sidx_stride, \
                                                      st_slot, st_head, N, H, Hg, scale); } while (0)
  if (bvj == 64) CSB(64);
  else if (bvj == 32) CSB(32);
  else CSB(16);
#undef CSB
}
}  // namespace cs_jm
