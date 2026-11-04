/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (milestone 3, run 167, owner note 168 item 1; run 169, owner note 173).
 *
 * xe_fmha256.cpp -- sycl-tla's flash-attention forward (Intel's CUTLASS, BSD-3-Clause, third_party/sycl-tla at
 * 7b56b93) instantiated at this model's head size with the per-query page selection as a mask. The epilogue and the
 * scheduler are the tree's; the mainloop is the tree's with the mask added, and the kernel the tree's with the kv
 * length readable from device memory (fmha/flash_attention_v2/..., found first by include order). One sequence of T
 * query rows over a context of ctx keys, bf16 Q / K / V / O, causal, no paging: K and V dense [kv_heads][ctx][256].
 * The 256-query x 32-key tiles, 32 sub-groups of 16 lanes and 256 registers a lane are vllm-xpu-kernels'
 * chunk_policy_head256 (FM1 / FQ1, run 165: 2.7 / 3.8 ms a 4,096-row chunk of 12 heads on the B70).
 *
 * The launch (run 169): the kernel as a functor with its properties (sub-group 16, GRF 256) and the dynamic
 * work-group memory of the engine's launch path (xe_lanes.h launch_as): while the engine records or glues the queue
 * into its native Level Zero lists, the launch is appended there through the hooks the caller resolved; otherwise a
 * plain submit on the queue. Either way it is ordered behind the engine's kernels and the caller's prep kernels. */
#include "xe_fmha256.h"
#include "rad_l0_rec.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include <cute/tensor.hpp>

#include <cstring>
#include <new>

namespace {
using namespace cute;
using Element = cutlass::bfloat16_t;
using ShapeQK = Shape<_256, _32, _32>;
using ShapePV = Shape<_256, _32, _32>;
using ShapeOut = Shape<_256, _256>;
using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
constexpr int kStages = 2;
using MMAOperation = XE_DPAS_TT<8, float, Element>;
using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
constexpr int VTiles = 8;   /* 256 / 32 */
using StrideQ = Stride<int, _1, int, int>;
using StrideK = Stride<int, _1, int, int>;
using StrideV = Stride<_1, int, int, int>;
using StrideO = Stride<int, _1, int, int>;
using StrideS = Stride<_1, int, int, int>;
template <class T, class S>
using DummyTensor = decltype(make_tensor(make_gmem_ptr(static_cast<T*>(nullptr)), make_layout(repeat<rank_v<S>>(1), S{})));
using TensorQ = DummyTensor<Element, StrideQ>;
using TensorK = DummyTensor<Element, StrideK>;
using TensorV = DummyTensor<Element, StrideV>;
using TensorO = DummyTensor<Element, StrideO>;
using TensorS = DummyTensor<float, StrideS>;
using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<false>;
using MainloopDispatchPolicy = cutlass::fmha::XeDefault<kStages>;
using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
    MainloopDispatchPolicy, /*Causal*/ true, /*BlockScale*/ false, /*F8kvF16mma*/ false, /*PerTensorScale*/ false,
    /*CachedKV*/ false, /*PagedKV*/ false, TiledMMAQK, TiledMMAPV, VTiles,
    TensorQ, TensorK, TensorV, TensorS, TensorS, TensorS, TensorK, TensorV,
    void, void, void, void, void>;
using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, ShapeOut, TensorO, void>;
using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<true, false>;   /* one batch, grouped heads */
using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

namespace sxe = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

/* the dynamic work-group memory the launch asks for: the kernel's SharedStorage, and never 0 bytes (a valid
 * pointer for the empty storage the head-256 policy has) */
constexpr size_t kLds = FMHAKernel::SharedStorageSize > 64 ? (size_t)FMHAKernel::SharedStorageSize : (size_t)64;

/* The kernel functor: one by-value parameter (the params) plus the launch path's dynamic work-group memory, so the
 * recorder's descriptor (rad_l0_rec::kernel_of) finds the layout it expects; the properties are the kernel's
 * (compile-time), the same ones compat::launch gave the probes FM1 / FQ1. */
struct FmhaFn {
    FMHAKernel::Params p;
    void operator()(sycl::nd_item<3>) const {
        char* smem = static_cast<char*>(sxe::get_work_group_scratch_memory());
        FMHAKernel op;
        op(p, smem);
    }
    auto get(sxe::properties_tag) const {
        return sxe::properties{sxe::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    }
};
}  // namespace

extern "C" int xe_fmha256_run(void* queue, const XeFmhaArgs* ap, const XeFmhaHooks* hooks) {
    if (!queue || !ap) return -1;
    const XeFmhaArgs& a = *ap;
    if (!a.q || !a.k || !a.v || !a.out || !a.sel_bits || a.T <= 0 || a.ctx <= 0 || a.heads <= 0 ||
        a.kv_heads <= 0 || a.heads % a.kv_heads != 0 || a.sel_words <= 0)
        return -1;
    sycl::queue& q = *static_cast<sycl::queue*>(queue);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    ProblemShapeType shape;
    shape.batch = 1;
    shape.num_heads_q = a.heads;
    shape.num_heads_kv = a.kv_heads;
    shape.seq_len_qo = a.T;
    shape.seq_len_kv = a.ctx;          /* the length, or its bound when ctx_dev is set (the kernel takes the min) */
    shape.seq_len_kv_cache = 0;
    shape.head_size_qk = 256;
    shape.head_size_vo = 256;
    StrideQ dQ{}; get<0>(dQ) = (int)a.q_rs; get<2>(dQ) = (int)a.q_hs; get<3>(dQ) = 0;
    StrideK dK{}; get<0>(dK) = 256;         get<2>(dK) = (int)a.k_hs; get<3>(dK) = 0;
    StrideV dV{}; get<1>(dV) = 256;         get<2>(dV) = (int)a.v_hs; get<3>(dV) = 0;
    StrideO dO{}; get<0>(dO) = (int)a.o_rs; get<2>(dO) = (int)a.o_hs; get<3>(dO) = 0;
    typename FMHAKernel::Arguments args{
        { shape,
          reinterpret_cast<const Element*>(a.q), dQ,
          reinterpret_cast<const Element*>(a.k), dK,
          reinterpret_cast<const Element*>(a.v), dV,
          reinterpret_cast<Element*>(a.out), dO,
          nullptr, {}, nullptr, {}, nullptr, {},
          1.f, 1.f, 1.f, 32,
          nullptr, {}, nullptr, {},
          nullptr,
          a.ctx_dev },
        { a.scale, nullptr, 0, nullptr, a.sel_bits, a.sel_words },
        {},
        hw_info };
    if (!FMHAKernel::can_implement(args)) return -2;
    FmhaFn fn{FMHAKernel::to_underlying_arguments(args, nullptr)};
    const auto block = FMHAKernel::get_block_shape();
    const auto grid = FMHAKernel::get_grid_shape(fn.p);
    const sycl::range<3> local(block.z, block.y, block.x);
    const sycl::range<3> global((size_t)grid.z * block.z, (size_t)grid.y * block.y, (size_t)grid.x * block.x);

    /* the engine's native path (recording, or the glue behind a native play): as xe_lanes.h launch_as */
    if (hooks && hooks->rec_device && hooks->rec_launch) {
        const int rd = hooks->rec_device(&q);
        if (rd >= 0) {
            const uint32_t g3[3] = {grid.x, grid.y, grid.z}, b3[3] = {block.x, block.y, block.z};
            const RadL0RecKernel* k = rad_l0_rec::kernel_of<FmhaFn>(q, rd);
            if (!k) {
                /* no kernel object: told to the backend, which orders the plain launch that follows */
                (void)hooks->rec_launch(&q, nullptr, g3, b3, nullptr, 0, 0);
            } else {
                alignas(FmhaFn) unsigned char vz[sizeof(FmhaFn)] = {};   /* zeroed padding, as launch_as */
                new (vz) FmhaFn(fn);
                if (hooks->rec_launch(&q, k, g3, b3, vz, sizeof vz, kLds) == 0) return 0;
            }
        }
    }
    try {
        sxe::launch_config cfg(sycl::nd_range<3>(global, local), sxe::properties{sxe::work_group_scratch_size(kLds)});
        q.submit([&](sycl::handler& h) { sxe::nd_launch(h, cfg, fn); });
    } catch (const sycl::exception&) {
        return -3;
    }
    return 0;
}
