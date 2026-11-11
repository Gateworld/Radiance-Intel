#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3). Refresh images/m1rt/ctx from build-m1l0's products
# (plain cp, as the permission rules allow) and build the card1 runtime image localhost/ri-m3rt:N
# (images/m1rt/Containerfile: the AOT binaries + the runtime libraries from intel/llvm + Intel's GPU
# stack at intel-host's versions). Then `podman save localhost/ri-m3rt:N | ssh intel-host podman load`.
#   m2/make-rt.sh N [load]
set -u
N=${1:?tag number}; LOAD=${2:-}
W=$HOME/code/radiance-intel; B=$W/build-m1l0; C=$W/images/m1rt/ctx
for f in radiance rad-kbench rad-info; do cp $B/bin/$f $C/bin/$f || exit 1; done
cp $B/lib/librad_l0k.so $C/lib/librad_l0k.so || exit 1
for f in device_test xe_event_stress xe_gemv_i8_test xe_lane_test xe_moe_w4_test xe_tape_bench xe_tape_lifetime_test xe_moe_planes_bench xe_allreduce_test xe_kernel_props xe_native_launch_test xe_native_tape_test xe_matrix_query xe_e4m3_decode_test xe_moe_bf16_test xe_list_pool_probe xe_hostbind_probe; do
  [ -f $B/tests/$f ] && cp $B/tests/$f $C/tests/$f
done
cp $B/radiance_home/kernels/libxe.so $C/radiance_home/kernels/libxe.so || exit 1
cp $B/radiance_home/kernels/libref.so $C/radiance_home/kernels/libref.so || exit 1
# run 52 (owner note 47 item 2): the spechub plugin (the MTP coupling's schemas and host reference rows; its device
# rows are HIP's) rides along when the build made it, for --kernels libxe:libref:spechub
[ -f $B/radiance_home/kernels/spechub.so ] && cp $B/radiance_home/kernels/spechub.so $C/radiance_home/kernels/spechub.so
# run 169 (owner notes 166 / 174): libavx, the production host rows -- its embed_lookup_q reads the n-gram table through
# an io_uring row cache with a read-ahead thread (libavx/avx_ngram.cpp) where libref's page-faults the mmap row by row;
# served only when --kernels names it (m3-prefill.sh's ...avx configs) and the container allows io_uring
cp $B/radiance_home/kernels/libavx.so $C/radiance_home/kernels/libavx.so || exit 1
# run 167 (owner note 168 item 1): the flash-attention module (sycl-tla's kernel, its own SYCL module), loaded by
# libxe beside itself when attn_qsa_fmha runs
if [ -f $B/radiance_home/kernels/xe/libxe_fmha.so ]; then
    mkdir -p $C/radiance_home/kernels/xe
    cp $B/radiance_home/kernels/xe/libxe_fmha.so $C/radiance_home/kernels/xe/libxe_fmha.so || exit 1
fi
# run 53: the architecture plugins too -- the image had carried the m1 build's (run 53's coupled start found the
# engine and sampler new and qwen4exp_fp8.so old: "the architecture declared no coupled MTP planes")
for f in $B/radiance_home/architectures/*.so; do cp $f $C/radiance_home/architectures/$(basename $f) || exit 1; done
cp $B/radiance_home/kernels.rkb $C/radiance_home/kernels.rkb; cp $B/radiance_home/kernels_moe.rkb $C/radiance_home/kernels_moe.rkb
L=$W/logs/build-rt-m3-$N.txt
df -B1 $W | tail -1
timeout 1h podman build --cpu-quota 800000 --memory 20g --label ri-probe=1 -t localhost/ri-m3rt:$N $W/images/m1rt > "$L" 2>&1
rc=$?; echo "image build rc=$rc ($L)"; [ $rc = 0 ] || { tail -20 "$L"; exit $rc; }
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m3rt:$N"
if [ "$LOAD" = load ]; then
  timeout 40m podman save localhost/ri-m3rt:$N | ssh intel-host podman load 2>&1 | tail -2
fi
