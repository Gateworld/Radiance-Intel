#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 113, owner notes 69 / 144 item 2): m3/make-rt.sh for the 26.35
# compiler's build -- refresh images/m1rt/ctx from build-m1l0neo's products and build localhost/ri-m3rtneo:N with
# images/m1rt/Containerfile.neo2635 (NEO 26.35 / IGC 2.41.5 at run time too). The next m3/make-rt.sh puts
# build-m1l0's products back into ctx. Then `podman save ... | ssh intel-host podman load` with `load`.
#   m3/make-rt-neo2.sh N [load|-] [TREE CONTAINERFILE [HCPF_MODULE]]
# run 207 (owner note 202 A.1): TREE (default build-m1l0neo) and CONTAINERFILE for another compiler's products and
# runtime: e.g. /srv/bulk/ri/build-m1l0neo2640 and Containerfile.neo2640 (NEO 26.40 / IGC 2.42.6 at run time) -> still
# localhost/ri-m3rtneo:N
# run 271 (owner note 208 item 1 "adopt if clean", the coordinator 16:3x 11-10): the CONTAINERFILE default is
# Containerfile.neo2640l0135 -- NEO 26.40 / IGC 2.42.6 with the Level Zero LOADER 1.35.0 in the runtime stage (image
# 296: the peer probe, the 7K logprobs and the decode text identical to image 295's, speeds within noise).
# Containerfile.neo2640 stays as the 1.29.0 fallback. The TREE is still given explicitly (the 2640 tree).
set -u
N=${1:?tag number}; LOAD=${2:-}
W=$HOME/code/radiance-intel; B=${3:-$W/build-m1l0neo}; CF=${4:-Containerfile.neo2640l0135}; C=$W/images/m1rt/ctx
# run 284: a tree configured with XE_GDN_SG_DIAG=ON (the GDN sweep's timing-only kernel variants, bench builds only)
# never becomes an image -- configure it with -DXE_GDN_SG_DIAG=OFF (build-l0-neo.sh's EXTRA) and rebuild first
if grep -q "^XE_GDN_SG_DIAG:BOOL=ON" "$B/CMakeCache.txt" 2>/dev/null; then
  echo "refused: $B is configured with XE_GDN_SG_DIAG=ON (bench build only) -- rebuild it with -DXE_GDN_SG_DIAG=OFF first"
  exit 1
fi
for f in radiance rad-kbench rad-info; do cp $B/bin/$f $C/bin/$f || exit 1; done
cp $B/lib/librad_l0k.so $C/lib/librad_l0k.so || exit 1
for f in device_test xe_event_stress xe_gemv_i8_test xe_lane_test xe_moe_w4_test xe_tape_bench xe_tape_lifetime_test xe_moe_planes_bench xe_allreduce_test xe_kernel_props xe_native_launch_test xe_native_tape_test xe_matrix_query xe_e4m3_decode_test xe_moe_bf16_test xe_list_pool_probe xe_hostbind_probe; do
  [ -f $B/tests/$f ] && cp $B/tests/$f $C/tests/$f
done
cp $B/radiance_home/kernels/libxe.so $C/radiance_home/kernels/libxe.so || exit 1
# run 261 (owner note 209; the supervisor's fix approved 21:20 11-07): hc_read's prefill matrix-unit kernels as built
# by IGC 2.28.4 (build-m1l0), whatever TREE gives the rest -- 2.42.6's build of them hangs the first prefill chunk.
# run 280 (owner note 223 step 3): a fifth argument names another module file -- the 2640 tree's own (IGC 2.42.6) build
# of the repaired kernel (32-bit offsets) once HC2 has shown it byte-equal; the default stays build-m1l0's.
mkdir -p $C/radiance_home/kernels/xe
HCPF=${5:-$W/build-m1l0/radiance_home/kernels/xe/libxe_hcpf.so}
cp $HCPF $C/radiance_home/kernels/xe/libxe_hcpf.so || exit 1
echo "libxe_hcpf.so from $HCPF ($(sha256sum $HCPF | cut -c1-16))"
cp $B/radiance_home/kernels/libref.so $C/radiance_home/kernels/libref.so || exit 1
# run 52 (owner note 47 item 2): the spechub plugin (the MTP coupling's schemas and host reference rows; its device
# rows are HIP's) rides along when the build made it, for --kernels libxe:libref:spechub
[ -f $B/radiance_home/kernels/spechub.so ] && cp $B/radiance_home/kernels/spechub.so $C/radiance_home/kernels/spechub.so
# run 53: the architecture plugins too -- the image had carried the m1 build's (run 53's coupled start found the
# engine and sampler new and qwen4exp_fp8.so old: "the architecture declared no coupled MTP planes")
for f in $B/radiance_home/architectures/*.so; do cp $f $C/radiance_home/architectures/$(basename $f) || exit 1; done
cp $B/radiance_home/kernels.rkb $C/radiance_home/kernels.rkb; cp $B/radiance_home/kernels_moe.rkb $C/radiance_home/kernels_moe.rkb
L=$W/logs/build-rtneo-m3-$N.txt
df -B1 $W | tail -1
timeout 1h podman build --cpu-quota 800000 --memory 20g --label ri-probe=1 -f $W/images/m1rt/$CF \
  -t localhost/ri-m3rtneo:$N $W/images/m1rt > "$L" 2>&1
rc=$?; echo "image build rc=$rc ($L)"; [ $rc = 0 ] || { tail -20 "$L"; exit $rc; }
podman images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "ri-m3rtneo:$N"
if [ "$LOAD" = load ]; then
  timeout 40m podman save localhost/ri-m3rtneo:$N | ssh intel-host podman load 2>&1 | tail -2
fi
