#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 263; the owner's desk item of 2026-11-07 ~00:3x: upstream
# 1.17.0's prefill MoE GEMM for gfx1201 read as ISA). Compiles ONE HIP unit for gfx1201 inside localhost/ri-m1rocm:2
# (the HIP-check image; ~/code/radiance-intel at /w; no card) with --save-temps, the flags rad_hipify gives libr4d
# (-O3 -DNDEBUG -g0 -ffp-contract=off --no-offload-compress; VOPD pairing off as on an unpatched LLVM 24), and prints
# every kernel's resource lines from the device assembly (.vgpr_count, .sgpr_count, .group_segment_fixed_size = LDS
# bytes, .vgpr_spill_count, the ; Occupancy comment). CPU only, one container, 4 CPUs, ~minutes.
#   bash m3/hip-isa.sh SRC_ROOT UNIT OUTDIR          e.g. upstream-1.17.0 libr4d/r4d_moe.hip igcdump-amd-moe17
#   SRC_ROOT: upstream-1.17.0 (its libr4d + core, the fork's src/abi for the ABI headers) or src (the fork)
set -u
ROOT=${1:?source root under ~/code/radiance-intel}; UNIT=${2:?unit path under the root}; OUT=${3:?output dir name}
W=$HOME/code/radiance-intel
mkdir -p "$W/$OUT"
L=$W/logs/hip-isa-$OUT.txt
# the include order: the unit's own tree first (upstream's libr4d headers differ from the fork's), then the ABI
INC="-I/w/$ROOT/libr4d -I/w/$ROOT/core -I/w/src/abi -I/w/src/libr4d -I/w/src/core"
timeout 1h podman run --rm --name ri-m3-hipisa --cpus 4 --memory 8g -v "$W:/w" localhost/ri-m1rocm:2 \
  nice -n 19 bash -c "cd /w/$OUT && hipcc -x hip --offload-arch=gfx1201 -std=c++20 -O3 -DNDEBUG -g0 -ffp-contract=off \
    --no-offload-compress -mllvm=-amdgpu-enable-vopd=false $INC -fPIC --save-temps -c /w/$ROOT/$UNIT -o unit.o" > "$L" 2>&1
rc=$?; echo "hipcc rc=$rc log $L"
ls "$W/$OUT" | head -20
S=$(ls "$W/$OUT"/*gfx1201*.s 2>/dev/null | head -1)
[ -n "$S" ] || { echo "no gfx1201 .s produced"; exit 1; }
echo "device assembly: $S ($(wc -l < "$S") lines)"
# one block per kernel: its symbol, then the resource lines of its metadata / comments
grep -n -E "^\s+\.(vgpr_count|sgpr_count|group_segment_fixed_size|vgpr_spill_count|sgpr_spill_count|wavefront_size):|^\s+\.name:|; Occupancy:|; NumVgprs:|; ScratchSize:|; LDSByteSize:|; NumSgprs:" "$S" | cut -c1-120
