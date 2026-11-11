#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 279, owner note 218 stage 0): compile one ESIMD test source alone,
# ahead of time for the B70 (intel_gpu_bmg_g31) under the 26.40 / IGC 2.42.6 build image (localhost/ri-m1buildneo:2), with
# IGC's shader dumps on, into an executable under $W/OUTNAME. The stage-0 gate is the compile's rc (ocloc runs at the
# link: run 265's invoke_simd probe crashed there); the dumps give the kernel's ISA (instruction count, spills).
#   timeout 1800 bash m3/esimd-dump.sh [SRC=src/libxe/test/esimd_fc_probe.cpp] [OUTNAME=igcdump-esimd0] [IMAGE]
set -u
SRC=${1:-src/libxe/test/esimd_fc_probe.cpp}; OUT=${2:-igcdump-esimd0}; IMG=${3:-localhost/ri-m1buildneo:2}
XF=${4:-}   # extra compiler flags (B.0's clock probe: the SPIR-V shader-clock extension)
BIN=$(basename "$SRC" .cpp)
W=$HOME/code/radiance-intel
L=$W/logs/esimd-dump-$OUT-$(date -u +%Y%m%dT%H%M%S).txt
mkdir -p "$W/$OUT"
timeout 1h podman run --rm --name ri-m3-esimddump --cpus 4 --memory 8g -v "$W:/w" \
    -e IGC_ShaderDumpEnable=1 -e IGC_DumpToCustomDir=/w/$OUT \
    $IMG nice -n 19 /opt/sycl/bin/clang++ -O2 -DNDEBUG -std=gnu++20 -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
    -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ffp-contract=off $XF -o /w/$OUT/$BIN /w/$SRC > "$L" 2>&1
rc=$?
echo "esimd-dump rc=$rc log $L"
tail -20 "$L"
ls "$W/$OUT" | grep -c asm
exit $rc
