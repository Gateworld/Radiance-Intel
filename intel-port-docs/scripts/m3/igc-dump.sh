#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 115, owner note 141 item 1). Inside localhost/ri-m1build:3:
# AOT-compile one libxe source alone (bmg_g31 only) with IGC's shader dumps on, to read the ISA IGC made of each
# kernel -- here the GEMV's two weight-scale instances, whose float arithmetic differs on the card.
#   podman run ... localhost/ri-m1build:3 bash /w/m3/igc-dump.sh SRC OUTDIR [extra compiler flags...]
# run 266: extra flags after OUTDIR go to the compile (the invoke_simd probe needs -fsycl-allow-func-ptr and
# -fno-sycl-device-code-split-esimd: the SYCL kernel and the ESIMD function in one module)
set -u
SRC=${1:-/w/src/libxe/xe_gemv_i8.cpp}; OUT=${2:-/w/igcdump}
XF=""; [ $# -gt 2 ] && XF="${*:3}"
mkdir -p "$OUT"
export IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$OUT"
timeout 1800 /opt/sycl/bin/clang++ -I/w/src/libxe/compat -I/w/src/libxe -I/w/src/libr4d -I/w/src/abi \
  -I/w/build-m1l0/rad_include -O2 -DNDEBUG -std=gnu++20 -fPIC -w -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
  -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ffp-contract=off -shared -Wl,--unresolved-symbols=ignore-all \
  $XF -o "$OUT/one.so" "$SRC"
echo "rc=$?"
ls "$OUT" | grep -c asm
