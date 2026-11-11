#!/bin/bash
# Run 212 (owner note 206 item 3): m3/igc-dump.sh with extra ocloc backend options, to see whether IGC's scheduling /
# unroll knobs reach our kernels in an AOT build (IGC's env registry keys are ignored by the release build).
#   podman run ... localhost/ri-m1buildneo:2 bash /w/logs/igcdump-opts.sh SRC OUTDIR "BACKEND OPTIONS"
set -u
SRC=$1; OUT=$2; BO=$3
mkdir -p "$OUT"
export IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$OUT"
timeout 1800 /opt/sycl/bin/clang++ -I/w/src/libxe/compat -I/w/src/libxe -I/w/src/libr4d -I/w/src/abi \
  -I/w/build-m1l0/rad_include -O2 -DNDEBUG -std=gnu++20 -fPIC -w -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
  -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ffp-contract=off -Xsycl-target-backend "$BO" \
  -shared -Wl,--unresolved-symbols=ignore-all -o "$OUT/one.so" "$SRC"
echo "rc=$?"
ls "$OUT" | grep -c asm
