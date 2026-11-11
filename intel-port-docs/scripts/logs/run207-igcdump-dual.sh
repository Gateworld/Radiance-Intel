#!/bin/bash
# Run 207: m3/igc-dump.sh's compile with libxe's real offload targets (intel_gpu_bmg_g31,spir64) and the dp4a
# define given to the card frontend alone, as libxe/CMakeLists.txt does -- does the card pass see XE_DP4A_GPU?
#   podman run ... localhost/ri-m1buildneo:2 bash /w/logs/run207-igcdump-dual.sh OUTDIR
set -u
OUT=${1:-/w/igcdump-gemv-dual}
mkdir -p "$OUT"
export IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$OUT"
timeout 1800 /opt/sycl/bin/clang++ -I/w/src/libxe/compat -I/w/src/libxe -I/w/src/libr4d -I/w/src/abi \
  -I/w/build-m1l0/rad_include -O2 -DNDEBUG -std=gnu++20 -fPIC -w -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 \
  -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -ffp-contract=off \
  -Xsycl-target-frontend=spir64 -DXE_DP4A_PORTABLE=1 \
  -shared -Wl,--unresolved-symbols=ignore-all -o "$OUT/one.so" /w/src/libxe/xe_gemv_i8.cpp
echo "rc=$?"
ls "$OUT" | grep -c asm
