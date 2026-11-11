#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 280, owner note 219 B.0). Inside a build image, a dry run: does the
# two-target device compile (intel_gpu_bmg_g31,spir64 -- libxe's) define __SYCL_TARGET_INTEL_GPU_BMG_G31__ in the B70's
# pass and NOT in the spir64 pass? xe_allreduce.cpp's ar_clock() compiles the SPIR-V clock read under that macro only, so
# the CPU device's image never carries it. Prints each device-side clang invocation's target and the macro's presence.
#   bash /w/m3/macro-test.sh
cd /tmp || exit 1
echo 'int main(){}' > m.cpp
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -c m.cpp -o m.o 2>&1 \
  | grep -E '"-fsycl-is-device"' | while IFS= read -r line; do
    tgt=$(printf '%s' "$line" | grep -o -- '-triple" "[^"]*' | head -1 | cut -d'"' -f3)
    aot=$(printf '%s' "$line" | grep -o -- '-fsycl-targets=[^" ]*\|bmg_g31' | head -1)
    if printf '%s' "$line" | grep -q '__SYCL_TARGET_INTEL_GPU_BMG_G31__=1'; then m="BMG macro =1"; else m="no BMG macro"; fi
    n=$(printf '%s' "$line" | grep -o -- '-D__SYCL_TARGET_INTEL_GPU_[A-Z0-9_]*=1' | wc -l)
    echo "device pass: triple $tgt ${aot:+($aot)} -> $m ($n target macros set)"
  done
