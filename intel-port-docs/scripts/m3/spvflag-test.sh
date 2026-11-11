#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 280, owner note 219 B.0). Inside a build image: a dry run of the
# two-target link (intel_gpu_bmg_g31,spir64, libxe's) with the SPIR-V translator flag that allows SPV_KHR_shader_clock in
# its per-target forms, to see which llvm-spirv call (the AOT one, whose output names bmg_g31, or the JIT one) receives
# the extension, and whether a call gets two -spirv-ext arguments. Run 167's flagtest asked the same for the FMHA module
# and chose a single-target module instead; the all-reduce's stamps live in libxe itself, so the two-target form matters.
#   bash /w/m3/spvflag-test.sh
cd /tmp || exit 1
echo 'int main(){}' > t.cpp
/opt/sycl/bin/clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -c t.cpp -o t.o || exit 1
summarize() {
  grep -E '"llvm-spirv"|unused|error|cannot deduce' | while IFS= read -r line; do
    n=$(printf '%s' "$line" | grep -o -- '-spirv-ext=' | wc -l)
    m=$(printf '%s' "$line" | grep -o 'SPV_KHR_shader_clock' | wc -l)
    if printf '%s' "$line" | grep -q bmg_g31; then k=AOT; else k=JIT; fi
    case "$line" in *llvm-spirv*) echo "  call $k: $n -spirv-ext args, $m with shader_clock" ;; *) echo "  $(printf '%s' "$line" | cut -c1-160)" ;; esac
  done
}
echo "== -Xspirv-translator=intel_gpu_bmg_g31 --spirv-ext=+SPV_KHR_shader_clock"
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -Xspirv-translator=intel_gpu_bmg_g31 --spirv-ext=+SPV_KHR_shader_clock t.o -o t.out 2>&1 | summarize
echo "== -Xspirv-translator=spir64 --spirv-ext=+SPV_KHR_shader_clock"
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -Xspirv-translator=spir64 --spirv-ext=+SPV_KHR_shader_clock t.o -o t.out 2>&1 | summarize
echo "== -Xspirv-translator=spir64_gen --spirv-ext=+SPV_KHR_shader_clock"
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -Xspirv-translator=spir64_gen --spirv-ext=+SPV_KHR_shader_clock t.o -o t.out 2>&1 | summarize
echo "== both: =intel_gpu_bmg_g31 and =spir64"
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -Xspirv-translator=intel_gpu_bmg_g31 --spirv-ext=+SPV_KHR_shader_clock -Xspirv-translator=spir64 --spirv-ext=+SPV_KHR_shader_clock t.o -o t.out 2>&1 | summarize
echo "== single target (the clock probe's form), for reference"
/opt/sycl/bin/clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31 -c t.cpp -o t1.o || exit 1
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xspirv-translator --spirv-ext=+SPV_KHR_shader_clock t1.o -o t1.out 2>&1 | summarize
echo "== the full llvm-spirv line of the AOT call under =intel_gpu_bmg_g31 (ext list)"
/opt/sycl/bin/clang++ -### -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -Xspirv-translator=intel_gpu_bmg_g31 --spirv-ext=+SPV_KHR_shader_clock t.o -o t.out 2>&1 | grep '"llvm-spirv"' | grep -o -- '-spirv-ext=[^"]*' | cut -c1-120
echo "== any SYCL clock API in the headers?"
grep -rln "ReadClockKHR\|kernel_clock\|read_clock\|device_clock" /opt/sycl/include/sycl 2>/dev/null | head -5
