#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 280, owner note 219 B.0). Inside a build image: the device pass's
# PREDEFINED macros (-dM -E, device only) for the B70 target and for the generic spir64 target -- is
# __SYCL_TARGET_INTEL_GPU_BMG_G31__ 1 in the first and absent (or 0) in the second? The driver's command lines carry no -D
# for it (m3/macro-test.sh), so if the frontend predefines it per target, this shows it.
#   bash /w/m3/macro-test2.sh
cd /tmp || exit 1
echo '' > e.cpp
for t in intel_gpu_bmg_g31 spir64; do
  echo "== -fsycl-targets=$t (device pass, predefined macros matching SYCL_TARGET / SYCL_DEVICE / bmg)"
  /opt/sycl/bin/clang++ -fsycl -fsycl-device-only -fsycl-targets=$t -dM -E -x c++ e.cpp 2>&1 | grep -iE "SYCL_TARGET|bmg|SYCL_DEVICE_ONLY|SPIR" | head -12
done
echo "== the two-target form (what libxe gets): the macros of each pass"
/opt/sycl/bin/clang++ -fsycl -fsycl-device-only -fsycl-targets=intel_gpu_bmg_g31,spir64 -dM -E -x c++ e.cpp 2>&1 | grep -iE "SYCL_TARGET|bmg|error|warning" | head -12
echo "== device_architecture.hpp's view: what does the header see under the B70 target?"
cat > a.cpp <<'EOF'
#include <sycl/sycl.hpp>
#if __SYCL_TARGET_INTEL_GPU_BMG_G31__
#warning "BMG macro is 1 in this pass"
#else
#warning "BMG macro is 0 / undefined in this pass"
#endif
int main() { sycl::queue q; q.single_task([](){}).wait(); return 0; }
EOF
/opt/sycl/bin/clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -c a.cpp -o a.o 2>&1 | grep -E "warning: \"BMG|error" | head -6
