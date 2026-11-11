#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner note 141 item 1). Inside localhost/ri-m1build:3 on
# amd-host's CPU SYCL device: xe_gemv_i8_test with the GEMV's default form, then with RADIANCE_XE_GEMV_GL=1 and =2 --
# each case's y hash must be equal and "0 of N checked differ from the GEMV's order".
#   podman run ... localhost/ri-m1build:3 bash /w/m3/gemv-gl-cpu.sh
export RI_DEVICE=cpu ONEAPI_DEVICE_SELECTOR=opencl:cpu OMP_NUM_THREADS=8 SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/w/sycl-cache
T=/w/build-m1l0/tests/xe_gemv_i8_test
for gl in 0 1 2; do
  echo "=== RADIANCE_XE_GEMV_GL=$gl"
  env RADIANCE_XE_GEMV_GL=$gl timeout 900 $T | grep -E "M +[0-9]+ N|gemv_i8_test:"
done
