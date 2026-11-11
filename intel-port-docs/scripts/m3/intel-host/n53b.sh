#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, owner note 53). N53's second half alone: libavx's moe_gemm_q over
# Qwen3.8-Flash-Next's served experts (w4nl64a8h, the model's geometry, a new routing a call) at THREADS threads,
# without the bench's FMA ceiling probes, which took intel-host's Tctl to 79 C in 9 s (N53). The DRAM read ceiling at
# 1 thread came from N53 (29.5 GB/s); --moe prints the expert bytes a call reads and the time.
#   m3-step.sh N53b 300 <image> bash /out/n53b.sh THREADS
set -u
T=${1:-1}
cp /out/rad-avx-bench /tmp/b && cp /out/libavx.so /tmp/libavx.so && chmod +x /tmp/b || { echo "n53b: binaries missing"; exit 2; }
echo "== threads $T: moe_gemm_q w4nl64a8h"
timeout 200 /tmp/b --avx /tmp/libavx.so --ref /opt/ri/radiance_home/kernels/libref.so --moe --moe-dtype w4nl64a8h \
    --threads "$T" --no-ref --iters 5 2>&1 | tail -30
echo "n53b: done"
