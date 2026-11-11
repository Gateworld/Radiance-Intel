#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, owner note 53's "now" measurements). On intel-host's CPU, no model and no
# card work: (1) the host memory ceilings at 1 and 2 threads (rad-avx-bench --peak: the DRAM triad), and (2) the CPU
# time of libavx's moe_gemm_q over Qwen3.8-Flash-Next's served experts (w4nl64a8h, 512 experts, top_k 10, a new
# routing every call so the experts come from DRAM as a served decode step's would) at 1 and 2 threads -- the cost a
# host thread would pay for the units that miss VRAM. The binaries come from amd-host's build via /out (copied to /tmp
# in case the mount is noexec); libref from the image for the bench's reference handle (not timed: --no-ref).
#   m3-step.sh N53 600 <image> bash /out/n53.sh
set -u
cp /out/rad-avx-bench /tmp/b && cp /out/libavx.so /tmp/libavx.so && chmod +x /tmp/b || { echo "n53: binaries missing"; exit 2; }
R=/opt/ri/radiance_home/kernels/libref.so
echo "n53: cpus $(nproc), $(grep -m1 'model name' /proc/cpuinfo)"
for t in 1 2; do
  echo "== threads $t: ceilings"
  timeout 120 /tmp/b --avx /tmp/libavx.so --ref "$R" --peak --threads "$t" --no-ref 2>&1 | head -40
  echo "== threads $t: moe_gemm_q w4nl64a8h"
  timeout 240 /tmp/b --avx /tmp/libavx.so --ref "$R" --moe --moe-dtype w4nl64a8h --threads "$t" --no-ref 2>&1 | tail -25
done
echo "n53: done"
