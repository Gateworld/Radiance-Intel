#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 53, owner note 47 item 2). The B70's FP64: the matrix query
# binary from amd-host's build (xe_matrix_query with the fp64 aspect line and a 1,000-term double sum on the device),
# copied to /tmp because the mount may be noexec.
#   m3-step.sh Q1 300 <image> bash /out/q1.sh
set -u
cp /out/xe_matrix_query /tmp/q && chmod +x /tmp/q || { echo "q1: no binary in /out"; exit 2; }
timeout 120 /tmp/q 0
echo "q1: rc=$?"
