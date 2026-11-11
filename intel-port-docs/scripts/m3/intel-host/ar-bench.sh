#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 102, owner note 109 item 1). The two-rank all_reduce's handshake
# forms (RADIANCE_XE_AR_PROTO 0-3), each first through allreduce_test's correctness cases (every value against the
# host sum, both ranks equal), then timed on the device at the decode message (AR_BENCH: 2560 x 1 and x 4 bf16).
# libxe.so and xe_allreduce_test come from the step folder (a build newer than the image). A stalled pair gives up
# in seconds (XE_AR_SPIN), never minutes.
#   m3-step.sh STEP 900 <image> bash /out/ar-bench.sh [PROTOS=0,1,2,3] [N=2000]
set -u
L=/out/libxe.so; T=/out/xe_allreduce_test
export XE_AR_SPIN=${XE_AR_SPIN:-20000000}
# the instances' bound 4x the message, as the engine's band bound exceeds a decode message (the LL form, PROTO=4, needs 2x)
export AR_BOUND_X=${AR_BOUND_X:-4}
for p in $(echo "${1:-0,1,2,3}" | tr ',' ' '); do
  echo "=== RADIANCE_XE_AR_PROTO=$p: correctness ($(date -u +%H:%M:%S))"
  RADIANCE_XE_AR_PROTO=$p timeout 300 $T $L 20 | grep -E "RESULT|stalled|differ|wrong"
  for rows in 1 4; do
    echo "=== RADIANCE_XE_AR_PROTO=$p: AR_BENCH ${2:-2000}, rows $rows"
    RADIANCE_XE_AR_PROTO=$p AR_BENCH=${2:-2000} AR_ROWS=$rows timeout 300 $T $L | grep RESULT
  done
done
echo "done ($(date -u +%H:%M:%S))"
