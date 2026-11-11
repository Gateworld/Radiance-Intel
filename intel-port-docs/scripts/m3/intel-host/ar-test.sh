#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, runs 102-103). The two-rank all_reduce test against the IMAGE's
# libxe.so (the shipped kernels), REPS times, so a failure is told from a one-off: on iommu=pt (owner note 116) AB1
# found the peer form's large messages (262,144 / 65,536 elements) differing between the ranks. Extra environment
# (a form's flag, e.g. RADIANCE_XE_AR_PROTO=4) as arguments; owner note 120 asks for 20 repetitions of the large cases.
#   m3-step.sh STEP 900 <image> bash /out/ar-test.sh [REPS=2] [ITERS=20] [VAR=value ...]
set -u
chmod +x /out/xe_allreduce_test 2>/dev/null   # run 122: a copied binary arrives without its mode
export XE_AR_SPIN=${XE_AR_SPIN:-20000000}
REPS=${1:-2}; ITERS=${2:-20}; shift 2 2>/dev/null
pass=0; fail=0
for r in $(seq "$REPS"); do
  echo "=== rep $r $* ($(date -u +%H:%M:%S))"
  out=$(env "$@" timeout 300 /out/xe_allreduce_test /opt/ri/radiance_home/kernels/libxe.so "$ITERS")
  echo "$out" | grep -E "RESULT allreduce |differ|wrong|stalled|FAIL" | grep -v " PASS,"
  if echo "$out" | grep -q "RESULT allreduce_test PASS"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done
echo "reps: $pass PASS, $fail FAIL"
