#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 116, owner note 146 item 1). Inside the step's container (card1,
# STEP_ENGINE=1 for /d): GL's suspected prefill defect. gemm_nt_q on the served container's weights at prompt-sized
# row counts with the matrix units off (RADIANCE_XE_GEMM_DPAS=0, as the replay ran), the GEMV's default form and
# RADIANCE_XE_GEMV_GL=1 -- each case's worst error against libref side by side; a defect shows as a failed case or
# an error far above the GEMV's.
#   bash /out/kb-gl2.sh [MAXTOK=64 ...]
cp /out/kbench-all.sh /tmp/kb.sh
for mt in ${*:-64 256}; do
  bash /tmp/kb.sh g0-$mt "$mt" gemm_nt_q RADIANCE_XE_GEMM_DPAS=0
  bash /tmp/kb.sh g1-$mt "$mt" gemm_nt_q RADIANCE_XE_GEMM_DPAS=0 RADIANCE_XE_GEMV_GL=1
  echo "== max-tok $mt: cases whose line differs (GEMV < > GL)"
  diff /out/r-g0-$mt-gemm_nt_q.md /out/r-g1-$mt-gemm_nt_q.md | grep -E "^[<>] \| .gemm_nt_q" | head -40
done
