#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner note 141 item 1). Inside the step's container (card1,
# STEP_ENGINE=1 for /d): gemm_nt_q on the served container's weights at up to 4 and 15 rows, the GEMV's default form
# and RADIANCE_XE_GEMV_GL=1 -- the reports' md5 must match if GL keeps every bit on the real shapes (M187's text
# differed from M97' with GL; the synthetic test's hashes did not).
#   bash /out/kb-gl.sh
cp /out/kbench-all.sh /tmp/kb.sh
for mt in 4 15; do
  bash /tmp/kb.sh gl0-$mt "$mt" gemm_nt_q
  bash /tmp/kb.sh gl1-$mt "$mt" gemm_nt_q RADIANCE_XE_GEMV_GL=1
done
for mt in 4 15; do
  echo "== max-tok $mt: case lines that differ (GEMV < > GL)"
  diff /out/r-gl0-$mt-gemm_nt_q.md /out/r-gl1-$mt-gemm_nt_q.md | head -20
done
