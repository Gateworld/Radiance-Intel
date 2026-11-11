#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 279, owner note 221 rule 2: the smallest reproducer of the hc prefill
# module's IGC 2.42.6 hang). Inside the step's container (one card, no engine, no model): libxe's hc row test
# (xe_hc_persist_test: hc_read on deterministic inputs, the outputs to a file; RI_HC_ROWS = one row count) at 64 rows and
# then at 4,096 rows (a prefill chunk), three runs each on the SAME libxe.so (the 2640 tree's, IGC 2.42.6):
#   (1) RADIANCE_XE_HC_PF_DPAS=0: the per-row kernels (the reference bytes);
#   (2) RADIANCE_XE_HC_PF_DPAS=1 with the IGC 2.28.4 module (RADIANCE_XE_HC_PF_MOD_PATH=/out/libxe_hcpf_228.so: the module
#       the images serve prefill with today) -- the matrix-unit kernels hc_pf_dpas_kernel<0> / <1> from 64 rows;
#   (3) RADIANCE_XE_HC_PF_DPAS=1 with the IGC 2.42.6 module (/out/libxe_hcpf_2426.so: the 2640 tree's own build, the one
#       that hung the card in run 262's engine steps).
# Each run under a host timeout (a hang shows as rc 124 and in the step's klog; the kernel itself has no spin); the bytes
# compared pairwise; 4,096 rows only if 64 ran clean. The "[xe] hc_read PF module: <path>" line proves which module ran.
#   bash /out/hcpf-probe.sh [TIMEOUT=120]
TO=${1:-120}
P=/out/xe_hc_persist_test
L=/out/libxe.so
chmod +x $P
E="RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_GU=0 RADIANCE_XE_HC_D16=0"   # the ctest's pins: the plain path, the forms compared on it
echo "=== hcpf-probe ($(date -u +%H:%M:%S)): timeout $TO s a run"
ls -la /out/libxe_hcpf_228.so /out/libxe_hcpf_2426.so
for rows in 64 4096; do
    echo "--- rows $rows"
    env $E RI_HC_ROWS=$rows RADIANCE_XE_HC_PF_DPAS=0 timeout $TO $P $L /out/hc_rows_$rows.bin 2> /out/hc_rows_$rows.err
    echo "rows $rows per-row (PF_DPAS=0) rc=$? ($(date -u +%H:%M:%S))"
    env $E RI_HC_ROWS=$rows RADIANCE_XE_HC_PF_DPAS=1 RADIANCE_XE_HC_PF_MOD_PATH=/out/libxe_hcpf_228.so timeout $TO $P $L /out/hc_pf228_$rows.bin 2> /out/hc_pf228_$rows.err
    rc228=$?; echo "rows $rows module 2.28.4 rc=$rc228 ($(date -u +%H:%M:%S))"
    grep -h "matrix units\|PF module" /out/hc_pf228_$rows.err | head -2
    env $E RI_HC_ROWS=$rows RADIANCE_XE_HC_PF_DPAS=1 RADIANCE_XE_HC_PF_MOD_PATH=/out/libxe_hcpf_2426.so timeout $TO $P $L /out/hc_pf2426_$rows.bin 2> /out/hc_pf2426_$rows.err
    rc2426=$?; echo "rows $rows module 2.42.6 rc=$rc2426 ($(date -u +%H:%M:%S))"
    grep -h "matrix units\|PF module" /out/hc_pf2426_$rows.err | head -2
    cmp /out/hc_rows_$rows.bin /out/hc_pf228_$rows.bin && echo "RESULT rows $rows: 2.28.4 module = per-row bytes"
    cmp /out/hc_rows_$rows.bin /out/hc_pf2426_$rows.bin && echo "RESULT rows $rows: 2.42.6 module = per-row bytes"
    if cmp /out/hc_pf228_$rows.bin /out/hc_pf2426_$rows.bin; then echo "RESULT rows $rows: 2.42.6 module = 2.28.4 module bytes"; same=1; else same=0; fi
    if [ "$rc2426" != 0 ] || [ "$rc228" != 0 ]; then echo "STOP: a module run did not finish (rc $rc228 / $rc2426) -- no larger chunk (owner note 221 rule 3)"; break; fi
    # the coordinator's addition (23:4x): a byte DIFFERENCE between the two modules also stops here -- that is hypothesis 2
    # (the a32 argument read), reported as such, not as a pass
    if [ "$same" != 1 ]; then echo "STOP: the 2.42.6 module's bytes DIFFER from the 2.28.4 module's at $rows rows (hypothesis 2: a wrong argument read) -- no larger chunk"; break; fi
done
echo "rc=0 ($(date -u +%H:%M:%S))"
