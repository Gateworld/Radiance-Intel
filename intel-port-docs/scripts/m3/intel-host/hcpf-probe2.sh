#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 280, owner note 223 step 2: HC2). Inside the step's container
# (one card, no engine, no model): libxe's hc row test (xe_hc_persist_test, RI_HC_ROWS = one row count) at 64 rows and
# then at 4,096 rows (a prefill chunk), on the SAME libxe.so (the 2640 tree's, IGC 2.42.6), four runs a row count under
# a host timeout each:
#   (1) RADIANCE_XE_HC_PF_DPAS=0: the per-row kernels (the reference bytes; they differ from the matrix-unit form at byte
#       513 by that form's own summation order -- known, B199 / B155 -- not a failure);
#   (2) PF_DPAS=1 with TODAY's module (libxe_hcpf_228u.so: the IGC 2.28.4 build of the unpatched kernel, the images');
#   (3) PF_DPAS=1 with the PATCHED module built by IGC 2.28.4 (libxe_hcpf_228p.so);
#   (4) PF_DPAS=1 with the PATCHED module built by IGC 2.42.6 (libxe_hcpf_2426p.so: the one HC1's unpatched build hung on).
# The gate: (2) = (3) = (4) byte for byte at both row counts, every rc 0. A hang shows as rc 124 (and in the step's klog);
# a non-zero rc, or a byte difference between the modules, stops before the larger chunk (owner note 223 / 221 rule 3).
#   bash /out/hcpf-probe2.sh [TIMEOUT=120]
TO=${1:-120}
P=/out/xe_hc_persist_test
L=/out/libxe.so
chmod +x $P
E="RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_GU=0 RADIANCE_XE_HC_D16=0"   # the ctest's pins: the plain path, the forms compared on it
echo "=== hcpf-probe2 ($(date -u +%H:%M:%S)): timeout $TO s a run"
ls -la /out/libxe_hcpf_228u.so /out/libxe_hcpf_228p.so /out/libxe_hcpf_2426p.so
for rows in 64 4096; do
    echo "--- rows $rows"
    env $E RI_HC_ROWS=$rows RADIANCE_XE_HC_PF_DPAS=0 timeout $TO $P $L /out/hc_rows_$rows.bin 2> /out/hc_rows_$rows.err
    echo "rows $rows per-row (PF_DPAS=0) rc=$? ($(date -u +%H:%M:%S))"
    stop=0
    for m in 228u 228p 2426p; do
        env $E RI_HC_ROWS=$rows RADIANCE_XE_HC_PF_DPAS=1 RADIANCE_XE_HC_PF_MOD_PATH=/out/libxe_hcpf_$m.so timeout $TO $P $L /out/hc_pf${m}_$rows.bin 2> /out/hc_pf${m}_$rows.err
        rc=$?; echo "rows $rows module $m rc=$rc ($(date -u +%H:%M:%S))"
        grep -h "matrix units\|PF module" /out/hc_pf${m}_$rows.err | head -2
        if [ "$rc" != 0 ]; then echo "STOP: module $m did not finish at $rows rows (rc $rc) -- no further run (owner note 223 / 221 rule 3)"; stop=1; break; fi
    done
    [ $stop = 1 ] && break
    cmp /out/hc_rows_$rows.bin /out/hc_pf228u_$rows.bin > /dev/null 2>&1 && echo "RESULT rows $rows: today's module = per-row bytes" || echo "RESULT rows $rows: today's module differs from the per-row bytes (the known order difference)"
    same=1
    cmp /out/hc_pf228u_$rows.bin /out/hc_pf228p_$rows.bin && echo "RESULT rows $rows: patched 2.28.4 module = today's module bytes" || same=0
    cmp /out/hc_pf228u_$rows.bin /out/hc_pf2426p_$rows.bin && echo "RESULT rows $rows: patched 2.42.6 module = today's module bytes" || same=0
    if [ "$same" != 1 ]; then echo "STOP: a patched module's bytes DIFFER from today's at $rows rows -- no larger chunk"; break; fi
done
echo "rc=0 ($(date -u +%H:%M:%S))"
