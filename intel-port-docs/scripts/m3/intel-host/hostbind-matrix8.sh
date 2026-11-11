#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 93). Inside the step's container (both cards): does a kernel's
# indirect host access make card 1 bind every host allocation of the process, the other rank's too? `pair` makes set A
# (the other rank's pool, 16.6 GiB as 2 GiB pieces, never used) and set B (this rank's, the same), card 1 copies from
# and reads only B. Every piece is 2 GiB (owner note 87).
#   b: NEO's DisableIndirectAccess=1 with B made resident on card 1 explicitly -- expected: reads right, A not bound
#   a: as the engine runs today -- if the hypothesis holds, the kernel makes card 1 bind A too and it sticks past ~26 GiB
#   bash /out/hostbind-matrix8.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
echo "=== b-pair-noindirect: NEOReadDebugKeys=1 DisableIndirectAccess=1 RI_HB_RESIDENT=1 xe_hostbind_probe 16.593 1 2048 1 pair" >> /out/hostbind.log
NEOReadDebugKeys=1 DisableIndirectAccess=1 RI_HB_RESIDENT=1 timeout 120 xe_hostbind_probe 16.593 1 2048 1 pair >> /out/hostbind.log 2>&1
echo "b-pair-noindirect rc=$? $(grep -E 'STUCK|every|failed|WRONG' /out/hostbind.log | tail -1)" >> $S
echo "=== a-pair-default: xe_hostbind_probe 16.593 1 2048 1 pair" >> /out/hostbind.log
timeout 120 xe_hostbind_probe 16.593 1 2048 1 pair >> /out/hostbind.log 2>&1
echo "a-pair-default rc=$? $(grep -E 'STUCK|every|failed|WRONG' /out/hostbind.log | tail -1)" >> $S
cat $S
