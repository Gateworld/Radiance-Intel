#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner note 145 item 1). vram-fill.sh with NEO's
# global-memory percentage keys at 100, to see whether they remove NEO 26.35's 5 % reserve from the free figure.
#   bash /out/vram-fill-pct.sh [block_mib cap_mib floor_mib]
chmod +x /out/xe_vram_fill_probe
export NEOReadDebugKeys=1 ClDeviceGlobalMemSizeAvailablePercent=100
timeout 300 /out/xe_vram_fill_probe "$@"
