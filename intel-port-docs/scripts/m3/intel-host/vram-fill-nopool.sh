#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner note 145 item 1). vram-fill.sh with NEO's USM
# allocation pools off (debug keys), to see whether they are the 1.6 GiB NEO 26.35 holds at start (VF2).
#   bash /out/vram-fill-nopool.sh
chmod +x /out/xe_vram_fill_probe
export NEOReadDebugKeys=1 EnableDeviceUsmAllocationPool=0 EnableHostUsmAllocationPool=0
timeout 300 /out/xe_vram_fill_probe 256
