#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner note 145 item 1). Inside the step's container
# (card1, no model): how much VRAM the card really allocates under the image's runtime (xe_vram_fill_probe).
#   bash /out/vram-fill.sh [block_mib cap_mib floor_mib]
chmod +x /out/xe_vram_fill_probe
timeout 300 /out/xe_vram_fill_probe "$@"
