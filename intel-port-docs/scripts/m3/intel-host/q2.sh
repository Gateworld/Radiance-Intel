#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 63, owner note 52 item 2). The branch-free E4M3 decode on the
# card: the image's xe_e4m3_decode_test (all 256 codes through both builtins in a kernel, against the ABI's decoder,
# bit for bit), on the Level Zero device m3-step.sh selects. Copied to /tmp in case the mount is noexec.
#   m3-step.sh Q2 300 <image> bash /out/q2.sh
set -u
cp /opt/ri/tests/xe_e4m3_decode_test /tmp/t && chmod +x /tmp/t || { echo "q2: no test binary in the image"; exit 2; }
timeout 120 /tmp/t
echo "q2: rc=$?"
