#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, kernel 4). Inside ri-m3rt on one card: every registered
# kernel's properties (xe_kernel_props: SIMD width, register use, spill/scratch bytes) for libxe.so on
# device 0, to /out/kernel_props.txt; the GDN recurrent update's instantiations are grepped for the step's
# output.
/opt/ri/tests/xe_kernel_props /opt/ri/radiance_home/kernels/libxe.so 0 > /out/kernel_props.txt 2>&1
echo "kernel_props: exit $? ($(grep -c . /out/kernel_props.txt) lines)"
grep -i "recurrent_update" /out/kernel_props.txt | cut -c1-240 | head -20
