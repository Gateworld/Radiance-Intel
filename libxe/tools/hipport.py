#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Deadcode and the radiance contributors
# Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
"""hipport.py -- the mechanical first pass of porting a libr4d unit to libxe.

    hipport.py libr4d/r4d_model_bf16.hip libxe/ports/r4d_model_bf16.cpp

The output is a starting point that is then edited by hand and committed; it is not regenerated at
build time. What it changes, and nothing else:
  - a header naming the source unit, the commit and this tool;
  - `__shared__ T name[A];` / `[A][B]` -> XE_SHARED / XE_SHARED2 (xe_lanes.h);
  - `extern __shared__ T name[];` -> XE_DYN_LDS;
  - `(c ? r4dchain_K : K)` -> `(K)`: chained launches are AQL-only, rad_dev_chain_enabled is 0 here.
The HIP names themselves (thread indices, lane builtins, hipLaunchKernelGGL) are xe_lanes.h's job,
reached through libxe/compat/hip/hip_runtime.h. Prints what it changed and what it left for a hand.
"""
import re
import sys

SRC_COMMIT = '9eaf10d'

shared2 = re.compile(r'__shared__\s+((?:__align__\(\d+\)\s+)?[\w:]+(?:\s*<[^;>]*>)?(?:\s+\w+)*?)\s+(\w+)\s*\[([^\]]+)\]\s*\[([^\]]+)\]\s*;')
shared1 = re.compile(r'__shared__\s+((?:__align__\(\d+\)\s+)?[\w:]+(?:\s*<[^;>]*>)?(?:\s+\w+)*?)\s+(\w+)\s*\[([^\]]+)\]\s*;')
dyn = re.compile(r'extern\s+__shared__\s+((?:__align__\(\d+\)\s+)?[\w:]+)\s+(\w+)\s*\[\s*\]\s*;')
chain = re.compile(r'\(\s*\w+\s*\?\s*r4dchain_\w+(?:\s*<[^>]*>)?\s*:\s*(\w+(?:\s*<[^>]*>)?)\s*\)')


def strip_align(t):
    return re.sub(r'__align__\(\d+\)\s+', '', t).strip()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    name = src.split('/')[-1]
    n = {'shared': 0, 'shared2': 0, 'dyn': 0, 'chain': 0}

    def r2(m):
        n['shared2'] += 1
        return 'XE_SHARED2(%s, %s, %s, %s);' % (strip_align(m.group(1)), m.group(2), m.group(3), m.group(4))

    def r1(m):
        n['shared'] += 1
        return 'XE_SHARED(%s, %s, %s);' % (strip_align(m.group(1)), m.group(2), m.group(3))

    def rd(m):
        n['dyn'] += 1
        return 'XE_DYN_LDS(%s, %s);' % (strip_align(m.group(1)), m.group(2))

    def rc(m):
        n['chain'] += 1
        return '(%s)' % m.group(1)

    text = dyn.sub(rd, text)
    text = shared2.sub(r2, text)
    text = shared1.sub(r1, text)
    text = chain.sub(rc, text)
    head = ('/* SPDX-License-Identifier: Apache-2.0\n'
            ' * Copyright 2026 Deadcode and the radiance contributors\n'
            ' * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).\n'
            ' *\n'
            ' * Ported from libr4d/%s at %s: first pass by libxe/tools/hipport.py, then edited by hand.\n'
            ' * Compiled as SYCL behind libxe/xe_lanes.h (sub-group 32); the original header follows. */\n'
            % (name, SRC_COMMIT))
    open(dst, 'w').write(head + text)
    left = [i + 1 for i, l in enumerate(text.split('\n')) if '__shared__' in l or 'r4dchain_' in l and '?' in l]
    print('%s -> %s: %s; left for a hand at lines %s' % (src, dst, n, left[:40]))


if __name__ == '__main__':
    main()
