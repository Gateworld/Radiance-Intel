#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Deadcode and the radiance contributors
# Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
"""hipport.py -- the mechanical first pass of porting a libr4d unit to libxe.

    hipport.py libr4d/r4d_model_bf16.hip libxe/ports/r4d_model_bf16.cpp

The output is a starting point that is then edited by hand and committed; it is not regenerated at
build time. What it changes, and nothing else:
  - a header naming the source unit, the commit and this tool;
  - `__shared__ T a[A], b[B][C], c;` -> one XE_SHARED / XE_SHARED2 / XE_SHARED_VAR per declarator
    (xe_lanes.h), alignment attributes dropped (work-group memory is aligned for its type);
  - `extern __shared__ T name[];` -> XE_DYN_LDS;
  - `(c ? r4dchain_K : K)` -> `(K)`: chained launches are AQL-only, rad_dev_chain_enabled is 0 here.
The HIP names themselves (thread indices, lane builtins, hipLaunchKernelGGL) are xe_lanes.h's job,
reached through libxe/compat/hip/hip_runtime.h. Prints what it changed and what it left for a hand.
"""
import re
import sys

SRC_COMMIT = '9eaf10d'

TYPE = r'((?:(?:unsigned|signed|long|short|const|volatile)\s+)*[\w:]+(?:\s*<[^;>]*>)?)'
ALIGN = r'(?:__align__\(\d+\)\s+|alignas\(\d+\)\s+|__attribute__\(\(aligned\(\d+\)\)\)\s+)*'
dyn = re.compile(r'extern\s+__shared__\s+' + ALIGN + TYPE + r'\s+' + ALIGN + r'(\w+)\s*\[\s*\]\s*;')
shared = re.compile(r'__shared__\s+' + ALIGN + TYPE + r'\s+' + ALIGN + r'([^;{}]+);')
decl = re.compile(r'^\s*(\w+)\s*((?:\[[^\]]+\]\s*)*)$', re.S)
chain = re.compile(r'\(\s*\w+\s*\?\s*r4dchain_\w+(?:\s*<[^>]*>)?[\s\\]*:\s*(\w+(?:\s*<[^>]*>)?)\s*\)')


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    name = src.split('/')[-1]
    n = {'shared': 0, 'dyn': 0, 'chain': 0}
    bad = []

    def rd(m):
        n['dyn'] += 1
        return 'XE_DYN_LDS(%s, %s);' % (m.group(1).strip(), m.group(2))

    def rs(m):
        t, rest = m.group(1).strip(), m.group(2)
        out = []
        for d in rest.split(','):
            dm = decl.match(d)
            if not dm:
                bad.append(m.group(0))
                return m.group(0)
            dims = re.findall(r'\[([^\]]+)\]', dm.group(2))
            v = dm.group(1)
            if len(dims) == 0:   out.append('XE_SHARED_VAR(%s, %s);' % (t, v))
            elif len(dims) == 1: out.append('XE_SHARED(%s, %s, %s);' % (t, v, dims[0].strip()))
            elif len(dims) == 2: out.append('XE_SHARED2(%s, %s, %s, %s);' % (t, v, dims[0].strip(), dims[1].strip()))
            else:
                bad.append(m.group(0))
                return m.group(0)
        n['shared'] += len(out)
        return ' '.join(out)

    def rc(m):
        n['chain'] += 1
        return '(%s)' % m.group(1)

    # comments are left alone: only code lines are rewritten
    lines = text.split('\n')
    for i, l in enumerate(lines):
        code = l.split('//')[0]
        if '__shared__' in code and not code.lstrip().startswith('*') and not code.lstrip().startswith('/*'):
            l2 = dyn.sub(rd, l)
            l2 = shared.sub(rs, l2)
            lines[i] = l2
    text = chain.sub(rc, '\n'.join(lines))
    head = ('/* SPDX-License-Identifier: Apache-2.0\n'
            ' * Copyright 2026 Deadcode and the radiance contributors\n'
            ' * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).\n'
            ' *\n'
            ' * Ported from libr4d/%s at %s: first pass by libxe/tools/hipport.py, then edited by hand.\n'
            ' * Compiled as SYCL behind libxe/xe_lanes.h (sub-group 32); the original header follows. */\n'
            % (name, SRC_COMMIT))
    open(dst, 'w').write(head + text)
    left = []
    for i, l in enumerate(text.split('\n')):
        code = l.split('//')[0]
        if ('__shared__' in code and not code.lstrip().startswith(('*', '/*'))) or ('r4dchain_' in code and '?' in code):
            left.append(i + 7)
    print('%s: %s; left for a hand at lines %s' % (name, n, left[:40]))


if __name__ == '__main__':
    main()
