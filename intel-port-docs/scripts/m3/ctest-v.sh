#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 113). One ctest pattern of build-m1l0, verbose (the test's own
# lines, which ctest hides on a pass), in localhost/ri-m1build:3 within m3/build-l0.sh's limits.
#   m3/ctest-v.sh PATTERN
set -u
PAT=${1:?test pattern}
W=$HOME/code/radiance-intel
timeout 30m podman run --rm --name ri-m3-ctest --cpus 16 --memory 28g -v "$W:/w" -e OMP_NUM_THREADS=8 \
    localhost/ri-m1build:3 nice -n 19 ctest --test-dir /w/build-m1l0 -R "$PAT" -V 2>&1 | tail -400
