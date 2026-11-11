#!/bin/bash
export RADIANCE_XE_AR_PROTO=6 XE_AR_TEST_TEAR=2048
echo "proto $RADIANCE_XE_AR_PROTO tear $XE_AR_TEST_TEAR iters 50"
timeout 1500 /opt/ri/tests/xe_allreduce_test /opt/ri/radiance_home/kernels/libxe.so 50 2>&1 | grep -E "device|peer access|case |RESULT|FAIL|stalled|differ|canar|re-reads|rc" | tail -60
