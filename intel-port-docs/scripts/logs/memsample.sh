#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 179): intel-host's /proc/meminfo every 15 s for SECONDS (MemAvailable, AnonPages, Cached,
# Shmem, Dirty, Mlocked, Unevictable, in MiB) -- what grows under a long step: anonymous memory, page cache or pins.
#   bash logs/memsample.sh SECONDS > logs/mem-STEP.csv
set -u
ssh intel-host "end=\$(( \$(date +%s) + $1 )); echo t,avail,anon,cached,shmem,dirty,mlocked,unevict; \
  while [ \$(date +%s) -lt \$end ]; do \
    awk -v t=\$(date +%s) '/^(MemAvailable|AnonPages|Cached|Shmem|Dirty|Mlocked|Unevictable):/ {v[\$1]=int(\$2/1024)} \
      END {print t\",\"v[\"MemAvailable:\"]\",\"v[\"AnonPages:\"]\",\"v[\"Cached:\"]\",\"v[\"Shmem:\"]\",\"v[\"Dirty:\"]\",\"v[\"Mlocked:\"]\",\"v[\"Unevictable:\"]}' /proc/meminfo; \
    sleep 15; done"
