#!/bin/bash
# Run 207 (supervisor, note 205): while ri-igc-build runs, stop it if amd-host's MemAvailable dips under 40 GiB.
while sleep 30; do
    a=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$a" -lt 40 ]; then
        podman stop ri-igc-build
        echo "STOPPED ri-igc-build: MemAvailable $a GiB at $(date -u +%H:%M:%S)"
        exit 0
    fi
    podman ps -q -f name=ri-igc-build | grep -q . || { echo "ri-igc-build gone at $(date -u +%H:%M:%S) (MemAvailable $a GiB)"; exit 0; }
done
