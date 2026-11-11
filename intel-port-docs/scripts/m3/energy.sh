#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 128 item 1: the clock-lock A/B). One line: TAG,
# the local epoch (s), each card's energy1_input (uJ, "card": the whole card; 0b = card0, 0f = card1) and min_freq
# (MHz). Two lines around a step give each card's mean power: (E1 - E0) / (t1 - t0) / 1e6 W.
#   bash m3/energy.sh TAG >> logs/energy.txt
S=/sys/bus/pci/devices
echo "$1 $(date +%s) $(ssh intel-host cat $S/0000:0b:00.0/hwmon/hwmon4/energy1_input $S/0000:0f:00.0/hwmon/hwmon5/energy1_input $S/0000:0b:00.0/tile0/gt0/freq0/min_freq $S/0000:0f:00.0/tile0/gt0/freq0/min_freq | tr '\n' ' ')"
