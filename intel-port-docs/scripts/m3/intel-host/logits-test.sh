#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 123, owner note 151 C). Inside the step's container (card1):
# xe_logits_i8_test against the step folder's libxe.so -- the int8 lm_head's 4-row form against the rows form
# (RADIANCE_XE_LOGITS_R=2 / 4): the same bytes, and the time a call -- at the served head's shape (V 124,160, K 2560).
#   bash /out/logits-test.sh [V=124160] [K=2560] [ITERS=20]
chmod +x /out/xe_logits_i8_test
timeout 300 /out/xe_logits_i8_test /out/libxe.so "${1:-124160}" "${2:-2560}" "${3:-20}"
