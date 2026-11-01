/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * xe_moe_w4nl_i8.cpp -- moe_gemm_q at w4nl64i8h ("format C", owner note 147, run 116): xe_moe_i4i8.cpp's run form
 * over today's joined w4nl64a8h entries against int8 activations, compiled here as its own translation unit (its
 * own device image). The kernel and its launch are xe_moe_i4i8.cpp's, with XE_I4I8_FC = 1.
 */
#define XE_I4I8_FC 1
#include "xe_moe_i4i8.cpp"
