# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Deadcode and the radiance contributors
# Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
#
# libxe -- the SYCL kernel library for Intel Xe2 (Arc Pro B70, BMG-G31), the Level Zero build's
# counterpart of libr4d. Links radiance::abi and nothing else of the engine, like every kernel
# library; built only with RAD_WITH_L0=ON (cmake/RadL0.cmake), AOT for the B70 plus SPIR-V for the CPU
# SYCL device.
file(GLOB XE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ports/*.cpp)
# libr4d's host-side weight layouts (E4M3 / int8 planes joined at load), compiled unchanged: the rows
# libxe copies from libr4d name them (rows_from_r4d.py keeps a hook defined here).
list(APPEND XE_SRC ${CMAKE_SOURCE_DIR}/libr4d/r4d_fp8_layout.cpp ${CMAKE_SOURCE_DIR}/libr4d/r4d_layout.cpp)
# The MTP coupling's argument readers and shape hooks (plugin/spechub, upstream d912c69), compiled unchanged: the
# Xe rows in xe_mtp_couple.cpp read their operands through them (milestone 3, owner note 47 item 2).
list(APPEND XE_SRC ${CMAKE_SOURCE_DIR}/plugin/spechub/spechub.cpp)
# milestone 3, owner note 56: hc_read's mixing matrices expanded to FP16 at load under RADIANCE_XE_HC_F16=1 (the
# hooks wrap libr4d's, which stay as they are for the AMD build)
list(APPEND XE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/xe_hc_f16_layout.cpp)

rad_add_plugin(libxe kernels SOURCES ${XE_SRC})
rad_l0_sycl(libxe)
# Run 280 (owner note 219 B.0): the all-reduce's phase stamps read the device clock through SPIR-V's OpReadClockKHR
# (xe_allreduce.cpp, ar_clock; the B70's pass only). No translator flag: this toolchain's default extension list
# already allows SPV_KHR_shader_clock on both targets (m3/spvflag-test.sh, a dry run of the two-target link), and the
# bare -Xspirv-translator cannot name a target with two of them ("cannot deduce implicit triple").
# Run 207 (owner note 203 item 1): the int8 dots by IGC's dp4a builtin on the card's pass alone (xe_dp4a.h); the
# spir64 pass, which the CPU device JIT-compiles, keeps the portable form, told by a define to that triple's
# frontend (the intel_gpu_bmg_g31 form of the option reaches neither pass). Every target in this directory that
# compiles these units (libxe and the CPU tests) gets the same flag. Every unit that includes xe_dp4a.h, directly
# or through another .cpp (xe_moe_w4nl_i8.cpp includes xe_moe_i4i8.cpp), must be listed.
set_property(SOURCE xe_gemv_i8.cpp xe_gemv_w2a8.cpp xe_moe_i4i8.cpp xe_moe_w4nl_i8.cpp ports/r4d_hc_bf16.cpp
  APPEND PROPERTY COMPILE_OPTIONS -Xsycl-target-frontend=spir64 -DXE_DP4A_PORTABLE=1)
# Run 167 (owner note 168 item 1): the flash-attention form (fmha/xe_fmha256.cpp, used by xe_attn_qsa.cpp) is
# sycl-tla's kernel (Intel's CUTLASS, BSD-3-Clause, header-only, pinned at 7b56b93 in third_party/sycl-tla beside
# src/) with its mainloop copied under fmha/ and the page mask added; compiled only when that tree is present, with
# the SPIR-V extensions its kernels use (harmless for the other kernels: the CPU device never launches these).
set(_tla ${CMAKE_SOURCE_DIR}/../third_party/sycl-tla)
target_link_libraries(libxe PRIVATE ${CMAKE_DL_LIBS})   # xe_attn_qsa.cpp loads libxe_fmha.so beside libxe.so
if(EXISTS ${_tla}/include/cutlass/cutlass.h)
  # Its own module with the B70 target alone: with libxe's two targets the per-target translator flag lands
  # on the wrong calls (run 167's flagtest), and the kernel never runs on the CPU device anyway.
  add_library(xe_fmha MODULE ${CMAKE_CURRENT_SOURCE_DIR}/fmha/xe_fmha256.cpp)
  # under kernels/xe/: the engine's plugin scan takes every .so in kernels/ for a plugin (shipped_plugins_test)
  set_target_properties(xe_fmha PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/radiance_home/kernels/xe
                        PREFIX "lib" OUTPUT_NAME xe_fmha)
  target_include_directories(xe_fmha PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fmha
                             ${_tla}/applications/flash_attention_v2/collective ${_tla}/include
                             ${_tla}/tools/util/include ${_tla}/applications)
  target_link_libraries(xe_fmha PRIVATE rad_abi)   # abi/rad_l0_rec.h: the launch through the engine's recorder hooks
  target_compile_options(xe_fmha PRIVATE -fsycl -fsycl-targets=intel_gpu_bmg_g31 -DCUTLASS_ENABLE_SYCL
    -DSYCL_INTEL_TARGET -Wno-unused-local-typedef -Wno-unused-but-set-variable -Wno-uninitialized -Wno-reorder-ctor
    -Wno-logical-op-parentheses -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable)
  target_link_options(xe_fmha PRIVATE -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xspirv-translator
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
  add_dependencies(libxe xe_fmha)
  message(STATUS "libxe: the flash-attention module from ${_tla}")
else()
  message(STATUS "libxe: no third_party/sycl-tla -- attn_qsa_fmha unsupported")
endif()
# Run 261 (owner note 209; the supervisor's fix approved 21:20 11-07): hc_read's prefill matrix-unit kernels as their
# own module, kernels/xe/libxe_hcpf.so, so the image takes this one unit from the IGC 2.28.4 tree (2.42.6's build of
# them hangs the first prefill chunk). r4d_hc_bf16.cpp loads it beside itself at the first >= 64-row chunk. libxe's
# flags and includes, and rad_l0k as libxe links it: the module's launches resolve the same recorder hooks.
add_library(xe_hcpf MODULE ${CMAKE_CURRENT_SOURCE_DIR}/hcpf/xe_hcpf.cpp)
set_target_properties(xe_hcpf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/radiance_home/kernels/xe
                      PREFIX "lib" OUTPUT_NAME xe_hcpf)
rad_l0_sycl(xe_hcpf)
target_include_directories(xe_hcpf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/libr4d)
target_compile_options(xe_hcpf PRIVATE -ffp-contract=off -Wno-unused-function -Wno-unused-const-variable)
target_link_libraries(xe_hcpf PRIVATE rad_abi rad_l0k)
add_dependencies(libxe xe_hcpf)
# ports/ are libr4d units behind the shim: <hip/hip_runtime.h> resolves to compat/ (the shim), a
# header edited for Xe2 to its copy in ports/ (beside the unit), and the rest to libr4d's own.
target_include_directories(libxe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/libr4d)
# As libr4d: the ops are held bit for bit to unfused chains and to libref's rounding.
target_compile_options(libxe PRIVATE -ffp-contract=off -Wno-unused-function)
# milestone 2: the all-reduce rows allocate at init through rad_l0k (rad_l0k_device)
target_link_libraries(libxe PRIVATE rad_l0k)
# Run 102 (owner note 111): the lane layer's xor patterns as permute_group_by_xor (xe_lanes.h, XE_NATIVE_XOR). The same
# partner lanes, so the same values; off unless the owner turns it on (note 96).
# A plain variable, not a cached option, so the default here takes effect at every configure; -DRAD_XE_NATIVE_XOR=ON
# still overrides it. (Run 102's image :120 was built with it ON for its gates: E34 / KA4 / M124 / M125.)
if(NOT DEFINED RAD_XE_NATIVE_XOR)
  set(RAD_XE_NATIVE_XOR OFF)
endif()
if(RAD_XE_NATIVE_XOR)
  target_compile_definitions(libxe PRIVATE XE_NATIVE_XOR=1)
endif()
# Run 105 (owner note 109 item 5): the emulated WMMA's operand moves as broadcasts and lane ^ 16 (xe_lanes.h,
# XE_WMMA_BCAST): the same elements in the same order. A plain variable as RAD_XE_NATIVE_XOR; off in the tree (image
# :127 was built with it ON for its gates: xe_lane_cpu, KC1 / KC2, B41).
if(NOT DEFINED RAD_XE_WMMA_BCAST)
  set(RAD_XE_WMMA_BCAST OFF)
endif()
if(RAD_XE_WMMA_BCAST)
  target_compile_definitions(libxe PRIVATE XE_WMMA_BCAST=1)
endif()
# Run 284 (the coordinator's GDN diagnostic sweep): XE_GDN_SG_DIAG=ON compiles the slice-group decode kernel's
# TIMING-ONLY variants (RADIANCE_XE_GDN_SG_DIAG=1..6, ports/r4d_gdn_recurrent_update_*.cpp) into libxe -- a bench build
# alone, never an image (m3/make-rt-neo2.sh refuses a tree whose cache has it ON). A cached option: the tree keeps it
# until a configure with -DXE_GDN_SG_DIAG=OFF.
option(XE_GDN_SG_DIAG "the GDN slice-group kernel's timing-only variants (bench builds only)" OFF)
if(XE_GDN_SG_DIAG)
  target_compile_definitions(libxe PRIVATE XE_GDN_SG_DIAG=1)
endif()

if(RAD_BUILD_TESTS)
  # This directory is added before the top level's enable_testing(), so enable it here.
  enable_testing()
  # The lane shim against tests/r4d_emu: the same HIP-dialect cases through the emulator (plain C++)
  # and through xe_lanes.h (SYCL). ctest runs it on the CPU SYCL device; on the card it runs as is.
  set(_emu ${CMAKE_SOURCE_DIR}/tests/r4d_emu/emu)
  add_executable(xe_lane_test test/lane_test.cpp test/lane_emu.cpp ${_emu}/emu.cpp)
  set_source_files_properties(test/lane_test.cpp PROPERTIES COMPILE_OPTIONS "${RAD_SYCL_FLAGS}")
  set_source_files_properties(test/lane_emu.cpp ${_emu}/emu.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${_emu}" COMPILE_OPTIONS "-ffp-contract=off;-Wno-psabi")
  target_include_directories(xe_lane_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(xe_lane_test PRIVATE rad_abi)
  target_link_options(xe_lane_test PRIVATE ${RAD_SYCL_FLAGS})
  set_target_properties(xe_lane_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # the shim is tested in the form libxe is built with (run 102: XE_NATIVE_XOR)
  if(RAD_XE_NATIVE_XOR)
    target_compile_definitions(xe_lane_test PRIVATE XE_NATIVE_XOR=1)
  endif()
  if(RAD_XE_WMMA_BCAST)
    target_compile_definitions(xe_lane_test PRIVATE XE_WMMA_BCAST=1)
  endif()
  # The trunk W8A8 GEMV against libref's quant_dot order (host re-computation), canaries on y.
  add_executable(xe_gemv_i8_test test/gemv_i8_test.cpp xe_gemv_i8.cpp)
  rad_l0_sycl(xe_gemv_i8_test)
  target_include_directories(xe_gemv_i8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_compile_options(xe_gemv_i8_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_gemv_i8_test PRIVATE rad_abi)
  set_target_properties(xe_gemv_i8_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_gemv_i8_cpu COMMAND xe_gemv_i8_test)
  set_tests_properties(xe_gemv_i8_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Milestone 3, stage B: gemm_nt over bf16 weights -- the sub-group GEMV and the matrix-unit form for M >= 64 (the
  # latter on a GPU) -- against a double reference, canaries on y.
  add_executable(xe_gemv_bf16_test test/gemv_bf16_test.cpp xe_gemv_bf16.cpp)
  rad_l0_sycl(xe_gemv_bf16_test)
  target_include_directories(xe_gemv_bf16_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_compile_options(xe_gemv_bf16_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_gemv_bf16_test PRIVATE rad_abi)
  set_target_properties(xe_gemv_bf16_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_gemv_bf16_cpu COMMAND xe_gemv_bf16_test)
  set_tests_properties(xe_gemv_bf16_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 900)
  # Run 267 (owner lever #3): the protected experts' bf16 GEMM on the matrix units (xe_moe_bf16_dpas.cpp) against the
  # WMMA form's arithmetic on the host -- the ulp report on a GPU (the card probe), the launcher's guards on the CPU
  add_executable(xe_moe_bf16_dpas_test test/moe_bf16_dpas_test.cpp xe_moe_bf16_dpas.cpp)
  rad_l0_sycl(xe_moe_bf16_dpas_test)
  target_include_directories(xe_moe_bf16_dpas_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_compile_options(xe_moe_bf16_dpas_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_moe_bf16_dpas_test PRIVATE rad_abi)
  set_target_properties(xe_moe_bf16_dpas_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_moe_bf16_dpas_cpu COMMAND xe_moe_bf16_dpas_test)
  set_tests_properties(xe_moe_bf16_dpas_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Run 267 (owner lever #2, step 0 -- "card test only"): do two in-order queues run kernels concurrently on one card,
  # and how much does a memory-bound kernel slow beside a (bounded) spinner. A card probe; no CPU test.
  add_executable(two_queue_probe test/two_queue_probe.cpp)
  rad_l0_sycl(two_queue_probe)
  target_link_libraries(two_queue_probe PRIVATE ${RAD_ZE_LOADER})   # the command-queue groups' listing
  set_target_properties(two_queue_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Run 271 (owner lever #2's pricing probe, the coordinator's item 2): the engine's 16 MiB all-reduce beside the trunk's
  # int8 matrix-unit GEMM on a second in-order queue of each card, alone vs together, through libxe's entries (dlopen).
  # A two-GPU probe; no CPU test.
  add_executable(ar_overlap_probe test/ar_overlap_probe.cpp)
  rad_l0_sycl(ar_overlap_probe)
  target_include_directories(ar_overlap_probe PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(ar_overlap_probe PRIVATE rad_abi rad_l0k ${CMAKE_DL_LIBS} Threads::Threads)
  set_target_properties(ar_overlap_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(ar_overlap_probe libxe)
  # Run 207 (owner note 203 item 2): both GEMVs at every other rows-a-work-group choice (RADIANCE_XE_GEMV_WG), whose
  # small cases' N (33, 40, 77, 130, 200 ...) leave a partial last work-group at each
  # Run 209 (owner note 206 item 1): the K-specialised NX instances (RADIANCE_XE_GEMV_KC=1); `full` adds the trunk's
  # K 6144 / 2560 shapes to the small cases (K 640, 2560 and the odd ones on the runtime-K instance)
  add_test(NAME xe_gemv_bf16_kc_cpu COMMAND xe_gemv_bf16_test)
  set_tests_properties(xe_gemv_bf16_kc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_GEMV_KC=1" TIMEOUT 900)
  # Run 211 (owner note 206 item 2): the cache-hinted weight loads (RADIANCE_XE_GEMV_WCC; on the CPU the plain load,
  # so this checks the dispatch) on the NX form
  add_test(NAME xe_gemv_i8_wcc_cpu COMMAND xe_gemv_i8_test)
  set_tests_properties(xe_gemv_i8_wcc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_GEMV_NX=1;RADIANCE_XE_GEMV_WCC=5" TIMEOUT 600)
  add_test(NAME xe_gemv_i8_kc_cpu COMMAND xe_gemv_i8_test full)
  set_tests_properties(xe_gemv_i8_kc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_GEMV_NX=1;RADIANCE_XE_GEMV_KC=1" TIMEOUT 1800)
  foreach(_wg 4 16 32)
    add_test(NAME xe_gemv_i8_wg${_wg}_cpu COMMAND xe_gemv_i8_test)
    set_tests_properties(xe_gemv_i8_wg${_wg}_cpu PROPERTIES LABELS "xe"
      ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_GEMV_WG=${_wg}" TIMEOUT 600)
    add_test(NAME xe_gemv_bf16_wg${_wg}_cpu COMMAND xe_gemv_bf16_test)
    set_tests_properties(xe_gemv_bf16_wg${_wg}_cpu PROPERTIES LABELS "xe"
      ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_GEMV_WG=${_wg}" TIMEOUT 900)
  endforeach()
  # Milestone 3 (the brief's #11): moe_gemm at bf16, the decode GEMV and the WMMA form, against a double reference.
  add_executable(xe_moe_bf16_test test/moe_bf16_test.cpp)
  rad_l0_sycl(xe_moe_bf16_test)
  target_include_directories(xe_moe_bf16_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_compile_options(xe_moe_bf16_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_moe_bf16_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_moe_bf16_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_moe_bf16_test libxe)
  add_test(NAME xe_moe_bf16_cpu COMMAND xe_moe_bf16_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_moe_bf16_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  add_test(NAME xe_moe_bf16_wmma_cpu COMMAND xe_moe_bf16_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_moe_bf16_wmma_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_MOE_BF16_GEMV=0" TIMEOUT 600)
  # Run 114 (owner notes 131 / 134 item 3): format B's grouped GEMM (w4g64i8h) against a double reference.
  add_executable(xe_moe_i4i8_test test/moe_i4i8_test.cpp)
  rad_l0_sycl(xe_moe_i4i8_test)
  target_link_libraries(xe_moe_i4i8_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_moe_i4i8_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_moe_i4i8_test libxe)
  add_test(NAME xe_moe_i4i8_cpu COMMAND xe_moe_i4i8_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_moe_i4i8_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  add_test(NAME xe_moe_i4i8_kc_cpu COMMAND xe_moe_i4i8_test $<TARGET_FILE:libxe>)   # run 209: RADIANCE_XE_I4I8_KC=1
  set_tests_properties(xe_moe_i4i8_kc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_I4I8_KC=1" TIMEOUT 600)
  # run 205 (owner note 202 A.3): the run form's select-chain decode (RADIANCE_XE_I4I8_RUN_SEL=1) against the SLM
  # table, every case's output bytes compared
  add_test(NAME xe_moe_i4i8_sel_cpu COMMAND sh -c
    "rm -f i4i8_tab.bin i4i8_sel.bin && RI_DUMP=i4i8_tab.bin $<TARGET_FILE:xe_moe_i4i8_test> $<TARGET_FILE:libxe> > /dev/null && RADIANCE_XE_I4I8_RUN_SEL=1 RI_DUMP=i4i8_sel.bin $<TARGET_FILE:xe_moe_i4i8_test> $<TARGET_FILE:libxe> > /dev/null && cmp i4i8_tab.bin i4i8_sel.bin && echo 'i4i8 run SEL: bit-identical'")
  set_tests_properties(xe_moe_i4i8_sel_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1200)
  # Run 169 (owner note 173 item 1): attn_qsa_fmha's checks and prep kernels on the CPU device with the operands in the
  # engine's forms (RADIANCE_XE_ATTN_QSA_PREP_ONLY=1: the flash-attention kernel itself is card-only).
  add_executable(xe_attn_qsa_test test/attn_qsa_test.cpp)
  rad_l0_sycl(xe_attn_qsa_test)
  target_link_libraries(xe_attn_qsa_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_attn_qsa_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_attn_qsa_test libxe)
  add_test(NAME xe_attn_qsa_cpu COMMAND xe_attn_qsa_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_attn_qsa_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 900)
  # Run 123 (owner note 151 C): the int8 lm_head's 4-row form against the rows form (the same bytes), and a microbench.
  # The CPU test at a small vocabulary; the card at the real one.
  add_executable(xe_logits_i8_test test/logits_i8_test.cpp)
  rad_l0_sycl(xe_logits_i8_test)
  target_link_libraries(xe_logits_i8_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_logits_i8_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_logits_i8_test libxe)
  add_test(NAME xe_logits_i8_cpu COMMAND xe_logits_i8_test $<TARGET_FILE:libxe> 4096 2048 2)
  set_tests_properties(xe_logits_i8_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Milestone 3 (owner note 52 item 2): the shim's branch-free E4M3 decode against the ABI's, all 256 codes.
  add_executable(xe_e4m3_decode_test test/e4m3_decode_test.cpp)
  rad_l0_sycl(xe_e4m3_decode_test)
  target_include_directories(xe_e4m3_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(xe_e4m3_decode_test PRIVATE rad_abi)
  set_target_properties(xe_e4m3_decode_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_e4m3_decode_cpu COMMAND xe_e4m3_decode_test)
  set_tests_properties(xe_e4m3_decode_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 300)
  # The MTP draft head's w2a8 GEMV, through its layout hooks, against libref's gemm_zp order.
  add_executable(xe_gemv_w2a8_test test/gemv_w2a8_test.cpp xe_gemv_w2a8.cpp)
  rad_l0_sycl(xe_gemv_w2a8_test)
  target_include_directories(xe_gemv_w2a8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_compile_options(xe_gemv_w2a8_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_gemv_w2a8_test PRIVATE rad_abi)
  set_target_properties(xe_gemv_w2a8_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_gemv_w2a8_cpu COMMAND xe_gemv_w2a8_test)
  set_tests_properties(xe_gemv_w2a8_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # The shim's 2D copy (cast_bf16_bf16's), plain and recorded into a SYCL graph.
  add_executable(xe_copy2d_test test/copy2d_test.cpp xe_support.cpp)
  rad_l0_sycl(xe_copy2d_test)
  target_include_directories(xe_copy2d_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_link_libraries(xe_copy2d_test PRIVATE rad_abi)
  set_target_properties(xe_copy2d_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_copy2d_cpu COMMAND xe_copy2d_test)
  set_tests_properties(xe_copy2d_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Step 6's model: a tiny random qwen4exp checkpoint (host code, tests/tiny_model.h's helpers).
  add_executable(xe_tiny_qwen4exp test/tiny_qwen4exp.cpp)
  target_include_directories(xe_tiny_qwen4exp PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/vendor)
  target_link_libraries(xe_tiny_qwen4exp PRIVATE rad_abi)
  set_target_properties(xe_tiny_qwen4exp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # The w4nl64a8h expert GEMM through libxe's own adapter (dlopen of libxe.so) against libref's
  # arithmetic; rad-kbench cannot allocate the row's packed weight plane.
  add_executable(xe_moe_w4_test test/moe_w4_test.cpp)
  rad_l0_sycl(xe_moe_w4_test)
  target_include_directories(xe_moe_w4_test PRIVATE ${CMAKE_SOURCE_DIR}/libr4d ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_compile_options(xe_moe_w4_test PRIVATE -ffp-contract=off)
  target_link_libraries(xe_moe_w4_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_moe_w4_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_moe_w4_test libxe)
  add_test(NAME xe_moe_w4_cpu COMMAND xe_moe_w4_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_moe_w4_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1800)
  # Milestone 2, stage 2: the two-rank all_reduce rows (peer stores / host mailbox) through the plugin
  # entry points, two rank threads, against the host sum; on the CPU device as a ctest, on two GPUs timed.
  # Run 279 / 280 (owner note 219 B.0): the device clock's tick length, measured (B.0's stamps are this read at the
  # all-reduce's phase boundaries); GPU only, no ctest
  add_executable(xe_clock_probe test/clock_probe.cpp)
  rad_l0_sycl(xe_clock_probe)   # SPV_KHR_shader_clock is in the translator's default list (see libxe above)
  set_target_properties(xe_clock_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_executable(xe_allreduce_test test/allreduce_test.cpp)
  rad_l0_sycl(xe_allreduce_test)
  target_include_directories(xe_allreduce_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_allreduce_test PRIVATE rad_abi rad_l0k ${CMAKE_DL_LIBS} Threads::Threads)
  set_target_properties(xe_allreduce_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_allreduce_test libxe)
  add_test(NAME xe_allreduce_cpu COMMAND xe_allreduce_test $<TARGET_FILE:libxe> 20)
  # XE_AR_BLOCKS=4: the CPU device (16 cores) must hold both ranks' blocks at once (see xe_allreduce.cpp)
  set_tests_properties(xe_allreduce_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;XE_AR_BLOCKS=4" TIMEOUT 1800)
  # Milestone 2, owner note 24: every registered kernel's Level Zero properties (spill = scratch, private
  # memory, SIMD width); libxe through dlopen, librad_l0k's core kernels through the link.
  add_executable(xe_kernel_props test/kernel_props.cpp)
  rad_l0_sycl(xe_kernel_props)
  target_link_libraries(xe_kernel_props PRIVATE rad_l0k ${CMAKE_DL_LIBS})
  set_target_properties(xe_kernel_props PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_kernel_props libxe)
  add_test(NAME xe_kernel_props_cpu COMMAND xe_kernel_props $<TARGET_FILE:libxe>)
  set_tests_properties(xe_kernel_props_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1800)
  # Milestone 3, stage B: the device's joint_matrix (DPAS) combinations from the runtime's query, for the
  # prefill GEMM kernels' tile shapes. A card probe; on the CPU device the list may be empty.
  add_executable(xe_matrix_query test/matrix_query.cpp)
  rad_l0_sycl(xe_matrix_query)
  set_target_properties(xe_matrix_query PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_matrix_query_cpu COMMAND xe_matrix_query)
  set_tests_properties(xe_matrix_query_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Milestone 3, owner note 59 phase 0: what the driver holds per recorded command list, the GPU time a command in
  # a played list, the immediate round trip, and whether a polled ring keeps the engine busy at idle. A card probe
  # only (its own context and kernel, no model); no CPU test.
  add_executable(xe_list_pool_probe test/list_pool_probe.cpp)
  rad_l0_sycl(xe_list_pool_probe)
  target_link_libraries(xe_list_pool_probe PRIVATE ${RAD_ZE_LOADER} ${CMAKE_DL_LIBS})
  set_target_properties(xe_list_pool_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Run 111 (owner note 137 step 1): each card's command-queue groups and whether a second copy queue can be had.
  # A card probe; no CPU test.
  add_executable(xe_queue_groups_probe test/queue_groups_probe.cpp)
  rad_l0_sycl(xe_queue_groups_probe)
  target_link_libraries(xe_queue_groups_probe PRIVATE ${RAD_ZE_LOADER})
  set_target_properties(xe_queue_groups_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Run 112 (owner note 137): a second copy path measured -- bcs, a compute-group list, both; and their cost to compute.
  add_executable(xe_copy_paths_probe test/copy_paths_probe.cpp)
  rad_l0_sycl(xe_copy_paths_probe)
  target_link_libraries(xe_copy_paths_probe PRIVATE ${RAD_ZE_LOADER})
  set_target_properties(xe_copy_paths_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Milestone 3, run 88 (owner note 84): the two-card first step's stuck bind without the model -- host memory of a
  # context over both cards, touched by one card's copy queue. A two-card probe only; no CPU test.
  add_executable(xe_hostbind_probe test/hostbind_probe.cpp)
  rad_l0_sycl(xe_hostbind_probe)
  target_link_libraries(xe_hostbind_probe PRIVATE ${RAD_ZE_LOADER})
  set_target_properties(xe_hostbind_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Run 114 (owner note 145 item 1): how much VRAM one card really allocates, under the runtime it runs on.
  add_executable(xe_vram_fill_probe test/vram_fill_probe.cpp)
  rad_l0_sycl(xe_vram_fill_probe)
  set_target_properties(xe_vram_fill_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Milestone 3, owner note 61 phase 1: xe direct's standalone proof (card1 only): a SYCL kernel's zebin run from
  # the engine's own VM, buffers, batch and compute queue on the xe render node. A card probe; no CPU test.
  add_executable(xe_direct_proof test/xe_direct_proof.cpp ${CMAKE_SOURCE_DIR}/core/device/xe_direct.cpp)
  rad_l0_sycl(xe_direct_proof)
  target_link_libraries(xe_direct_proof PRIVATE ${RAD_ZE_LOADER} ${CMAKE_DL_LIBS})
  set_target_properties(xe_direct_proof PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # the zebins embedded in a host file, with their sections and relocations (CPU only; a tool, no test)
  add_executable(zebin_dump test/zebin_dump.cpp)
  set_target_properties(zebin_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # its CPU side: the command writers, the walker's fields, the cross-thread data, the zebin reader's refusals
  add_executable(xe_direct_cmds_test test/xe_direct_cmds_test.cpp ${CMAKE_SOURCE_DIR}/core/device/xe_direct.cpp)
  set_target_properties(xe_direct_cmds_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_direct_cmds_cpu COMMAND xe_direct_cmds_test)
  set_tests_properties(xe_direct_cmds_cpu PROPERTIES LABELS "xe;fast" TIMEOUT 60)
  # Milestone 3, owner note 47 item 2: libxe's MTP coupling rows against the spechub plugin's host reference rows
  # on the same bytes (dlopen of both libraries; the CPU SYCL device here, card1 in the card steps).
  add_executable(xe_mtp_couple_test test/mtp_couple_test.cpp)
  rad_l0_sycl(xe_mtp_couple_test)
  target_include_directories(xe_mtp_couple_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_mtp_couple_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_mtp_couple_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_mtp_couple_test libxe spechub)
  add_test(NAME xe_mtp_couple_cpu COMMAND xe_mtp_couple_test $<TARGET_FILE:libxe> $<TARGET_FILE:spechub>)
  set_tests_properties(xe_mtp_couple_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Run 312 (owner notes 243 item 2 / 246): the verify's two device forms against the same host reference bytes at
  # widths 64, 129, 1024 and 1025 (past the cooperative form's LDS plane: its serial fallback).
  foreach(_w 64 129 1024 1025)
    foreach(_coop 0 1)
      add_test(NAME xe_mtp_couple_w${_w}_c${_coop}_cpu COMMAND xe_mtp_couple_test $<TARGET_FILE:libxe> $<TARGET_FILE:spechub>)
      set_tests_properties(xe_mtp_couple_w${_w}_c${_coop}_cpu PROPERTIES LABELS "xe"
        ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RI_MTP_NCAND=${_w};RADIANCE_XE_MTP_COOP=${_coop}" TIMEOUT 600)
    endforeach()
  endforeach()
  # Milestone 2, owner note 28: the feasibility probe for tapes on direct Level Zero command lists (a
  # launch_as kernel launched natively, and the cost of a submission: SYCL graphs vs native lists).
  add_executable(xe_native_launch_test test/native_launch_test.cpp)
  rad_l0_sycl(xe_native_launch_test)
  target_include_directories(xe_native_launch_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_native_launch_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_native_launch_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_native_launch_cpu COMMAND xe_native_launch_test 10)
  set_tests_properties(xe_native_launch_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Milestone 2: device timing of the planes expert GEMV at Flash-Next's decode shapes (dlopen of
  # libxe.so; RADIANCE_XE_MOE_V1=1 for the old kernel). Not a ctest: a bench for card1.
  add_executable(xe_moe_planes_bench test/moe_planes_bench.cpp)
  rad_l0_sycl(xe_moe_planes_bench)
  target_include_directories(xe_moe_planes_bench PRIVATE ${CMAKE_SOURCE_DIR}/libr4d ${CMAKE_SOURCE_DIR}/abi
                             ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_link_libraries(xe_moe_planes_bench PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_moe_planes_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_moe_planes_bench libxe)
  # Run 110 (owner note 131): the i4 x i8 expert format probe, timed against libxe's expert GEMV; REF mode is the
  # host reference (amd-host). Not a ctest.
  add_executable(xe_i4i8_probe test/i4i8_probe.cpp)
  rad_l0_sycl(xe_i4i8_probe)
  target_include_directories(xe_i4i8_probe PRIVATE ${CMAKE_SOURCE_DIR}/libr4d ${CMAKE_SOURCE_DIR}/abi
                             ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_link_libraries(xe_i4i8_probe PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_i4i8_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_i4i8_probe libxe)
  # Run 190 (owner notes 188 / 189): the expert-layout probe -- format C's decode GEMM against the same arithmetic
  # over a load-time repack of the same entries, the bf16 outputs compared byte for byte, then timed per launch on
  # the card (LAUNCHES > 0). The ctest is the byte check on the CPU device (LAUNCHES 0).
  add_executable(xe_fc_tile_probe test/fc_tile_probe.cpp)
  rad_l0_sycl(xe_fc_tile_probe)
  target_include_directories(xe_fc_tile_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/libr4d
                             ${CMAKE_SOURCE_DIR}/abi ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_link_libraries(xe_fc_tile_probe PRIVATE rad_abi ${CMAKE_DL_LIBS})
  target_compile_options(xe_fc_tile_probe PRIVATE -ffp-contract=off)   # libxe's: no fused multiply-adds, same bits
  set_target_properties(xe_fc_tile_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_fc_tile_probe libxe)
  add_test(NAME xe_fc_tile_cpu COMMAND xe_fc_tile_probe $<TARGET_FILE:libxe> 0)
  set_tests_properties(xe_fc_tile_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1200)
  # Run 209 (owner note 206 item 1): format C's K-specialised gate_up instance (RADIANCE_XE_I4I8_KC=1) against the
  # probe's T1 bytes at gate_up's shape (K 2560 / 2560)
  # Run 212: format C down's decode instance at 1 and 4 columns a team (RADIANCE_XE_I4I8_DOWN_R) against T1's bytes
  foreach(_r 1 4)
    add_test(NAME xe_fc_tile_downr${_r}_cpu COMMAND xe_fc_tile_probe $<TARGET_FILE:libxe> 0)
    set_tests_properties(xe_fc_tile_downr${_r}_cpu PROPERTIES LABELS "xe"
      ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_I4I8_DOWN_R=${_r}" TIMEOUT 1200)
    add_test(NAME xe_moe_i4i8_downr${_r}_cpu COMMAND xe_moe_i4i8_test $<TARGET_FILE:libxe>)
    set_tests_properties(xe_moe_i4i8_downr${_r}_cpu PROPERTIES LABELS "xe"
      ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_I4I8_DOWN_R=${_r}" TIMEOUT 600)
  endforeach()
  # Run 279 (owner note 218 stage 0): format C's gate_up decode GEMM as a whole-kernel ESIMD form (a thread = 16 columns
  # x one of today's eight partial sums; the pair table in registers; dp4a), ahead of time for the B70. GPU only: the
  # CPU device has no ESIMD, so the ctest is the build plus the probe's "skip" exit; the card run is stage 2's.
  add_executable(xe_esimd_fc_probe test/esimd_fc_probe.cpp)
  rad_l0_sycl(xe_esimd_fc_probe)
  target_compile_options(xe_esimd_fc_probe PRIVATE -ffp-contract=off)
  set_target_properties(xe_esimd_fc_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_esimd_fc_cpu COMMAND xe_esimd_fc_probe 0)
  set_tests_properties(xe_esimd_fc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 300)
  add_test(NAME xe_fc_tile_wcc_cpu COMMAND xe_fc_tile_probe $<TARGET_FILE:libxe> 0)   # run 211: RADIANCE_XE_I4I8_WCC=5
  set_tests_properties(xe_fc_tile_wcc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_I4I8_WCC=5" TIMEOUT 1200)
  add_test(NAME xe_fc_tile_kc_cpu COMMAND xe_fc_tile_probe $<TARGET_FILE:libxe> 0)
  set_tests_properties(xe_fc_tile_kc_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_I4I8_KC=1" TIMEOUT 1200)
  # run 193 (owner note 191 item 2): the pair mode once (1 launch, 1 run) on the CPU device -- its three queue forms'
  # outputs compared byte for byte (the times mean nothing there)
  add_test(NAME xe_fc_tile_pair_cpu COMMAND xe_fc_tile_probe $<TARGET_FILE:libxe> 1 1)
  set_tests_properties(xe_fc_tile_pair_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RI_PAIR=1" TIMEOUT 1200)
  # Run 110 (owner note 129 item 4): router_topk_scatter's time a launch, against the selection alone and an empty
  # launch. Not a ctest.
  add_executable(xe_router_bench test/router_bench.cpp)
  rad_l0_sycl(xe_router_bench)
  target_include_directories(xe_router_bench PRIVATE ${CMAKE_SOURCE_DIR}/libr4d ${CMAKE_SOURCE_DIR}/abi
                             ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_link_libraries(xe_router_bench PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_router_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_router_bench libxe)
  # Run 111 (owner note 135 item 3): libxe's gram_accum (libr4d's unit, ported) against a host reference, with
  # canaries; small on the CPU device (amd-host), `full` adds the expert shapes for a card. Not a ctest.
  add_executable(xe_gram_accum_test test/gram_accum_test.cpp)
  rad_l0_sycl(xe_gram_accum_test)
  target_include_directories(xe_gram_accum_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_gram_accum_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_gram_accum_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_gram_accum_test libxe)
  # Run 112 (owner note 138): the decode GDN fold, fused against split -- output hashes and time a call. Not a ctest.
  add_executable(xe_gdn_split_bench test/gdn_split_bench.cpp)
  rad_l0_sycl(xe_gdn_split_bench)
  target_include_directories(xe_gdn_split_bench PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_gdn_split_bench PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_gdn_split_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_gdn_split_bench libxe)
  # Run 101 (owner note 108 item 1): device timing of the decode GDN update at one rank's shape. Not a ctest.
  add_executable(xe_gdn_decode_bench test/gdn_decode_bench.cpp)
  rad_l0_sycl(xe_gdn_decode_bench)
  target_include_directories(xe_gdn_decode_bench PRIVATE ${CMAKE_SOURCE_DIR}/libr4d ${CMAKE_SOURCE_DIR}/abi
                             ${CMAKE_CURRENT_SOURCE_DIR}/compat)
  target_link_libraries(xe_gdn_decode_bench PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_gdn_decode_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_gdn_decode_bench libxe)
  # Run 102 (owner note 111): the lane layer's emulated xor butterfly against permute_group_by_xor. Not a ctest.
  add_executable(xe_lane_xor_bench test/lane_xor_bench.cpp)
  rad_l0_sycl(xe_lane_xor_bench)
  target_include_directories(xe_lane_xor_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_lane_xor_bench PRIVATE rad_abi)
  set_target_properties(xe_lane_xor_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # Owner note 1's per-launch figure: N tiny kernels launched plainly and replayed from a tape.
  add_executable(xe_tape_bench test/tape_bench.cpp)
  rad_l0_sycl(xe_tape_bench)
  target_include_directories(xe_tape_bench PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(xe_tape_bench PRIVATE stdc++ rad_core)
  set_target_properties(xe_tape_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_tape_bench_cpu COMMAND xe_tape_bench 200 3 dev)
  set_tests_properties(xe_tape_bench_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RADIANCE_L0_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Owner note 7: event records far past run 24's ~74,000 moves, no stream sync; live events flat.
  add_executable(xe_event_stress test/event_stress.cpp)
  rad_l0_sycl(xe_event_stress)
  target_include_directories(xe_event_stress PRIVATE ${CMAKE_SOURCE_DIR}/core)
  target_link_libraries(xe_event_stress PRIVATE stdc++ rad_core)
  set_target_properties(xe_event_stress PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_event_stress_cpu COMMAND xe_event_stress 1000000 64)
  set_tests_properties(xe_event_stress_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RADIANCE_L0_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1800)
  # GPU faults 3-4 (owner note 10): tapes destroyed while a play runs, per-request buffers reallocated.
  add_executable(xe_tape_lifetime_test test/tape_lifetime_test.cpp)
  rad_l0_sycl(xe_tape_lifetime_test)
  target_include_directories(xe_tape_lifetime_test PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(xe_tape_lifetime_test PRIVATE stdc++ rad_core)
  set_target_properties(xe_tape_lifetime_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_tape_lifetime_cpu COMMAND xe_tape_lifetime_test 200 16)
  set_tests_properties(xe_tape_lifetime_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RADIANCE_L0_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1800)
  # Milestone 3, owner note 28: tapes on direct Level Zero command lists -- the native recorder through
  # the backend's tape API and launch_as (mode, LDS kernel, order against the runtime, events, bypass).
  add_executable(xe_native_tape_test test/native_tape_test.cpp)
  rad_l0_sycl(xe_native_tape_test)
  target_include_directories(xe_native_tape_test PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(xe_native_tape_test PRIVATE stdc++ rad_core)
  set_target_properties(xe_native_tape_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_native_tape_cpu COMMAND xe_native_tape_test 20)
  set_tests_properties(xe_native_tape_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RADIANCE_L0_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 1800)
  # The OpenCL CPU compiler crash (note 5's oracle on amd-host): the ported MoE unit alone, SPIR-V only,
  # one device image a kernel, and a harness that builds them one at a time (test/cpu_probe.cpp).
  add_library(xe_moe_cpuprobe SHARED ports/r4d_moe.cpp)
  target_include_directories(xe_moe_cpuprobe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/libr4d)
  target_compile_options(xe_moe_cpuprobe PRIVATE -fsycl -fsycl-targets=spir64 -foffload-fp32-prec-div
                         -foffload-fp32-prec-sqrt -ffp-contract=off -Wno-unused-function)
  target_link_options(xe_moe_cpuprobe PRIVATE -fsycl -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel)
  target_link_libraries(xe_moe_cpuprobe PRIVATE rad_abi)
  set_target_properties(xe_moe_cpuprobe PROPERTIES EXCLUDE_FROM_ALL ON)
  add_executable(xe_cpu_probe test/cpu_probe.cpp)
  target_compile_options(xe_cpu_probe PRIVATE -fsycl)
  target_link_options(xe_cpu_probe PRIVATE -fsycl)
  target_link_libraries(xe_cpu_probe PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(xe_cpu_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests EXCLUDE_FROM_ALL ON)
  add_test(NAME xe_lane_cpu COMMAND xe_lane_test)
  set_tests_properties(xe_lane_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # Milestone 3, owner note 186 item 3b (the megakernel's stage 1): xe_persist.h's phase barrier -- the chain over
  # resident work-groups and the bounded spin; the CPU SYCL device here, a card as is.
  add_executable(xe_persist_test test/persist_test.cpp)
  set_source_files_properties(test/persist_test.cpp PROPERTIES COMPILE_OPTIONS "${RAD_SYCL_FLAGS}")
  target_include_directories(xe_persist_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_options(xe_persist_test PRIVATE ${RAD_SYCL_FLAGS})
  set_target_properties(xe_persist_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_persist_cpu COMMAND xe_persist_test)
  set_tests_properties(xe_persist_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 300)
  # ... and stage 1a: hc_read at a verify's rows with RADIANCE_XE_HC_PERSIST off and on (two processes: the switch is
  # read once), the outputs compared byte for byte
  # run 206 (the fusion map's pair 4): the gated attention's sigmoid + mul + quant_act_i8g against the gate_quant_fp8
  # row (RADIANCE_XE_GATE_QUANT=1), byte for byte
  add_executable(xe_gate_quant_test test/gate_quant_test.cpp)
  rad_l0_sycl(xe_gate_quant_test)
  target_include_directories(xe_gate_quant_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_gate_quant_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_gate_quant_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_gate_quant_test libxe)
  add_test(NAME xe_gate_quant_cpu COMMAND sh -c
    "RI_GQ=0 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gq0.bin && RADIANCE_XE_GATE_QUANT=1 RI_GQ=1 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gq1.bin && cmp gq0.bin gq1.bin && echo 'gate_quant_fp8 (pair 4): bit-identical' && RI_GQ=2 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gq2.bin && RADIANCE_XE_GATED_HAD=1 RI_GQ=3 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gq3.bin && cmp gq2.bin gq3.bin && echo 'gated_had_quant_i8g (pair 2): bit-identical'")
  set_tests_properties(xe_gate_quant_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # run 273 (owner note 214 item 3, the decode step's adjacent-launch pair 1): the routed gated_had_quant_i8g and the
  # shared gated_quant_fp8 (int8 codes) against the gated_quant_pair row (RADIANCE_XE_GATED_QUANT_PAIR=1), both sets'
  # codes, scales and bf16 product byte for byte
  add_test(NAME xe_gated_quant_pair_cpu COMMAND sh -c
    "RI_GQ=4 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gqp4.bin && RADIANCE_XE_GATED_QUANT_PAIR=1 RI_GQ=5 $<TARGET_FILE:xe_gate_quant_test> $<TARGET_FILE:libxe> gqp5.bin && cmp gqp4.bin gqp5.bin && echo 'gated_quant_pair (the decode step pair 1): bit-identical'")
  set_tests_properties(xe_gated_quant_pair_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # run 273 (the decode step's pair 3 as libr4d's fused prologue): rmsnorm q (the [q|gate] column view) + rmsnorm k +
  # rope q + rope k against the qk_norm_rope_gate row (RADIANCE_XE_QK_PROLOGUE=1), q and k byte for byte over three
  # shapes; the same binary runs on the card (logs/qkstep.sh, the MB form)
  add_executable(xe_qk_prologue_test test/qk_prologue_test.cpp)
  rad_l0_sycl(xe_qk_prologue_test)
  target_include_directories(xe_qk_prologue_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_qk_prologue_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_qk_prologue_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_qk_prologue_test libxe)
  add_test(NAME xe_qk_prologue_cpu COMMAND sh -c
    "RADIANCE_XE_QK_PROLOGUE=1 $<TARGET_FILE:xe_qk_prologue_test> $<TARGET_FILE:libxe>")
  set_tests_properties(xe_qk_prologue_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # run 273 (the decode step's pair 2): router_topk_scatter + had_quant_act_i8g against the router_topk_scatter_quant
  # row (RADIANCE_XE_ROUTER_QUANT_PAIR=1) at M 1 / 4 / 8 / 12, every output byte for byte; the same binary runs on the
  # card (logs/rqstep.sh, the MB form)
  add_executable(xe_router_quant_test test/router_quant_test.cpp)
  rad_l0_sycl(xe_router_quant_test)
  target_include_directories(xe_router_quant_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_router_quant_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_router_quant_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_router_quant_test libxe)
  add_test(NAME xe_router_quant_cpu COMMAND sh -c
    "RADIANCE_XE_ROUTER_QUANT_PAIR=1 $<TARGET_FILE:xe_router_quant_test> $<TARGET_FILE:libxe>")
  set_tests_properties(xe_router_quant_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # run 285 (group A of owner note 232's audit): router_gemm_topk_scatter_quant + the standalone int8 GEMV against the
  # router_gemm_topk_scatter_quant_sh row (RADIANCE_XE_GEMV_MULTI2=1; the NX form the paired kernel copies) at M 1 / 4 /
  # 8 / 12, every output byte for byte; the same binary runs on the card
  add_executable(xe_router_sh_test test/router_sh_test.cpp)
  rad_l0_sycl(xe_router_sh_test)
  target_include_directories(xe_router_sh_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_router_sh_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_router_sh_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_router_sh_test libxe)
  add_test(NAME xe_router_sh_cpu COMMAND sh -c
    "RADIANCE_XE_GEMV_MULTI2=1 RADIANCE_XE_GEMV_NX=1 $<TARGET_FILE:xe_router_sh_test> $<TARGET_FILE:libxe>")
  set_tests_properties(xe_router_sh_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  # run 274 (the decode step's pairs 4 / 5): three gemm_nt_q rows vs gemm_nt_q_x3 and gemm_nt_q + gemm_nt vs
  # gemm_nt_q_bf16_pair (RADIANCE_XE_GEMV_MULTI=1; the NX form the paired kernels copy), M 1 / 4, every output byte for
  # byte; the same binary runs on the card (logs/gmstep.sh, the MB form)
  add_executable(xe_gemv_multi_test test/gemv_multi_test.cpp)
  rad_l0_sycl(xe_gemv_multi_test)
  target_include_directories(xe_gemv_multi_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_gemv_multi_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_gemv_multi_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_gemv_multi_test libxe)
  add_test(NAME xe_gemv_multi_cpu COMMAND sh -c
    "RADIANCE_XE_GEMV_MULTI=1 RADIANCE_XE_GEMV_MULTI2=b RADIANCE_XE_GEMV_NX=1 $<TARGET_FILE:xe_gemv_multi_test> $<TARGET_FILE:libxe>")
  set_tests_properties(xe_gemv_multi_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
  add_executable(xe_hc_persist_test test/hc_persist_test.cpp)
  rad_l0_sycl(xe_hc_persist_test)
  target_include_directories(xe_hc_persist_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  target_link_libraries(xe_hc_persist_test PRIVATE rad_abi ${CMAKE_DL_LIBS})
  set_target_properties(xe_hc_persist_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_dependencies(xe_hc_persist_test libxe)
  add_test(NAME xe_hc_persist_cpu COMMAND sh -c
    "RADIANCE_XE_HC_PERSIST=0 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_off.bin && RADIANCE_XE_HC_PERSIST=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_on.bin && cmp hc_persist_off.bin hc_persist_on.bin && echo 'hc_persist: bit-identical' && RADIANCE_XE_HC_PERSIST=2 RADIANCE_XE_HC_PERSIST_GPC=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_tk.bin && cmp hc_persist_off.bin hc_persist_tk.bin && echo 'hc_persist (stage 1b, ticketed): bit-identical' && RADIANCE_XE_HC_AB=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_ab.bin 2> hc_persist_ab.err && grep -q 'one ticketed grid' hc_persist_ab.err && cmp hc_persist_off.bin hc_persist_ab.bin && echo 'hc_read AB (upstream 1.14 merge): bit-identical' && RADIANCE_XE_HC_NR=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_nr.bin 2> hc_persist_nr.err && grep -q 'one shape-matched launch' hc_persist_nr.err && cmp hc_persist_off.bin hc_persist_nr.bin && echo 'hc_read NR (norm + columns, shape-matched): bit-identical' && RADIANCE_XE_HC_GU=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_gu.bin 2> hc_persist_gu.err && grep -q 'gains and up mix as one shape-matched launch' hc_persist_gu.err && cmp hc_persist_off.bin hc_persist_gu.bin && echo 'hc_read GU (gains + up mix, shape-matched): bit-identical' && RADIANCE_XE_HC_D16=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_d16.bin && cmp hc_persist_off.bin hc_persist_d16.bin && echo 'hc_read D16 (the up mix reads d as bf16): bit-identical' && RADIANCE_XE_HC_GU=1 RADIANCE_XE_HC_D16=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_persist_gud.bin && cmp hc_persist_off.bin hc_persist_gud.bin && echo 'hc_read GU + D16: bit-identical' && RI_HCWR=0 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_wr0.bin && RI_HCWR=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_wr1.bin 2> hc_wr1.err && grep -q 'hc_write folded' hc_wr1.err && cmp hc_wr0.bin hc_wr1.bin && echo 'hc_write_read (the write in the norm): bit-identical' && RADIANCE_XE_HC_PERSIST=1 RI_HCWR=1 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_wr1p.bin && cmp hc_wr0.bin hc_wr1p.bin && echo 'hc_write_read (the write first, persistent read): bit-identical'")
  # run 193 (owner note 190): RADIANCE_XE_HC_AB=1 -- the columns, gains and up mix as one ticketed grid; the grep proves
  # the grid ran (it falls back silently where its shape conditions fail)
  # run 189 (owner note 187): =2 is stage 1b (work by ticket, per-row flags); three runs of the row test. GPC=1: one
  # group a compute unit, as =1's (the build container's 4 CPUs time-slice 32 groups, and a descheduled holder of a
  # unit lets a waiter's bounded spin run out -- correct, then a fail-fast run, but not the bytes this compares)
  # run 207: GU and D16 are the defaults now; the base runs pin them off so every form is compared with the plain path
  # (the forms' own runs set =1 on their command, which overrides)
  set_tests_properties(xe_hc_persist_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_HC_V2=8;RADIANCE_XE_HC_GU=0;RADIANCE_XE_HC_D16=0" TIMEOUT 1200)
  # run 212 (the decode desk list): RADIANCE_XE_HC_DCOLS=4 / 8, the plain columns launch at 4 / 8 columns a workgroup,
  # against the 2-column default, byte for byte
  add_test(NAME xe_hc_dcols_cpu COMMAND sh -c
    "$<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_dcols2.bin && RADIANCE_XE_HC_DCOLS=4 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_dcols4.bin 2> hc_dcols4.err && grep -q 'columns at 4 a workgroup' hc_dcols4.err && cmp hc_dcols2.bin hc_dcols4.bin && echo 'hc_read columns, 4 a workgroup: bit-identical' && RADIANCE_XE_HC_DCOLS=8 $<TARGET_FILE:xe_hc_persist_test> $<TARGET_FILE:libxe> hc_dcols8.bin 2> hc_dcols8.err && grep -q 'columns at 8 a workgroup' hc_dcols8.err && cmp hc_dcols2.bin hc_dcols8.bin && echo 'hc_read columns, 8 a workgroup: bit-identical'")
  set_tests_properties(xe_hc_dcols_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_HC_V2=8;RADIANCE_XE_HC_GU=0;RADIANCE_XE_HC_D16=0" TIMEOUT 1200)
endif()
