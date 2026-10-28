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

rad_add_plugin(libxe kernels SOURCES ${XE_SRC})
rad_l0_sycl(libxe)
# ports/ are libr4d units behind the shim: <hip/hip_runtime.h> resolves to compat/ (the shim), a
# header edited for Xe2 to its copy in ports/ (beside the unit), and the rest to libr4d's own.
target_include_directories(libxe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/libr4d)
# As libr4d: the ops are held bit for bit to unfused chains and to libref's rounding.
target_compile_options(libxe PRIVATE -ffp-contract=off -Wno-unused-function)
# milestone 2: the all-reduce rows allocate at init through rad_l0k (rad_l0k_device)
target_link_libraries(libxe PRIVATE rad_l0k)

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
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_MOE_BF16_GEMV=1" TIMEOUT 600)
  add_test(NAME xe_moe_bf16_wmma_cpu COMMAND xe_moe_bf16_test $<TARGET_FILE:libxe>)
  set_tests_properties(xe_moe_bf16_wmma_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu;RADIANCE_XE_MOE_BF16_GEMV=0" TIMEOUT 600)
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
endif()
