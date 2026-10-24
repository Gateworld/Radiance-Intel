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
list(APPEND XE_SRC ${CMAKE_SOURCE_DIR}/libr4d/r4d_fp8_layout.cpp)

rad_add_plugin(libxe kernels SOURCES ${XE_SRC})
rad_l0_sycl(libxe)
# ports/ are libr4d units behind the shim: <hip/hip_runtime.h> resolves to compat/ (the shim), a
# header edited for Xe2 to its copy in ports/ (beside the unit), and the rest to libr4d's own.
target_include_directories(libxe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/libr4d)
# As libr4d: the ops are held bit for bit to unfused chains and to libref's rounding.
target_compile_options(libxe PRIVATE -ffp-contract=off -Wno-unused-function)

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
  # Step 6's model: a tiny random qwen4exp checkpoint (host code, tests/tiny_model.h's helpers).
  add_executable(xe_tiny_qwen4exp test/tiny_qwen4exp.cpp)
  target_include_directories(xe_tiny_qwen4exp PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/vendor)
  target_link_libraries(xe_tiny_qwen4exp PRIVATE rad_abi)
  set_target_properties(xe_tiny_qwen4exp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME xe_lane_cpu COMMAND xe_lane_test)
  set_tests_properties(xe_lane_cpu PROPERTIES LABELS "xe"
    ENVIRONMENT "RI_DEVICE=cpu;ONEAPI_DEVICE_SELECTOR=opencl:cpu" TIMEOUT 600)
endif()
