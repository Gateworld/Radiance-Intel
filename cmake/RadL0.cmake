# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Deadcode and the radiance contributors
# Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
#
# RadL0.cmake -- the Level Zero build: detection and the helpers the core and libxe use.
#
# The Level Zero build compiles the whole tree with intel/llvm's clang++ (CMAKE_CXX_COMPILER), which is
# an ordinary clang for every file that does not ask for SYCL. Only core/device/l0.cpp and the libxe
# kernel library are compiled with -fsycl. The backend drives the card through the SYCL runtime on its
# Level Zero plugin (in-order queues, USM, graphs) and calls Level Zero directly where SYCL has no
# query (the PCI address, the memory counters). It holds no device code itself, so executables linking
# the core need libsycl at link time but no device link; libxe is a MODULE that links its own device
# images, ahead of time for the B70 (intel_gpu_bmg_g31) plus SPIR-V for the CPU SYCL device.
#
#   cmake -S . -B build-l0 -G Ninja -DRAD_WITH_HIP=OFF -DRAD_WITH_L0=ON \
#         -DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ -DCMAKE_C_COMPILER=/opt/sycl/bin/clang

include(CheckCXXCompilerFlag)

if(RAD_HIP_FOUND)
  message(FATAL_ERROR "RAD_WITH_L0=ON needs RAD_WITH_HIP=OFF: one device backend per build.")
endif()

check_cxx_compiler_flag(-fsycl RAD_CXX_HAS_FSYCL)
if(NOT RAD_CXX_HAS_FSYCL)
  message(FATAL_ERROR
    "RAD_WITH_L0=ON but ${CMAKE_CXX_COMPILER} does not accept -fsycl. Configure with "
    "-DCMAKE_CXX_COMPILER=<intel/llvm>/bin/clang++.")
endif()

get_filename_component(_rad_sycl_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_rad_sycl_root "${_rad_sycl_bin}" DIRECTORY)
find_library(RAD_SYCL_LIB sycl HINTS ${_rad_sycl_root}/lib NO_DEFAULT_PATH)
find_library(RAD_ZE_LOADER ze_loader)
find_path(RAD_ZE_INCLUDE level_zero/ze_api.h)
if(NOT RAD_SYCL_LIB OR NOT RAD_ZE_LOADER OR NOT RAD_ZE_INCLUDE)
  message(FATAL_ERROR
    "RAD_WITH_L0=ON: libsycl (${RAD_SYCL_LIB}), libze_loader (${RAD_ZE_LOADER}) and "
    "level_zero/ze_api.h (${RAD_ZE_INCLUDE}) are all required.")
endif()

# The offload targets: AOT for the B70 so nothing JIT-compiles on the card's host, and SPIR-V for the
# CPU SYCL device the plain kernels are checked on first.
set(RAD_L0_TARGETS "intel_gpu_bmg_g31,spir64" CACHE STRING "-fsycl-targets of the Level Zero build")
set(RAD_SYCL_FLAGS -fsycl -fsycl-targets=${RAD_L0_TARGETS})
set(RAD_L0_FOUND ON)
# Ubuntu's gcc links with --build-id by default and clang does not; rad-kbench names code by its
# build id (tests/kbench_test.cpp, the_code_identity_follows_build_ids).
add_link_options(LINKER:--build-id=sha1)
message(STATUS "radiance: Level Zero backend ON, SYCL targets ${RAD_L0_TARGETS} (${RAD_SYCL_LIB})")

# The core half: RAD_HAVE_L0, l0.cpp as SYCL host code, and the runtimes every executable then needs.
function(rad_l0_core target)
  target_compile_definitions(${target} PUBLIC RAD_HAVE_L0=1)
  target_include_directories(${target} SYSTEM PRIVATE ${RAD_ZE_INCLUDE})
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device/l0.cpp
               APPEND PROPERTY COMPILE_OPTIONS -fsycl)
  target_link_libraries(${target} PUBLIC ${RAD_SYCL_LIB} ${RAD_ZE_LOADER})
  get_filename_component(_libdir "${RAD_SYCL_LIB}" DIRECTORY)
  target_link_options(${target} INTERFACE "LINKER:-rpath,${_libdir}")
endfunction()

# A SYCL MODULE (a kernel library): compiled and device-linked with the offload targets.
function(rad_l0_sycl target)
  target_compile_options(${target} PRIVATE ${RAD_SYCL_FLAGS})
  target_link_options(${target} PRIVATE ${RAD_SYCL_FLAGS})
  target_include_directories(${target} SYSTEM PRIVATE ${RAD_ZE_INCLUDE})
endfunction()

# clang is stricter than g++ about target features: libavx's AVX-512 level calls _pdep_u64 (BMI2)
# without -mbmi2 (avx_gemm_tile.cpp), which g++ accepts and clang refuses. Every AVX-512 CPU has
# BMI2, so the level's dispatch condition already implies it. Called after libavx is added.
function(rad_l0_clang_fixes)
  if(TARGET avx_lvl3)
    target_compile_options(avx_lvl3 PRIVATE -mbmi2)
  endif()
endfunction()
