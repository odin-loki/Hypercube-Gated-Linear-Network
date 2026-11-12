# SPDX-License-Identifier: AGPL-3.0-or-later
# hgln_gpu: the HGLN kernel (cuda/hgln_kernel.h) run through FORGE's gpu::cuda_hip runtime on CUDA and HIP devices.
#  * Device images, built with clang only (no nvcc / CUDA SDK / ROCm), from the header-free shim cuda/hgln_ptx.cu,
#    for float64 and float32: PTX sm_80 and sm_70 (the driver JITs it), and AMDGPU code objects for HGLN_HIP_ARCHS
#    (default: FORGE_HIP_ARCHS, gfx90a gfx942 gfx1100; needs lld). Same recipe as FORGE's own images. Skipped with a
#    message when clang lacks a backend or lld is missing; the library then reports "no image" at open().
#  * The images are checked for unresolved external calls and embedded into hgln_gpu_images.gen.hpp.
#  * Tests: mock libcuda.so.1 / libamdhip64.so.6 (tests/mock_hgln.c + cuda/hgln_emu_launch.cpp) that run the kernel
#    source on the CPU, and tests/test_gpu.cpp driving Net -> Bank -> FORGE runtime -> mock -> results.
# Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
set(HGLN_GPU_CLANG "${FORGE_GPU_CLANG}" CACHE FILEPATH "clang used for HGLN's PTX / AMDGPU images (default: FORGE's)")
set(HGLN_GPU_LLD "${FORGE_LLD}" CACHE FILEPATH "ld.lld for the AMDGPU code objects (default: FORGE's)")
if(NOT FORGE_HIP_ARCHS)
  set(FORGE_HIP_ARCHS gfx90a gfx942 gfx1100)
endif()
set(HGLN_HIP_ARCHS ${FORGE_HIP_ARCHS} CACHE STRING "AMDGPU targets embedded for HIP")
set(HGLN_CUDA_ARCHS sm_80 sm_70 CACHE STRING "PTX targets embedded for CUDA, in the order they are tried")
find_program(HGLN_LLVM_NM NAMES llvm-nm llvm-nm-18 llvm-nm-19 llvm-nm-20)

set(_dir ${CMAKE_BINARY_DIR}/hgln_gpu_images)
file(MAKE_DIRECTORY ${_dir})
set(_has_nvptx OFF)
set(_has_amdgpu OFF)
if(HGLN_GPU_CLANG)
  execute_process(COMMAND ${HGLN_GPU_CLANG} -print-targets OUTPUT_VARIABLE _t ERROR_QUIET)
  if(_t MATCHES "nvptx64")
    set(_has_nvptx ON)
  endif()
  if(_t MATCHES "amdgcn")
    set(_has_amdgpu ON)
  endif()
endif()
set(_src ${PROJECT_SOURCE_DIR}/cuda/hgln_ptx.cu)
set(_deps ${_src} ${PROJECT_SOURCE_DIR}/cuda/hgln_kernel.h ${PROJECT_SOURCE_DIR}/cuda/hgln_dims.h ${PROJECT_SOURCE_DIR}/include/hgln_gate.h
          ${PROJECT_SOURCE_DIR}/cuda/hgln_q_kernel.h ${PROJECT_SOURCE_DIR}/cuda/hgln_q_dims.h)
set(_common -O3 -ffp-contract=off -Wall -Werror -I${PROJECT_SOURCE_DIR}/cuda -I${PROJECT_SOURCE_DIR}/include)
set(_images "")
set(_list "")
foreach(_real double float)
  if(_real STREQUAL double)
    set(_rb 8)
    set(_rn f64)
  else()
    set(_rb 4)
    set(_rn f32)
  endif()
  if(_has_nvptx)
    foreach(_sm ${HGLN_CUDA_ARCHS})
      set(_o ${_dir}/hgln_${_rn}.${_sm}.ptx)
      add_custom_command(OUTPUT ${_o}
        COMMAND ${HGLN_GPU_CLANG} -x cuda --cuda-device-only -nocudainc -nocudalib -Wno-unknown-cuda-version --cuda-gpu-arch=${_sm}
                ${_common} -DHG_REAL=${_real} -S ${_src} -o ${_o}
        DEPENDS ${_deps} COMMENT "HGLN PTX (${_sm}, ${_real})" VERBATIM)
      list(APPEND _images ${_o})
      string(APPEND _list "0|${_rb}|ptx ${_sm}|${_o}\n")
    endforeach()
  endif()
  if(_has_amdgpu AND HGLN_GPU_LLD)
    get_filename_component(_lldir ${HGLN_GPU_LLD} DIRECTORY)
    foreach(_gfx ${HGLN_HIP_ARCHS})
      set(_o ${_dir}/hgln_${_rn}.${_gfx}.hsaco)
      add_custom_command(OUTPUT ${_o}
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${_lldir}:$ENV{PATH}" ${HGLN_GPU_CLANG} -x hip --cuda-device-only --no-gpu-bundle-output
                -nogpuinc -nogpulib -cuid=hgln_${_rn} --offload-arch=${_gfx} ${_common} --ld-path=${HGLN_GPU_LLD} -DHG_REAL=${_real} ${_src} -o ${_o}
        DEPENDS ${_deps} COMMENT "HGLN AMDGPU code object (${_gfx}, ${_real})" VERBATIM)
      list(APPEND _images ${_o})
      string(APPEND _list "1|${_rb}|${_gfx}|${_o}\n")
    endforeach()
  endif()
endforeach()
if(NOT _has_nvptx)
  message(STATUS "HGLN: '${HGLN_GPU_CLANG}' has no NVPTX backend: no CUDA images (Bank::open(cuda) will report it)")
endif()
if(NOT _has_amdgpu OR NOT HGLN_GPU_LLD)
  message(STATUS "HGLN: no AMDGPU backend or no lld (clang '${HGLN_GPU_CLANG}', lld '${HGLN_GPU_LLD}'): no HIP images")
endif()
file(WRITE ${_dir}/images.list.in "${_list}")
configure_file(${_dir}/images.list.in ${_dir}/images.list COPYONLY)   # rewritten only when the set changes
set(_gen ${_dir}/include/hgln_gpu_images.gen.hpp)
add_custom_command(OUTPUT ${_gen}
  COMMAND ${CMAKE_COMMAND} -DLIST=${_dir}/images.list -DOUT=${_gen} -DNM=${HGLN_LLVM_NM} -P ${PROJECT_SOURCE_DIR}/src/gpu/hgln_embed.cmake
  DEPENDS ${_images} ${_dir}/images.list ${PROJECT_SOURCE_DIR}/src/gpu/hgln_embed.cmake
  COMMENT "Embedding HGLN GPU images" VERBATIM)
add_custom_target(hgln_gpu_images DEPENDS ${_gen})
message(STATUS "HGLN: GPU images: ${_images}")

add_library(hgln_gpu STATIC src/gpu/hgln_gpu.cpp ${_gen})
add_dependencies(hgln_gpu hgln_gpu_images)
target_include_directories(hgln_gpu PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${PROJECT_SOURCE_DIR}/cuda ${_dir}/include)
target_compile_features(hgln_gpu PRIVATE cxx_std_23)
set_target_properties(hgln_gpu PROPERTIES CXX_EXTENSIONS OFF)   # as FORGE: with GNU extensions `linux` is a macro
target_link_libraries(hgln_gpu PUBLIC hgln PRIVATE forge_gpu forge_ops forge_core ${CMAKE_DL_LIBS})
# FORGE's own workaround (its CMakeLists.txt applies it to its directory only): clang < 19 reports __cpp_concepts
# 201907L, so libstdc++ 13/14 hides std::expected, which FORGE's headers use.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19 AND NOT FORGE_STD_EXPECTED_OK)
  target_compile_options(hgln_gpu PRIVATE -Wno-builtin-macro-redefined -U__cpp_concepts -D__cpp_concepts=202002L)
endif()

add_executable(hgln_gpu_cli src/gpu/hgln_gpu_cli.cpp)
set_target_properties(hgln_gpu_cli PROPERTIES OUTPUT_NAME hgln_gpu)
target_compile_options(hgln_gpu_cli PRIVATE -ffp-contract=off)
target_link_libraries(hgln_gpu_cli PRIVATE hgln_gpu)

# research/nlp: the text8 / enwik8 character-level language-modelling benchmark (GPU training through Bank, CPU evaluation)
add_executable(hgln_charlm research/nlp/charlm_hgln.cpp)
target_compile_options(hgln_charlm PRIVATE -ffp-contract=off)
target_link_libraries(hgln_charlm PRIVATE hgln_gpu Threads::Threads)

if(HGLN_BUILD_TESTS)
  # Mock drivers: the real kernel source compiled for the CPU, once per real type, behind the cu*/hip* entry points.
  foreach(_real double float)
    add_library(hgln_emu_${_real} OBJECT cuda/hgln_emu_launch.cpp)
    target_compile_definitions(hgln_emu_${_real} PRIVATE HG_REAL=${_real})
    target_include_directories(hgln_emu_${_real} PRIVATE ${PROJECT_SOURCE_DIR}/cuda ${PROJECT_SOURCE_DIR}/include)
    target_compile_features(hgln_emu_${_real} PRIVATE cxx_std_20)
    target_compile_options(hgln_emu_${_real} PRIVATE -ffp-contract=off)
    set_target_properties(hgln_emu_${_real} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
  endforeach()
  foreach(_m cuda hip)
    add_library(hgln_mock_${_m} SHARED tests/mock_hgln.c $<TARGET_OBJECTS:hgln_emu_double> $<TARGET_OBJECTS:hgln_emu_float>)
    if(_m STREQUAL cuda)
      set_target_properties(hgln_mock_${_m} PROPERTIES OUTPUT_NAME cuda SOVERSION 1)
    else()
      target_compile_definitions(hgln_mock_${_m} PRIVATE MOCK_HIP)
      set_target_properties(hgln_mock_${_m} PROPERTIES OUTPUT_NAME amdhip64 SOVERSION 6)
    endif()
    set_target_properties(hgln_mock_${_m} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/hgln_mock${_m}
                          LINKER_LANGUAGE CXX C_VISIBILITY_PRESET hidden)
    target_link_libraries(hgln_mock_${_m} PRIVATE Threads::Threads)
  endforeach()
  add_executable(hgln_test_gpu tests/test_gpu.cpp)
  target_compile_options(hgln_test_gpu PRIVATE -ffp-contract=off)
  target_link_libraries(hgln_test_gpu PRIVATE hgln_gpu ${CMAKE_DL_LIBS})
  add_dependencies(hgln_test_gpu hgln_mock_cuda hgln_mock_hip)
  set(_mc "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/hgln_mockcuda")
  set(_mh "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/hgln_mockhip")
  # name | api | environment (beyond the mock's library path) | extra arguments (Params::units / fanout variants)
  set(_cases
    "cuda|cuda|"
    "hip|hip|"
    "cuda_sm70|cuda|MOCK_ARCH=sm_70"
    "hip_gfx1100|hip|MOCK_ARCH=gfx1100"
    "cuda_small_smem|cuda|MOCK_SMEM_OPTIN=16384"
    "cuda_u12|cuda||units=12"
    "hip_u8|hip||units=8"
    "cuda_u16_fanout|cuda||units=16 fanout"
    "hip_u12_fanout|hip||units=12 fanout"
    "cuda_u8_fanout_small_smem|cuda|MOCK_SMEM_OPTIN=16384|units=8 fanout"
    # version 5.0 options: v5=1 latch + credit + global-error gates (the v5 float settings), 2 latch + own error, 3 gate_err 2
    "cuda_u8_fanout_v5|cuda||units=8 fanout v5=1"
    "hip_u16_chain_v5|hip||units=16 v5=1"
    "cuda_u12_fanout_v5_latch|cuda||units=12 fanout v5=2"
    "hip_u8_chain_v5_gerr|hip||units=8 v5=3"
    "cuda_u16_v5_dense|cuda||units=16 v5=4"
    "hip_u12_fanout_top8|hip||units=12 fanout v5=5"
    # the integer network (NetQ) through BankQ: bit-identical
    "cuda_int|cuda||int"
    "hip_int|hip||int"
    "cuda_sm70_int|cuda|MOCK_ARCH=sm_70|int")
  foreach(_c ${_cases})
    string(REPLACE "|" ";" _f "${_c}")
    list(GET _f 0 _name)
    list(GET _f 1 _api)
    list(LENGTH _f _nf)
    set(_env ${_mc})
    if(_api STREQUAL hip)
      set(_env ${_mh})
    endif()
    if(_nf GREATER 2)
      list(GET _f 2 _extra)
      if(_extra)
        list(APPEND _env ${_extra})
      endif()
    endif()
    set(_args "")
    if(_nf GREATER 3)
      list(GET _f 3 _a)
      separate_arguments(_args UNIX_COMMAND "${_a}")
    endif()
    add_test(NAME hgln_gpu_mock_${_name} COMMAND hgln_test_gpu ${_api} ${_args})
    set_tests_properties(hgln_gpu_mock_${_name} PROPERTIES ENVIRONMENT "${_env}" LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 900)
  endforeach()
  # without a driver library on the path: open() must fail with a reason (skipped when a real driver is installed)
  add_test(NAME hgln_gpu_no_driver COMMAND hgln_test_gpu nodriver)
  set_tests_properties(hgln_gpu_no_driver PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  foreach(_api cuda hip)
    set(_env ${_mc})
    if(_api STREQUAL hip)
      set(_env ${_mh})
    endif()
    add_test(NAME hgln_gpu_cli_selftest_${_api} COMMAND hgln_gpu_cli selftest 8 2 600 ${_api})
    set_tests_properties(hgln_gpu_cli_selftest_${_api} PROPERTIES ENVIRONMENT "${_env}" LABELS gpu TIMEOUT 600)
  endforeach()
  add_test(NAME hgln_gpu_cli_selftest_cuda_u12_fanout COMMAND hgln_gpu_cli selftest 8 2 600 cuda units=12 fanout v42)
  set_tests_properties(hgln_gpu_cli_selftest_cuda_u12_fanout PROPERTIES ENVIRONMENT "${_mc}" LABELS gpu TIMEOUT 600)
  add_test(NAME hgln_gpu_cli_selftest_cuda_v5 COMMAND hgln_gpu_cli selftest 8 2 600 cuda units=8 fanout v5)
  add_test(NAME hgln_gpu_cli_selftest_cuda_int COMMAND hgln_gpu_cli selftest 8 2 600 cuda int)
  set_tests_properties(hgln_gpu_cli_selftest_cuda_int PROPERTIES ENVIRONMENT "${_mc}" LABELS gpu TIMEOUT 600)
  set_tests_properties(hgln_gpu_cli_selftest_cuda_v5 PROPERTIES ENVIRONMENT "${_mc}" LABELS gpu TIMEOUT 600)
endif()
