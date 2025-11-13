# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(CMAKE_COMPILER_IS_GNUCXX)
  # TODO(hcho3): Remove Wno-free-nonheap-object once GCC 15 is released. The flag is needed to work
  # around spurious warnings emitted by GCC 14.x. See
  # https://github.com/rapidsai/cuml/pull/7471#issuecomment-3525796585 for more details.
  list(APPEND CUML_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-free-nonheap-object)
  if(CUML_BUILD_TESTS OR CUML_BUILD_BENCHMARKS)
    # Suppress parentheses warning which causes gmock to fail
    list(APPEND CUML_CUDA_FLAGS -Xcompiler=-Wno-parentheses)
  endif()
endif()

list(APPEND CUML_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(CUDA_WARNINGS_AS_ERRORS)
  list(APPEND CUML_CUDA_FLAGS -Werror=all-warnings)
endif()
list(
  APPEND
  CUML_CUDA_FLAGS
  -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations,-Wno-error=sign-compare,-Wno-free-nonheap-object
)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUML_CXX_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
  list(APPEND CUML_CUDA_FLAGS -Wno-deprecated-declarations -Xcompiler=-Wno-deprecated-declarations
       -DRAFT_HIDE_DEPRECATION_WARNINGS
  )
endif()

# make sure we produce smallest binary size
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE CUML_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINE_INFO)
  list(APPEND CUML_CUDA_FLAGS -lineinfo)
endif()

if(CUDA_ENABLE_KERNEL_INFO)
  list(APPEND CUML_CUDA_FLAGS -Xptxas=-v)
endif()

if(OpenMP_FOUND)
  list(APPEND CUML_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUML: Building with debugging flags")
  list(APPEND CUML_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()
