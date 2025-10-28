# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("pointer_utils.pyx")

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       opg_data_utils_mg.pyx
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}"
  MODULE_PREFIX common_
)

if(${CUML_UNIVERSAL})
# todo: ml_cuda_utils.h should be in the include folder of cuML or the functionality
# moved to another file, pointer_utils.pyx needs it
# https://github.com/rapidsai/cuml/issues/4841
  target_include_directories(common_pointer_utils PRIVATE "../../../../cpp/src/")
endif()
