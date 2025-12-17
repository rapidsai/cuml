# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("linear_regression.pyx" ${linearregression_algo} ${linear_model_algo})
add_module_gpu_default("ridge.pyx" ${ridge_algo} ${linear_model_algo})

if(NOT SINGLEGPU)
  list(APPEND cython_sources base_mg.pyx linear_regression_mg.pyx logistic_regression_mg.pyx
       ridge_mg.pyx
  )

endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}" MODULE_PREFIX linear_model_
)

if(NOT SINGLEGPU)
  target_include_directories(linear_model_base_mg PRIVATE "../../../../cpp/src_prims/")
  target_include_directories(linear_model_linear_regression_mg PRIVATE "../../../../cpp/src_prims/")
  target_include_directories(
    linear_model_logistic_regression_mg PRIVATE "../../../../cpp/src_prims/"
  )
  target_include_directories(linear_model_ridge_mg PRIVATE "../../../../cpp/src_prims/")
endif()
