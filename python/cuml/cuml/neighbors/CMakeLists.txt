# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("kneighbors_classifier.pyx" ${kneighbors_classifier_algo} ${neighbors_algo})
add_module_gpu_default("kneighbors_regressor.pyx" ${kneighbors_regressor_algo} ${neighbors_algo})
add_module_gpu_default("nearest_neighbors.pyx" ${nearest_neighbors_algo} ${neighbors_algo})

if(NOT SINGLEGPU)
  list(APPEND cython_sources kneighbors_classifier_mg.pyx kneighbors_regressor_mg.pyx
       nearest_neighbors_mg.pyx
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}" MODULE_PREFIX neighbors_
)
