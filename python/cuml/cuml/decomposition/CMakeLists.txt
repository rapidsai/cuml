# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================


set(cython_sources "")
add_module_gpu_default("pca.pyx" ${pca_algo} ${decomposition_algo})
add_module_gpu_default("tsvd.pyx" ${tsvd_algo} ${decomposition_algo})

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       pca_mg.pyx
       tsvd_mg.pyx
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}"
  MODULE_PREFIX decomposition_
)
