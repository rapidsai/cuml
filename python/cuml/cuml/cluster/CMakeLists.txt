# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("agglomerative.pyx" ${agglomerative_algo} ${cluster_algo})
add_module_gpu_default("dbscan.pyx" ${dbscan_algo} ${cluster_algo})
add_module_gpu_default("kmeans.pyx" ${kmeans_algo} ${cluster_algo})
add_module_gpu_default("spectral_clustering.pyx" ${cluster_algo})

add_subdirectory(hdbscan)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}" MODULE_PREFIX cluster_
)
