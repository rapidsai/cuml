# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("t_sne.pyx" ${t_sne_algo} ${manifold_algo})
add_module_gpu_default("spectral_embedding.pyx")

add_subdirectory(umap)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}" MODULE_PREFIX manifold_
)
