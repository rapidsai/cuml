# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("kl_divergence.pyx" ${kl_divergence_algo} ${metrics_algo})
add_module_gpu_default("pairwise_distances.pyx" ${pairwise_distances_algo} ${metrics_algo})
add_module_gpu_default("trustworthiness.pyx" ${trustworthiness_algo} ${metrics_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}" MODULE_PREFIX metrics_
)
