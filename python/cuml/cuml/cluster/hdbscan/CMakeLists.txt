# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("hdbscan.pyx" ${hdbscan_algo} ${cluster_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}" MODULE_PREFIX cluster_hdbscan_
)
