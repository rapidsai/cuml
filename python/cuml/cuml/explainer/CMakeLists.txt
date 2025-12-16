# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("base.pyx" ${kernel_shap_algo} ${permutation_shap_algo} ${explainer_algo})
add_module_gpu_default("kernel_shap.pyx" ${kernel_shap_algo} ${explainer_algo})
add_module_gpu_default("permutation_shap.pyx" ${permutation_shap_algo} ${explainer_algo})
add_module_gpu_default("tree_shap.pyx" ${tree_shap_algo} ${explainer_algo})

set(linked_libraries "${cuml_sg_libraries}" "${CUML_PYTHON_TREELITE_TARGET}")

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX explainer_
)
