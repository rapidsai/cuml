# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("arima.pyx" ${arima_algo} ${tsa_algo})
add_module_gpu_default("auto_arima.pyx" ${auto_arima_algo} ${tsa_algo})
add_module_gpu_default("holtwinters.pyx" ${holtwinters_algo} ${tsa_algo})
add_module_gpu_default("stationarity.pyx" ${stationarity_algo} ${tsa_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}" MODULE_PREFIX tsa_
)
