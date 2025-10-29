# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_executable(symreg_example symreg_example.cpp)
target_include_directories(symreg_example PRIVATE ${CUML_INCLUDE_DIRECTORIES})
target_link_libraries(symreg_example cuml++ cuvs::cuvs)
