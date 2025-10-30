# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources internals.pyx logger.pyx treelite.pyx)

set(linked_libraries "${cuml_sg_libraries}" "${CUML_PYTHON_TREELITE_TARGET}")

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX internals_
)

# We need to include for callbacks_implements.h in the internals folder
target_include_directories(internals_internals PRIVATE ${CMAKE_CURRENT_LIST_DIR})
