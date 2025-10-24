/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/constants.hpp>
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/specialization_types.hpp>
#include <cuml/fil/tree_layout.hpp>

#include <variant>

/* Macro which, given a variant index, will extract the type of the
 * corresponding variant from the specialization_variant type. This allows us
 * to specify all forest variants we wish to support in one location and then
 * reference them by index elsewhere. */
#define CUML_FIL_SPEC(variant_index) \
  std::variant_alternative_t<variant_index, fil::detail::specialization_variant>

/* Macro which expands to a full declaration of a forest type corresponding to
 * the given variant index. */
#define CUML_FIL_FOREST(variant_index)                          \
  forest<CUML_FIL_SPEC(variant_index)::layout,                  \
         typename CUML_FIL_SPEC(variant_index)::threshold_type, \
         typename CUML_FIL_SPEC(variant_index)::index_type,     \
         typename CUML_FIL_SPEC(variant_index)::metadata_type,  \
         typename CUML_FIL_SPEC(variant_index)::offset_type>
