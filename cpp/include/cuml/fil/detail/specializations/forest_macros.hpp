/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
