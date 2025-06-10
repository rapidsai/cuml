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
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/specialization_types.hpp>
#include <cuml/fil/detail/specializations/forest_macros.hpp>
#include <cuml/fil/infer_kind.hpp>

#include <cstddef>
#include <variant>

/* Macro which expands to the valid arguments to an inference call for a forest
 * model without vector leaves or non-local categorical data.*/
#define CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index)                 \
  (CUML_FIL_FOREST(variant_index) const&,                              \
   postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&, \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   std::nullptr_t,                                                     \
   std::nullptr_t,                                                     \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model with vector leaves but without non-local categorical data.*/
#define CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index)                 \
  (CUML_FIL_FOREST(variant_index) const&,                              \
   postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&, \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   std::nullptr_t,                                                     \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model without vector leaves but with non-local categorical data.*/
#define CUML_FIL_SCALAR_NONLOCAL_ARGS(dev, variant_index)              \
  (CUML_FIL_FOREST(variant_index) const&,                              \
   postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&, \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   std::nullptr_t,                                                     \
   CUML_FIL_SPEC(variant_index)::index_type*,                          \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model with vector leaves and with non-local categorical data.*/
#define CUML_FIL_VECTOR_NONLOCAL_ARGS(dev, variant_index)              \
  (CUML_FIL_FOREST(variant_index) const&,                              \
   postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&, \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   CUML_FIL_SPEC(variant_index)::threshold_type*,                      \
   CUML_FIL_SPEC(variant_index)::index_type*,                          \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index */
#define CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, categorical) \
  template_type void infer<dev, categorical, CUML_FIL_FOREST(variant_index)>

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves or categorical nodes*/
#define CUML_FIL_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, false)              \
  CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves and with only local categorical nodes*/
#define CUML_FIL_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)                  \
  CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves and with non-local categorical nodes*/
#define CUML_FIL_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)                     \
  CUML_FIL_SCALAR_NONLOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and without categorical nodes*/
#define CUML_FIL_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, false)              \
  CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and with only local categorical nodes*/
#define CUML_FIL_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)                  \
  CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and with non-local categorical nodes*/
#define CUML_FIL_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)                     \
  CUML_FIL_VECTOR_NONLOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of all valid inference templates for
 * the given device on the forest type specified by the given variant index */
#define CUML_FIL_INFER_ALL(template_type, dev, variant_index)                    \
  CUML_FIL_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index)       \
  CUML_FIL_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index)    \
  CUML_FIL_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  CUML_FIL_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index)       \
  CUML_FIL_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index)    \
  CUML_FIL_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)
