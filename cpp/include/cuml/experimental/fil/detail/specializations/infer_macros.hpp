#pragma once
#include <cstddef>
#include <variant>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/postprocessor.hpp>
#include <cuml/experimental/fil/detail/specialization_types.hpp>
#include <cuml/experimental/fil/detail/specializations/forest_macros.hpp>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>

#define HERRING_SCALAR_LOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  std::nullptr_t,\
  std::nullptr_t,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_VECTOR_LOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::nullptr_t,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_SCALAR_NONLOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  std::nullptr_t,\
  HERRING_SPEC(variant_index)::index_type*,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_VECTOR_NONLOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::index_type*,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_INFER_TEMPLATE(template_type, dev, variant_index, categorical) template_type void infer<\
  dev, categorical, HERRING_FOREST(variant_index)>

#define HERRING_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, false)HERRING_SCALAR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_SCALAR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_SCALAR_NONLOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, false)HERRING_VECTOR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_VECTOR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_VECTOR_NONLOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_ALL(template_type, dev, variant_index) HERRING_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)
