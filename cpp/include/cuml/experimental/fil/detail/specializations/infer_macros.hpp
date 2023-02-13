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

#define CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index)(\
  CUML_FIL_FOREST(variant_index) const&,\
  postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  std::nullptr_t,\
  std::nullptr_t,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index)(\
  CUML_FIL_FOREST(variant_index) const&,\
  postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  std::nullptr_t,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define CUML_FIL_SCALAR_NONLOCAL_ARGS(dev, variant_index)(\
  CUML_FIL_FOREST(variant_index) const&,\
  postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  std::nullptr_t,\
  CUML_FIL_SPEC(variant_index)::index_type*,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define CUML_FIL_VECTOR_NONLOCAL_ARGS(dev, variant_index)(\
  CUML_FIL_FOREST(variant_index) const&,\
  postprocessor<CUML_FIL_SPEC(variant_index)::threshold_type> const&,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  index_type,\
  index_type,\
  index_type,\
  CUML_FIL_SPEC(variant_index)::threshold_type*,\
  CUML_FIL_SPEC(variant_index)::index_type*,\
  std::optional<index_type>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, categorical) template_type void infer<\
  dev, categorical, CUML_FIL_FOREST(variant_index)>

#define CUML_FIL_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, false)CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)CUML_FIL_SCALAR_LOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)CUML_FIL_SCALAR_NONLOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, false)CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)CUML_FIL_VECTOR_LOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) CUML_FIL_INFER_TEMPLATE(template_type, dev, variant_index, true)CUML_FIL_VECTOR_NONLOCAL_ARGS(dev, variant_index);

#define CUML_FIL_INFER_ALL(template_type, dev, variant_index) CUML_FIL_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index)\
  CUML_FIL_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  CUML_FIL_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)\
  CUML_FIL_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index)\
  CUML_FIL_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  CUML_FIL_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)
