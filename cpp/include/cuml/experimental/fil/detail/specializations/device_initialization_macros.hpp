#pragma once
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/fil/detail/specializations/forest_macros.hpp>
/* Declare device initialization function for the types specified by the given
 * variant index */
#define CUML_FIL_INITIALIZE_DEVICE(template_type, variant_index) template_type void initialize_device<\
  CUML_FIL_FOREST(variant_index),\
  kayak::device_type::gpu\
>(kayak::device_id<kayak::device_type::gpu>);
