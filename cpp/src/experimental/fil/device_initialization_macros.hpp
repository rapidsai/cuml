#pragma once
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <cuml/experimental/fil/specializations/forest_macros.hpp>
#define HERRING_INITIALIZE_DEVICE(template_type, variant_index) template_type void initialize_device<\
  HERRING_FOREST(variant_index),\
  kayak::device_type::gpu\
>(kayak::device_id<kayak::device_type::gpu>);
