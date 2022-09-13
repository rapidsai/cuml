#pragma once

#include <variant>
#include <cuml/experimental/fil/detail/device_initialization/cpu.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/fil/detail/device_initialization/gpu.hpp>
#endif

namespace herring {
namespace detail {
template<typename forest_t, kayak::device_type D>
void initialize_device(kayak::device_id<D> device) {
  device_initialization::initialize_device<forest_t>(device);
}

template<typename forest_t>
void initialize_device(kayak::device_id_variant device) {
  std::visit([](auto&& concrete_device) {
    device_initialization::initialize_device<forest_t>(concrete_device);
  }, device);
}
}
}
