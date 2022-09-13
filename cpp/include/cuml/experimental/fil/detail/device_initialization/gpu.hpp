#pragma once
#include <type_traits>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_setter.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
namespace herring {
namespace detail {
namespace device_initialization {

template<typename forest_t, kayak::device_type D>
std::enable_if_t<cuml/experimental/kayak::GPU_ENABLED && D==kayak::device_type::gpu, void> initialize_device(kayak::device_id<D> device);

}
}
}

