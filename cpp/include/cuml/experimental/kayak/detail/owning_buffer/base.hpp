#pragma once
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <type_traits>

namespace kayak {
namespace detail {

template<device_type D, typename T>
struct owning_buffer {
  owning_buffer() {}
  owning_buffer(device_id<D> device_id, std::size_t size, cuda_stream stream) {}
  auto* get() const { return static_cast<T*>(nullptr); }
};

}
}
