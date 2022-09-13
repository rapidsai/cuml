#pragma once
#include <cuda_runtime_api.h>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <kayak/device_setter.hpp>
#include <kayak/detail/owning_buffer/base.hpp>
#include <rmm/device_buffer.hpp>
#include <type_traits>

namespace kayak {
namespace detail {
template<typename T>
struct owning_buffer<device_type::gpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;
  owning_buffer() : data_{} {}

  owning_buffer(device_id<device_type::gpu> device_id, std::size_t size, cudaStream_t stream) noexcept(false)
    : data_{[&device_id, &size, &stream]() {
      auto device_context = device_setter{device_id};
      return rmm::device_buffer{size * sizeof(value_type), rmm::cuda_stream_view{stream}};
    }()}
  {
  }

  auto* get() const { return reinterpret_cast<T*>(data_.data()); }

 private:
  mutable rmm::device_buffer data_;
};
}
}
