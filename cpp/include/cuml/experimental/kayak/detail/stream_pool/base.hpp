#pragma once
#include <cstddef>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>

namespace kayak {
namespace detail {

template<device_type D>
struct stream_pool {
  explicit stream_pool(std::size_t pool_size = std::size_t{}) {
  }
  auto get_stream() const noexcept {
    return cuda_stream{};
  }
  auto get_stream(std::size_t stream_id) const {
    return cuda_stream{};
  }
  auto get_pool_size() const noexcept {
    return std::size_t{};
  }
  void sync_all() const noexcept {
  }
};

}
}
