#pragma once
#include <cstddef>
#include <memory>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/detail/stream_pool/base.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/exceptions.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/error.hpp>

namespace kayak {
namespace detail {

template<>
struct stream_pool<device_type::gpu> {
  explicit stream_pool(std::shared_ptr<rmm::cuda_stream_pool> existing_pool)
    : rmm_pool{[&existing_pool] () {
      auto result = std::shared_ptr<rmm::cuda_stream_pool>{nullptr};
      if(existing_pool != nullptr) {
        result = existing_pool;
      } else {
        result = std::make_shared<rmm::cuda_stream_pool>();
      }
      return result;
    }()} {}
  explicit stream_pool(std::size_t pool_size = rmm::cuda_stream_pool::default_size)
    : rmm_pool{std::make_shared<rmm::cuda_stream_pool>(pool_size)} {}

  auto get_stream() const noexcept {
    return rmm_pool->get_stream().value();
  }
  auto get_stream(std::size_t stream_id) const {
    return rmm_pool->get_stream(stream_id).value();
  }
  auto get_pool_size() const noexcept {
    return rmm_pool->get_pool_size();
  }

  void sync_all() const noexcept(false) {
    try {
      for (auto i = std::size_t{}; i < get_pool_size(); ++i) {
        rmm_pool->get_stream(i).synchronize();
      }
    } catch (rmm::cuda_error const& err) {
      throw(bad_cuda_call(err.what()));
    }
  }
 private:
  std::shared_ptr<rmm::cuda_stream_pool> rmm_pool;
};

}
}

