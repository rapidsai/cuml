#pragma once
#include <cuda_runtime_api.h>
#include <cuml/experimental/kayak/cuda_check.hpp>
#include <cuml/experimental/kayak/detail/device_setter/base.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/device_id.hpp>

namespace kayak {
namespace detail {

/** Struct for setting current device within a code block */
template <>
struct device_setter<device_type::gpu> {
  device_setter(kayak::device_id<device_type::gpu> device) noexcept(false) : prev_device_{} {
    kayak::cuda_check(cudaSetDevice(device.value()));
  }

  ~device_setter() {
    cudaSetDevice(prev_device_.value());
  }
 private:
  device_id<device_type::gpu> prev_device_;
};

}
}
