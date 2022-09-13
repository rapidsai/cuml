#pragma once
#include <cuml/experimental/kayak/detail/cuda_check/base.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/kayak/detail/cuda_check/gpu.hpp>
#endif
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {
template <typename error_t>
void cuda_check(error_t const& err) noexcept(!GPU_ENABLED) {
  detail::cuda_check<device_type::gpu>(err);
}
}
