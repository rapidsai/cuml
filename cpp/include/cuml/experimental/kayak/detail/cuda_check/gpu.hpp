#pragma once
#include <cuda_runtime_api.h>
#include <kayak/detail/cuda_check/base.hpp>
#include <kayak/device_type.hpp>
#include <kayak/exceptions.hpp>
namespace kayak {
namespace detail {

template <>
inline void cuda_check<device_type::gpu, cudaError_t>(cudaError_t const& err) noexcept(false) {
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw bad_cuda_call(cudaGetErrorString(err));
  }
}

}
}
