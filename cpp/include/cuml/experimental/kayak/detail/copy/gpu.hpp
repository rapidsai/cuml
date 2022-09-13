#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <kayak/cuda_check.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/gpu_support.hpp>
#include <type_traits>

namespace kayak {
namespace detail {

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<(dst_type == device_type::gpu || src_type == device_type::gpu) && GPU_ENABLED, void> copy(T* dst, T const* src, uint32_t size, cuda_stream stream) {
  kayak::cuda_check(cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDefault, stream));
}

}
}
