#pragma once
#include <cuda_runtime_api.h>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/kayak/cuda_check.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <vector>

namespace ML {
namespace experimental {
namespace fil {
namespace detail {

inline auto get_max_shared_mem_per_block(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto static cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    kayak::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev=0; dev < device_count; ++dev) {
      kayak::cuda_check(
        cudaDeviceGetAttribute(
          &(cache[dev]),
          cudaDevAttrMaxSharedMemoryPerBlockOptin,
          dev
        )
      );
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_sm_count(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto static cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    kayak::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev=0; dev < device_count; ++dev) {
      kayak::cuda_check(
        cudaDeviceGetAttribute(
          &(cache[dev]),
          cudaDevAttrMultiProcessorCount,
          dev
        )
      );
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_max_threads_per_block(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerBlock,
      device_id.value()
    )
  );
  return index_type(result);
}

inline auto get_max_threads_per_sm(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerMultiProcessor,
      device_id.value()
    )
  );
  return index_type(result);
}

inline auto get_max_shared_mem_per_sm(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto static cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    kayak::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev=0; dev < device_count; ++dev) {
      kayak::cuda_check(
        cudaDeviceGetAttribute(
          &(cache[dev]),
          cudaDevAttrMaxSharedMemoryPerMultiprocessor,
          dev
        )
      );
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_mem_clock_rate(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMemoryClockRate,
      device_id.value()
    )
  );
  return index_type(result);
}

inline auto get_core_clock_rate(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrClockRate,
      device_id.value()
    )
  );
  return index_type(result);
}

template <typename T>
auto get_max_active_blocks_per_sm(
  T kernel, index_type block_size, index_type dynamic_smem_size=index_type{}
) {
  auto result = int{};
  kayak::cuda_check(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &result, kernel, block_size, dynamic_smem_size
    )
  );
  return index_type(result);
}

/* The maximum number of bytes that can be read in a single instruction */
auto constexpr static const MAX_READ_CHUNK = index_type{128};
auto constexpr static const MAX_BLOCKS = index_type{65536};
auto constexpr static const WARP_SIZE = index_type{32};
auto constexpr static const MAX_THREADS_PER_BLOCK = index_type{256};
#ifdef __CUDACC__
#if __CUDA_ARCH__ == 750
auto constexpr static const MAX_THREADS_PER_SM = index_type{1024};
#else
auto constexpr static const MAX_THREADS_PER_SM = index_type{2048};
#endif
#else
auto constexpr static const MAX_THREADS_PER_SM = index_type{2048};
#endif

auto constexpr static const MIN_BLOCKS_PER_SM = MAX_THREADS_PER_SM / MAX_THREADS_PER_BLOCK;

}
}
}
}
