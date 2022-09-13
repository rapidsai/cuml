#pragma once
#include <cstddef>
#include <stddef.h>
#include <type_traits>
#include <cuml/experimental/fil/detail/index_type.hpp>

namespace herring {

struct shared_memory_buffer {
  __device__ shared_memory_buffer(std::byte* buffer=nullptr, index_type size=index_type{}) :
    data{buffer}, total_size{size}, remaining_data{buffer}, remaining_size{size} {}

  template <typename T>
  __device__ auto* copy(
    T* source, index_type row_count, index_type col_count, index_type row_pad=index_type{}
  ) {
    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);
    auto source_count = row_count * col_count;
    auto dest_count = row_count * (col_count + row_pad);

    auto copy_data = (dest_count * sizeof(T) <= remaining_size);

    source_count *= copy_data;
    for (auto i = threadIdx.x; i < source_count; i += blockDim.x) {
      dest[i + row_pad * (i / col_count)] = source[i];
    }

    auto* result = copy_data ? static_cast<T*>(dest) : source;
    requires_sync = requires_sync || copy_data;

    auto offset = dest_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  template <typename T>
  __device__ auto* copy(T* source, index_type element_count) {
    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);

    auto copy_data = (element_count * index_type(sizeof(T)) <= remaining_size);

    element_count *= copy_data;
    for (auto i = threadIdx.x; i < element_count; i += blockDim.x) {
      dest[i] = source[i];
    }
    auto* result = copy_data ? static_cast<T*>(dest) : source;
    requires_sync = requires_sync || copy_data;

    auto offset = element_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  template <typename T>
  __device__ auto* fill(index_type element_count, T value=T{}) {
    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);

    auto copy_data = (element_count * index_type(sizeof(T)) <= remaining_size);

    element_count *= copy_data;
    for (auto i = threadIdx.x; i < element_count; i += blockDim.x) {
      dest[i] = value;
    }

    auto* result = copy_data ? static_cast<T*>(dest) : static_cast<T*>(nullptr);
    requires_sync = requires_sync || copy_data;

    auto offset = element_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  __device__ auto* clear() {
    remaining_size = total_size;
    remaining_data = data;
    return remaining_data;
  }

  template <typename T>
  __device__ void align() {
    auto pad_required = (total_size - remaining_size) % index_type(sizeof(T));
    remaining_data += pad_required;
    remaining_size -= pad_required;
  }

  __device__ void sync() {
    if (requires_sync) {
      __syncthreads();
    }
    requires_sync = false;
  }

  __device__ auto remaining() {
    return remaining_size;
  }

 private:
  std::byte* data;
  index_type total_size;
  std::byte* remaining_data;
  index_type remaining_size;
  bool requires_sync;
};

}
