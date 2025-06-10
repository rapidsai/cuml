/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cuml/fil/detail/index_type.hpp>

#include <stddef.h>

#include <cstddef>
#include <type_traits>

namespace ML {
namespace fil {

/* A struct used to simplify complex access to a buffer of shared memory
 *
 * @param buffer A pointer to the shared memory allocation
 * @param size The size in bytes of the shared memory allocation
 */
struct shared_memory_buffer {
  __device__ shared_memory_buffer(std::byte* buffer = nullptr, index_type size = index_type{})
    : data{buffer}, total_size{size}, remaining_data{buffer}, remaining_size{size}
  {
  }

  /* If possible, copy the given number of rows with the given number of columns from source
   * to the end of this buffer, padding each row by the given number of
   * elements (usually to reduce memory bank conflicts). If there is not enough
   * room, no copy is performed. Return a pointer to the desired data, whether
   * that is in the original location or copied to shared memory. */
  template <typename T>
  __device__ auto* copy(T* source,
                        index_type row_count,
                        index_type col_count,
                        index_type row_pad = index_type{})
  {
    auto* dest        = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);
    auto source_count = row_count * col_count;
    auto dest_count   = row_count * (col_count + row_pad);

    auto copy_data = (dest_count * sizeof(T) <= remaining_size);

    source_count *= copy_data;
    for (auto i = threadIdx.x; i < source_count; i += blockDim.x) {
      dest[i + row_pad * (i / col_count)] = source[i];
    }

    auto* result  = copy_data ? static_cast<T*>(dest) : source;
    requires_sync = requires_sync || copy_data;

    auto offset = dest_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  /* If possible, copy the given number of elements from source to the end of this buffer
   * If there is not enough room, no copy is performed. Return a pointer to the
   * desired data, whether that is in the original location or copied to shared
   * memory. */
  template <typename T>
  __device__ auto* copy(T* source, index_type element_count)
  {
    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);

    auto copy_data = (element_count * index_type(sizeof(T)) <= remaining_size);

    element_count *= copy_data;
    for (auto i = threadIdx.x; i < element_count; i += blockDim.x) {
      dest[i] = source[i];
    }
    auto* result  = copy_data ? static_cast<T*>(dest) : source;
    requires_sync = requires_sync || copy_data;

    auto offset = element_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  /* If possible, fill the next element_count elements with given value. If
   * there is not enough room, the fill is not performed. Return a pointer to
   * the start of the desired data if the fill was possible or else nullptr. */
  template <typename T>
  __device__ auto* fill(index_type element_count, T value = T{}, T* fallback_buffer = nullptr)
  {
    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);

    auto copy_data = (element_count * index_type(sizeof(T)) <= remaining_size);

    element_count *= copy_data;
    for (auto i = threadIdx.x; i < element_count; i += blockDim.x) {
      dest[i] = value;
    }

    auto* result  = copy_data ? static_cast<T*>(dest) : fallback_buffer;
    requires_sync = requires_sync || copy_data;

    auto offset = element_count * index_type(sizeof(T));
    remaining_data += offset;
    remaining_size -= offset;

    return result;
  }

  /* Clear all stored data and return a pointer to the beginning of available
   * shared memory */
  __device__ auto* clear()
  {
    remaining_size = total_size;
    remaining_data = data;
    return remaining_data;
  }

  /* Pad stored data to ensure correct alignment for given type */
  template <typename T>
  __device__ void align()
  {
    auto pad_required = (total_size - remaining_size) % index_type(sizeof(T));
    remaining_data += pad_required;
    remaining_size -= pad_required;
  }

  /* If necessary, sync threads. Note that this can cause a deadlock if not all
   * threads call this method. */
  __device__ void sync()
  {
    if (requires_sync) { __syncthreads(); }
    requires_sync = false;
  }

  /* Return the remaining size in bytes left in this buffer */
  __device__ auto remaining() { return remaining_size; }

 private:
  std::byte* data;
  index_type total_size;
  std::byte* remaining_data;
  index_type remaining_size;
  bool requires_sync;
};

}  // namespace fil
}  // namespace ML
