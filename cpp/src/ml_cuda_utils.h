/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace ML {

inline int get_device(const void* ptr)
{
  cudaPointerAttributes att;
  cudaPointerGetAttributes(&att, ptr);
  return att.device;
}

inline cudaMemoryType memory_type(const void* p)
{
  cudaPointerAttributes att;
  cudaError_t err = cudaPointerGetAttributes(&att, p);
  ASSERT(err == cudaSuccess || err == cudaErrorInvalidValue, "%s", cudaGetErrorString(err));

  if (err == cudaErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = cudaGetLastError();
    ASSERT(err == cudaErrorInvalidValue, "%s", cudaGetErrorString(err));
  }
  return att.type;
}

inline bool is_device_or_managed_type(const void* p)
{
  cudaMemoryType p_memory_type = memory_type(p);
  return p_memory_type == cudaMemoryTypeDevice || p_memory_type == cudaMemoryTypeManaged;
}

}  // namespace ML
