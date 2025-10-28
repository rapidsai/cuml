/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
