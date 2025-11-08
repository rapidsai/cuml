/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#ifdef CUML_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

namespace raft_proto {
#ifdef CUML_ENABLE_GPU
using cuda_stream = cudaStream_t;
#else
using cuda_stream = int;
#endif
inline void synchronize(cuda_stream stream)
{
#ifdef CUML_ENABLE_GPU
  cudaStreamSynchronize(stream);
#endif
}
}  // namespace raft_proto
