/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
