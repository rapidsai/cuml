/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace ML {

/**
 * Chunk a single host array up into one or many GPUs (determined by the provided
 * list of device ids)
 *
 * @param ptr       an array in host memory to chunk over devices
 * @param n         number of elements in ptr
 * @param D         number of cols in ptr
 * @param devices   array of device ids for chunking the ptr
 * @param n_chunks  number of elements in gpus
 * @param output    vector containing chunks in the form of rmm::device_uvector
 * @param stream    cuda stream to use
 */
template <typename OutType, typename T = size_t>
void chunk_to_device(const OutType* ptr,
                     T n,
                     int D,
                     int* devices,
                     int n_chunks,
                     std::vector<rmm::device_uvector<OutType>>& output,
                     cudaStream_t stream)
{
  size_t chunk_size = raft::ceildiv<size_t>((size_t)n, (size_t)n_chunks);

#pragma omp parallel for
  for (int i = 0; i < n_chunks; i++) {
    T length = chunk_size;
    if (length * (i + 1) > n) length = length - ((chunk_size * (i + 1)) - n);

    int device = devices[i];
    RAFT_CUDA_TRY(cudaSetDevice(device));
    output.emplace_back(length * D, stream);
    raft::update_device(output.back().data(), ptr + (chunk_size * i), length * D, stream);
  }
};

};  // end namespace ML
