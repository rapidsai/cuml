/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <common/allocatorAdapter.hpp>
#include <raft/cuda_utils.cuh>
#include "../common.cuh"
#include "pack.h"

#include <raft/sparse/convert/csr.cuh>

using namespace thrust;

namespace ML {
namespace Dbscan {
namespace AdjGraph {
namespace Algo {

using namespace MLCommon;

static const int TPB_X = 256;

/**
 * Takes vertex degree array (vd) and CSR row_ind array (ex_scan) to produce the
 * CSR row_ind_ptr array (adj_graph)
 */
template <typename Index_ = int>
void launcher(const raft::handle_t &handle, Pack<Index_> data,
              Index_ batch_size, cudaStream_t stream) {
  device_ptr<Index_> dev_vd = device_pointer_cast(data.vd);
  device_ptr<Index_> dev_ex_scan = device_pointer_cast(data.ex_scan);

  ML::thrustAllocatorAdapter alloc(handle.get_device_allocator(), stream);
  exclusive_scan(thrust::cuda::par(alloc).on(stream), dev_vd,
                 dev_vd + batch_size, dev_ex_scan);

  raft::sparse::convert::csr_adj_graph_batched<Index_, TPB_X>(
    data.ex_scan, data.N, data.adjnnz, batch_size, data.adj, data.adj_graph,
    stream);

  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML