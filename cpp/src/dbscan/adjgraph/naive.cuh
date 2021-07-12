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

#include <raft/cudart_utils.h>
#include <cuml/common/host_buffer.hpp>
#include <raft/cuda_utils.cuh>
#include "../common.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace AdjGraph {
namespace Naive {

template <typename Index_ = int>
void launcher(const raft::handle_t& handle,
              Pack<Index_> data,
              Index_ batch_size,
              cudaStream_t stream)
{
  Index_ k = 0;
  Index_ N = data.N;
  MLCommon::host_buffer<Index_> host_vd(handle.get_host_allocator(), stream, batch_size + 1);
  MLCommon::host_buffer<bool> host_adj(handle.get_host_allocator(), stream, batch_size * N);
  MLCommon::host_buffer<Index_> host_ex_scan(handle.get_host_allocator(), stream, batch_size);
  raft::update_host(host_adj.data(), data.adj, batch_size * N, stream);
  raft::update_host(host_vd.data(), data.vd, batch_size + 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  size_t adjgraph_size = size_t(host_vd[batch_size]);
  MLCommon::host_buffer<Index_> host_adj_graph(handle.get_host_allocator(), stream, adjgraph_size);
  for (Index_ i = 0; i < batch_size; i++) {
    for (Index_ j = 0; j < N; j++) {
      /// TODO: change layout or remove; cf #3414
      if (host_adj[i * N + j]) {
        host_adj_graph[k] = j;
        k                 = k + 1;
      }
    }
  }
  host_ex_scan[0] = Index_(0);
  for (Index_ i = 1; i < batch_size; i++)
    host_ex_scan[i] = host_ex_scan[i - 1] + host_vd[i - 1];
  raft::update_device(data.adj_graph, host_adj_graph.data(), adjgraph_size, stream);
  raft::update_device(data.ex_scan, host_ex_scan.data(), batch_size, stream);
}
}  // namespace Naive
}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML