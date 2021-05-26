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

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

template <typename value_t, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void l2SelectMinK(
  faiss::gpu::Tensor<value_t, 2, true> inner_products,
  faiss::gpu::Tensor<value_t, 1, true> sq_norms,
  faiss::gpu::Tensor<value_t, 1, true> core_dists,
  faiss::gpu::Tensor<value_t, 2, true> out_dists,
  faiss::gpu::Tensor<int, 2, true> out_inds, int batch_offset, int k,
  value_t initK, value_t alpha) {
  // Each block handles a single row of the distances (results)
  constexpr int kNumWarps = ThreadsPerBlock / 32;

  __shared__ value_t smemK[kNumWarps * NumWarpQ];
  __shared__ int smemV[kNumWarps * NumWarpQ];

  faiss::gpu::BlockSelect<value_t, int, false, faiss::gpu::Comparator<value_t>,
                          NumWarpQ, NumThreadQ, ThreadsPerBlock>
    heap(initK, -1, smemK, smemV, k);

  int row = blockIdx.x;

  // Whole warps must participate in the selection
  int limit = faiss::gpu::utils::roundDown(inner_products.getSize(1), 32);
  int i = threadIdx.x;

  for (; i < limit; i += blockDim.x) {
    value_t v = sqrt(faiss::gpu::Math<value_t>::add(
      sq_norms[row + batch_offset],
      faiss::gpu::Math<value_t>::add(sq_norms[i], inner_products[row][i])));

    v = max(core_dists[i], max(core_dists[row + batch_offset], alpha * v));
    heap.add(v, i);
  }

  if (i < inner_products.getSize(1)) {
    value_t v = sqrt(faiss::gpu::Math<value_t>::add(
      sq_norms[row + batch_offset],
      faiss::gpu::Math<value_t>::add(sq_norms[i], inner_products[row][i])));

    v = max(core_dists[i], max(core_dists[row + batch_offset], alpha * v));
    heap.addThreadQ(v, i);
  }

  heap.reduce();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    out_dists[row][i] = smemK[i];
    out_inds[row][i] = smemV[i];
  }
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML