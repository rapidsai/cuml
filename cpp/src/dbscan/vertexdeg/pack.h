/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/ball_cover.hpp>

namespace ML {
namespace Dbscan {
namespace VertexDeg {

template <typename Type, typename Index_>
struct Pack {
  /** optional rbc index */
  void* rbc_index;
  /**
   * vertex degree array
   * Last position is the sum of all elements in this array (excluding it)
   * Hence, its length is one more than the number of points
   */
  Index_* vd;
  /** weighted vertex degree */
  Type* weight_sum;
  /** the CSR adjacency matrix */
  Index_* ia;
  rmm::device_uvector<Index_>* ja;
  /** iff > 0 maximum expected rowlength */
  Index_ max_k;
  /** the dense adjacency matrix */
  bool* adj;
  /** input dataset */
  const Type* x;
  /** weighted vertex degree */
  const Type* sample_weight;
  /** epsilon neighborhood thresholding param */
  Type eps;
  /** number of points in the dataset */
  Index_ N;
  /** dataset dimensionality */
  Index_ D;

  /**
   * @brief reset the output array before calling the actual kernel
   * @param stream cuda stream where to perform this operation
   * @param vdlen length of the vertex degree array
   */
  void resetArray(cudaStream_t stream, Index_ vdlen)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(vd, 0, sizeof(Index_) * vdlen, stream));
  }
};

}  // namespace VertexDeg
}  // namespace Dbscan
}  // namespace ML
