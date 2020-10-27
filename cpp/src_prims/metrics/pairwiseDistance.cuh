/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <raft/cuda_utils.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <distance/distance.cuh>
#include <raft/linalg/binary_op.cuh>

namespace MLCommon {
namespace Metrics {

/**
 * @brief Function to calculate the distance between each ij pair in the input
          array
 * 
 * @tparam DataT type of the data samples
 * @tparam IndexT typeof the index
 * @param x pointer to the input data samples array (mRows x kCols)
 * @param y pointer to the second input data samples array. Can use the same
 *          pointer as x (nRows x kCols)
 * @param dist output pointer where the results will be stored (mRows x nCols)
 * @param m number of rows in x
 * @param n number of rows in y
 * @param k number of cols in x and y (must be the same)
 * @param metric the distance metric to use for the calculation
 * @param allocator default allocator to allocate device memory
 * @param stream the cuda stream where to launch this kernel
 * @param isRowMajor specifies whether the x and y data pointers are row (C
 *                   type array) or col (F type array) major
 */
template <typename DataT, typename IndexT>
void pairwiseDistance(const DataT *x, const DataT *y, DataT *dist, IndexT m,
                      IndexT n, IndexT k, ML::Distance::DistanceType metric,
                      std::shared_ptr<MLCommon::deviceAllocator> allocator,
                      cudaStream_t stream, bool isRowMajor = true) {
  //Allocate workspace
  MLCommon::device_buffer<char> workspace(allocator, stream, 1);

  //Call the distance function
  Distance::pairwiseDistance(x, y, dist, m, n, k, workspace, metric, stream,
                             isRowMajor);
}

};  // namespace Metrics
};  // namespace MLCommon
