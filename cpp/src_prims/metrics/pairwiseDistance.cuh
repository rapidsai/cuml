/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <math.h>
#include <algorithm>
#include <cub/cub.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <distance/distance.cuh>
#include <iostream>
#include <linalg/binary_op.cuh>
#include <numeric>
#include "common/device_buffer.hpp"
#include "cuda_utils.cuh"

namespace MLCommon {
namespace Metrics {


/**
* @brief main function that returns the average silhouette score for a given set of data and its clusterings
* @tparam DataT: type of the data samples
* @tparam LabelT: type of the labels
* @param X_in: pointer to the input Data samples array (nRows x nCols)
* @param nRows: number of data samples
* @param nCols: number of features
* @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
* @param nLabels: number of Labels
* @param silhouetteScorePerSample: pointer to the array that is optionally taken in as input and is populated with the silhouette score for every sample (1 x nRows)
* @param allocator: default allocator to allocate device memory
* @param stream: the cuda stream where to launch this kernel 
* @param metric: the numerical value that maps to the type of distance metric to be used in the calculations
*/
template <typename DataT, typename IndexT>
void pairwiseDistance(const DataT *x, const DataT *y, DataT *dist, IndexT m,
                      IndexT n, IndexT k, int metric,
                      std::shared_ptr<MLCommon::deviceAllocator> allocator,
                      cudaStream_t stream) {
  //TODO: Assert valid config
  // ASSERT(nLabels >= 2 && nLabels <= (nRows - 1),
  //        "silhouette Score not defined for the given number of labels!");

  //Allocate workspace
  MLCommon::device_buffer<char> workspace(allocator, stream, 1);

  //Call the distance function
  Distance::pairwiseDistance(x, y, dist, m, n, k, workspace, static_cast<Distance::DistanceType>(metric), stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

};  // namespace Metrics
};  // namespace MLCommon
