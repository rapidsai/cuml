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
/**
* @file vMeasure.h
*/

#include "metrics/homogeneityScore.h"

namespace MLCommon {

namespace Metrics {

/**
* @brief Function to calculate the v-measure between two clusters
* 
* @param truthClusterArray: the array of truth classes of type T
* @param predClusterArray: the array of predicted classes of type T
* @param size: the size of the data points of type int
* @param lowerLabelRange: the lower bound of the range of labels
* @param upperLabelRange: the upper bound of the range of labels
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
* @param beta: vmeasure parameter
*/
template <typename T>
double vMeasure(const T *truthClusterArray, const T *predClusterArray, int size,
                T lowerLabelRange, T upperLabelRange,
                std::shared_ptr<MLCommon::deviceAllocator> allocator,
                cudaStream_t stream, double beta = 1.0) {
  double computedHomogeity, computedCompleteness, computedVMeasure;

  computedHomogeity = MLCommon::Metrics::homogeneityScore(
    truthClusterArray, predClusterArray, size, lowerLabelRange, upperLabelRange,
    allocator, stream);
  computedCompleteness = MLCommon::Metrics::homogeneityScore(
    predClusterArray, truthClusterArray, size, lowerLabelRange, upperLabelRange,
    allocator, stream);

  if (computedCompleteness + computedHomogeity == 0.0)
    computedVMeasure = 0.0;
  else
    computedVMeasure = ((1 + beta) * computedHomogeity * computedCompleteness /
                        (beta * computedHomogeity + computedCompleteness));

  return computedVMeasure;
}

};  //end namespace Metrics
};  //end namespace MLCommon
