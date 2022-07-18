/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
 * @file completeness_score.cuh
 *
 * @brief A clustering result satisfies completeness if all the data points
 * that are members of a given class are elements of the same cluster.
 */

#pragma once

#include "entropy.cuh"
#include "mutual_info_score.cuh"

namespace MLCommon {
namespace Metrics {

/**
 * @brief Function to calculate the completeness score between two clusters
 *
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double completeness_score(const T* truthClusterArray,
                          const T* predClusterArray,
                          int size,
                          T lowerLabelRange,
                          T upperLabelRange,
                          cudaStream_t stream)
{
  if (size == 0) return 1.0;

  double computedMI, computedEntropy;

  computedMI = MLCommon::Metrics::mutual_info_score(
    truthClusterArray, predClusterArray, size, lowerLabelRange, upperLabelRange, stream);
  computedEntropy =
    MLCommon::Metrics::entropy(predClusterArray, size, lowerLabelRange, upperLabelRange, stream);

  double completeness;

  if (computedEntropy) {
    completeness = computedMI / computedEntropy;
  } else
    completeness = 1.0;

  return completeness;
}

};  // end namespace Metrics
};  // end namespace MLCommon
