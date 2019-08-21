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
* @file entropy.h
* @brief Calculates the entropy for a labeling in nats.(ie, uses natural logarithm for the calculations)
*/

#include <math.h>
#include <cub/cub.cuh>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/divide.h"
#include "linalg/map_then_reduce.h"

namespace MLCommon {

/**
* @brief Lambda to calculate the entropy of a sample given its probability value
*
* @param p: the input to the functional mapping
* @param q: dummy param
*/
struct entropyOp {
  HDI double operator()(double p, double q) {
    if (p)
      return -1 * (p) * (log(p));
    else
      return 0.0;
  }
};

namespace Metrics {

/**
* @brief function to calculate the bincounts of number of samples in every label
*
* @tparam LabelT: type of the labels
* @param labels: the pointer to the array containing labels for every data sample 
* @param binCountArray: pointer to the 1D array that contains the count of samples per cluster
* @param nRows: number of data samples
* @param lowerLabelRange
* @param upperLabelRange
* @param workspace: device buffer containing workspace memory
* @param allocator: default allocator to allocate memory
* @param stream: the cuda stream where to launch this kernel
*/
template <typename LabelT>
void countLabels(const LabelT *labels, double *binCountArray, int nRows,
                 LabelT lowerLabelRange, LabelT upperLabelRange,
                 MLCommon::device_buffer<char> &workspace,
                 std::shared_ptr<MLCommon::deviceAllocator> allocator,
                 cudaStream_t stream) {
  int num_levels = upperLabelRange - lowerLabelRange + 2;
  LabelT lower_level = lowerLabelRange;
  LabelT upper_level = upperLabelRange + 1;
  size_t temp_storage_bytes = 0;

  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    nullptr, temp_storage_bytes, labels, binCountArray, num_levels, lower_level,
    upper_level, nRows, stream));

  workspace.resize(temp_storage_bytes, stream);

  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    workspace.data(), temp_storage_bytes, labels, binCountArray, num_levels,
    lower_level, upper_level, nRows, stream));
}

/**
* @brief Function to calculate entropy
* <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
*
* @param clusterArray: the array of classes of type T
* @param size: the size of the data points of type int
* @param lowerLabelRange: the lower bound of the range of labels
* @param upperLabelRange: the upper bound of the range of labels
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
* @return the entropy score
*/
template <typename T>
double entropy(const T *clusterArray, const int size, const T lowerLabelRange,
               const T upperLabelRange,
               std::shared_ptr<MLCommon::deviceAllocator> allocator,
               cudaStream_t stream) {
  if (!size) return 1.0;

  T numUniqueClasses = upperLabelRange - lowerLabelRange + 1;

  //declaring, allocating and initializing memory for bincount array and entropy values
  MLCommon::device_buffer<double> prob(allocator, stream, numUniqueClasses);
  CUDA_CHECK(
    cudaMemsetAsync(prob.data(), 0, numUniqueClasses * sizeof(double), stream));
  MLCommon::device_buffer<double> d_entropy(allocator, stream, 1);
  CUDA_CHECK(cudaMemsetAsync(d_entropy.data(), 0, sizeof(double), stream));

  //workspace allocation
  device_buffer<char> workspace(allocator, stream, 1);

  //calculating the bincounts and populating the prob array
  countLabels(clusterArray, prob.data(), size, lowerLabelRange, upperLabelRange,
              workspace, allocator, stream);

  //scalar dividing by size
  MLCommon::LinAlg::divideScalar<double>(prob.data(), prob.data(), (double)size,
                                         numUniqueClasses, stream);

  //calculating the aggregate entropy
  MLCommon::LinAlg::mapThenSumReduce<double, entropyOp>(
    d_entropy.data(), numUniqueClasses, entropyOp(), stream, prob.data(),
    prob.data());

  //updating in the host memory
  double h_entropy;
  MLCommon::updateHost(&h_entropy, d_entropy.data(), 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return h_entropy;
}

};  //end namespace Metrics
};  //end namespace MLCommon
