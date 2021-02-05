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

#pragma once

#include <math.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <algorithm>
#include <cub/cub.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/device_buffer.hpp>
#include <distance/distance.cuh>
#include <iostream>
#include <linalg/reduce_cols_by_key.cuh>
#include <numeric>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>

namespace MLCommon {
namespace Metrics {

/**
* @brief kernel that calculates the average intra-cluster distance for every sample data point and updates the cluster distance to max value
* @tparam DataT: type of the data samples
* @tparam LabelT: type of the labels
* @param sampleToClusterSumOfDistances: the pointer to the 2D array that contains the sum of distances from every sample to every cluster (nRows x nLabels)
* @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x nLabels)
* @param d_aArray: the pointer to the array of average intra-cluster distances for every sample in device memory (1 x nRows)
* @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
* @param nRows: number of data samples
* @param nLabels: number of Labels
* @param MAX_VAL: DataT specific upper limit
*/
template <typename DataT, typename LabelT>
__global__ void populateAKernel(DataT *sampleToClusterSumOfDistances,
                                DataT *binCountArray, DataT *d_aArray,
                                LabelT *labels, int nRows, int nLabels,
                                const DataT MAX_VAL) {
  //getting the current index
  int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (sampleIndex >= nRows) return;

  //sampleDistanceVector is an array that stores that particular row of the distanceMatrix
  DataT *sampleToClusterSumOfDistancesVector =
    &sampleToClusterSumOfDistances[sampleIndex * nLabels];

  LabelT sampleCluster = labels[sampleIndex];

  int sampleClusterIndex = (int)sampleCluster;

  if (binCountArray[sampleClusterIndex] - 1 <= 0) {
    d_aArray[sampleIndex] = -1;
    return;

  }

  else {
    d_aArray[sampleIndex] =
      (sampleToClusterSumOfDistancesVector[sampleClusterIndex]) /
      (binCountArray[sampleClusterIndex] - 1);

    //modifying the sampleDistanceVector to give sample average distance
    sampleToClusterSumOfDistancesVector[sampleClusterIndex] = MAX_VAL;
  }
}

/**
* @brief function to calculate the bincounts of number of samples in every label
* @tparam DataT: type of the data samples
* @tparam LabelT: type of the labels
* @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
* @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x nLabels)
* @param nRows: number of data samples
* @param nUniqueLabels: number of Labels
* @param workspace: device buffer containing workspace memory
* @param allocator: default allocator to allocate memory
* @param stream: the cuda stream where to launch this kernel
*/
template <typename DataT, typename LabelT>
void countLabels(LabelT *labels, DataT *binCountArray, int nRows,
                 int nUniqueLabels, MLCommon::device_buffer<char> &workspace,
                 std::shared_ptr<MLCommon::deviceAllocator> allocator,
                 cudaStream_t stream) {
  int num_levels = nUniqueLabels + 1;
  LabelT lower_level = 0;
  LabelT upper_level = nUniqueLabels;
  size_t temp_storage_bytes = 0;

  device_buffer<int> countArray(allocator, stream, nUniqueLabels);

  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    nullptr, temp_storage_bytes, labels, binCountArray, num_levels, lower_level,
    upper_level, nRows, stream));

  workspace.resize(temp_storage_bytes, stream);

  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    workspace.data(), temp_storage_bytes, labels, binCountArray, num_levels,
    lower_level, upper_level, nRows, stream));
}

/**
* @brief stucture that defines the division Lambda for elementwise op
*/
template <typename DataT>
struct DivOp {
  HDI DataT operator()(DataT a, int b, int c) {
    if (b == 0)
      return ULLONG_MAX;
    else
      return a / b;
  }
};

/**
* @brief stucture that defines the elementwise operation to calculate silhouette score using params 'a' and 'b'
*/
template <typename DataT>
struct SilOp {
  HDI DataT operator()(DataT a, DataT b) {
    if (a == 0 && b == 0 || a == b)
      return 0;
    else if (a == -1)
      return 0;
    else if (a > b)
      return (b - a) / a;
    else
      return (b - a) / b;
  }
};

/**
* @brief stucture that defines the reduction Lambda to find minimum between elements
*/
template <typename DataT>
struct MinOp {
  HDI DataT operator()(DataT a, DataT b) {
    if (a > b)
      return b;
    else
      return a;
  }
};

/**
* @brief main function that returns the average silhouette score for a given set of data and its clusterings
* @tparam DataT: type of the data samples
* @tparam LabelT: type of the labels
* @param X_in: pointer to the input Data samples array (nRows x nCols)
* @param nRows: number of data samples
* @param nCols: number of features
* @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
* @param nLabels: number of Labels
* @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and is populated with the silhouette score for every sample (1 x nRows)
* @param allocator: default allocator to allocate device memory
* @param stream: the cuda stream where to launch this kernel
* @param metric: the numerical value that maps to the type of distance metric to be used in the calculations
*/
template <typename DataT, typename LabelT>
DataT silhouette_score(DataT *X_in, int nRows, int nCols, LabelT *labels,
                       int nLabels, DataT *silhouette_scorePerSample,
                       std::shared_ptr<MLCommon::deviceAllocator> allocator,
                       cudaStream_t stream,
                       raft::distance::DistanceType metric =
                         raft::distance::DistanceType::L2Unexpanded) {
  ASSERT(nLabels >= 2 && nLabels <= (nRows - 1),
         "silhouette Score not defined for the given number of labels!");

  //compute the distance matrix
  MLCommon::device_buffer<DataT> distanceMatrix(allocator, stream,
                                                nRows * nRows);
  MLCommon::device_buffer<char> workspace(allocator, stream, 1);

  Distance::pairwise_distance(
    X_in, X_in, distanceMatrix.data(), nRows, nRows, nCols, workspace,
    static_cast<raft::distance::DistanceType>(metric), stream);

  //deciding on the array of silhouette scores for each dataPoint
  MLCommon::device_buffer<DataT> silhouette_scoreSamples(allocator, stream, 0);
  DataT *perSampleSilScore = nullptr;
  if (silhouette_scorePerSample == nullptr) {
    silhouette_scoreSamples.resize(nRows, stream);
    perSampleSilScore = silhouette_scoreSamples.data();
  } else {
    perSampleSilScore = silhouette_scorePerSample;
  }
  CUDA_CHECK(
    cudaMemsetAsync(perSampleSilScore, 0, nRows * sizeof(DataT), stream));

  //getting the sample count per cluster
  MLCommon::device_buffer<DataT> binCountArray(allocator, stream, nLabels);
  CUDA_CHECK(
    cudaMemsetAsync(binCountArray.data(), 0, nLabels * sizeof(DataT), stream));
  countLabels(labels, binCountArray.data(), nRows, nLabels, workspace,
              allocator, stream);

  //calculating the sample-cluster-distance-sum-array
  device_buffer<DataT> sampleToClusterSumOfDistances(allocator, stream,
                                                     nRows * nLabels);
  CUDA_CHECK(cudaMemsetAsync(sampleToClusterSumOfDistances.data(), 0,
                             nRows * nLabels * sizeof(DataT), stream));
  MLCommon::LinAlg::reduce_cols_by_key(distanceMatrix.data(), labels,
                                       sampleToClusterSumOfDistances.data(),
                                       nRows, nRows, nLabels, stream);

  //creating the a array and b array
  device_buffer<DataT> d_aArray(allocator, stream, nRows);
  device_buffer<DataT> d_bArray(allocator, stream, nRows);
  CUDA_CHECK(
    cudaMemsetAsync(d_aArray.data(), 0, nRows * sizeof(DataT), stream));
  CUDA_CHECK(
    cudaMemsetAsync(d_bArray.data(), 0, nRows * sizeof(DataT), stream));

  //kernel that populates the d_aArray
  //kernel configuration
  dim3 numThreadsPerBlock(32, 1, 1);
  dim3 numBlocks(raft::ceildiv<int>(nRows, numThreadsPerBlock.x), 1, 1);

  //calling the kernel
  populateAKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
    sampleToClusterSumOfDistances.data(), binCountArray.data(), d_aArray.data(),
    labels, nRows, nLabels, std::numeric_limits<DataT>::max());

  //elementwise dividing by bincounts
  device_buffer<DataT> averageDistanceBetweenSampleAndCluster(allocator, stream,
                                                              nRows * nLabels);
  CUDA_CHECK(cudaMemsetAsync(averageDistanceBetweenSampleAndCluster.data(), 0,
                             nRows * nLabels * sizeof(DataT), stream));

  raft::linalg::matrixVectorOp<DataT, DivOp<DataT>>(
    averageDistanceBetweenSampleAndCluster.data(),
    sampleToClusterSumOfDistances.data(), binCountArray.data(),
    binCountArray.data(), nLabels, nRows, true, true, DivOp<DataT>(), stream);

  //calculating row-wise minimum
  raft::linalg::reduce<DataT, DataT, int, raft::Nop<DataT>, MinOp<DataT>>(
    d_bArray.data(), averageDistanceBetweenSampleAndCluster.data(), nLabels,
    nRows, std::numeric_limits<DataT>::max(), true, true, stream, false,
    raft::Nop<DataT>(), MinOp<DataT>());

  //calculating the silhouette score per sample using the d_aArray and d_bArray
  raft::linalg::binaryOp<DataT, SilOp<DataT>>(perSampleSilScore,
                                              d_aArray.data(), d_bArray.data(),
                                              nRows, SilOp<DataT>(), stream);

  //calculating the sum of all the silhouette score
  device_buffer<DataT> d_avgSilhouetteScore(allocator, stream, 1);
  CUDA_CHECK(
    cudaMemsetAsync(d_avgSilhouetteScore.data(), 0, sizeof(DataT), stream));

  DataT avgSilhouetteScore;

  raft::linalg::mapThenSumReduce<double, raft::Nop<DataT>>(
    d_avgSilhouetteScore.data(), nRows, raft::Nop<DataT>(), stream,
    perSampleSilScore, perSampleSilScore);

  raft::update_host(&avgSilhouetteScore, d_avgSilhouetteScore.data(), 1,
                    stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  avgSilhouetteScore /= nRows;

  return avgSilhouetteScore;
}

};  // namespace Metrics
};  // namespace MLCommon
