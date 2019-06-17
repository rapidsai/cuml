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

#include <distance/distance.h>
#include <linalg/binary_op.h>
#include <math.h>
#include <ml_cuda_utils.h>
#include <algorithm>
#include <common/host_buffer.hpp>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/eltwise.h"
#include "linalg/map_then_reduce.h"
#include "linalg/matrix_vector_op.h"
#include "linalg/reduce.h"
#include "linalg/reduce_cols_by_key.h"

namespace MLCommon {
namespace Metrics {

/**
* @brief helper debuggger functions that displays the runtime device parameters
* @tparam DataT type of input data to be displayed
* @param matrix the 2D/1D matrix to be displayed for debugging
* @param nRows number of rows
* @param nCols number of columns
* @param stream the device stream
*/

template <typename DataT>
void display(DataT *matrix, int nRows, int nCols, cudaStream_t stream) {
  DataT *temp = (DataT *)malloc(nRows * nCols * sizeof(DataT *));

  updateHost(temp, matrix, nRows * nCols, stream);

  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
      printf("%f\t", temp[i * nCols + j]);
    }
    printf("\n");
  }
  printf("\n");
}
template <typename DataT>
void displayD(DataT *matrix, int nRows, int nCols, cudaStream_t stream) {
  DataT *temp = (DataT *)malloc(nRows * nCols * sizeof(DataT *));

  updateHost(temp, matrix, nRows * nCols, stream);

  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
      printf("%d\t", temp[i * nCols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

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

  if (binCountArray[sampleClusterIndex] - 1 == 0) {
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
      return 0;
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
    else if (a > b)
      return b / a - 1;
    else
      return 1 - a / b;
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
* @param silhouetteScorePerSample: pointer to the array that is optionally taken in as input and is populated with the silhouette score for every sample (1 x nRows)
* @param allocator: default allocator to allocate device memory
* @param stream: the cuda stream where to launch this kernel 
* @param metric: the numerical value that maps to the type of distance metric to be used in the calculations
*/
template <typename DataT, typename LabelT>
DataT silhouetteScore(DataT *X_in, int nRows, int nCols, LabelT *labels,
                      int nLabels, DataT *silhouetteScorePerSample,
                      std::shared_ptr<MLCommon::deviceAllocator> allocator,
                      cudaStream_t stream, int metric = 4) {
  /**  TODO: after 'pairwise-distance' prim is available, use runtime distance metric over hardcoding distance metric
 */
  constexpr auto distance_type_const =
    MLCommon::Distance::DistanceType::EucUnexpandedL2;

  //compute the distance matrix
  MLCommon::device_buffer<DataT> distanceMatrix(allocator, stream,
                                                nRows * nRows);
  MLCommon::device_buffer<DataT> X_in_dup(allocator, stream, nRows * nCols);
  copy(X_in_dup.data(), X_in, nRows * nCols, stream);
  size_t workspaceSize =
    Distance::getWorkspaceSize<distance_type_const, DataT, DataT, DataT, int>(
      X_in, X_in_dup.data(), nRows, nRows, nCols);
  MLCommon::device_buffer<char> workspace(allocator, stream, workspaceSize);
  MLCommon::Distance::distance<distance_type_const, DataT, DataT, DataT,
                               cutlass::Shape<8, 128, 128>, int>(
    X_in, X_in_dup.data(), distanceMatrix.data(), nRows, nRows, nCols,
    (void *)workspace.data(), workspaceSize, stream);

  //deciding on the array of silhouette scores for each dataPoint
  MLCommon::device_buffer<DataT> silhouetteScoreSamples(allocator, stream, 0);
  DataT *perSampleSilScore = nullptr;
  if (silhouetteScorePerSample == nullptr) {
    silhouetteScoreSamples.resize(nRows, stream);
    perSampleSilScore = silhouetteScoreSamples.data();
  } else {
    perSampleSilScore = silhouetteScorePerSample;
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
  LinAlg::reduce_cols_by_key<DataT, LabelT>(
    distanceMatrix.data(), labels, sampleToClusterSumOfDistances.data(), nRows,
    nRows, nLabels, stream);

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
  dim3 numBlocks(ceildiv<int>(nRows, numThreadsPerBlock.x), 1, 1);

  //calling the kernel
  populateAKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
    sampleToClusterSumOfDistances.data(), binCountArray.data(), d_aArray.data(),
    labels, nRows, nLabels, std::numeric_limits<DataT>::max());

  //elementwise dividing by bincounts
  device_buffer<DataT> averageDistanceBetweenSampleAndCluster(allocator, stream,
                                                              nRows * nLabels);
  CUDA_CHECK(cudaMemsetAsync(averageDistanceBetweenSampleAndCluster.data(), 0,
                             nRows * nLabels * sizeof(DataT), stream));

  LinAlg::matrixVectorOp<DataT, DivOp<DataT>>(
    averageDistanceBetweenSampleAndCluster.data(),
    sampleToClusterSumOfDistances.data(), binCountArray.data(),
    binCountArray.data(), nLabels, nRows, true, true, DivOp<DataT>(), stream);

  //calculating row-wise minimum
  LinAlg::reduce<DataT, DataT, int, Nop<DataT>, MinOp<DataT>>(
    d_bArray.data(), averageDistanceBetweenSampleAndCluster.data(), nLabels,
    nRows, std::numeric_limits<DataT>::max(), true, true, stream, false,
    Nop<DataT>(), MinOp<DataT>());

  //calculating the silhouette score per sample using the d_aArray and d_bArray
  LinAlg::binaryOp<DataT, SilOp<DataT>>(perSampleSilScore, d_aArray.data(),
                                        d_bArray.data(), nRows, SilOp<DataT>(),
                                        stream);

  //calculating the sum of all the silhouette score
  device_buffer<DataT> d_avgSilhouetteScore(allocator, stream, 1);
  CUDA_CHECK(
    cudaMemsetAsync(d_avgSilhouetteScore.data(), 0, sizeof(DataT), stream));

  DataT avgSilhouetteScore;

  MLCommon::LinAlg::mapThenSumReduce<double, Nop<DataT>>(
    d_avgSilhouetteScore.data(), nRows, Nop<DataT>(), stream, perSampleSilScore,
    perSampleSilScore);

  updateHost(&avgSilhouetteScore, d_avgSilhouetteScore.data(), 1, stream);

  avgSilhouetteScore /= nRows;

  return avgSilhouetteScore;
}

};  // namespace Metrics
};  // namespace MLCommon