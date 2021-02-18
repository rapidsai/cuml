/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>
#include <iostream>
#include <metrics/batched/silhouette_score.cuh>
#include <metrics/silhouette_score.cuh>
#include <random>
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

//parameter structure definition
struct silhouetteScoreParam {
  int nRows;
  int nCols;
  int nLabels;
  raft::distance::DistanceType metric;
  int chunk;
  double tolerance;
};

//test fixture class
template <typename LabelT, typename DataT>
class silhouetteScoreTest
  : public ::testing::TestWithParam<silhouetteScoreParam> {
 protected:
  void host_silhouette_score() {
    //generating random value test input
    std::vector<double> h_X(nElements, 0.0);
    std::vector<int> h_labels(nRows, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(0, nLabels - 1);
    std::uniform_real_distribution<double> realGenerator(0, 100);

    std::generate(h_X.begin(), h_X.end(), [&]() { return realGenerator(dre); });
    std::generate(h_labels.begin(), h_labels.end(),
                  [&]() { return intGenerator(dre); });

    //allocating and initializing memory to the GPU
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(d_X, nElements, true);
    raft::allocate(d_labels, nElements, true);
    raft::allocate(sampleSilScore, nElements);

    raft::update_device(d_X, &h_X[0], (int)nElements, stream);
    raft::update_device(d_labels, &h_labels[0], (int)nElements, stream);

    //finding the distance matrix

    device_buffer<double> d_distanceMatrix(allocator, stream, nRows * nRows);
    device_buffer<char> workspace(allocator, stream, 1);
    double *h_distanceMatrix =
      (double *)malloc(nRows * nRows * sizeof(double *));

    MLCommon::Distance::pairwise_distance(d_X, d_X, d_distanceMatrix.data(),
                                          nRows, nRows, nCols, workspace,
                                          params.metric, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    raft::update_host(h_distanceMatrix, d_distanceMatrix.data(), nRows * nRows,
                      stream);

    //finding the bincount array

    double *binCountArray = (double *)malloc(nLabels * sizeof(double *));
    memset(binCountArray, 0, nLabels * sizeof(double));

    for (int i = 0; i < nRows; ++i) {
      binCountArray[h_labels[i]] += 1;
    }

    //finding the average intra cluster distance for every element

    double *a = (double *)malloc(nRows * sizeof(double *));

    for (int i = 0; i < nRows; ++i) {
      int myLabel = h_labels[i];
      double sumOfIntraClusterD = 0;

      for (int j = 0; j < nRows; ++j) {
        if (h_labels[j] == myLabel) {
          sumOfIntraClusterD += h_distanceMatrix[i * nRows + j];
        }
      }

      if (binCountArray[myLabel] <= 1)
        a[i] = -1;
      else
        a[i] = sumOfIntraClusterD / (binCountArray[myLabel] - 1);
    }

    //finding the average inter cluster distance for every element

    double *b = (double *)malloc(nRows * sizeof(double *));

    for (int i = 0; i < nRows; ++i) {
      int myLabel = h_labels[i];
      double minAvgInterCD = ULLONG_MAX;

      for (int j = 0; j < nLabels; ++j) {
        int curClLabel = j;
        if (curClLabel == myLabel) continue;
        double avgInterCD = 0;

        for (int k = 0; k < nRows; ++k) {
          if (h_labels[k] == curClLabel) {
            avgInterCD += h_distanceMatrix[i * nRows + k];
          }
        }

        if (binCountArray[curClLabel])
          avgInterCD /= binCountArray[curClLabel];
        else
          avgInterCD = ULLONG_MAX;
        minAvgInterCD = min(minAvgInterCD, avgInterCD);
      }

      b[i] = minAvgInterCD;
    }

    //finding the silhouette score for every element

    double *truthSampleSilScore = (double *)malloc(nRows * sizeof(double *));
    for (int i = 0; i < nRows; ++i) {
      if (a[i] == -1)
        truthSampleSilScore[i] = 0;
      else if (a[i] == 0 && b[i] == 0)
        truthSampleSilScore[i] = 0;
      else
        truthSampleSilScore[i] = (b[i] - a[i]) / max(a[i], b[i]);
      truthSilhouetteScore += truthSampleSilScore[i];
    }

    truthSilhouetteScore /= nRows;
  }

  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<silhouetteScoreParam>::GetParam();

    nRows = params.nRows;
    nCols = params.nCols;
    nLabels = params.nLabels;
    chunk = params.chunk;
    nElements = nRows * nCols;

    allocator = std::make_shared<raft::mr::device::default_allocator>();

    host_silhouette_score();

    //calling the silhouette_score CUDA implementation
    computedSilhouetteScore = MLCommon::Metrics::silhouette_score(
      d_X, nRows, nCols, d_labels, nLabels, sampleSilScore, allocator, stream,
      params.metric);

    batchedSilhouetteScore =
      Batched::silhouette_score(handle, d_X, nRows, nCols, d_labels, nLabels,
                                sampleSilScore, chunk, params.metric);
  }

  //the destructor
  void TearDown() override {
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  silhouetteScoreParam params;
  int nLabels;
  DataT *d_X = nullptr;
  DataT *sampleSilScore = nullptr;
  LabelT *d_labels = nullptr;
  int nRows;
  int nCols;
  int nElements;
  double truthSilhouetteScore = 0;
  double computedSilhouetteScore = 0;
  double batchedSilhouetteScore = 0;
  cudaStream_t stream;
  raft::handle_t handle;
  int chunk;
  std::shared_ptr<MLCommon::deviceAllocator> allocator;
};

//setting test parameter values
const std::vector<silhouetteScoreParam> inputs = {
  {4, 2, 3, raft::distance::DistanceType::L2Expanded, 4, 0.00001},
  {4, 2, 2, raft::distance::DistanceType::L2SqrtUnexpanded, 2, 0.00001},
  {8, 8, 3, raft::distance::DistanceType::L2Unexpanded, 4, 0.00001},
  {11, 2, 5, raft::distance::DistanceType::L2Expanded, 3, 0.00001},
  {40, 2, 8, raft::distance::DistanceType::L2Expanded, 10, 0.00001},
  {12, 7, 3, raft::distance::DistanceType::CosineExpanded, 8, 0.00001},
  {7, 5, 5, raft::distance::DistanceType::L1, 2, 0.00001}};

//writing the test suite
typedef silhouetteScoreTest<int, double> silhouetteScoreTestClass;
TEST_P(silhouetteScoreTestClass, Result) {
  ASSERT_NEAR(computedSilhouetteScore, truthSilhouetteScore, params.tolerance);
  ASSERT_NEAR(batchedSilhouetteScore, truthSilhouetteScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(silhouetteScore, silhouetteScoreTestClass,
                        ::testing::ValuesIn(inputs));

}  //end namespace Metrics
}  //end namespace MLCommon
