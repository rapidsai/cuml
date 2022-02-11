/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <gtest/gtest.h>
#include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/label/classlabels.hpp>
#include <raft/random/make_blobs.hpp>
#include <raft/spatial/knn/knn.hpp>
#include <rmm/device_uvector.hpp>
#include <selection/knn.cuh>
#include <vector>

namespace MLCommon {
namespace Selection {

struct KNNClassifyInputs {
  int rows;
  int cols;
  int n_labels;
  float cluster_std;
  int k;
};

class KNNClassifyTest : public ::testing::TestWithParam<KNNClassifyInputs> {
 public:
  KNNClassifyTest()
    : params(::testing::TestWithParam<KNNClassifyInputs>::GetParam()),
      stream(handle.get_stream()),
      train_samples(params.rows * params.cols, stream),
      train_labels(params.rows, stream),
      pred_labels(params.rows, stream),
      knn_indices(params.rows * params.k, stream),
      knn_dists(params.rows * params.k, stream)
  {
    basicTest();
  }

 protected:
  void basicTest()
  {
    raft::random::make_blobs<float, int>(train_samples.data(),
                                         train_labels.data(),
                                         params.rows,
                                         params.cols,
                                         params.n_labels,
                                         stream,
                                         true,
                                         nullptr,
                                         nullptr,
                                         params.cluster_std);

    rmm::device_uvector<int> unique_labels(0, stream);
    auto n_classes =
      raft::label::getUniquelabels(unique_labels, train_labels.data(), params.rows, stream);

    std::vector<float*> ptrs(1);
    std::vector<int> sizes(1);
    ptrs[0]  = train_samples.data();
    sizes[0] = params.rows;

    raft::spatial::knn::brute_force_knn(handle,
                                        ptrs,
                                        sizes,
                                        params.cols,
                                        train_samples.data(),
                                        params.rows,
                                        knn_indices.data(),
                                        knn_dists.data(),
                                        params.k);

    std::vector<int*> y;
    y.push_back(train_labels.data());

    std::vector<int*> uniq_labels;
    uniq_labels.push_back(unique_labels.data());

    std::vector<int> n_unique;
    n_unique.push_back(n_classes);

    knn_classify(handle,
                 pred_labels.data(),
                 knn_indices.data(),
                 y,
                 params.rows,
                 params.rows,
                 params.k,
                 uniq_labels,
                 n_unique);

    handle.sync_stream(stream);
  }

 protected:
  KNNClassifyInputs params;
  raft::handle_t handle;
  cudaStream_t stream;

  rmm::device_uvector<float> train_samples;
  rmm::device_uvector<int> train_labels;

  rmm::device_uvector<int> pred_labels;

  rmm::device_uvector<int64_t> knn_indices;
  rmm::device_uvector<float> knn_dists;
};

typedef KNNClassifyTest KNNClassifyTestF;
TEST_P(KNNClassifyTestF, Fit)
{
  ASSERT_TRUE(
    devArrMatch(train_labels.data(), pred_labels.data(), params.rows, raft::Compare<int>()));
}

const std::vector<KNNClassifyInputs> inputsf = {{100, 10, 2, 0.01f, 2},
                                                {1000, 10, 5, 0.01f, 2},
                                                {10000, 10, 5, 0.01f, 2},
                                                {100, 10, 2, 0.01f, 10},
                                                {1000, 10, 5, 0.01f, 10},
                                                {10000, 10, 5, 0.01f, 10},
                                                {100, 10, 2, 0.01f, 50},
                                                {1000, 10, 5, 0.01f, 50},
                                                {10000, 10, 5, 0.01f, 50}};

INSTANTIATE_TEST_CASE_P(KNNClassifyTest, KNNClassifyTestF, ::testing::ValuesIn(inputsf));

};  // end namespace Selection
};  // namespace MLCommon
