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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <selection/haversine_knn.cuh>
#include <vector>
#include "test_utils.h"

namespace MLCommon {
namespace Selection {

template <typename value_idx, typename value_t>
class HaversineKNNTest : public ::testing::Test {
 protected:
  void basicTest() {
    auto alloc = std::make_shared<raft::mr::device::default_allocator>();

    // Allocate input
    raft::allocate(d_train_inputs, n * d);

    // Allocate reference arrays
    raft::allocate<value_idx>(d_ref_I, n * n);
    raft::allocate(d_ref_D, n * n);

    // Allocate predicted arrays
    raft::allocate<value_idx>(d_pred_I, n * n);
    raft::allocate(d_pred_D, n * n);

    // make testdata on host
    std::vector<value_t> h_train_inputs = {
      0.71113885, -1.29215058, 0.59613176, -2.08048115,
      0.74932804, -1.33634042, 0.51486728, -1.65962873,
      0.53154002, -1.47049808, 0.72891737, -1.54095137};

    h_train_inputs.resize(n);
    raft::update_device(d_train_inputs, h_train_inputs.data(), n * d, 0);

    std::vector<value_t> h_res_D = {
      0., 0.05041587, 0.18767063, 0.23048252, 0.35749438, 0.62925595,
      0., 0.36575755, 0.44288665, 0.5170737,  0.59501296, 0.62925595,
      0., 0.05041587, 0.152463,   0.2426416,  0.34925285, 0.59501296,
      0., 0.16461092, 0.2345792,  0.34925285, 0.35749438, 0.36575755,
      0., 0.16461092, 0.20535265, 0.23048252, 0.2426416,  0.5170737,
      0., 0.152463,   0.18767063, 0.20535265, 0.2345792,  0.44288665};
    h_res_D.resize(n * n);
    raft::update_device(d_ref_D, h_res_D.data(), n * n, 0);

    std::vector<value_idx> h_res_I = {0, 2, 5, 4, 3, 1, 1, 3, 5, 4, 2, 0,
                                      2, 0, 5, 4, 3, 1, 3, 4, 5, 2, 0, 1,
                                      4, 3, 5, 0, 2, 1, 5, 2, 0, 4, 3, 1};
    h_res_I.resize(n * n);
    raft::update_device<value_idx>(d_ref_I, h_res_I.data(), n * n, 0);

    std::vector<value_t *> input_vec = {d_train_inputs};
    std::vector<value_idx> sizes_vec = {n};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    raft::selection::haversine_knn(d_pred_I, d_pred_D, d_train_inputs,
                                   d_train_inputs, n, n, k, stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(d_train_inputs));
    CUDA_CHECK(cudaFree(d_pred_I));
    CUDA_CHECK(cudaFree(d_pred_D));
    CUDA_CHECK(cudaFree(d_ref_I));
    CUDA_CHECK(cudaFree(d_ref_D));
  }

 protected:
  value_t *d_train_inputs;

  int n = 6;
  int d = 2;

  int k = 6;

  value_idx *d_pred_I;
  value_t *d_pred_D;

  value_idx *d_ref_I;
  value_t *d_ref_D;
};

typedef HaversineKNNTest<int, float> HaversineKNNTestF;

TEST_F(HaversineKNNTestF, Fit) {
  ASSERT_TRUE(raft::devArrMatch(d_ref_D, d_pred_D, n * n,
                                raft::CompareApprox<float>(1e-3)));
  ASSERT_TRUE(
    raft::devArrMatch(d_ref_I, d_pred_I, n * n, raft::Compare<int>()));
}

};  // end namespace Selection
};  // namespace MLCommon
