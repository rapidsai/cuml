/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <iostream>
#include <selection/knn.cuh>
#include <vector>
#include "test_utils.h"

namespace MLCommon {
namespace Selection {

/**
 *
 * NOTE: Not exhaustively testing the kNN implementation since
 * we are using FAISS for this. Just testing API to verify the
 * knn.cu class is accepting inputs and providing outputs as
 * expected.
 */
template <typename T>
class KNNTest : public ::testing::Test {
 protected:
  void basicTest() {
    auto alloc = std::make_shared<raft::mr::device::default_allocator>();

    // Allocate input
    raft::allocate(d_train_inputs, n * d);

    // Allocate reference arrays
    raft::allocate<long>(d_ref_I, n * n);
    raft::allocate(d_ref_D, n * n);

    // Allocate predicted arrays
    raft::allocate<long>(d_pred_I, n * n);
    raft::allocate(d_pred_D, n * n);

    // make testdata on host
    std::vector<T> h_train_inputs = {1.0, 50.0, 51.0};
    h_train_inputs.resize(n);
    raft::update_device(d_train_inputs, h_train_inputs.data(), n * d, 0);

    std::vector<T> h_res_D = {0.0, 49.0, 50.0, 0.0, 1.0, 49.0, 0.0, 1.0, 50.0};
    h_res_D.resize(n * n);
    raft::update_device(d_ref_D, h_res_D.data(), n * n, 0);

    std::vector<long> h_res_I = {0, 1, 2, 1, 2, 0, 2, 1, 0};
    h_res_I.resize(n * n);
    raft::update_device<long>(d_ref_I, h_res_I.data(), n * n, 0);

    std::vector<float *> input_vec = {d_train_inputs};
    std::vector<int> sizes_vec = {n};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    brute_force_knn(input_vec, sizes_vec, d, d_train_inputs, n, d_pred_I,
                    d_pred_D, n, alloc, stream);

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
  T *d_train_inputs;

  int n = 3;
  int d = 1;

  long *d_pred_I;
  T *d_pred_D;

  long *d_ref_I;
  T *d_ref_D;
};

typedef KNNTest<float> KNNTestF;
TEST_F(KNNTestF, Fit) {
  ASSERT_TRUE(raft::devArrMatch(d_ref_D, d_pred_D, n * n,
                                raft::CompareApprox<float>(1e-3)));
  ASSERT_TRUE(
    raft::devArrMatch(d_ref_I, d_pred_I, n * n, raft::Compare<long>()));
}

};  // end namespace Selection
};  // namespace MLCommon
