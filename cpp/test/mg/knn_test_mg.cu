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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <iostream>
#include <vector>
#include "knn/knn.hpp"
#include "knn/util.h"

namespace ML {

using namespace MLCommon;

/**
 *
 * NOTE: Not exhaustively testing the kNN implementation since
 * we are using FAISS for this. Just testing API to verify the
 * knn.cu class is accepting inputs and providing outputs as
 * expected.
 */
template<typename T>
class KNN_MGTest: public ::testing::Test {
protected:
	void basicTest() {

    // make test data on host
    std::vector<T> h_train_inputs = {1.0, 50.0, 51.0, 1.0, 50.0, 51.0};
    h_train_inputs.resize(n*2);

    std::vector<T> h_search = {1.0, 50.0, 51.0};
    h_search.resize(n);

    int* devices = new int[2]{0, 1};

    knn->fit_from_host(h_train_inputs.data(), n*2, devices, 2);

    allocate<float>(d_search, n);

    // Allocate reference arrays
    allocate<long>(d_ref_I, n*n);
    allocate(d_ref_D, n*n);

    // Allocate predicted arrays
    allocate<long>(d_pred_I, n*n);
    allocate(d_pred_D, n*n);

    updateDevice(d_search, h_search.data(), n*d, 0);

    std::vector<T> h_res_D = { 0.0, 0.0, 2401.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
    h_res_D.resize(n*n);
    updateDevice(d_ref_D, h_res_D.data(), n*n, 0);

    std::vector<long> h_res_I = { 0, 3, 1, 1, 4, 2, 2, 5, 1 };
    h_res_I.resize(n*n);
    updateDevice<long>(d_ref_I, h_res_I.data(), n*n, 0);

    knn->search(d_search, n, d_pred_I, d_pred_D, n);
  }

	void TearDown() override {
		CUDA_CHECK(cudaFree(d_search));
		CUDA_CHECK(cudaFree(d_pred_I));
		CUDA_CHECK(cudaFree(d_pred_D));
		CUDA_CHECK(cudaFree(d_ref_I));
		CUDA_CHECK(cudaFree(d_ref_D));
	}

	T* d_search;

  int n = 3;
  int d = 1;

  long* d_pred_I;
  T* d_pred_D;

  long* d_ref_I;
  T* d_ref_D;

  cumlHandle handle;
  kNN* knn = new kNN(handle, d);
};

typedef KNN_MGTest<float> KNNTestF;
TEST_F(KNNTestF, Fit) {
  ASSERT_TRUE(devArrMatch(d_ref_D, d_pred_D, n * n, Compare<float>()));
  ASSERT_TRUE(devArrMatch(d_ref_I, d_pred_I, n * n, Compare<long>()));
}

}  // end namespace ML
