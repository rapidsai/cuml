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

#include "knn/knn.hpp"
#include <vector>
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>

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

		// Allocate input
        cudaSetDevice(0);
        allocate(d_train_inputs_dev1, n * d);
        cudaSetDevice(1);
        allocate(d_train_inputs_dev2, n * d);

        // Allocate reference arrays
        allocate<long>(d_ref_I, n*n);
        allocate(d_ref_D, n*n);

        // Allocate predicted arrays
        allocate<long>(d_pred_I, n*n);
        allocate(d_pred_D, n*n);

        // make test data on host
        std::vector<T> h_train_inputs = {1.0, 50.0, 51.0};
        h_train_inputs.resize(n);

        updateDevice(d_train_inputs_dev1, h_train_inputs.data(), n*d, 0);
        updateDevice(d_train_inputs_dev2, h_train_inputs.data(), n*d, 0);

        std::vector<T> h_res_D = { 0.0, 0.0, 2401.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
        h_res_D.resize(n*n);
        updateDevice(d_ref_D, h_res_D.data(), n*n, 0);

        std::vector<long> h_res_I = { 0, 3, 1, 1, 4, 2, 2, 5, 1 };
        h_res_I.resize(n*n);
        updateDevice<long>(d_ref_I, h_res_I.data(), n*n, 0);

        float **ptrs = new float*[2];
        int *sizes = new int[2];
        ptrs[0] = d_train_inputs_dev1;
        sizes[0] = n;

        ptrs[1] = d_train_inputs_dev2;
        sizes[1] = n;

        cudaSetDevice(0);

        knn->fit(ptrs, sizes, 2);
        knn->search(d_train_inputs_dev1, n, d_pred_I, d_pred_D, n);

        delete ptrs;
        delete sizes;
    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(d_train_inputs_dev1));
		CUDA_CHECK(cudaFree(d_train_inputs_dev2));
		CUDA_CHECK(cudaFree(d_pred_I));
		CUDA_CHECK(cudaFree(d_pred_D));
		CUDA_CHECK(cudaFree(d_ref_I));
		CUDA_CHECK(cudaFree(d_ref_D));
	}

protected:

	T* d_train_inputs_dev1;
	T* d_train_inputs_dev2;

	int n = 3;
	int d = 1;

    long *d_pred_I;
    T* d_pred_D;

    long *d_ref_I;
    T* d_ref_D;

    cumlHandle handle;
    kNN *knn = new kNN(handle, d);
};


typedef KNN_MGTest<float> KNNTestF;
TEST_F(KNNTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(d_ref_D, d_pred_D, n*n, Compare<float>()));
	ASSERT_TRUE(
			devArrMatch(d_ref_I, d_pred_I, n*n, Compare<long>()));
}

} // end namespace ML
