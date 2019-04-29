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

#include "random_projection/rproj_c.h"
#include <vector>
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>

namespace ML {

using namespace MLCommon;

template<typename T>
class RPROJTest: public ::testing::Test {
protected:
	void gaussianTest() {
        cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

        const int N = 1000;
        const int M = 100;
        const int D = 10;

        params1 = new paramsRPROJ();
        *params1 = {
            N,              // number of samples
            M,              // number of features
            D,              // number of components
            0.1,            // error tolerance
            true,           // gaussian or sparse method
            -1.0,           // auto density
            false,          // not used
            42              // random seed
        };

        std::vector<T> v(N*M);
        std::generate(v.begin(), v.end(), std::rand);
        allocate(d_input1, v.size());
        updateDevice(d_input1, v.data(), v.size(), stream);
        allocate(d_output1, N*D);
        random_matrix1 = new rand_mat<T>();
        RPROJfit(random_matrix1, params1);
        RPROJtransform(d_input1, random_matrix1, d_output1, params1);

        CUDA_CHECK(cudaStreamDestroy(stream));
    }


    void sparseTest() {
        cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

        const int N = 500;
        const int M = 700;

        params2 = new paramsRPROJ();
        *params2 = {
            N,              // number of samples
            M,              // number of features
            -1,             // number of components (-1: auto-deduction)
            0.3,            // error tolerance
            false,          // gaussian or sparse method
            -1.0,           // auto density (-1: auto-deduction)
            false,          // not used
            42              // random seed
        };

        std::vector<T> v(N*M);
        std::generate(v.begin(), v.end(), std::rand);
        allocate(d_input2, v.size());
        updateDevice(d_input2, v.data(), v.size(), stream);
        random_matrix2 = new rand_mat<T>();
        RPROJfit(random_matrix2, params2);
        allocate(d_output2, N*params2->n_components);
        RPROJtransform(d_input2, random_matrix2, d_output2, params2);

        CUDA_CHECK(cudaStreamDestroy(stream));
    }

 	void SetUp() override {
        gaussianTest();
        sparseTest();
	}

	void TearDown() override {
        CUDA_CHECK(cudaFree(d_input1));
        CUDA_CHECK(cudaFree(d_output1));
        CUDA_CHECK(cudaFree(d_input2));
        CUDA_CHECK(cudaFree(d_output2));
        delete params1;
        delete random_matrix1;
        delete params2;
        delete random_matrix2;
	}

protected:
    paramsRPROJ* params1;
    T* d_input1;
    rand_mat<T> *random_matrix1;
    T* d_output1;

    paramsRPROJ* params2;
    T* d_input2;
    rand_mat<T> *random_matrix2;
    T* d_output2;
};


typedef RPROJTest<float> RPROJTestF;
TEST_F(RPROJTestF, Result) {
    int M = params1->n_features;
    int D = params1->n_components;
    ASSERT_TRUE(D == 10);
    ASSERT_TRUE(random_matrix1->dense_data);

    float* h_dense_data = new float[M*D];
    updateHost(h_dense_data, random_matrix1->dense_data, M*D, NULL);
    ASSERT_TRUE(!std::any_of(h_dense_data, &h_dense_data[M*D-1], [](int i){return i < -1.0 || i > 1.0;}));
    delete[] h_dense_data;

    int N = params2->n_samples;
    M = params2->n_features;
    D = params2->n_components;
    ASSERT_TRUE(D == 690);
    ASSERT_TRUE(params2->density == 1/sqrt(M));
    ASSERT_TRUE(random_matrix2->indices);
    ASSERT_TRUE(random_matrix2->indptr);
    ASSERT_TRUE(random_matrix2->sparse_data);
    ASSERT_TRUE(random_matrix2->sparse_data_size = N*D);
}

typedef RPROJTest<double> RPROJTestD;
TEST_F(RPROJTestD, Result) {
    int M = params1->n_features;
    int D = params1->n_components;
    ASSERT_TRUE(D == 10);
    ASSERT_TRUE(random_matrix1->dense_data);

    double* h_dense_data = new double[M*D];
    updateHost(h_dense_data, random_matrix1->dense_data, M*D, NULL);
    ASSERT_TRUE(!std::any_of(h_dense_data, &h_dense_data[M*D-1], [](int i){return i < -1.0 || i > 1.0;}));
    delete[] h_dense_data;

    int N = params2->n_samples;
    M = params2->n_features;
    D = params2->n_components;
    ASSERT_TRUE(D == 690);
    ASSERT_TRUE(params2->density == 1/sqrt(M));
    ASSERT_TRUE(random_matrix2->indices);
    ASSERT_TRUE(random_matrix2->indptr);
    ASSERT_TRUE(random_matrix2->sparse_data);
    ASSERT_TRUE(random_matrix2->sparse_data_size = N*D);
}

} // end namespace ML
