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

#include <cuml/random_projection/rproj_c.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/transpose.h>
#include <test_utils.h>
#include <cuml/metrics/metrics.hpp>
#include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.hpp>
#include <random>
#include <vector>

namespace ML {

template <typename T, int N, int M>
class RPROJTest : public ::testing::Test {
 protected:
  T* transpose(T* in, int n_rows, int n_cols)
  {
    cudaStream_t stream          = h.get_stream();
    cublasHandle_t cublas_handle = h.get_cublas_handle();
    T* result;
    raft::allocate(result, n_rows * n_cols, stream);
    raft::linalg::transpose(h, in, result, n_rows, n_cols, stream);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaFree(in));
    return result;
  }

  void generate_data()
  {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> dist(0, 1);

    h_input.resize(N * M);
    for (auto& i : h_input) {
      i = dist(rng);
    }
    raft::allocate(d_input, h_input.size(), h.get_stream());
    raft::update_device(d_input, h_input.data(), h_input.size(), h.get_stream());
    // d_input = transpose(d_input, N, M);
    // From row major to column major (this operation is only useful for non-random datasets)
  }

  void gaussianTest()
  {
    params1  = new paramsRPROJ();
    *params1 = {
      N,        // number of samples
      M,        // number of features
      -1,       // number of components
      epsilon,  // error tolerance
      true,     // gaussian or sparse method
      -1.0,     // auto density
      false,    // not used
      42        // random seed
    };

    cudaStream_t stream = h.get_stream();
    random_matrix1      = std::make_unique<rand_mat<T>>(stream);
    RPROJfit(h, random_matrix1.get(), params1);
    raft::allocate(d_output1, N * params1->n_components, stream);
    RPROJtransform(h, d_input, random_matrix1.get(), d_output1, params1);
    d_output1 = transpose(d_output1, N, params1->n_components);  // From column major to row major
  }

  void sparseTest()
  {
    params2  = new paramsRPROJ();
    *params2 = {
      N,        // number of samples
      M,        // number of features
      -1,       // number of components (-1: auto-deduction)
      epsilon,  // error tolerance
      false,    // gaussian or sparse method
      -1.0,     // auto density (-1: auto-deduction)
      false,    // not used
      42        // random seed
    };

    cudaStream_t stream = h.get_stream();
    random_matrix2      = std::make_unique<rand_mat<T>>(stream);
    RPROJfit(h, random_matrix2.get(), params2);

    raft::allocate(d_output2, N * params2->n_components, stream);

    RPROJtransform(h, d_input, random_matrix2.get(), d_output2, params2);

    d_output2 = transpose(d_output2, N, params2->n_components);  // From column major to row major
  }

  void SetUp() override
  {
    epsilon = 0.2;
    generate_data();
    gaussianTest();
    sparseTest();
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output1));
    CUDA_CHECK(cudaFree(d_output2));
    delete params1;
    delete params2;
  }

  void random_matrix_check()
  {
    size_t D = johnson_lindenstrauss_min_dim(N, epsilon);

    ASSERT_TRUE(params1->n_components == D);
    ASSERT_TRUE(random_matrix1->dense_data.size() > 0);
    ASSERT_TRUE(random_matrix1->type == dense);

    ASSERT_TRUE(params2->n_components == D);
    ASSERT_TRUE(params2->density == 1 / sqrt(M));
    ASSERT_TRUE(random_matrix2->indices.size() > 0);
    ASSERT_TRUE(random_matrix2->indptr.size() > 0);
    ASSERT_TRUE(random_matrix2->sparse_data.size() > 0);
    ASSERT_TRUE(random_matrix2->type == sparse);
  }

  void epsilon_check()
  {
    int D = johnson_lindenstrauss_min_dim(N, epsilon);

    constexpr auto distance_type = raft::distance::DistanceType::L2SqrtUnexpanded;

    cudaStream_t stream = h.get_stream();

    T* d_pdist;
    raft::allocate(d_pdist, N * N, stream);
    ML::Metrics::pairwise_distance(h, d_input, d_input, d_pdist, N, N, M, distance_type);
    CUDA_CHECK(cudaPeekAtLastError());

    T* h_pdist = new T[N * N];
    raft::update_host(h_pdist, d_pdist, N * N, stream);
    CUDA_CHECK(cudaFree(d_pdist));

    T* d_pdist1;
    raft::allocate(d_pdist1, N * N, stream);
    ML::Metrics::pairwise_distance(h, d_output1, d_output1, d_pdist1, N, N, D, distance_type);
    CUDA_CHECK(cudaPeekAtLastError());

    T* h_pdist1 = new T[N * N];
    raft::update_host(h_pdist1, d_pdist1, N * N, stream);
    CUDA_CHECK(cudaFree(d_pdist1));

    T* d_pdist2;
    raft::allocate(d_pdist2, N * N, stream);
    ML::Metrics::pairwise_distance(h, d_output2, d_output2, d_pdist2, N, N, D, distance_type);
    CUDA_CHECK(cudaPeekAtLastError());

    T* h_pdist2 = new T[N * N];
    raft::update_host(h_pdist2, d_pdist2, N * N, stream);
    CUDA_CHECK(cudaFree(d_pdist2));

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j <= i; j++) {
        T pdist  = h_pdist[i * N + j];
        T pdist1 = h_pdist1[i * N + j];
        T pdist2 = h_pdist2[i * N + j];

        T lower_bound = (1.0 - epsilon) * pdist;
        T upper_bound = (1.0 + epsilon) * pdist;

        ASSERT_TRUE(lower_bound <= pdist1 && pdist1 <= upper_bound);
        ASSERT_TRUE(lower_bound <= pdist2 && pdist2 <= upper_bound);
      }
    }

    delete[] h_pdist;
    delete[] h_pdist1;
    delete[] h_pdist2;
  }

 protected:
  raft::handle_t h;
  paramsRPROJ* params1;
  T epsilon;

  std::vector<T> h_input;
  T* d_input;

  std::unique_ptr<rand_mat<T>> random_matrix1;
  T* d_output1;

  paramsRPROJ* params2;
  std::unique_ptr<rand_mat<T>> random_matrix2;
  T* d_output2;
};

typedef RPROJTest<float, 500, 2000> RPROJTestF1;
TEST_F(RPROJTestF1, RandomMatrixCheck) { random_matrix_check(); }
TEST_F(RPROJTestF1, EpsilonCheck) { epsilon_check(); }

typedef RPROJTest<double, 500, 2000> RPROJTestD1;
TEST_F(RPROJTestD1, RandomMatrixCheck) { random_matrix_check(); }
TEST_F(RPROJTestD1, EpsilonCheck) { epsilon_check(); }

typedef RPROJTest<float, 5000, 3500> RPROJTestF2;
TEST_F(RPROJTestF2, RandomMatrixCheck) { random_matrix_check(); }
TEST_F(RPROJTestF2, EpsilonCheck) { epsilon_check(); }

typedef RPROJTest<double, 5000, 3500> RPROJTestD2;
TEST_F(RPROJTestD2, RandomMatrixCheck) { random_matrix_check(); }
TEST_F(RPROJTestD2, EpsilonCheck) { epsilon_check(); }

}  // end namespace ML
