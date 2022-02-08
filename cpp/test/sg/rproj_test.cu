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

#include <cuml/metrics/metrics.hpp>
#include <cuml/random_projection/rproj_c.h>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/linalg/transpose.hpp>
#include <random>
#include <test_utils.h>
#include <vector>

namespace ML {

template <typename T, int N, int M>
class RPROJTest : public ::testing::Test {
 public:
  RPROJTest()
    : stream(handle.get_stream()),
      random_matrix1(stream),
      random_matrix2(stream),
      d_input(0, stream),
      d_output1(0, stream),
      d_output2(0, stream)
  {
  }

 protected:
  void generate_data()
  {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> dist(0, 1);

    h_input.resize(N * M);
    for (auto& i : h_input) {
      i = dist(rng);
    }
    d_input.resize(h_input.size(), stream);
    raft::update_device(d_input.data(), h_input.data(), h_input.size(), stream);
    // transpose(d_input, d_input, N, M);
    // From row major to column major (this operation is only useful for non-random datasets)
  }

  void gaussianTest()
  {
    params1 = {
      N,        // number of samples
      M,        // number of features
      -1,       // number of components
      epsilon,  // error tolerance
      true,     // gaussian or sparse method
      -1.0,     // auto density
      false,    // not used
      42        // random seed
    };

    RPROJfit(handle, &random_matrix1, &params1);

    d_output1.resize(N * params1.n_components, stream);
    rmm::device_uvector<T> tmp(d_output1.size(), stream);
    RPROJtransform(handle, d_input.data(), &random_matrix1, tmp.data(), &params1);

    raft::linalg::transpose(handle,
                            tmp.data(),
                            d_output1.data(),
                            N,
                            params1.n_components,
                            stream);  // From column major to row major

    handle.sync_stream(stream);
  }

  void sparseTest()
  {
    params2 = {
      N,        // number of samples
      M,        // number of features
      -1,       // number of components (-1: auto-deduction)
      epsilon,  // error tolerance
      false,    // gaussian or sparse method
      -1.0,     // auto density (-1: auto-deduction)
      false,    // not used
      42        // random seed
    };

    RPROJfit(handle, &random_matrix2, &params2);

    d_output2.resize(N * params2.n_components, stream);
    rmm::device_uvector<T> tmp(d_output2.size(), stream);
    RPROJtransform(handle, d_input.data(), &random_matrix2, tmp.data(), &params2);

    raft::linalg::transpose(handle,
                            tmp.data(),
                            d_output2.data(),
                            N,
                            params2.n_components,
                            stream);  // From column major to row major

    handle.sync_stream(stream);
  }

  void SetUp() override
  {
    epsilon = 0.2;
    generate_data();
    gaussianTest();
    sparseTest();
  }

  void random_matrix_check()
  {
    int D = johnson_lindenstrauss_min_dim(N, epsilon);

    ASSERT_TRUE(params1.n_components == D);
    ASSERT_TRUE(random_matrix1.dense_data.size() > 0);
    ASSERT_TRUE(random_matrix1.type == dense);

    ASSERT_TRUE(params2.n_components == D);
    ASSERT_TRUE(params2.density == 1 / sqrt(M));
    ASSERT_TRUE(random_matrix2.indices.size() > 0);
    ASSERT_TRUE(random_matrix2.indptr.size() > 0);
    ASSERT_TRUE(random_matrix2.sparse_data.size() > 0);
    ASSERT_TRUE(random_matrix2.type == sparse);
  }

  void epsilon_check()
  {
    int D                        = johnson_lindenstrauss_min_dim(N, epsilon);
    constexpr auto distance_type = raft::distance::DistanceType::L2SqrtUnexpanded;

    rmm::device_uvector<T> d_pdist(N * N, stream);
    ML::Metrics::pairwise_distance(
      handle, d_input.data(), d_input.data(), d_pdist.data(), N, N, M, distance_type);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    T* h_pdist = new T[N * N];
    raft::update_host(h_pdist, d_pdist.data(), N * N, stream);

    rmm::device_uvector<T> d_pdist1(N * N, stream);
    ML::Metrics::pairwise_distance(
      handle, d_output1.data(), d_output1.data(), d_pdist1.data(), N, N, D, distance_type);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    T* h_pdist1 = new T[N * N];
    raft::update_host(h_pdist1, d_pdist1.data(), N * N, stream);

    rmm::device_uvector<T> d_pdist2(N * N, stream);
    ML::Metrics::pairwise_distance(
      handle, d_output2.data(), d_output2.data(), d_pdist2.data(), N, N, D, distance_type);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    T* h_pdist2 = new T[N * N];
    raft::update_host(h_pdist2, d_pdist2.data(), N * N, stream);

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
  raft::handle_t handle;
  cudaStream_t stream = 0;

  T epsilon;

  std::vector<T> h_input;
  rmm::device_uvector<T> d_input;

  paramsRPROJ params1;
  rand_mat<T> random_matrix1;
  rmm::device_uvector<T> d_output1;

  paramsRPROJ params2;
  rand_mat<T> random_matrix2;
  rmm::device_uvector<T> d_output2;
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
