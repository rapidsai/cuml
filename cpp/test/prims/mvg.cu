/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/cudart_utils.h>
#include <random/mvg.cuh>
#include <random>
#include <rmm/device_uvector.hpp>

// mvg.h takes in matrices that are colomn major (as in fortan)
#define IDX2C(i, j, ld) (j * ld + i)

namespace MLCommon {
namespace Random {

// helper kernels
/// @todo Duplicate called vctwiseAccumulate in utils.h (Kalman Filters,
// i think that is much better to use., more general)
template <typename T>
__global__ void En_KF_accumulate(const int nPoints, const int dim, const T* X, T* x)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints) raft::myAtomicAdd(x + col, X[idx]);
}

template <typename T>
__global__ void En_KF_normalize(const int divider, const int dim, T* x)
{
  int xi = threadIdx.x + blockDim.x * blockIdx.x;
  if (xi < dim) x[xi] = x[xi] / divider;
}

template <typename T>
__global__ void En_KF_dif(const int nPoints, const int dim, const T* X, const T* x, T* X_diff)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints) X_diff[idx] = X[idx] - x[col];
}

// for specialising tests
enum Correlation : unsigned char {
  CORRELATED,  // = 0
  UNCORRELATED
};

template <typename T>
struct MVGInputs {
  T tolerance;
  typename MultiVarGaussian<T>::Decomposer method;
  Correlation corr;
  int dim, nPoints;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MVGInputs<T>& dims)
{
  return os;
}

template <typename T>
class MVGTest : public ::testing::TestWithParam<MVGInputs<T>> {
 protected:
  MVGTest()
    : workspace_d(0, stream),
      P_d(0, stream),
      x_d(0, stream),
      X_d(0, stream),
      Rand_cov(0, stream),
      Rand_mean(0, stream)
  {
  }

  void SetUp() override
  {
    // getting params
    params    = ::testing::TestWithParam<MVGInputs<T>>::GetParam();
    dim       = params.dim;
    nPoints   = params.nPoints;
    method    = params.method;
    corr      = params.corr;
    tolerance = params.tolerance;

    RAFT_CUBLAS_TRY(cublasCreate(&cublasH));
    RAFT_CUSOLVER_TRY(cusolverDnCreate(&cusolverH));
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // preparing to store stuff
    P.resize(dim * dim);
    x.resize(dim);
    X.resize(dim * nPoints);
    P_d.resize(dim * dim, stream);
    X_d.resize(nPoints * dim, stream);
    x_d.resize(dim, stream);
    Rand_cov.resize(dim * dim, stream);
    Rand_mean.resize(dim, stream);

    // generating random mean and cov.
    srand(params.seed);
    for (int j = 0; j < dim; j++)
      x.data()[j] = rand() % 100 + 5.0f;

    // for random Cov. martix
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // P (developing a +ve definite symm matrix)
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < j + 1; i++) {
        T k = distribution(generator);
        if (corr == UNCORRELATED) k = 0.0;
        P.data()[IDX2C(i, j, dim)] = k;
        P.data()[IDX2C(j, i, dim)] = k;
        if (i == j) P.data()[IDX2C(i, j, dim)] += dim;
      }
    }

    // porting inputs to gpu
    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
    raft::update_device(x_d.data(), x.data(), dim, stream);

    // initilizing the mvg
    mvg      = new MultiVarGaussian<T>(dim, method);
    size_t o = mvg->init(cublasH, cusolverH, stream);

    // give the workspace area to mvg
    workspace_d.resize(o, stream);
    mvg->set_workspace(workspace_d.data());

    // get gaussians in X_d | P_d is destroyed.
    mvg->give_gaussian(nPoints, P_d.data(), X_d.data(), x_d.data());

    // saving the mean of the randoms in Rand_mean
    //@todo can be swapped with a API that calculates mean
    RAFT_CUDA_TRY(cudaMemset(Rand_mean.data(), 0, dim * sizeof(T)));
    dim3 block = (64);
    dim3 grid  = (raft::ceildiv(nPoints * dim, (int)block.x));
    En_KF_accumulate<<<grid, block>>>(nPoints, dim, X_d.data(), Rand_mean.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    grid = (raft::ceildiv(dim, (int)block.x));
    En_KF_normalize<<<grid, block>>>(nPoints, dim, Rand_mean.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // storing the error wrt random point mean in X_d
    grid = (raft::ceildiv(dim * nPoints, (int)block.x));
    En_KF_dif<<<grid, block>>>(nPoints, dim, X_d.data(), Rand_mean.data(), X_d.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // finding the cov matrix, placing in Rand_cov
    T alfa = 1.0 / (nPoints - 1), beta = 0.0;
    cublasHandle_t handle;
    RAFT_CUBLAS_TRY(cublasCreate(&handle));
    RAFT_CUBLAS_TRY(raft::linalg::cublasgemm(handle,
                                             CUBLAS_OP_N,
                                             CUBLAS_OP_T,
                                             dim,
                                             dim,
                                             nPoints,
                                             &alfa,
                                             X_d.data(),
                                             dim,
                                             X_d.data(),
                                             dim,
                                             &beta,
                                             Rand_cov.data(),
                                             dim,
                                             stream));

    // restoring cov provided into P_d
    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
  }

  void TearDown() override
  {
    // deleting mvg
    mvg->deinit();
    delete mvg;

    RAFT_CUBLAS_TRY(cublasDestroy(cublasH));
    RAFT_CUSOLVER_TRY(cusolverDnDestroy(cusolverH));
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  MVGInputs<T> params;
  std::vector<T> P, x, X;
  rmm::device_uvector<T> workspace_d, P_d, x_d, X_d, Rand_cov, Rand_mean;
  int dim, nPoints;
  typename MultiVarGaussian<T>::Decomposer method;
  Correlation corr;
  MultiVarGaussian<T>* mvg = NULL;
  T tolerance;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cudaStream_t stream = 0;
};  // end of MVGTest class

///@todo find out the reason that Un-correlated covs are giving problems (in qr)
// Declare your inputs
const std::vector<MVGInputs<float>> inputsf = {
  {0.3f, MultiVarGaussian<float>::Decomposer::chol_decomp, Correlation::CORRELATED, 5, 30000, 6ULL},
  {0.1f,
   MultiVarGaussian<float>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED,
   5,
   30000,
   6ULL},
  {0.25f, MultiVarGaussian<float>::Decomposer::jacobi, Correlation::CORRELATED, 5, 30000, 6ULL},
  {0.1f, MultiVarGaussian<float>::Decomposer::jacobi, Correlation::UNCORRELATED, 5, 30000, 6ULL},
  {0.2f, MultiVarGaussian<float>::Decomposer::qr, Correlation::CORRELATED, 5, 30000, 6ULL},
  // { 0.2f,          MultiVarGaussian<float>::Decomposer::qr,
  // Correlation::UNCORRELATED, 5, 30000, 6ULL}
};
const std::vector<MVGInputs<double>> inputsd = {
  {0.25,
   MultiVarGaussian<double>::Decomposer::chol_decomp,
   Correlation::CORRELATED,
   10,
   3000000,
   6ULL},
  {0.1,
   MultiVarGaussian<double>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED,
   10,
   3000000,
   6ULL},
  {0.25, MultiVarGaussian<double>::Decomposer::jacobi, Correlation::CORRELATED, 10, 3000000, 6ULL},
  {0.1, MultiVarGaussian<double>::Decomposer::jacobi, Correlation::UNCORRELATED, 10, 3000000, 6ULL},
  {0.2, MultiVarGaussian<double>::Decomposer::qr, Correlation::CORRELATED, 10, 3000000, 6ULL},
  // { 0.2,          MultiVarGaussian<double>::Decomposer::qr,
  // Correlation::UNCORRELATED, 10, 3000000, 6ULL}
};

// make the tests
typedef MVGTest<float> MVGTestF;
typedef MVGTest<double> MVGTestD;
TEST_P(MVGTestF, MeanIsCorrectF)
{
  EXPECT_TRUE(
    raft::devArrMatch(x_d.data(), Rand_mean.data(), dim, raft::CompareApprox<float>(tolerance)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestF, CovIsCorrectF)
{
  EXPECT_TRUE(
    raft::devArrMatch(P_d.data(), Rand_cov.data(), dim, dim, raft::CompareApprox<float>(tolerance)))
    << " in CovIsCorrect";
}
TEST_P(MVGTestD, MeanIsCorrectD)
{
  EXPECT_TRUE(
    raft::devArrMatch(x_d.data(), Rand_mean.data(), dim, raft::CompareApprox<double>(tolerance)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestD, CovIsCorrectD)
{
  EXPECT_TRUE(raft::devArrMatch(
    P_d.data(), Rand_cov.data(), dim, dim, raft::CompareApprox<double>(tolerance)))
    << " in CovIsCorrect";
}

// call the tests
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestD, ::testing::ValuesIn(inputsd));

};  // end of namespace Random
};  // end of namespace MLCommon
