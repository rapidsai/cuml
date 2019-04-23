/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <cmath>
#include <iostream>
#include <random>
#include "random/mvg.h"
#include "test_utils.h"

// mvg.h takes in matrices that are colomn major (as in fortan)
#define IDX2C(i, j, ld) (j * ld + i)

namespace MLCommon {
namespace Random {

// helper kernels
/// @todo Duplicate called vctwiseAccumulate in utils.h (Kalman Filters,
// i think that is much better to use., more general)
template <typename T>
__global__ void En_KF_accumulate(const int nPoints, const int dim, const T *X,
                                 T *x) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints)
    myAtomicAdd(x + col, X[idx]);
}

template <typename T>
__global__ void En_KF_normalize(const int divider, const int dim, T *x) {
  int xi = threadIdx.x + blockDim.x * blockIdx.x;
  if (xi < dim)
    x[xi] = x[xi] / divider;
}

template <typename T>
__global__ void En_KF_dif(const int nPoints, const int dim, const T *X,
                          const T *x, T *X_diff) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints)
    X_diff[idx] = X[idx] - x[col];
}

// for specialising tests
enum Correlation : unsigned char {
  CORRELATED, // = 0
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
::std::ostream &operator<<(::std::ostream &os, const MVGInputs<T> &dims) {
  return os;
}

template <typename T>
class MVGTest : public ::testing::TestWithParam<MVGInputs<T>> {
protected:
  void SetUp() override {
    // getting params
    params = ::testing::TestWithParam<MVGInputs<T>>::GetParam();
    dim = params.dim;
    nPoints = params.nPoints;
    method = params.method;
    corr = params.corr;
    tolerance = params.tolerance;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // preparing to store stuff
    P = (T *)malloc(sizeof(T) * dim * dim);
    x = (T *)malloc(sizeof(T) * dim);
    X = (T *)malloc(sizeof(T) * dim * nPoints);
    CUDA_CHECK(cudaMalloc((void **)&P_d, sizeof(T) * dim * dim));
    CUDA_CHECK(cudaMalloc((void **)&X_d, sizeof(T) * nPoints * dim));
    CUDA_CHECK(cudaMalloc((void **)&x_d, sizeof(T) * dim));
    CUDA_CHECK(cudaMalloc((void **)&Rand_cov, sizeof(T) * dim * dim));
    CUDA_CHECK(cudaMalloc((void **)&Rand_mean, sizeof(T) * dim));

    // generating random mean and cov.
    srand(params.seed);
    for (int j = 0; j < dim; j++)
      x[j] = rand() % 100 + 5.0f;

    // for random Cov. martix
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // P (developing a +ve definite symm matrix)
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < j + 1; i++) {
        T k = distribution(generator);
        if (corr == UNCORRELATED)
          k = 0.0;
        P[IDX2C(i, j, dim)] = k;
        P[IDX2C(j, i, dim)] = k;
        if (i == j)
          P[IDX2C(i, j, dim)] += dim;
      }
    }

    // porting inputs to gpu
    updateDevice(P_d, P, dim * dim);
    updateDevice(x_d, x, dim);

    // initilizing the mvg
    mvg = new MultiVarGaussian<T>(dim, method);
    size_t o = mvg->init(cublasH, cusolverH, stream);

    // give the workspace area to mvg
    CUDA_CHECK(cudaMalloc((void **)&workspace_d, o));
    mvg->set_workspace(workspace_d);

    // get gaussians in X_d | P_d is destroyed.
    mvg->give_gaussian(nPoints, P_d, X_d, x_d);

    // saving the mean of the randoms in Rand_mean
    //@todo can be swapped with a API that calculates mean
    CUDA_CHECK(cudaMemset(Rand_mean, 0, dim * sizeof(T)));
    dim3 block = (64);
    dim3 grid = (ceildiv(nPoints * dim, (int)block.x));
    En_KF_accumulate<<<grid, block>>>(nPoints, dim, X_d, Rand_mean);
    CUDA_CHECK(cudaPeekAtLastError());
    grid = (ceildiv(dim, (int)block.x));
    En_KF_normalize<<<grid, block>>>(nPoints, dim, Rand_mean);
    CUDA_CHECK(cudaPeekAtLastError());

    // storing the error wrt random point mean in X_d
    grid = (ceildiv(dim * nPoints, (int)block.x));
    En_KF_dif<<<grid, block>>>(nPoints, dim, X_d, Rand_mean, X_d);
    CUDA_CHECK(cudaPeekAtLastError());

    // finding the cov matrix, placing in Rand_cov
    T alfa = 1.0 / (nPoints - 1), beta = 0.0;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(LinAlg::cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim, dim,
                                    nPoints, &alfa, X_d, dim, X_d, dim, &beta,
                                    Rand_cov, dim, stream));

    // restoring cov provided into P_d
    updateDevice(P_d, P, dim * dim);
  }

  void TearDown() override {
    // freeing mallocs
    CUDA_CHECK(cudaFree(P_d));
    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(workspace_d));
    free(P);
    free(x);
    free(X);

    // deleting mvg
    mvg->deinit();
    delete mvg;

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

protected:
  MVGInputs<T> params;
  T *P, *x, *X, *workspace_d, *P_d, *x_d, *X_d;
  int dim, nPoints;
  typename MultiVarGaussian<T>::Decomposer method;
  Correlation corr;
  MultiVarGaussian<T> *mvg = NULL;
  T *Rand_cov, *Rand_mean, tolerance;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cudaStream_t stream;
}; // end of MVGTest class

///@todo find out the reason that Un-correlated covs are giving problems (in qr)
// Declare your inputs
const std::vector<MVGInputs<float>> inputsf = {
  {0.3f, MultiVarGaussian<float>::Decomposer::chol_decomp,
   Correlation::CORRELATED, 5, 30000, 6ULL},
  {0.1f, MultiVarGaussian<float>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED, 5, 30000, 6ULL},
  {0.25f, MultiVarGaussian<float>::Decomposer::jacobi, Correlation::CORRELATED,
   5, 30000, 6ULL},
  {0.1f, MultiVarGaussian<float>::Decomposer::jacobi, Correlation::UNCORRELATED,
   5, 30000, 6ULL},
  {0.2f, MultiVarGaussian<float>::Decomposer::qr, Correlation::CORRELATED, 5,
   30000, 6ULL},
  // { 0.2f,          MultiVarGaussian<float>::Decomposer::qr,
  // Correlation::UNCORRELATED, 5, 30000, 6ULL}
};
const std::vector<MVGInputs<double>> inputsd = {
  {0.25, MultiVarGaussian<double>::Decomposer::chol_decomp,
   Correlation::CORRELATED, 10, 3000000, 6ULL},
  {0.1, MultiVarGaussian<double>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED, 10, 3000000, 6ULL},
  {0.25, MultiVarGaussian<double>::Decomposer::jacobi, Correlation::CORRELATED,
   10, 3000000, 6ULL},
  {0.1, MultiVarGaussian<double>::Decomposer::jacobi, Correlation::UNCORRELATED,
   10, 3000000, 6ULL},
  {0.2, MultiVarGaussian<double>::Decomposer::qr, Correlation::CORRELATED, 10,
   3000000, 6ULL},
  // { 0.2,          MultiVarGaussian<double>::Decomposer::qr,
  // Correlation::UNCORRELATED, 10, 3000000, 6ULL}
};

// make the tests
typedef MVGTest<float> MVGTestF;
typedef MVGTest<double> MVGTestD;
TEST_P(MVGTestF, MeanIsCorrectF) {
  EXPECT_TRUE(devArrMatch(x_d, Rand_mean, dim, CompareApprox<float>(tolerance)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestF, CovIsCorrectF) {
  EXPECT_TRUE(
    devArrMatch(P_d, Rand_cov, dim, dim, CompareApprox<float>(tolerance)))
    << " in CovIsCorrect";
}
TEST_P(MVGTestD, MeanIsCorrectD) {
  EXPECT_TRUE(
    devArrMatch(x_d, Rand_mean, dim, CompareApprox<double>(tolerance)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestD, CovIsCorrectD) {
  EXPECT_TRUE(
    devArrMatch(P_d, Rand_cov, dim, dim, CompareApprox<double>(tolerance)))
    << " in CovIsCorrect";
}

// call the tests
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestD, ::testing::ValuesIn(inputsd));

}; // end of namespace Random
}; // end of namespace MLCommon
