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

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include "kalman_filter/KalmanFilter.cuh"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"

namespace kf {
namespace linear {

using namespace MLCommon;

template <typename T>
struct LKFInputs {
  T tolerance;
  int dim_x, dim_z, iterations;
  Option option;
  unsigned long long int seed;
  // 0 = Long, 1 = Shrt_Exp, 2 = Shrt_Imp (option)
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const LKFInputs<T> &dims) {
  return os;
}

template <typename T>
class LKFTest : public ::testing::TestWithParam<LKFInputs<T>> {
 protected:  // functionsv
  void SetUp() override {
    // getting params
    params = ::testing::TestWithParam<LKFInputs<T>>::GetParam();
    dim_x = params.dim_x;
    dim_z = params.dim_z;
    iterations = params.iterations;
    option = (Option)params.option;
    tolerance = params.tolerance;

    // cpu mallocs
    Phi = (T *)malloc(dim_x * dim_x * sizeof(T));
    x_up = (T *)malloc(dim_x * 1 * sizeof(T));
    x_est = (T *)malloc(dim_x * 1 * sizeof(T));
    P_up = (T *)malloc(dim_x * dim_x * sizeof(T));
    Q = (T *)malloc(dim_x * dim_x * sizeof(T));
    H = (T *)malloc(dim_z * dim_x * sizeof(T));
    R = (T *)malloc(dim_z * dim_z * sizeof(T));
    z = (T *)malloc(dim_z * 1 * sizeof(T));

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle = NULL;
    cudaStream_t stream;

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // making sane model
    x_up[0] = 0.0;
    x_up[1] = 1.0;
    Phi[0] = 1.0;
    Phi[1] = 0.0;
    Phi[2] = 1.0;
    Phi[3] = 1.0;
    P_up[0] = 100.0;
    P_up[1] = 0.0;
    P_up[2] = 0.0;
    P_up[3] = 100.0;
    R[0] = 100.0;
    T var = 0.001;
    Q[0] = 0.25 * var;
    Q[1] = 0.5 * var;
    Q[2] = 0.5 * var;
    Q[3] = 1.1 * var;
    H[0] = 1.0;
    H[1] = 0.0;

    // gpu mallocs
    CUDA_CHECK(cudaMalloc((void **)&x_est_d, dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&x_up_d, dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&Phi_d, dim_x * dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&P_est_d, dim_x * dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&P_up_d, dim_x * dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&Q_d, dim_x * dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&R_d, dim_z * dim_z * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&H_d, dim_z * dim_x * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&z_d, dim_z * sizeof(T)));

    // copy data to gpu (available in ml-common/cuda_utils.h)
    updateDevice(Phi_d, Phi, dim_x * dim_x, stream);
    updateDevice(x_up_d, x_up, dim_x, stream);
    updateDevice(P_up_d, P_up, dim_x * dim_x, stream);
    updateDevice(Q_d, Q, dim_x * dim_x, stream);
    updateDevice(R_d, R, dim_z * dim_z, stream);
    updateDevice(H_d, H, dim_z * dim_x, stream);

    // kf initialization
    Variables<T> vars;
    size_t workspaceSize;
    init(vars, dim_x, dim_z, option, x_est_d, x_up_d, Phi_d, P_est_d, P_up_d,
         Q_d, R_d, H_d, nullptr, workspaceSize, cusolver_handle);
    CUDA_CHECK(cudaMalloc((void **)&workspace_l, workspaceSize));
    init(vars, dim_x, dim_z, option, x_est_d, x_up_d, Phi_d, P_est_d, P_up_d,
         Q_d, R_d, H_d, workspace_l, workspaceSize, cusolver_handle);

    // for random noise
    std::default_random_engine generator(params.seed);
    std::normal_distribution<T> distribution(0.0, 1.0);
    rmse_x = 0.0;
    rmse_v = 0.0;

    for (int q = 0; q < iterations; q++) {
      predict(vars, cublas_handle, stream);
      // generating measurement
      z[0] = q + distribution(generator);
      updateDevice(z_d, z, dim_z, stream);

      update(vars, z_d, cublas_handle, cusolver_handle, stream);
      // getting update
      updateHost(x_up, x_up_d, dim_x, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // summing squared ratios
      rmse_v += pow(x_up[1] - 1, 2);  // true velo is alwsy 1
      rmse_x += pow(x_up[0] - q, 2);
    }
    rmse_x /= iterations;
    rmse_v /= iterations;
    rmse_x = pow(rmse_x, 0.5);
    rmse_v = pow(rmse_v, 0.5);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    // freeing gpu mallocs
    CUDA_CHECK(cudaFree(workspace_l));
    CUDA_CHECK(cudaFree(Phi_d));
    CUDA_CHECK(cudaFree(P_up_d));
    CUDA_CHECK(cudaFree(P_est_d));
    CUDA_CHECK(cudaFree(Q_d));
    CUDA_CHECK(cudaFree(R_d));
    CUDA_CHECK(cudaFree(H_d));
    CUDA_CHECK(cudaFree(x_est_d));
    CUDA_CHECK(cudaFree(x_up_d));
    CUDA_CHECK(cudaFree(z_d));

    // freeing cpu mallocs
    free(Phi);
    free(x_up);
    free(x_est);
    free(P_up);
    free(Q);
    free(H);
    free(R);
    free(z);
  }

 protected:  // variables
  LKFInputs<T> params;
  Option option;
  T *Phi, *x_up, *x_est, *P_up, *Q, *H, *R, *z;  //cpu pointers
  T *x_est_d, *x_up_d, *Phi_d, *P_est_d, *P_up_d, *Q_d, *R_d, *H_d, *z_d,
    *workspace_l;               //gpu pointers
  T rmse_x, rmse_v, tolerance;  // root mean squared error
  int dim_z, dim_x, iterations;
};  // LKFTest

// float
const std::vector<LKFInputs<float>> inputsf = {
  {1.5f, 2, 1, 100, Option::LongForm, 6ULL},
  {1.5f, 2, 1, 100, Option::ShortFormExplicit, 6ULL},
  {1.3f, 2, 1, 100, Option::ShortFormImplicit, 6ULL}};
typedef LKFTest<float> LKFTestF;
TEST_P(LKFTestF, RMSEUnderToleranceF) {
  EXPECT_LT(rmse_x, tolerance) << " position out of tol.";
  EXPECT_LT(rmse_v, tolerance) << " velocity out of tol.";
}
INSTANTIATE_TEST_CASE_P(LKFTests, LKFTestF, ::testing::ValuesIn(inputsf));

// double
const std::vector<LKFInputs<double>> inputsd = {
  {1.5, 2, 1, 100, Option::LongForm, 6ULL},
  {1.5, 2, 1, 100, Option::ShortFormExplicit, 6ULL},
  {1.2, 2, 1, 100, Option::ShortFormImplicit, 6ULL}};
typedef LKFTest<double> LKFTestD;
TEST_P(LKFTestD, RMSEUnderToleranceD) {
  EXPECT_LT(rmse_x, tolerance) << " position out of tol.";
  EXPECT_LT(rmse_v, tolerance) << " velocity out of tol.";
}
INSTANTIATE_TEST_CASE_P(LKFTests, LKFTestD, ::testing::ValuesIn(inputsd));

};  // end namespace linear
};  // end namespace kf
