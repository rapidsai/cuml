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

#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"
#include "lkf.h"
#include "lkf_py.h"

namespace kf {
namespace linear {

size_t get_workspace_size_f32(Variables<float> &var, int dim_x, int dim_z,
                              Option solver, float *x_est, float *x_up,
                              float *Phi, float *P_est, float *P_up, float *Q,
                              float *R, float *H) {
  size_t workspaceSize;
  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  init<float>(var, dim_x, dim_z, solver, x_est, x_up, Phi, P_est, P_up, Q, R, H,
              (void *)nullptr, workspaceSize, cusolver_handle);
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  return workspaceSize;
}

void init_f32(Variables<float> &var, int dim_x, int dim_z, Option solver,
              float *x_est, float *x_up, float *Phi, float *P_est, float *P_up,
              float *Q, float *R, float *H, void *workspace,
              size_t &workspaceSize) {
  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  // CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
  init(var, dim_x, dim_z, solver, x_est, x_up, Phi, P_est, P_up, Q, R, H,
       workspace, workspaceSize, cusolver_handle);
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}
void predict_f32(Variables<float> &var) {
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  predict(var, cublas_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}
void update_f32(Variables<float> &var, float *_z) {
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  update(var, _z, cublas_handle, cusolver_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// Double precision functions

size_t get_workspace_size_f64(Variables<double> &var, int dim_x, int dim_z,
                              Option solver, double *x_est, double *x_up,
                              double *Phi, double *P_est, double *P_up,
                              double *Q, double *R, double *H) {
  size_t workspaceSize;
  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  init<double>(var, dim_x, dim_z, solver, x_est, x_up, Phi, P_est, P_up, Q, R,
               H, (void *)nullptr, workspaceSize, cusolver_handle);
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  return workspaceSize;
}

void init_f64(Variables<double> &var, int dim_x, int dim_z, Option solver,
              double *x_est, double *x_up, double *Phi, double *P_est,
              double *P_up, double *Q, double *R, double *H, void *workspace,
              size_t &workspaceSize) {
  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  // CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
  init(var, dim_x, dim_z, solver, x_est, x_up, Phi, P_est, P_up, Q, R, H,
       workspace, workspaceSize, cusolver_handle);
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}
void predict_f64(Variables<double> &var) {
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  predict(var, cublas_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}
void update_f64(Variables<double> &var, double *_z) {
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  update(var, _z, cublas_handle, cusolver_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

};  // end namespace linear
};  // end namespace kf
