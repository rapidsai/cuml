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

#pragma once
#include <stdio.h>
#include <iostream>
#include "cuda_utils.h"
#include "kf_variables.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"
#include "utils.h"

namespace kf {
namespace linear {

using namespace MLCommon;
using namespace MLCommon::LinAlg;

// initialize this structure with all relevant pointers
// during first call, pass workspace as a nullptr to evaluate the workspace
// size needed in B, then during the second call, pass the rightfully
// allocated workspace buffer as input
template <typename T>
void set(Variables<T> &var, int _dim_x, int _dim_z, Option _solver, T *_x_est,
         T *_x_up, T *_Phi, T *_P_est, T *_P_up, T *_Q, T *_R, T *_H,
         void *workspace, size_t &workspaceSize,
         cusolverDnHandle_t handle_sol) {
  var.solver = _solver;
  var.dim_x = _dim_x;
  var.dim_z = _dim_z;
  CUSOLVER_CHECK(cusolverDngetrf_bufferSize(handle_sol, var.dim_z, var.dim_z,
                                            var.R_cpy, var.dim_z, &var.Lwork));
  workspaceSize = 0;
  const size_t granularity = 256;
  var.R_cpy = (T *)workspaceSize;
  workspaceSize += alignTo(var.dim_z * var.dim_z * sizeof(T), granularity);
  var.K = (T *)workspaceSize;
  workspaceSize += alignTo(var.dim_x * var.dim_z * sizeof(T), granularity);
  var.piv = (int *)workspaceSize;
  workspaceSize += alignTo(var.dim_z * sizeof(int), granularity);
  var.placeHolder1 = (T *)workspaceSize;
  workspaceSize += alignTo(var.dim_z * var.dim_x * sizeof(T), granularity);
  var.workspace_lu = (T *)workspaceSize;
  workspaceSize += alignTo(var.Lwork * sizeof(T), granularity);
  var.info = (int *)workspaceSize;
  workspaceSize += alignTo(sizeof(int), granularity);
  // only need when we need to calculate kalman gain
  if (var.solver < 2) {
    var.placeHolder2 = (T *)workspaceSize;
    workspaceSize += alignTo(var.dim_z * var.dim_z * sizeof(T), granularity);
    var.placeHolder0 = (T *)workspaceSize;
    workspaceSize += alignTo(var.dim_z * var.dim_z * sizeof(T), granularity);
  }

  if (workspace) {
    ASSERT(!var.initialized, "kf::linear::set: already initialized!");
    var.x_est = _x_est;
    var.x_up = _x_up;
    var.Phi = _Phi;
    var.P_est = _P_est;
    var.P_up = _P_up;
    var.Q = _Q;
    var.R = _R;
    var.H = _H;
    // initialize all the workspace pointers
    var.R_cpy = (T *)((size_t)var.R_cpy + (size_t)workspace);
    var.K = (T *)((size_t)var.K + (size_t)workspace);
    var.piv = (int *)((size_t)var.piv + (size_t)workspace);
    var.placeHolder1 = (T *)((size_t)var.placeHolder1 + (size_t)workspace);
    var.workspace_lu = (T *)((size_t)var.workspace_lu + (size_t)workspace);
    var.info = (int *)((size_t)var.info + (size_t)workspace);
    if (var.solver < 2) {
      var.placeHolder2 = (T *)((size_t)var.placeHolder2 + (size_t)workspace);
      var.placeHolder0 = (T *)((size_t)var.placeHolder0 + (size_t)workspace);
    }
    if (var.solver < ShortFormImplicit)
      make_ID_matrix(var.placeHolder0, var.dim_z);
    var.initialized = true;
  }
}

template <typename T>
void predict_x(Variables<T> &var, cublasHandle_t handle, cudaStream_t stream) {
  T alfa = (T)1.0, beta = (T)0.0;
  CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N, var.dim_x, var.dim_x, &alfa,
                          var.Phi, var.dim_x, var.x_up, 1, &beta, var.x_est, 1,
                          stream));
}

template <typename T>
void predict_P(Variables<T> &var, cublasHandle_t handle, cudaStream_t stream) {
  T alfa = (T)1.0, beta = (T)0.0;
  CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                          var.dim_x, var.dim_x, &alfa, var.Phi, var.dim_x,
                          var.P_up, var.dim_x, &beta, var.P_est, var.dim_x,
                          stream));
  beta = (T)1.0;
  CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                          var.dim_x, var.dim_x, &alfa, var.P_est, var.dim_x,
                          var.Phi, var.dim_x, &beta, var.Q, var.dim_x, stream));
  // This is for making the matrix symmetric
  alfa = beta = (T)0.5;
  CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                          var.dim_x, &alfa, var.Q, var.dim_x, &beta, var.Q,
                          var.dim_x, var.P_est, var.dim_x, stream));
}

template <typename T>
void update_x(Variables<T> &var, cublasHandle_t handle,
              cusolverDnHandle_t handle_sol, cudaStream_t stream) {
  T alfa = (T)-1.0, beta = (T)1.0;  // z - H * x is stored in z
  CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N, var.dim_z, var.dim_x, &alfa,
                          var.H, var.dim_z, var.x_est, 1, &beta, var.z, 1,
                          stream));
  if (var.solver < ShortFormImplicit) {  // explicit KG
    alfa = 1.0;
    beta = 1.0;
    CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N, var.dim_x, var.dim_z, &alfa,
                            var.K, var.dim_x, var.z, 1, &beta, var.x_est, 1,
                            stream));
  } else {  // implicit Kalman Gain
    // finding Y = [inv(B)*(z - H*x)] and placing the result in z
    CUSOLVER_CHECK(cusolverDngetrs(
      handle_sol, CUBLAS_OP_N, var.dim_z, 1, (const T *)var.R_cpy, var.dim_z,
      (const int *)var.piv, var.z, var.dim_z, var.info, stream));
    int info_h;
    updateHost(&info_h, var.info, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT(info_h == 0,
           "kf::linear: implicit kalman gain"
           " {Y = [inv(B)*(z - H*x)]}, info returned val=%d",
           info_h);
    // finding x_est + A * w and placing in x_est
    alfa = beta = (T)1.0;
    CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N, var.dim_x, var.dim_z, &alfa,
                            var.K, var.dim_x, var.z, 1, &beta, var.x_est, 1,
                            stream));
  }
  // DUE TO GPU POINTER THINGGY, NEED TO COPY DATA AT X_update instead of
  // just swapping the pointers.
  CUDA_CHECK(cudaMemcpy(var.x_up, var.x_est, var.dim_x * sizeof(T),
                        cudaMemcpyDeviceToDevice));
}

template <typename T>
void update_P(Variables<T> &var, cublasHandle_t handle,
              cusolverDnHandle_t handle_sol, cudaStream_t stream) {
  if (var.solver == LongForm) {
    T alfa = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x, var.dim_x, var.dim_z, &alfa,
      var.K, var.dim_x, var.H, var.dim_z, &beta, var.P_up, var.dim_x, stream));
    alfa = (T)-1.0;
    beta = (T)1.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, var.dim_x, &alfa, var.P_up, var.dim_x,
                            var.P_est, var.dim_x, &beta, var.P_est, var.dim_x,
                            stream));
    alfa = (T)1.0;
    beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_z,
                            var.dim_x, var.dim_z, &alfa, var.R, var.dim_z,
                            var.K, var.dim_x, &beta, var.placeHolder1,
                            var.dim_z, stream));
    alfa = (T)-1.0;
    beta = (T)1.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                            var.dim_x, var.dim_x, &alfa, var.P_est, var.dim_x,
                            var.P_up, var.dim_x, &beta, var.P_est, var.dim_x,
                            stream));
    alfa = (T)1.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, var.dim_z, &alfa, var.K, var.dim_x,
                            var.placeHolder1, var.dim_z, &beta, var.P_est,
                            var.dim_x, stream));
    // making the error cov symmetric
    alfa = beta = (T)0.5;
    CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                            var.dim_x, &alfa, var.P_est, var.dim_x, &beta,
                            var.P_est, var.dim_x, var.P_up, var.dim_x, stream));
  } else if (var.solver == ShortFormExplicit) {
    T alfa = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x, var.dim_x, var.dim_z, &alfa,
      var.K, var.dim_x, var.H, var.dim_z, &beta, var.P_up, var.dim_x, stream));
    alfa = (T)-1.0;
    beta = (T)1.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, var.dim_x, &alfa, var.P_up, var.dim_x,
                            var.P_est, var.dim_x, &beta, var.P_est, var.dim_x,
                            stream));
    alfa = beta = (T)0.5;
    CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, &alfa, var.P_est, var.dim_x, &beta,
                            var.P_est, var.dim_x, var.P_up, var.dim_x, stream));
  } else {
    CUDA_CHECK(cudaMemcpy(var.placeHolder1, var.H,
                          var.dim_z * var.dim_x * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    // finding [inv(B)*H] and placing the result in var.placeHolder1_d
    CUSOLVER_CHECK(cusolverDngetrs(handle_sol, CUBLAS_OP_N, var.dim_z,
                                   var.dim_x, (const T *)var.R_cpy, var.dim_z,
                                   (const int *)var.piv, var.placeHolder1,
                                   var.dim_z, var.info, stream));
    int info_h;
    updateHost(&info_h, var.info, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT(info_h == 0,
           "kf::linear: implicit kalman gain with short form, finding "
           "[inv(B)*H] info returned val=%d",
           info_h);
    T alfa = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, var.dim_z, &alfa, var.K, var.dim_x,
                            var.placeHolder1, var.dim_z, &beta, var.K,
                            var.dim_x, stream));
    alfa = (T)-1.0;
    beta = (T)1.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_x, var.dim_x, &alfa, var.K, var.dim_x,
                            var.P_est, var.dim_x, &beta, var.P_est, var.dim_x,
                            stream));
    alfa = beta = (T)0.5;
    CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                            var.dim_x, &alfa, var.P_est, var.dim_x, &beta,
                            var.P_est, var.dim_x, var.P_up, var.dim_x, stream));
  }
}

template <typename T>
void find_kalman_gain(Variables<T> &var, cublasHandle_t handle,
                      cusolverDnHandle_t handle_sol, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpy(var.R_cpy, var.R, var.dim_z * var.dim_z * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  T alfa = (T)1.0, beta = (T)0.0;
  CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, var.dim_x,
                          var.dim_z, var.dim_x, &alfa, var.P_est, var.dim_x,
                          var.H, var.dim_z, &beta, var.K, var.dim_x, stream));
  alfa = beta = (T)1.0;
  CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_z,
                          var.dim_z, var.dim_x, &alfa, var.H, var.dim_z, var.K,
                          var.dim_x, &beta, var.R_cpy, var.dim_z, stream));
  CUSOLVER_CHECK(cusolverDngetrf(handle_sol, var.dim_z, var.dim_z, var.R_cpy,
                                 var.dim_z, var.workspace_lu, var.piv, var.info,
                                 stream));
  int info_h;
  updateHost(&info_h, var.info, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "kf::linear: LU decomp, info returned val=%d", info_h);
  if (var.solver < ShortFormImplicit) {
    // copying ID matrix
    CUDA_CHECK(cudaMemcpy(var.placeHolder2, var.placeHolder0,
                          var.dim_z * var.dim_z * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDngetrs(handle_sol, CUBLAS_OP_N, var.dim_z,
                                   var.dim_z, (const T *)var.R_cpy, var.dim_z,
                                   (const int *)var.piv, var.placeHolder0,
                                   var.dim_z, var.info, stream));
    int info_h;
    updateHost(&info_h, var.info, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT(info_h == 0,
           "kf::linear: Explicit var.kalman gain, inverse "
           "returned val=%d",
           info_h);
    // var.R_cpy contains junk, R contains the real R value.
    alfa = (T)1.0;
    beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, var.dim_x,
                            var.dim_z, var.dim_z, &alfa, var.K, var.dim_x,
                            var.placeHolder0, var.dim_z, &beta, var.K,
                            var.dim_x, stream));
    T *temp = var.placeHolder0;
    var.placeHolder0 = var.placeHolder2;
    var.placeHolder2 = temp;
    // junk in var.R_cpy and var.placeHolder2. and var.K contains the Kalman gain.
  }
}

/**
 * @brief Initialization method for the opaque data structure
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @param _dim_x state vector dimension
 * @param _dim_z measurement vector dimension
 * @param _solver solver type
 * @param _x_est estimated state
 * @param _x_up updated state
 * @param _Phi state transition matrix
 * @param _P_est estimated error covariance
 * @param _P_up updated error covariance
 * @param _Q process noise covariance matrix
 * @param _R measurent noise covariance matrix
 * @param _H state to measurement tranformation matrix
 * @param workspace workspace buffer. Pass nullptr to compute its size
 * @param workspaceSize workspace buffer size in B.
 * @note this must always be called first before calling predict/update
 */
template <typename T>
void init(Variables<T> &var, int _dim_x, int _dim_z, Option _solver, T *_x_est,
          T *_x_up, T *_Phi, T *_P_est, T *_P_up, T *_Q, T *_R, T *_H,
          void *workspace, size_t &workspaceSize,
          cusolverDnHandle_t handle_sol) {
  set(var, _dim_x, _dim_z, _solver, _x_est, _x_up, _Phi, _P_est, _P_up, _Q, _R,
      _H, workspace, workspaceSize, handle_sol);
}

/**
 * @brief Predict the state for the next step, before the measurements are taken
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @note it is assumed that the 'init' function call has already been made with
 * a legal workspace buffer! Also, calling the 'predict' and 'update' functions
 * out-of-order will lead to unknown state!
 */
template <typename T>
void predict(Variables<T> &var, cublasHandle_t handle, cudaStream_t stream) {
  ASSERT(var.initialized, "kf::linear::predict: 'init' not called!");
  predict_x(var, handle, stream);
  predict_P(var, handle, stream);
}

/**
 * @brief Update the state in-lieu of measurements
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @param _z the measurement vector
 * @note it is assumed that the 'init' function call has already been made with
 * a legal workspace buffer! Also, calling the 'predict' and 'update' functions
 * out-of-order will lead to unknown state!
 */
template <typename T>
void update(Variables<T> &var, T *_z, cublasHandle_t handle,
            cusolverDnHandle_t handle_sol, cudaStream_t stream) {
  ASSERT(var.initialized, "kf::linear::update: 'init' not called!");
  var.z = _z;
  find_kalman_gain(var, handle, handle_sol, stream);
  update_x(var, handle, handle_sol, stream);
  update_P(var, handle, handle_sol, stream);
}

};  // end namespace linear
};  // end namespace kf
