/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/**
 * @file linear_svm.cuh
 * @brief Fit linear SVM.
 */

#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/gemv.h>
#include <raft/linalg/transpose.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <common/nvtx.hpp>
#include <label/classlabels.cuh>
#include <matrix/kernelfactory.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/matrix.cuh>
#include <rmm/device_uvector.hpp>

#include <glm/ols.cuh>
#include <glm/qn/qn.cuh>

#include <cuml/svm/linear.hpp>

namespace ML {
namespace SVM {

namespace {

template <typename T>
__global__ void transpose(
  T* out, const T* in, const T* mul, const int nRows, const int nCols, const bool withBias)
{
  int nCols1 = withBias ? nCols + 1 : nCols;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCols1; i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < nRows; j += blockDim.y * gridDim.y) {
      out[i + j * nCols1] = mul[j] * (i == nCols ? T(1.0) : in[i * nRows + j]);
    }
  }
}

template <typename T, int BlockSize>
__global__ void mean(T* out, const T* in, const int nRows, const int nCols)
{
  typedef cub::BlockReduce<T, BlockSize> BlockSum;
  __shared__ typename BlockSum::TempStorage shm;
  int i = blockIdx.x;
  T t   = 0;
  T s   = 0;
  if (i < nCols) {
    for (int j = threadIdx.x; j < nRows; j += blockDim.x)
      t += in[i * nRows + j];
    s = BlockSum(shm).Sum(t);
  }
  if (threadIdx.x == 0) out[i] = s / T(nRows);
}

inline bool isRegression(LinearSVMParams::Loss loss)
{
  return loss == LinearSVMParams::EPSILON_INSENSITIVE ||
         loss == LinearSVMParams::SQUARED_EPSILON_INSENSITIVE;
}

template <typename T>
struct SignFun {
  const T H1_value;
  __device__ T operator()(const T x) const { return x == H1_value ? 1 : -1; }
};

};  // namespace

template <typename T>
LinearSVMModel<T>::LinearSVMModel(const raft::handle_t& handle,
                                  const LinearSVMParams params,
                                  const T* X,
                                  const int nRows,
                                  const int nCols,
                                  const T* y,
                                  const T* sampleWeight)
  : params(params), handle(handle), nRows(nRows), nCols(nCols), w(nCols + 1, handle.get_stream())
{
  ML::PUSH_RANGE("Trace::LinearSVMModel::fit");
  cudaStream_t stream = handle.get_stream();
  mean<T, 256><<<dim3(w.size(), 1, 1), dim3(256, 1, 1), 0, stream>>>(w.data(), X, nRows, nCols);

  auto nCols1   = (params.fit_intercept && params.penalized_intercept) ? nCols + 1 : nCols;
  int num_iters = 0;
  T target;
  T iC = params.C > 0 ? (1.0 / params.C) : 1.0;

  T* X1 = (T*)X;
  rmm::device_uvector<T> X1Buf(0, stream);
  if (params.fit_intercept && params.penalized_intercept) {
    X1Buf.resize(nCols1 * nRows, stream);
    X1 = X1Buf.data();
    CUDA_CHECK(cudaMemcpyAsync(X1, X, sizeof(T) * nCols * nRows, cudaMemcpyDeviceToDevice, stream));
    thrust::device_ptr<T> p(X1 + nCols * nRows);
    thrust::fill(thrust::cuda::par.on(stream), p, p + nRows, 1.0);
  }

  T* y1 = (T*)y;
  rmm::device_uvector<T> y1Buf(0, stream);
  if (!isRegression(params.loss)) {
    y1Buf.resize(nRows, stream);
    y1 = y1Buf.data();
    raft::linalg::unaryOp(y1, y, nRows, SignFun<T>{T(params.H1_value)}, stream);
  }

  int qn_loss = 99;
  switch (params.loss) {
    case LinearSVMParams::HINGE: qn_loss = 3; break;
    case LinearSVMParams::SQUARED_HINGE: qn_loss = 4; break;
    case LinearSVMParams::EPSILON_INSENSITIVE: qn_loss = 5; break;
    case LinearSVMParams::SQUARED_EPSILON_INSENSITIVE: qn_loss = 6; break;
    default: break;
  }
  GLM::qnFit<T>(handle,
                X1,
                true,
                y1,
                nRows,
                nCols1,
                1,
                params.fit_intercept && !params.penalized_intercept,
                T(params.penalty == LinearSVMParams::L1 ? iC : 0.0),
                T(params.penalty == LinearSVMParams::L2 ? iC : 0.0),
                params.max_iter,
                T(params.grad_tol),
                T(params.change_tol),
                params.linesearch_max_iter,
                params.lbfgs_memory,
                params.verbose,
                w.data(),
                &target,
                &num_iters,
                qn_loss,
                stream,
                (T*)sampleWeight,
                T(params.svr_sensitivity));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG(
    "LinearSVM finished fitting in %d iterations out of maximum %d.", num_iters, params.max_iter);

  ML::POP_RANGE();
}

template <typename T>
void LinearSVMModel<T>::predict(const T* X, const int nRows, const int nCols, T* out) const
{
  ASSERT(nCols == this->nCols,
         "Number of features passed to predict() must be the same as for fitting (%d != %d).",
         nCols,
         this->nCols);
  cudaStream_t stream = handle.get_stream();
  raft::linalg::gemv(handle, X, nRows, nCols, w.data(), out, false, stream);
  const T* p = w.data() + nCols;
  if (isRegression(params.loss)) {
    raft::linalg::unaryOp(
      out, out, nRows, [p] __device__(T x) -> T { return x + *p; }, stream);
  } else {
    raft::linalg::unaryOp(
      out, out, nRows, [p] __device__(T x) -> T { return T((x + *p) > 0); }, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template class LinearSVMModel<float>;
template class LinearSVMModel<double>;

}  // namespace SVM
}  // namespace ML
