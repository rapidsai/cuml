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
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
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

template <typename T>
__global__ void predictClass(
  T* out, const T* z, const T* classes, const int nRows, const int coefCols)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRows) return;
  T maxval = -std::numeric_limits<T>::infinity();
  int maxj = 0;
  for (int j = 0; j < coefCols; j++) {
    T t = z[i + j * nRows];
    if (t > maxval) {
      maxj   = j;
      maxval = t;
    }
  }
  out[i] = classes[coefCols == 1 ? maxval > 0 : maxj];
}

template <typename T>
__global__ void rowMajorGetCol(T* out, const T* in, const int i, const int nRows, const int nCols)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nRows) return;
  out[j] = in[i + j * nCols];
}

template <typename T>
__global__ void rowMajorSetCol(T* out, const T* in, const int i, const int nRows, const int nCols)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nRows) return;
  out[i + j * nCols] = in[j];
}

template <typename T, int BlockSize>
__global__ void initWeights(
  T* out, const T* in, const int nRows, const int nCols, const int coefCols)
{
  typedef cub::BlockReduce<T, BlockSize> BlockSum;
  __shared__ typename BlockSum::TempStorage shm;
  const int i = blockIdx.x;
  T t         = 0;
  T s         = 0;
  if (i < nCols) {
    for (int j = threadIdx.x; j < nRows; j += BlockSize)
      t += in[i * nRows + j];
    s = BlockSum(shm).Sum(t);
  }
  if (coefCols == 1) {
    if (threadIdx.x == 0) out[i] = s / T(nRows);
  } else {
    __shared__ T r;
    if (threadIdx.x == 0) r = s / T(nRows);
    __syncthreads();
    t = r;
    for (int j = threadIdx.x; j < coefCols; j += BlockSize)
      out[j + i * coefCols] = t;
  }
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

template <typename T>
struct OvrSelector {
  const T* classes;
  const int selected;
  __device__ T operator()(const T x) const { return x == classes[selected] ? 1 : -1; }
};

template <typename T>
struct IndicatorFun {
  const T H1_value;
  __device__ T operator()(const T x) const { return T(x == H1_value); }
};

template <typename T>
void predict_linear(const raft::handle_t& handle,
                    const T* X,
                    const T* w,
                    const int nRows,
                    const int nCols,
                    T* out,
                    cudaStream_t stream)
{
  raft::linalg::gemv(handle, X, nRows, nCols, w, out, false, stream);
  const T* p = w + nCols;
  raft::linalg::unaryOp(
    out, out, nRows, [p] __device__(T x) -> T { return x + *p; }, stream);
}

template <typename T>
void predict_indicator(const raft::handle_t& handle,
                       const T* X,
                       const T* w,
                       const int nRows,
                       const int nCols,
                       T* out,
                       cudaStream_t stream)
{
  raft::linalg::gemv(handle, X, nRows, nCols, w, out, false, stream);
  const T* p = w + nCols;
  raft::linalg::unaryOp(
    out, out, nRows, [p] __device__(T x) -> T { return T((x + *p) > 0); }, stream);
}

template <typename T>
void predict_prob(const raft::handle_t& handle,
                  const T* X,
                  const T* w,
                  const T* probScale,
                  const int nRows,
                  const int nCols,
                  T* out,
                  cudaStream_t stream)
{
  raft::linalg::gemv(handle, X, nRows, nCols, w, out, false, stream);
  const T* p = w + nCols;
  raft::linalg::unaryOp(
    out,
    out,
    nRows,
    [p, probScale] __device__(T x) -> T {
      T z = probScale[0] * (x + *p) + probScale[1];
      T t = raft::myExp(z < 0 ? z : -z);
      T q = 1 / (1 + t);
      return q * (z < 0 ? t : T(1.0));
    },
    stream);
}

template <typename T>
void predict_log_prob(const raft::handle_t& handle,
                      const T* X,
                      const T* w,
                      const T* probScale,
                      const int nRows,
                      const int nCols,
                      T* out,
                      cudaStream_t stream)
{
  raft::linalg::gemv(handle, X, nRows, nCols, w, out, false, stream);
  const T* p = w + nCols;
  raft::linalg::unaryOp(
    out,
    out,
    nRows,
    [p, probScale] __device__(T x) -> T {
      T z = probScale[0] * (x + *p) + probScale[1];
      T t = -raft::myLog(1 + raft::myExp(z < 0 ? z : -z));
      return t + (z < 0 ? z : T(0));
    },
    stream);
}

};  // namespace

template <typename T>
LinearSVMModel<T>::LinearSVMModel(const raft::handle_t& handle,
                                  const LinearSVMParams params,
                                  const T* X,
                                  const int nRows,
                                  const int nCols,
                                  const T* y,
                                  const T* sampleWeight)
  : params(params),
    handle(handle),
    nRows(nRows),
    nCols(nCols),
    classes(0, handle.get_stream()),
    probScale(0, handle.get_stream()),
    w(0, handle.get_stream())
{
  cudaStream_t stream = handle.get_stream();
  const int nClasses =
    isRegression(params.loss) ? 1 : raft::label::getUniquelabels(classes, (T*)y, nRows, stream);
  ASSERT(isRegression(params.loss) || nClasses > 1,
         "Found only one unique value in the target data, whereas at least two are required "
         "(one-class classification does not make sense)");
  // from now on, nClasses == 1 implies we solve the regression problem.
  const int coefCols = nClasses <= 2 ? 1 : nClasses;
  const int coefRows = nCols + params.fit_intercept;

  ML::PUSH_RANGE("Trace::LinearSVMModel::fit");
  w.resize(coefCols * coefRows, stream);
  initWeights<T, 256>
    <<<dim3(coefRows, 1, 1), dim3(256, 1, 1), 0, stream>>>(w.data(), X, nRows, nCols, coefCols);

  auto nCols1   = nCols + int(params.fit_intercept && params.penalized_intercept);
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

  int qn_loss = 99;
  switch (params.loss) {
    case LinearSVMParams::HINGE: qn_loss = 3; break;
    case LinearSVMParams::SQUARED_HINGE: qn_loss = 4; break;
    case LinearSVMParams::EPSILON_INSENSITIVE: qn_loss = 5; break;
    case LinearSVMParams::SQUARED_EPSILON_INSENSITIVE: qn_loss = 6; break;
    default: break;
  }

  T* y1 = (T*)y;
  T* w1 = w.data();
  rmm::device_uvector<T> y1Buf(0, stream);
  rmm::device_uvector<T> w1Buf(0, stream);
  if (nClasses > 1) {
    y1Buf.resize(nRows, stream);
    y1 = y1Buf.data();
  }
  if (coefCols > 1) {
    w1Buf.resize(coefRows, stream);
    w1 = w1Buf.data();
  }

  // one-vs-rest logic goes over each class
  for (int class_i = 0; class_i < coefCols; class_i++) {
    if (nClasses > 1) {
      raft::linalg::unaryOp(
        y1, y, nRows, OvrSelector<T>{classes.data(), nClasses == 2 ? 1 : class_i}, stream);
    }
    if (coefCols > 1)
      rowMajorGetCol<T><<<dim3(raft::ceildiv(coefRows, 256), 1, 1), dim3(256, 1, 1), 0, stream>>>(
        w1, w.data(), class_i, coefRows, coefCols);

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
                  w1,
                  &target,
                  &num_iters,
                  qn_loss,
                  stream,
                  (T*)sampleWeight,
                  T(params.svr_sensitivity));

    if (coefCols > 1)
      rowMajorSetCol<T><<<dim3(raft::ceildiv(coefRows, 256), 1, 1), dim3(256, 1, 1), 0, stream>>>(
        w.data(), w1, class_i, coefRows, coefCols);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUML_LOG_DEBUG(
      "LinearSVM finished fitting in %d iterations out of maximum %d.", num_iters, params.max_iter);
  }

  ML::POP_RANGE();
  // if (!params.probability) return;
  // ML::PUSH_RANGE("Trace::LinearSVMModel::fit-probabilities");
  // probScale.resize(coefRows + 1, stream);

  // rmm::device_uvector<T> xwBuf(nRows, stream);
  // T* xw = xwBuf.data();
  // predict_linear(handle, X, w.data(), nRows, nCols, xw, stream);
  // raft::linalg::unaryOp(y1, y, nRows, IndicatorFun<T>{T(params.H1_value)}, stream);

  // GLM::qnFit<T>(handle,
  //               xw,
  //               true,
  //               y1,
  //               nRows,
  //               1 /* D = 1 for only one parameter besides bias */,
  //               2 /* C = 2 classes forced by LogisticLoss */,
  //               true /* bias is the second parameter to fit */,
  //               0,
  //               0,
  //               params.max_iter,
  //               T(params.grad_tol),
  //               T(params.change_tol),
  //               params.linesearch_max_iter,
  //               params.lbfgs_memory,
  //               params.verbose,
  //               probScale.data(),
  //               &target,
  //               &num_iters,
  //               0 /* logistic loss*/,
  //               stream,
  //               (T*)sampleWeight);
  // CUML_LOG_DEBUG("LinearSVM finished fitting probabilities in %d iterations out of maximum %d.",
  //                num_iters,
  //                params.max_iter);

  // ML::POP_RANGE();
}

template <typename T>
void LinearSVMModel<T>::predict(const T* X, const int nRows, const int nCols, T* out) const
{
  ASSERT(nCols == this->nCols,
         "Number of features passed to predict() must be the same as for fitting (%d != %d).",
         nCols,
         this->nCols);
  cudaStream_t stream = handle.get_stream();
  if (isRegression(params.loss)) {
    predict_linear(handle, X, w.data(), nRows, nCols, out, stream);
  } else {
    const int nClasses = classes.size();
    // const int coefRows = nCols + params.fit_intercept;
    const int coefCols = nClasses <= 2 ? 1 : nClasses;

    rmm::device_uvector<T> temp(nRows * coefCols, stream);
    raft::linalg::gemm<T>(
      handle, temp.data(), (T*)X, (T*)w.data(), nRows, coefCols, nCols, true, true, false, stream);

    if (params.fit_intercept)
      raft::linalg::matrixVectorOp(
        temp.data(),
        temp.data(),
        w.data() + nCols * coefCols,
        nRows,
        coefCols,
        true,
        false,
        [] __device__(T a, T b) { return a + b; },
        stream);

    predictClass<T><<<dim3(raft::ceildiv(nRows, 256), 1, 1), dim3(256, 1, 1), 0, stream>>>(
      out, temp.data(), classes.data(), nRows, coefCols);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
void LinearSVMModel<T>::predict_proba(
  const T* X, const int nRows, const int nCols, const bool log, T* out) const
{
  ASSERT(nCols == this->nCols,
         "Number of features passed to predict() must be the same as for fitting (%d != %d).",
         nCols,
         this->nCols);
  ASSERT(!isRegression(params.loss),
         "Predicting probabilities is not available for the regression model");
  ASSERT(
    params.probability,
    "The model was not trained to output probabilities (LinearSVMParams.probability == false).");

  cudaStream_t stream = handle.get_stream();
  if (log) {
    predict_log_prob(handle, X, w.data(), probScale.data(), nRows, nCols, out, stream);
  } else {
    predict_prob(handle, X, w.data(), probScale.data(), nRows, nCols, out, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template class LinearSVMModel<float>;
template class LinearSVMModel<double>;

}  // namespace SVM
}  // namespace ML
