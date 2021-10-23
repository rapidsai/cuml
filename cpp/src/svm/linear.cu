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

template <typename T, int BX = 32, int BY = 8>
__global__ void predictClass(
  T* out, const T* z, const T* classes, const int nRows, const int coefCols)
{
  const int i = threadIdx.y + blockIdx.y * BY;
  if (i >= nRows) return;
  const T* row = z + i * coefCols;
  T maxval     = std::numeric_limits<T>::lowest();
  int maxj     = 0;
  for (int j = threadIdx.x; j < coefCols; j += BX) {
    T t = row[j];
    if (t > maxval) {
      maxj   = j;
      maxval = t;
    }
  }
  if (coefCols == 1 && threadIdx.x == 0) { out[i] = classes[maxval > 0]; }
  if constexpr (BX > 1) {
    typedef cub::WarpReduce<cub::KeyValuePair<int, T>, BX> WarpRed;
    __shared__ typename WarpRed::TempStorage warpStore[BY];
    auto maxkv =
      WarpRed(warpStore[threadIdx.y]).Reduce(cub::KeyValuePair(maxj, maxval), cub::ArgMax());
    if (threadIdx.x == 0) out[i] = classes[maxkv.key];
  }
}

template <typename T, int BlockSize = 256, int BX = 32>
struct PredictClass {
  static inline void run(
    T* out, const T* z, const T* classes, const int nRows, const int coefCols, cudaStream_t stream)
  {
    if constexpr (BX > 1) {
      if (coefCols <= (BX >> 1))
        return PredictClass<T, BlockSize, std::max<int>(BX >> 1, 1)>::run(
          out, z, classes, nRows, coefCols, stream);
    }
    const int BY = BlockSize / BX;
    const dim3 bs(BX, BY, 1);
    const dim3 gs(1, raft::ceildiv(nRows, BY), 1);
    printf("predictClass<T, %d, %d><<(%d, %d, %d), (%d, %d, %d)>>>()\n",
           BX,
           BY,
           gs.x,
           gs.y,
           gs.z,
           bs.x,
           bs.y,
           bs.z);
    predictClass<T, BX, BY><<<gs, bs, 0, stream>>>(out, z, classes, nRows, coefCols);
  }
};

template <typename T, bool Log, bool Binary, int BX = 32, int BY = 8>
__global__ void predictProba(T* out, const T* z, const int nRows, const int nClasses)
{
  typedef cub::WarpReduce<T, BX> WarpRed;
  __shared__ typename WarpRed::TempStorage shm[BY];
  typename WarpRed::TempStorage& warpStore = shm[threadIdx.y];

  const int i = threadIdx.y + blockIdx.y * BY;
  if (i >= nRows) return;
  const T* rowIn = z + i * (Binary ? 1 : nClasses);
  T* rowOut      = out + i * nClasses;

  // the largest 'z' in the row (for substract it from z for numeric stability).
  T t      = std::numeric_limits<T>::lowest();
  T maxVal = t;
  int j    = threadIdx.x;
  if constexpr (Binary) {
    t      = rowIn[0];
    maxVal = raft::myMax<T>(t, 0);
    t      = T(j) * t;  // set z[0] = 0, z[1] = t
  } else {
    for (; j < nClasses; j += BX) {
      t      = rowIn[j];
      maxVal = raft::myMax<T>(maxVal, t);
    }
    j -= BX;
    maxVal = WarpRed(warpStore).Reduce(maxVal, cub::Max());
    maxVal = cub::ShuffleIndex<BX>(maxVal, 0, 0xFFFFFFFFU);
  }
  // At this point, either `j` refers to the last valid column idx worked
  // by the current thread, or `j` is negative.
  // We traverse the columns array in the opposite direction in the next
  // block. This allows us to avoid extra global memory accesses when
  // BX >= nClasses, which is a very common case.

  T et;         // Numerator of the softmax.
  T smSum = 0;  // Denominator of the softmax.
  while (j >= 0) {
    et = raft::myExp<T>(t - maxVal);
    smSum += et;
    if (j < BX) break;
    j -= BX;
    t = rowIn[j];
  }
  smSum = WarpRed(warpStore).Reduce(smSum, cub::Sum());
  smSum = cub::ShuffleIndex<BX>(smSum, 0, 0xFFFFFFFFU);

  // Now, either `j` refers to the first valid column idx worked by the
  // current thread, or `j` is negative (no work at all).
  // Traverse in the forward direction again to save the results.
  // Note, no extra memory reads when BX >= nClasses!
  if (j < 0) return;
  T d = Log ? -maxVal - raft::myLog<T>(smSum) : 1 / smSum;
  while (j < nClasses) {
    rowOut[j] = Log ? t + d : et * d;
    j += BX;
    if (j >= nClasses) break;
    t = rowIn[j];
    if constexpr (Log) et = raft::myExp<T>(t - maxVal);
  }
}

template <typename T, int BlockSize = 256, int BX = 32>
struct PredictProba {
  static inline void run(
    T* out, const T* z, const int nRows, const int nClasses, const bool log, cudaStream_t stream)
  {
    if constexpr (BX > 2) {
      if (nClasses <= (BX >> 1))
        return PredictProba<T, BlockSize, std::max<int>(BX >> 1, 2)>::run(
          out, z, nRows, nClasses, log, stream);
    }
    const int BY      = BlockSize / BX;
    const bool Binary = BX == 2;
    const dim3 bs(BX, BY, 1);
    const dim3 gs(1, raft::ceildiv(nRows, BY), 1);
    printf("PredictProba<T, %d, %d><<(%d, %d, %d), (%d, %d, %d)>>>(binary = %d, log = %d)\n",
           BX,
           BY,
           gs.x,
           gs.y,
           gs.z,
           bs.x,
           bs.y,
           bs.z,
           Binary,
           log);
    if constexpr (Binary)
      ASSERT((void*)out != (void*)z, "PredictProba for the binary case cannot be inplace.");
    if (log)
      predictProba<T, true, Binary, BX, BY><<<gs, bs, 0, stream>>>(out, z, nRows, nClasses);
    else
      predictProba<T, false, Binary, BX, BY><<<gs, bs, 0, stream>>>(out, z, nRows, nClasses);
  }
};

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

// template <typename T>
// struct SignFun {
//   const T H1_value;
//   __device__ T operator()(const T x) const { return x == H1_value ? 1 : -1; }
// };

template <typename T>
struct OvrSelector {
  const T* classes;
  const int selected;
  __device__ T operator()(const T x) const { return x == classes[selected] ? 1 : -1; }
};

// template <typename T>
// struct IndicatorFun {
//   const T H1_value;
//   __device__ T operator()(const T x) const { return T(x == H1_value); }
// };

template <typename T>
void predict_linear(const raft::handle_t& handle,
                    const T* X,
                    const T* w,
                    const int nRows,
                    const int nCols,
                    const int coefCols,
                    const bool fitIntercept,
                    T* out,
                    cudaStream_t stream)
{
  raft::linalg::gemm<T>(
    handle, out, (T*)X, (T*)w, nRows, coefCols, nCols, false, true, false, stream);

  if (fitIntercept)
    raft::linalg::matrixVectorOp(
      out, out, w + nCols * coefCols, nRows, coefCols, false, false, cub::Sum(), stream);
}

};  // namespace

template <typename T>
LinearSVMModel<T>::LinearSVMModel(const raft::handle_t& handle, const LinearSVMParams params)
  : handle(handle),
    params(params),
    classes(0, handle.get_stream()),
    w(0, handle.get_stream()),
    probScale(0, handle.get_stream())
{
}

template <typename T>
LinearSVMModel<T>::LinearSVMModel(const raft::handle_t& handle,
                                  const LinearSVMParams params,
                                  const T* X,
                                  const int nRows,
                                  const int nCols,
                                  const T* y,
                                  const T* sampleWeight)
  : LinearSVMModel<T>::LinearSVMModel(handle, params)
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
  if (!params.probability) return;

  ML::PUSH_RANGE("Trace::LinearSVMModel::fit-probabilities");
  probScale.resize(coefCols * (coefCols + 1), stream);

  rmm::device_uvector<T> xwBuf(nRows * coefCols, stream);
  T* xw = xwBuf.data();
  predict_linear(handle, X, w.data(), nRows, nCols, coefCols, params.fit_intercept, xw, stream);

  // here we should encode the labels into corresponding `classes` indices.
  // for now, we assume labels are the values from 0 to classes.size() - 1.
  // raft::linalg::unaryOp(y1, y, nRows, IndicatorFun<T>{T(params.H1_value)}, stream);

  GLM::qnFit<T>(handle,
                xw,
                false,
                (T*)y,
                nRows,
                coefCols,
                nClasses,
                true,
                0,
                0,
                params.max_iter,
                T(params.grad_tol),
                T(params.change_tol),
                params.linesearch_max_iter,
                params.lbfgs_memory,
                params.verbose,
                probScale.data(),
                &target,
                &num_iters,
                nClasses == 2 ? 0 /* logistic loss*/ : 2 /** softmax */,
                stream,
                (T*)sampleWeight);

  CUML_LOG_DEBUG("LinearSVM finished fitting probabilities in %d iterations out of maximum %d.",
                 num_iters,
                 params.max_iter);

  ML::POP_RANGE();
}

template <typename T>
void LinearSVMModel<T>::predict(const T* X, const int nRows, const int nCols, T* out) const
{
  cudaStream_t stream = handle.get_stream();
  const int nClasses  = classes.size();
  const int coefCols  = nClasses <= 2 ? 1 : nClasses;
  if (isRegression(params.loss))
    return predict_linear(
      handle, X, w.data(), nRows, nCols, coefCols, params.fit_intercept, out, stream);

  rmm::device_uvector<T> temp(nRows * coefCols, stream);
  predict_linear(
    handle, X, w.data(), nRows, nCols, coefCols, params.fit_intercept, temp.data(), stream);
  PredictClass<T>::run(out, temp.data(), classes.data(), nRows, coefCols, stream);
}

template <typename T>
void LinearSVMModel<T>::predict_proba(
  const T* X, const int nRows, const int nCols, const bool log, T* out) const
{
  ASSERT(!isRegression(params.loss),
         "Predicting probabilities is not available for the regression model");
  ASSERT(
    params.probability,
    "The model was not trained to output probabilities (LinearSVMParams.probability == false).");

  cudaStream_t stream = handle.get_stream();
  const int nClasses  = classes.size();
  const int coefCols  = nClasses <= 2 ? 1 : nClasses;
  rmm::device_uvector<T> temp(nRows * coefCols, stream);
  predict_linear(
    handle, X, w.data(), nRows, nCols, coefCols, params.fit_intercept, temp.data(), stream);

  PredictProba<T>::run(out, temp.data(), nRows, classes.size(), log, handle.get_stream());
  // cudaStream_t stream = handle.get_stream();
}

template class LinearSVMModel<float>;
template class LinearSVMModel<double>;

}  // namespace SVM
}  // namespace ML
