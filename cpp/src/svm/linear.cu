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
#include <thread>

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

/** The cuda kernel for classification. Call it via PredictClass::run(..).
 *
 * @param out - [out] vector of classes (nRows,)
 * @param z   - [in] row-major matrix of scores (nRows, coefCols)
 * @param coefCols - nClasses > 2 ? nClasses : 1
 * */
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

/**
 * The wrapper struct on top of predictClass that recursively selects the best BX
 * (largest BX satisfying `BX < coefCols*2`) and then schedules the kernel launch.
 */
template <typename T, int BlockSize = 256, int BX = 32>
struct PredictClass {
  static_assert(BX <= 32, "BX must be not larger than warpSize");
  static_assert(BX <= BlockSize, "BX must be not larger than BlockSize");
  /**
   * Predict classes using the scores.
   *
   * @param out - [out] vector of classes (nRows,)
   * @param z   - [in] row-major matrix of scores (nRows, coefCols)
   * @param coefCols - nClasses > 2 ? nClasses : 1
   * */
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
    predictClass<T, BX, BY><<<gs, bs, 0, stream>>>(out, z, classes, nRows, coefCols);
  }
};

/**
 * The cuda kernel for classification. Call it via PredictProba::run(..).
 * @param out - [out] row-major matrix of probabilities (nRows, nClasses)
 * @param z   - [in] row-major matrix of scores (nRows, Binary ? 1 : nClasses)
 */
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

/**
 * The wrapper struct on top of predictProba that recursively selects the best BX
 * (largest BX satisfying `BX < coefCols*2`) and then schedules the kernel launch.
 */
template <typename T, int BlockSize = 256, int BX = 32>
struct PredictProba {
  static_assert(BX <= 32, "BX must be not larger than warpSize");
  static_assert(BX <= BlockSize, "BX must be not larger than BlockSize");
  /**
   * Predict probabilities using the scores.
   *
   * @param out - [out] row-major matrix of probabilities (nRows, nClasses)
   * @param z   - [in] row-major matrix of scores (nRows, Binary ? 1 : nClasses)
   */
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
    if constexpr (Binary)
      ASSERT((void*)out != (void*)z, "PredictProba for the binary case cannot be inplace.");
    if (log)
      predictProba<T, true, Binary, BX, BY><<<gs, bs, 0, stream>>>(out, z, nRows, nClasses);
    else
      predictProba<T, false, Binary, BX, BY><<<gs, bs, 0, stream>>>(out, z, nRows, nClasses);
  }
};

inline bool isRegression(LinearSVMParams::Loss loss)
{
  return loss == LinearSVMParams::EPSILON_INSENSITIVE ||
         loss == LinearSVMParams::SQUARED_EPSILON_INSENSITIVE;
}

/** A functor that maps the multiclass problem onto the one-vs-rest binary problem */
template <typename T>
struct OvrSelector {
  const T* classes;
  const int selected;
  __device__ T operator()(const T x) const { return x == classes[selected] ? 1 : -1; }
};

/** The linear part of the prediction. */
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
  CUDA_CHECK(cudaMemsetAsync(w.data(), 0, w.size() * sizeof(T), stream));

  auto nCols1 = nCols + int(params.fit_intercept && params.penalized_intercept);
  T iC        = params.C > 0 ? (1.0 / params.C) : 1.0;

  T* X1 = (T*)X;
  rmm::device_uvector<T> X1Buf(0, stream);
  if (params.fit_intercept && params.penalized_intercept) {
    X1Buf.resize(nCols1 * nRows, stream);
    X1 = X1Buf.data();
    raft::copy(X1, X, nCols * nRows, stream);
    thrust::device_ptr<T> p(X1 + nCols * nRows);
    thrust::fill(thrust::cuda::par.on(stream), p, p + nRows, 1.0);
  }

  auto qn_loss = (ML::GLM::QN_LOSS_TYPE)99;
  switch (params.loss) {
    case LinearSVMParams::HINGE: qn_loss = ML::GLM::QN_LOSS_SVC_L1; break;
    case LinearSVMParams::SQUARED_HINGE: qn_loss = ML::GLM::QN_LOSS_SVC_L2; break;
    case LinearSVMParams::EPSILON_INSENSITIVE: qn_loss = ML::GLM::QN_LOSS_SVR_L1; break;
    case LinearSVMParams::SQUARED_EPSILON_INSENSITIVE: qn_loss = ML::GLM::QN_LOSS_SVR_L2; break;
    default: break;
  }

  T* y1 = (T*)y;
  T* w1 = w.data();
  rmm::device_uvector<T> y1Buf(0, stream);
  rmm::device_uvector<T> w1Buf(0, stream);
  if (nClasses > 1) {
    y1Buf.resize(nRows * coefCols, stream);
    y1 = y1Buf.data();
  }
  if (coefCols > 1) {
    w1Buf.resize(w.size(), stream);
    w1 = w1Buf.data();
  }

  // one-vs-rest logic goes over each class
  std::vector<T> targets(coefCols);
  std::vector<int> num_iters(coefCols);
  int n_streams    = handle.get_num_internal_streams();
  bool parallel    = n_streams > 1 && coefCols > 1;
  T* classes1      = classes.data();
  auto solveBinary = [&handle,
                      y1,
                      w1,
                      parallel,
                      stream,
                      nClasses,
                      nRows,
                      coefRows,
                      params,
                      X1,
                      y,
                      n_streams,
                      classes1,
                      nCols1,
                      iC,
                      qn_loss,
                      sampleWeight](int class_i) {
    T* yi  = y1 + nRows * class_i;
    T* wi  = w1 + coefRows * class_i;
    auto s = parallel ? handle.get_internal_stream(class_i % n_streams) : stream;
    if (nClasses > 1) {
      raft::linalg::unaryOp(yi, y, nRows, OvrSelector<T>{classes1, nClasses == 2 ? 1 : class_i}, s);
    }
    T target;
    int num_iters;
    GLM::qnFit<T>(handle,
                  X1,
                  true,
                  yi,
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
                  wi,
                  &target,
                  &num_iters,
                  qn_loss,
                  s,
                  (T*)sampleWeight,
                  T(params.epsilon));
  };
  if (parallel) {
    std::vector<std::thread> threads;
    threads.reserve(coefCols);
    int class_i = 0;
    std::generate_n(std::back_inserter(threads), coefCols, [&solveBinary, &class_i] {
      return std::move(std::thread(solveBinary, class_i++));
    });
    for (auto& thread : threads)
      thread.join();                    // make sure all stream actions are recorded...
    handle.wait_on_internal_streams();  // ... and executed
  } else {
    for (int class_i = 0; class_i < coefCols; class_i++)
      solveBinary(class_i);
  }

  if (coefCols > 1) raft::linalg::transpose(handle, w1, w.data(), coefRows, coefCols, stream);

  ML::POP_RANGE();

  /** TODO: probabolisting calibration is disabled for now, multiclass case is not ready. */
  // if (!params.probability) return;

  // ML::PUSH_RANGE("Trace::LinearSVMModel::fit-probabilities");
  // probScale.resize(coefCols * (coefCols + 1), stream);

  // rmm::device_uvector<T> xwBuf(nRows * coefCols, stream);
  // T* xw = xwBuf.data();
  // predict_linear(handle, X, w.data(), nRows, nCols, coefCols, params.fit_intercept, xw, stream);

  // // here we should encode the labels into corresponding `classes` indices.
  // // for now, we assume labels are the values from 0 to classes.size() - 1.
  // // raft::linalg::unaryOp(y1, y, nRows, IndicatorFun<T>{T(params.H1_value)}, stream);

  // GLM::qnFit<T>(handle,
  //               xw,
  //               false,
  //               (T*)y,
  //               nRows,
  //               coefCols,
  //               nClasses,
  //               true,
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
  //               nClasses == 2 ? 0 /* logistic loss*/ : 2 /** softmax */,
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

  /** TODO: apply probScale calibration! */

  PredictProba<T>::run(out, temp.data(), nRows, classes.size(), log, handle.get_stream());
}

template class LinearSVMModel<float>;
template class LinearSVMModel<double>;

}  // namespace SVM
}  // namespace ML
