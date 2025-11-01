/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/nvtx.hpp>

#include <cuml/common/functional.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/linear_model/glm.hpp>
#include <cuml/svm/linear.hpp>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <cublas_v2.h>
#include <omp.h>

#include <random>
#include <type_traits>

namespace ML {
namespace SVM {
namespace linear {

inline int narrowDown(std::size_t n)
{
  ASSERT(std::size_t(std::numeric_limits<int>::max()) >= n,
         "LinearSVM supports input sizes only within `int` range at this point (got = %zu)",
         n);
  return int(n);
}

/**  The cuda kernel for classification. Call it via PredictProba::run(..). */
template <typename T, bool Binary, int BX = 32, int BY = 8>
CUML_KERNEL void predictProba(T* out, const T* z, const int nRows, const int nClasses)
{
  typedef cub::WarpReduce<T, BX> WarpRed;
  __shared__ typename WarpRed::TempStorage shm[BY];
  typename WarpRed::TempStorage& warpStore = shm[threadIdx.y];

  const int i = threadIdx.y + blockIdx.y * BY;
  if (i >= nRows) return;
  const T* rowIn = z + i * (Binary ? 1 : nClasses);
  T* rowOut      = out + i * nClasses;

  // the largest 'z' in the row (to subtract it from z for numeric stability).
  T t      = std::numeric_limits<T>::lowest();
  T maxVal = t;
  int j    = threadIdx.x;
  if constexpr (Binary) {
    t      = rowIn[0];
    maxVal = raft::max<T>(t, T{0});
    t      = T(j) * t;  // set z[0] = 0, z[1] = t
  } else {
    for (; j < nClasses; j += BX) {
      t      = rowIn[j];
      maxVal = raft::max<T>(maxVal, t);
    }
    j -= BX;
    maxVal = WarpRed(warpStore).Reduce(maxVal, ML::detail::maximum{});
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
    et = raft::exp<T>(t - maxVal);
    smSum += et;
    if (j < BX) break;
    j -= BX;
    t = rowIn[j];
  }
  smSum = WarpRed(warpStore).Reduce(smSum, cuda::std::plus{});
  smSum = cub::ShuffleIndex<BX>(smSum, 0, 0xFFFFFFFFU);

  // Now, either `j` refers to the first valid column idx worked by the
  // current thread, or `j` is negative (no work at all).
  // Traverse in the forward direction again to save the results.
  // Note, no extra memory reads when BX >= nClasses!
  if (j < 0) return;
  T d = 1 / smSum;
  while (j < nClasses) {
    rowOut[j] = et * d;
    j += BX;
    if (j >= nClasses) break;
    t  = rowIn[j];
    et = raft::exp<T>(t - maxVal);
  }
}

/**
 * The wrapper struct on top of predictProba that recursively selects the best BX
 * (largest BX satisfying `BX < coefCols*2`) and then schedules the kernel launch.
 *
 * @tparam T - the data element type (e.g. float/double).
 * @tparam BlockSize - the total size of the cuda thread block (BX * BY).
 * @tparam BX - the size of the block along rows (nClasses dim).
 */
template <typename T, int BlockSize = 256, int BX = 32>
struct PredictProba {
  static_assert(BX <= 32, "BX must be not larger than warpSize");
  static_assert(BX <= BlockSize, "BX must be not larger than BlockSize");
  /**
   * Predict probabilities using the scores.
   *
   * @param [out] out - row-major matrix of probabilities (nRows, nClasses).
   * @param [in] z   - row-major matrix of scores (nRows, Binary ? 1 : nClasses).
   * @param [in] nRows - number of rows in the data.
   * @param [in] nClasses - number of classes in the problem.
   * @param [in] stream - the work stream.
   */
  static inline void run(
    T* out, const T* z, const int nRows, const int nClasses, cudaStream_t stream)
  {
    if constexpr (BX > 2) {
      if (nClasses <= (BX >> 1))
        return PredictProba<T, BlockSize, std::max<int>(BX >> 1, 2)>::run(
          out, z, nRows, nClasses, stream);
    }
    const int BY      = BlockSize / BX;
    const bool Binary = BX == 2;
    const dim3 bs(BX, BY, 1);
    const dim3 gs(1, raft::ceildiv(nRows, BY), 1);
    if constexpr (Binary)
      ASSERT((void*)out != (void*)z, "PredictProba for the binary case cannot be inplace.");
    predictProba<T, Binary, BX, BY><<<gs, bs, 0, stream>>>(out, z, nRows, nClasses);
  }
};

/** The loss function is the main hint for whether we solve classification or regression. */
inline bool isRegression(Params::Loss loss)
{
  return loss == Params::EPSILON_INSENSITIVE || loss == Params::SQUARED_EPSILON_INSENSITIVE;
}

/** A functor that maps the multiclass problem onto the one-vs-rest binary problem */
template <typename T>
struct OvrSelector {
  const T* classes;
  const int selected;
  __device__ T operator()(const T x) const { return x == classes[selected] ? 1 : 0; }
};

/**
 * The linear part of the prediction.
 *
 * @param [in] handle - raft handle
 * @param [in] X - column-major matrix of size (nRows, nCols)
 * @param [in] w - row-major matrix of size [nCols + fitIntercept, coefCols]
 * @param [in] nRows - number of samples
 * @param [in] nCols - number of features
 * @param [in] coefCols - number of columns in `w` (`nClasses == 2 ? 1 : nClasses`)
 * @param [in] fitIntercept - whether to add the bias term
 * @param [out] out - row-major matrix of size [nRows, coefCols]
 * @param [in] stream - cuda stream (not synchronized)
 */
template <typename T>
void predictLinear(const raft::handle_t& handle,
                   const T* X,
                   const T* w,
                   const std::size_t nRows,
                   const std::size_t nCols,
                   const std::size_t coefCols,
                   const bool fitIntercept,
                   T* out,
                   cudaStream_t stream)
{
  raft::linalg::gemm<T>(handle,
                        out,
                        (T*)X,
                        (T*)w,
                        narrowDown(nRows),
                        narrowDown(coefCols),
                        narrowDown(nCols),
                        false,
                        true,
                        false,
                        stream);

  if (fitIntercept)
    raft::linalg::matrixVectorOp<true, true>(
      out, out, w + nCols * coefCols, coefCols, nRows, cuda::std::plus{}, stream);
}

/** A helper struct for selecting handle/stream depending on whether omp parallel is active. */
class WorkerHandle {
 private:
  raft::handle_t* handle_ptr = nullptr;

 public:
  int stream_id = 0;
  const raft::handle_t& handle;
  cudaStream_t stream;

  WorkerHandle(const raft::handle_t& handle, cudaStream_t stream) : handle(handle), stream(stream)
  {
  }

  WorkerHandle(const raft::handle_t& h, int stream_id)
    : handle_ptr{new raft::handle_t{h.get_next_usable_stream(stream_id)}},
      stream_id(stream_id),
      handle(*handle_ptr),
      stream(h.get_next_usable_stream(stream_id))
  {
  }

  ~WorkerHandle()
  {
    if (handle_ptr != nullptr) delete handle_ptr;
  }
};

template <typename T>
int fit(const raft::handle_t& handle,
        const Params& params,
        const std::size_t nRows,
        const std::size_t nCols,
        const int nClasses,
        const T* classes,
        const T* X,
        const T* y,
        const T* sampleWeight,
        T* w,
        T* probScale)
{
  if (isRegression(params.loss)) {
    ASSERT(nClasses == 0 && classes == nullptr, "Regression fit takes no classes");
    ASSERT(probScale == nullptr, "probability scaling not allowed for regressions");
  } else {
    ASSERT(nClasses > 1 && classes != nullptr, "Must have > 1 class for classification");
  }

  cudaStream_t stream = handle.get_stream();

  const int coefCols         = nClasses <= 2 ? 1 : nClasses;
  const std::size_t coefRows = nCols + int(params.fit_intercept);

  raft::common::nvtx::range fun_scope("ML::SVM::linear::fit");

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

  ML::GLM::qn_params qn_pams;
  switch (params.loss) {
    case Params::HINGE: qn_pams.loss = ML::GLM::QN_LOSS_SVC_L1; break;
    case Params::SQUARED_HINGE: qn_pams.loss = ML::GLM::QN_LOSS_SVC_L2; break;
    case Params::EPSILON_INSENSITIVE: qn_pams.loss = ML::GLM::QN_LOSS_SVR_L1; break;
    case Params::SQUARED_EPSILON_INSENSITIVE: qn_pams.loss = ML::GLM::QN_LOSS_SVR_L2; break;
    default: break;
  }
  qn_pams.fit_intercept       = params.fit_intercept && !params.penalized_intercept;
  qn_pams.penalty_l1          = params.penalty == Params::L1 ? iC : 0.0;
  qn_pams.penalty_l2          = params.penalty == Params::L2 ? iC : 0.0;
  qn_pams.penalty_normalized  = true;
  qn_pams.max_iter            = params.max_iter;
  qn_pams.grad_tol            = params.grad_tol;
  qn_pams.change_tol          = params.change_tol;
  qn_pams.linesearch_max_iter = params.linesearch_max_iter;
  qn_pams.lbfgs_memory        = params.lbfgs_memory;
  qn_pams.verbose             = static_cast<int>(params.verbose);

  ML::GLM::qn_params qn_pams_logistic = qn_pams;
  qn_pams_logistic.loss               = ML::GLM::QN_LOSS_LOGISTIC;
  qn_pams_logistic.fit_intercept      = true;
  qn_pams_logistic.penalty_l1         = 0;
  qn_pams_logistic.penalty_l2 = 1 / T(1 + nRows);  // L2 regularization reflects the flat prior.

  T* y1  = (T*)y;
  T* w1  = w;
  T* ps1 = probScale;
  rmm::device_uvector<T> y1Buf(0, stream);
  rmm::device_uvector<T> w1Buf(0, stream);
  rmm::device_uvector<T> psBuf(0, stream);
  if (nClasses > 1) {
    y1Buf.resize(nRows * coefCols, stream);
    y1 = y1Buf.data();
  }
  if (coefCols > 1) {
    w1Buf.resize(coefCols * coefRows, stream);
    w1 = w1Buf.data();
    if (probScale != nullptr) {
      psBuf.resize(2 * coefCols, stream);
      ps1 = psBuf.data();
    }
  }
  RAFT_CUDA_TRY(cudaMemsetAsync(w1, 0, coefCols * coefRows * sizeof(T), stream));
  if (probScale != nullptr) {
    thrust::device_ptr<thrust::tuple<T, T>> p((thrust::tuple<T, T>*)ps1);
    thrust::fill(thrust::cuda::par.on(stream), p, p + coefCols, thrust::make_tuple(T(1), T(0)));
  }

  // one-vs-rest logic goes over each class
  const int n_streams = coefCols > 1 ? handle.get_stream_pool_size() : 1;
  bool parallel       = n_streams > 1;
  int max_n_iter      = 0;
#pragma omp parallel for num_threads(n_streams) if (parallel) reduction(max : max_n_iter)
  for (int class_i = 0; class_i < coefCols; class_i++) {
    T* yi = y1 + nRows * class_i;
    T* wi = w1 + coefRows * class_i;
    auto worker =
      parallel ? WorkerHandle(handle, omp_get_thread_num()) : WorkerHandle(handle, stream);
    if (nClasses > 1) {
      raft::linalg::unaryOp(
        yi, y, nRows, OvrSelector<T>{classes, nClasses == 2 ? 1 : class_i}, worker.stream);
    }
    T target;
    int n_iter;
    GLM::qnFit<T>(worker.handle,
                  qn_pams,
                  X1,
                  true,
                  yi,
                  narrowDown(nRows),
                  narrowDown(nCols1),
                  // regression: C == 1; classification: C == 2
                  nClasses == 0 ? 1 : 2,
                  wi,
                  &target,
                  &n_iter,
                  (T*)sampleWeight,
                  T(params.epsilon));
    if (n_iter > max_n_iter) { max_n_iter = n_iter; }

    if (probScale == nullptr) continue;
    // Calibrate probabilities
    T* psi = ps1 + 2 * class_i;
    rmm::device_uvector<T> xwBuf(nRows, worker.stream);
    T* xw = xwBuf.data();
    predictLinear(worker.handle, X, wi, nRows, nCols, 1, params.fit_intercept, xw, worker.stream);

    GLM::qnFit<T>(worker.handle,
                  qn_pams_logistic,
                  xw,
                  false,
                  yi,
                  narrowDown(nRows),
                  1,
                  2,
                  psi,
                  &target,
                  &n_iter,
                  (T*)sampleWeight);

    if (n_iter > max_n_iter) { max_n_iter = n_iter; }
  }
  if (parallel) handle.sync_stream_pool();

  if (coefCols > 1) {
    raft::linalg::transpose(handle, w1, w, coefRows, coefCols, stream);
    if (probScale != nullptr) {
      raft::linalg::transpose(handle, ps1, probScale, 2, coefCols, stream);
    }
  }

  return max_n_iter;
}

template <typename T>
void computeProbabilities(const raft::handle_t& handle,
                          const std::size_t nRows,
                          const int nClasses,
                          const T* probScale,
                          T* scores,
                          T* out)
{
  auto stream                = handle.get_stream();
  const std::size_t coefCols = nClasses == 2 ? 1 : nClasses;

  // probability calibration
  raft::linalg::matrixVectorOp<true, true>(
    scores,
    scores,
    probScale,
    probScale + coefCols,
    coefCols,
    nRows,
    [] __device__(const T x, const T a, const T b) { return a * x + b; },
    stream);

  // apply sigmoid/softmax
  PredictProba<T>::run(out, scores, nRows, nClasses, stream);
}

// Explicit instantiations for library
template int fit<float>(const raft::handle_t& handle,
                        const Params& params,
                        const std::size_t nRows,
                        const std::size_t nCols,
                        const int nClasses,
                        const float* classes,
                        const float* X,
                        const float* y,
                        const float* sampleWeight,
                        float* w,
                        float* probScale);
template int fit<double>(const raft::handle_t& handle,
                         const Params& params,
                         const std::size_t nRows,
                         const std::size_t nCols,
                         const int nClasses,
                         const double* classes,
                         const double* X,
                         const double* y,
                         const double* sampleWeight,
                         double* w,
                         double* probScale);
template void computeProbabilities<float>(const raft::handle_t& handle,
                                          const std::size_t nRows,
                                          const int nClasses,
                                          const float* probScale,
                                          float* scores,
                                          float* out);
template void computeProbabilities<double>(const raft::handle_t& handle,
                                           const std::size_t nRows,
                                           const int nClasses,
                                           const double* probScale,
                                           double* scores,
                                           double* out);

}  // namespace linear
}  // namespace SVM
}  // namespace ML
