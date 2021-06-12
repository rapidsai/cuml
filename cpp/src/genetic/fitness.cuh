/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/strided_reduction.cuh>
#include <raft/linalg/unary_op.cuh>

#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <raft/cudart_utils.h>

namespace cuml {
namespace genetic {

template <typename math_t>
void _weighted_pearson(const raft::handle_t& h, const int n_samples,
                       const int n_progs, const math_t* Y, const math_t* X,
                       const math_t* W, math_t* out) {
  // Find Pearson's correlation coefficient

  cudaStream_t stream = h.get_stream();

  rmm::device_uvector<math_t> corr(n_samples * n_progs,
                                   stream);  // Per sample correlation

  rmm::device_uvector<math_t> y_tmp(n_samples, stream);
  rmm::device_uvector<math_t> x_tmp(n_samples * n_progs, stream);

  rmm::device_uvector<math_t> mu_y(1, stream);        // label mean
  rmm::device_uvector<math_t> mu_x(n_progs, stream);  // predicted label means

  rmm::device_uvector<math_t> y_diff(n_samples, stream);  // normalized labels
  rmm::device_uvector<math_t> x_diff(n_samples * n_progs,
                                     stream);  // normalized prediction labels

  rmm::device_uvector<math_t> y_std(1, stream);  // standard deviation of labels
  rmm::device_uvector<math_t> x_std(
    n_progs, stream);  // standard deviation of predicted labels

  rmm::device_uvector<math_t> dWS(1, stream);  // sum of weights
  math_t N = (math_t)n_samples;

  // Sum of weights
  raft::stats::sum(dWS.data(), W, 1, n_samples, false, stream);
  math_t WS = dWS.element(0, stream);

  // Find mu_y
  raft::linalg::matrixVectorOp(
    y_tmp.data(), Y, W, 1, n_samples, false, false,
    [N, WS] __device__(math_t y, math_t w) { return N * w * y / WS; }, stream);
  raft::stats::mean(mu_y.data(), y_tmp.data(), 1, n_samples, false, false,
                    stream);

  // Find mu_x
  raft::linalg::matrixVectorOp(
    x_tmp.data(), X, W, n_progs, n_samples, false, true,
    [N, WS] __device__(math_t x, math_t w) { return N * w * x / WS; }, stream);
  raft::stats::mean(mu_x.data(), x_tmp.data(), n_progs, n_samples, false, false,
                    stream);

  // Find y_diff
  raft::stats::meanCenter(y_diff.data(), Y, mu_y.data(), 1, n_samples, false,
                          true, stream);

  // Find x_diff
  raft::stats::meanCenter(x_diff.data(), X, mu_x.data(), n_progs, n_samples,
                          false, true, stream);

  // Find y_std
  raft::linalg::stridedReduction(
    y_std.data(), y_diff.data(), 1, n_samples, (math_t)0, stream, false,
    [W] __device__(math_t v, int i) { return v * v * W[i]; },
    raft::Sum<math_t>(), [] __device__(math_t in) { return raft::mySqrt(in); });
  math_t HYstd = y_std.element(0, stream);

  // Find x_std
  raft::linalg::stridedReduction(
    x_std.data(), x_diff.data(), n_progs, n_samples, (math_t)0, stream, false,
    [W] __device__(math_t v, int i) { return v * v * W[i]; },
    raft::Sum<math_t>(), [] __device__(math_t in) { return raft::mySqrt(in); });

  // Cross covariance
  raft::linalg::matrixVectorOp(
    corr.data(), x_diff.data(), y_diff.data(), W, n_progs, n_samples, false,
    false,
    [N, HYstd] __device__(math_t xd, math_t yd, math_t w) {
      return N * w * xd * yd / HYstd;
    },
    stream);

  // Find Correlation coeff
  raft::linalg::matrixVectorOp(
    corr.data(), corr.data(), x_std.data(), n_progs, n_samples, false, true,
    [] __device__(math_t c, math_t xd) { return c / xd; }, stream);
  raft::stats::mean(out, corr.data(), n_progs, n_samples, false, false, stream);
}

struct rank_functor {
  template <typename math_t>
  __host__ __device__ math_t operator()(math_t data) {
    if (data == 0)
      return 0;
    else
      return 1;
  }
};

template <typename math_t>
void _weighted_spearman(const raft::handle_t& h, const int n_samples,
                        const int n_progs, const math_t* Y,
                        const math_t* Y_pred, const math_t* W, math_t* out) {
  cudaStream_t stream = h.get_stream();

  // Get ranks for Y
  thrust::device_vector<math_t> Ycopy(Y, Y + n_samples);
  thrust::device_vector<math_t> rank_idx(n_samples, 0);
  thrust::device_vector<math_t> rank_diff(n_samples, 0);
  thrust::device_vector<math_t> Yrank(n_samples, 0);

  thrust::sequence(thrust::cuda::par.on(stream), rank_idx.begin(),
                   rank_idx.end(), 0);
  thrust::sort_by_key(thrust::cuda::par.on(stream), Ycopy.begin(), Ycopy.end(),
                      rank_idx.begin());
  thrust::adjacent_difference(thrust::cuda::par.on(stream), Ycopy.begin(),
                              Ycopy.end(), rank_diff.begin());
  thrust::transform(thrust::cuda::par.on(stream), rank_diff.begin(),
                    rank_diff.end(), rank_diff.begin(), rank_functor());
  rank_diff[0] = 1;
  thrust::inclusive_scan(thrust::cuda::par.on(stream), rank_diff.begin(),
                         rank_diff.end(), rank_diff.begin());
  thrust::copy(
    rank_diff.begin(), rank_diff.end(),
    thrust::make_permutation_iterator(Yrank.begin(), rank_idx.begin()));

  // Get ranks for Y_pred
  // TODO: Find a better way to batch this
  thrust::device_vector<math_t> Ypredcopy(Y_pred, Y_pred + n_samples * n_progs);
  thrust::device_vector<math_t> Ypredrank(n_samples * n_progs, 0);
  thrust::device_ptr<math_t> Ypredptr =
    thrust::device_pointer_cast<math_t>(Ypredcopy.data());
  thrust::device_ptr<math_t> Ypredrankptr =
    thrust::device_pointer_cast<math_t>(Ypredrank.data());

  for (int i = 0; i < n_progs; ++i) {
    thrust::sequence(thrust::cuda::par.on(stream), rank_idx.begin(),
                     rank_idx.end(), 0);
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        Ypredptr + (i * n_samples),
                        Ypredptr + ((i + 1) * n_samples), rank_idx.begin());
    thrust::adjacent_difference(
      thrust::cuda::par.on(stream), Ypredptr + (i * n_samples),
      Ypredptr + ((i + 1) * n_samples), rank_diff.begin());
    thrust::transform(thrust::cuda::par.on(stream), rank_diff.begin(),
                      rank_diff.end(), rank_diff.begin(), rank_functor());
    rank_diff[0] = 1;
    thrust::inclusive_scan(thrust::cuda::par.on(stream), rank_diff.begin(),
                           rank_diff.end(), rank_diff.begin());
    thrust::copy(rank_diff.begin(), rank_diff.end(),
                 thrust::make_permutation_iterator(
                   Ypredrankptr + (i * n_samples), rank_idx.begin()));
  }

  // Compute pearson's coefficient
  _weighted_pearson(h, n_samples, n_progs,
                    thrust::raw_pointer_cast(Yrank.data()),
                    thrust::raw_pointer_cast(Ypredrank.data()), W, out);
}

template <typename math_t>
void _mean_absolute_error(const raft::handle_t& h, const int n_samples,
                          const int n_progs, const math_t* Y,
                          const math_t* Y_pred, const math_t* W, math_t* out) {
  cudaStream_t stream = h.get_stream();
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_uvector<math_t> dWS(1, stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum(dWS.data(), W, 1, n_samples, false, stream);
  math_t WS = dWS.element(0, stream);

  // Compute absolute differences
  raft::linalg::matrixVectorOp(
    error.data(), Y_pred, Y, W, n_progs, n_samples, false, false,
    [N, WS] __device__(math_t y_p, math_t y, math_t w) {
      return N * w * raft::myAbs(y - y_p) / WS;
    },
    stream);

  // Average along rows
  raft::stats::mean(out, error.data(), n_progs, n_samples, false, false,
                    stream);
}

template <typename math_t>
void _mean_square_error(const raft::handle_t& h, const int n_samples,
                        const int n_progs, const math_t* Y,
                        const math_t* Y_pred, const math_t* W, math_t* out) {
  cudaStream_t stream = h.get_stream();
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_uvector<math_t> dWS(1, stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum(dWS.data(), W, 1, n_samples, false, stream);
  math_t WS = dWS.element(0, stream);

  // Compute square differences
  raft::linalg::matrixVectorOp(
    error.data(), Y_pred, Y, W, n_progs, n_samples, false, false,
    [N, WS] __device__(math_t y_p, math_t y, math_t w) {
      return N * w * (y_p - y) * (y_p - y) / WS;
    },
    stream);

  // Add up row values per column
  raft::stats::mean(out, error.data(), n_progs, n_samples, false, false,
                    stream);
}

template <typename math_t>
void _root_mean_square_error(const raft::handle_t& h, const int n_samples,
                             const int n_progs, const math_t* Y,
                             const math_t* Y_pred, const math_t* W,
                             math_t* out) {
  cudaStream_t stream = h.get_stream();

  // Find MSE
  _mean_square_error(h, n_samples, n_progs, Y, Y_pred, W, out);

  // Take sqrt on all entries
  raft::linalg::unaryOp(
    out, out, n_progs, [] __device__(math_t in) { return raft::mySqrt(in); },
    stream);
}

template <typename math_t>
void _log_loss(const raft::handle_t& h, const int n_samples, const int n_progs,
               const math_t* Y, const math_t* Y_pred, const math_t* W,
               math_t* out) {
  cudaStream_t stream = h.get_stream();
  // Logistic error per sample
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_uvector<math_t> dWS(1, stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum(dWS.data(), W, 1, n_samples, false, stream);
  math_t WS = dWS.element(0, stream);

  // Populate logistic error as matrix vector op
  raft::linalg::matrixVectorOp(
    error.data(), Y_pred, Y, W, n_progs, n_samples, false, false,
    [N, WS] __device__(math_t y_p, math_t y, math_t w) {
      return N * w * (-y * logf(y_p) - (1 - y) * logf(1 - y_p)) / WS;
    },
    stream);

  // Take average along rows
  raft::stats::mean(out, error.data(), n_progs, n_samples, false, false,
                    stream);
}

template void _weighted_pearson<float>(const raft::handle_t& h,
                                       const int n_samples, const int n_progs,
                                       const float* Y, const float* Y_pred,
                                       const float* W, float* out);
template void _weighted_spearman<float>(const raft::handle_t& h,
                                        const int n_samples, const int n_progs,
                                        const float* Y, const float* Y_pred,
                                        const float* W, float* out);
template void _mean_absolute_error<float>(const raft::handle_t& h,
                                          const int n_samples,
                                          const int n_progs, const float* Y,
                                          const float* Y_pred, const float* W,
                                          float* out);
template void _mean_square_error<float>(const raft::handle_t& h,
                                        const int n_samples, const int n_progs,
                                        const float* Y, const float* Y_pred,
                                        const float* W, float* out);
template void _root_mean_square_error<float>(const raft::handle_t& h,
                                             const int n_samples,
                                             const int n_progs, const float* Y,
                                             const float* Y_pred,
                                             const float* W, float* out);

}  // namespace genetic
}  // namespace cuml
