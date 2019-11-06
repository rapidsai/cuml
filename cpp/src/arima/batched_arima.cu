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

#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

#include "batched_arima.hpp"
#include "batched_kalman.hpp"
#include "cuda_utils.h"
#include "utils.h"

#include <common/nvtx.hpp>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/cuml.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/matrix_vector_op.h>
#include <stats/mean.h>
#include <matrix/batched_matrix.hpp>

namespace ML {

using std::vector;

void residual(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
              int d, int q, double* d_params, double* d_vs, bool trans) {
  std::vector<double> loglike;
  batched_loglike(handle, d_y, num_batches, nobs, p, d, q, d_params, loglike,
                  d_vs, trans);
}

void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int batch_size, int nobs, double* d_y, double* d_y_diff,
              double* d_vs, double* d_params, double* d_y_fc) {
  auto alloc = handle.getDeviceAllocator();
  const auto stream = handle.getStream();
  double* d_y_ = (double*)alloc->allocate((p + num_steps) * batch_size, stream);
  double* d_vs_ =
    (double*)alloc->allocate((q + num_steps) * batch_size, stream);
  const auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     if (p > 0) {
                       for (int ip = 0; ip < p; ip++) {
                         d_y_[(p + num_steps) * bid + ip] =
                           d_y_diff[(nobs - d) * bid + (nobs - d - p) + ip];
                       }
                     }
                     if (q > 0) {
                       for (int iq = 0; iq < q; iq++) {
                         d_vs_[(q + num_steps) * bid + iq] =
                           d_vs[(nobs - d) * bid + (nobs - d - q) + iq];
                       }
                     }
                   });

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     int N = p + d + q;
                     auto mu_ib = d_params[N * bid];
                     double ar_sum = 0.0;
                     for (int ip = 0; ip < p; ip++) {
                       double ar_i = d_params[N * bid + d + ip];
                       ar_sum += ar_i;
                     }
                     double mu_star = mu_ib * (1 - ar_sum);

                     for (int i = 0; i < num_steps; i++) {
                       auto it = num_steps * bid + i;
                       d_y_fc[it] = mu_star;
                       if (p > 0) {
                         double dot_ar_y = 0.0;
                         for (int ip = 0; ip < p; ip++) {
                           dot_ar_y += d_params[N * bid + d + ip] *
                                       d_y_[(p + num_steps) * bid + i + ip];
                         }
                         d_y_fc[it] += dot_ar_y;
                       }
                       if (q > 0 && i < q) {
                         double dot_ma_y = 0.0;
                         for (int iq = 0; iq < q; iq++) {
                           dot_ma_y += d_params[N * bid + d + p + iq] *
                                       d_vs_[(q + num_steps) * bid + i + iq];
                         }
                         d_y_fc[it] += dot_ma_y;
                       }
                       if (p > 0) {
                         d_y_[(p + num_steps) * bid + i + p] = d_y_fc[it];
                       }
                     }
                   });

  // undifference
  if (d > 0) {
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size,
      [=] __device__(int bid) {
        for (int i = 0; i < num_steps; i++) {
          // Undifference via cumsum, using last 'y' as initial value, in cumsum.
          // Then drop that first value.
          // In python:
          // xi = np.append(y[-1], fc)
          // return np.cumsum(xi)[1:]
          if (i == 0) {
            d_y_fc[bid * num_steps] += d_y[bid * nobs + (nobs - 1)];
          } else {
            d_y_fc[bid * num_steps + i] += d_y_fc[bid * num_steps + i - 1];
          }
        }
      });
  }
}

void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                       int nobs, int p, int d, int q, double* d_params,
                       double* d_vs, double* d_y_p) {
  residual(handle, d_y, num_batches, nobs, p, d, q, d_params, d_vs, false);
  auto stream = handle.getStream();
  double* d_y_diff;

  if (d == 0) {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + num_batches, [=] __device__(int bid) {
                       for (int i = 0; i < nobs; i++) {
                         int it = bid * nobs + i;
                         d_y_p[it] = d_y[it] - d_vs[it];
                       }
                     });
  } else {
    d_y_diff = (double*)handle.getDeviceAllocator()->allocate(
      sizeof(double) * num_batches * (nobs - 1), handle.getStream());
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + num_batches, [=] __device__(int bid) {
                       for (int i = 0; i < nobs - 1; i++) {
                         int it = bid * nobs + i;
                         int itd = bid * (nobs - 1) + i;
                         // note: d_y[it] + (d_y[it + 1] - d_y[it]) - d_vs[itd]
                         //    -> d_y[it+1] - d_vs[itd]
                         d_y_p[it] = d_y[it + 1] - d_vs[itd];
                         d_y_diff[itd] = d_y[it + 1] - d_y[it];
                       }
                     });
  }

  // due to `differencing` we need to forecast a single step to make the
  // in-sample prediction the same length as the original signal.
  if (d == 1) {
    double* d_y_fc = (double*)handle.getDeviceAllocator()->allocate(
      sizeof(double) * num_batches, handle.getStream());
    forecast(handle, 1, p, d, q, num_batches, nobs, d_y, d_y_diff, d_vs,
             d_params, d_y_fc);

    // append forecast to end of in-sample prediction
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + num_batches, [=] __device__(int bid) {
                       d_y_p[bid * nobs + (nobs - 1)] = d_y_fc[bid];
                     });
  }
}

void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_mu, double* d_ar,
                     double* d_ma, std::vector<double>& loglike, double* d_vs,
                     bool trans) {
  using std::get;
  using std::vector;

  ML::PUSH_RANGE(__FUNCTION__);

  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_Tar =
    (double*)allocator->allocate(sizeof(double) * num_batches * p, stream);
  double* d_Tma =
    (double*)allocator->allocate(sizeof(double) * num_batches * q, stream);

  if (trans) {
    batched_jones_transform(handle, p, q, num_batches, false, d_ar, d_ma, d_Tar,
                            d_Tma);
  } else {
    // non-transformed case: just use original parameters
    CUDA_CHECK(cudaMemcpyAsync(d_Tar, d_ar, sizeof(double) * num_batches * p,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Tma, d_ma, sizeof(double) * num_batches * q,
                               cudaMemcpyDeviceToDevice, stream));
  }

  if (d == 0) {
    // no diff
    batched_kalman_filter(handle, d_y, nobs, d_Tar, d_Tma, p, q, num_batches,
                          loglike, d_vs);
  } else if (d == 1) {
    ////////////////////////////////////////////////////////////
    // diff and center (with `mu`):
    ////////////////////////////////////////////////////////////

    // make device array and pointer
    double* y_diff = (double*)allocator->allocate(
      num_batches * (nobs - 1) * sizeof(double), stream);

    // TODO: check performance of this thrust code against the custom kernel
    // _batched_diff_kernel
    {
      auto counting = thrust::make_counting_iterator(0);
      // TODO: This for_each should probably go over samples, so batches
      // are in the inner loop.
      thrust::for_each(thrust::cuda::par.on(stream), counting,
                       counting + num_batches, [=] __device__(int bid) {
                         double mu_ib = d_mu[bid];
                         for (int i = 0; i < nobs - 1; i++) {
                           // diff and center (with `mu` parameter)
                           y_diff[bid * (nobs - 1) + i] =
                             (d_y[bid * nobs + i + 1] - d_y[bid * nobs + i]) -
                             mu_ib;
                         }
                       });
    }

    batched_kalman_filter(handle, y_diff, nobs - d, d_Tar, d_Tma, p, q,
                          num_batches, loglike, d_vs);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }
  allocator->deallocate(d_Tar, sizeof(double) * p * num_batches, stream);
  allocator->deallocate(d_Tma, sizeof(double) * q * num_batches, stream);
  ML::POP_RANGE();
}

void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_params,
                     std::vector<double>& loglike, double* d_vs, bool trans) {
  ML::PUSH_RANGE(__FUNCTION__);

  // unpack parameters
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_mu =
    (double*)allocator->allocate(sizeof(double) * num_batches, stream);
  double* d_ar =
    (double*)allocator->allocate(sizeof(double) * num_batches * p, stream);
  double* d_ma =
    (double*)allocator->allocate(sizeof(double) * num_batches * q, stream);

  // params -> (mu, ar, ma)
  unpack(d_params, d_mu, d_ar, d_ma, num_batches, p, d, q, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  batched_loglike(handle, d_y, num_batches, nobs, p, d, q, d_mu, d_ar, d_ma,
                  loglike, d_vs, trans);

  allocator->deallocate(d_mu, sizeof(double) * num_batches, stream);
  allocator->deallocate(d_ar, sizeof(double) * p * num_batches, stream);
  allocator->deallocate(d_ma, sizeof(double) * q * num_batches, stream);
  ML::POP_RANGE();
}

/**
 * TODO: doc
 */
void bic(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
         int d, int q, double* d_mu, double* d_ar, double* d_ma,
         std::vector<double>& ic) {
  ML::PUSH_RANGE(__FUNCTION__);

  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_vs = (double*)allocator->allocate(
    sizeof(double) * (nobs - d) * num_batches, stream);
  std::vector<double> loglike = std::vector<double>(num_batches);

  /* Compute log-likelihood */
  batched_loglike(handle, d_y, num_batches, nobs, p, d, q, d_mu, d_ar, d_ma,
                  loglike, d_vs, false);

  /* Compute Bayes information criterion (BIC) */
  /// TODO: worth doing that on gpu? (need to copy loglike and copy back BIC)
  double ic_base =
    log(static_cast<double>(nobs)) * static_cast<double>(p + d + q);
#pragma omp parallel for
  for (int i = 0; i < num_batches; i++) {
    ic[i] = ic_base - 2.0 * loglike[i];
  }

  allocator->deallocate(d_vs, sizeof(double) * (nobs - d) * num_batches,
                        stream);
  ML::POP_RANGE();
}

/*
 * TODO: move to batched matrix prims + write doc
 * Note: the block id is the batch id and the thread id is the starting index
 */
static __global__ void batched_offset_copy_kernel(double* out, double* in,
                                                  int offset, int ld, int len) {
  const double* batch_in = in + blockIdx.x * ld + offset;
  double* batch_out = out + blockIdx.x * len;

  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    batch_out[i] = batch_in[i];
  }
}

/*
 * TODO: move to batched matrix prims + write doc
 * Note: the block id is the batch id and the thread id is the starting index
 */
static __global__ void _batched_ls_set_kernel(double* lagged_series,
                                              const double* data, int ls_width,
                                              int ls_height, int offset, int ld,
                                              int ls_batch_offset,
                                              int ls_batch_stride) {
  const double* batch_in = data + blockIdx.x * ld + offset - 1;
  double* batch_out =
    lagged_series + blockIdx.x * ls_batch_stride + ls_batch_offset;

  for (int lag = 0; lag < ls_width; lag++) {
    for (int i = threadIdx.x; i < ls_height; i += blockDim.x) {
      batch_out[lag * ls_height + i] = batch_in[i + ls_width - lag];
    }
  }
}

/*
 * TODO: move to batched matrix (e.g broadcast_vector)?
 * TODO: quick doc (internal auxiliary function)
 * Note: the block id is the batch id and the thread id is the starting index
 */
static __global__ void _batched_fill_kernel(double* out, const double* in,
                                            int n_fill, int stride) {
  double* batch_out = out + blockIdx.x * n_fill;
  const double batch_in = in[blockIdx.x * stride];
  for (int i = threadIdx.x; i < n_fill; i += blockDim.x) {
    batch_out[i] = batch_in;
  }
}

static double _ar_bic(cumlHandle& handle, double* d_y, int num_batches,
                      int nobs, int p_lags) {
  const int TPB = nobs > 512 ? 256 : 128;  // Quick heuristics for block size
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Create lagged series set
  int ls_height = nobs - p_lags;
  MLCommon::Matrix::BatchedMatrix<double> bm_ls(
    ls_height, p_lags, num_batches, cublas_handle, allocator, stream, false);
  _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
    bm_ls.raw_data(), d_y, p_lags, ls_height, 0, nobs, 0, p_lags * ls_height);
  CUDA_CHECK(cudaPeekAtLastError());

  // Init AR fit (note: larger dimensions because gels works in-place)
  MLCommon::Matrix::BatchedMatrix<double> bm_ar_fit(
    ls_height, 1, num_batches, cublas_handle, allocator, stream, false);
  batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
    bm_ar_fit.raw_data(), d_y, p_lags, nobs, ls_height);

  // Note: this overwrites bm_ls
  int ar_fit_info;
  CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
    cublas_handle, CUBLAS_OP_N, ls_height, p_lags, 1, bm_ls.data(), ls_height,
    bm_ar_fit.data(), ls_height, &ar_fit_info, nullptr, num_batches));

  // Copy AR fit in matrix of the right shape
  MLCommon::Matrix::BatchedMatrix<double> bm_ar(
    p_lags, 1, num_batches, cublas_handle, allocator, stream, false);
  batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
    bm_ar.raw_data(), bm_ar_fit.raw_data(), 0, ls_height, p_lags);

  // TODO: offset y?
  std::vector<double> all_bic = std::vector<double>(num_batches);
  bic(handle, d_y, num_batches, nobs, p_lags, 0, 0, nullptr, bm_ar.raw_data(),
      nullptr, all_bic);

  // Aggregate results ; TODO: change aggregation method?
  double sum_bic = 0.0;
#pragma omp parallel for reduction(+ : sum_bic)
  for (int ib = 0; ib < num_batches; ib++) {
    sum_bic += all_bic[ib];
  }
  return sum_bic / static_cast<double>(num_batches);
}

/**
 * TODO: quick doc (internal auxiliary function)
 *
 * @note: d_yd is mutated!
 *        
 * Determine reasonable starting mu, AR, and MA parameters
 */
static void _start_params(cumlHandle& handle, double* d_mu, double* d_ar,
                          double* d_ma, double* d_y, int num_batches, int nobs,
                          int p, int d, int q, int p_lags = -1) {
  const int TPB = nobs > 512 ? 256 : 128;  // Quick heuristics for block size

  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Initialize params
  cudaMemsetAsync(d_ar, 0, sizeof(double) * p * num_batches, stream);
  cudaMemsetAsync(d_ma, 0, sizeof(double) * q * num_batches, stream);

  if (d > 0) {
    // Compute means and write them in mu
    MLCommon::Stats::mean(d_mu, d_y, num_batches, nobs, false, false, stream);

    // Center the series around their means in-place
    MLCommon::LinAlg::matrixVectorOp(
      d_y, d_y, d_mu, num_batches, nobs, false, true,
      [] __device__(double a, double b) { return a - b; }, stream);
  }

  if (p == 0 && q == 0) {
    return;
  } else if (p != 0) {
    /* Select the number of lags for the initial AR fit */
    if (p_lags == -1) {
      int maxlags = std::min(
        static_cast<int>(
          std::round(12.0 * std::pow(static_cast<double>(nobs) / 100.0, 0.25))),
        p);
      /* statsmodels uses BIC to pick the best p for its AR fit
       * Requires to fit AR from 1 to maxlags (usually around 10)
       * This is quite expensive... TODO: keep or remove? */
      float best_bic;
      /* Note: maxlags is greater than 1 */
      for (int lag = 1; lag <= maxlags; lag++) {
        float current_bic = _ar_bic(handle, d_y, num_batches, nobs, lag);
        if (lag == 1 || current_bic < best_bic) {
          best_bic = current_bic;
          p_lags = lag;
        }
      }
    } else if (p_lags > p) {
      p_lags = p;
    }
    // if (p_lags >= nobs / 2) {
    //   p_lags = (nobs - 1) / 2;
    // }

    // Create lagged series set
    int ls_height = nobs - p_lags;
    MLCommon::Matrix::BatchedMatrix<double> bm_ls(
      ls_height, p_lags, num_batches, cublas_handle, allocator, stream, false);
    _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
      bm_ls.raw_data(), d_y, p_lags, ls_height, 0, nobs, 0, p_lags * ls_height);
    CUDA_CHECK(cudaPeekAtLastError());

    // Initial AR fit (note: larger dimensions because gels works in-place)
    MLCommon::Matrix::BatchedMatrix<double> bm_ar_fit(
      ls_height, 1, num_batches, cublas_handle, allocator, stream, false);
    batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
      bm_ar_fit.raw_data(), d_y, p_lags, nobs, ls_height);

    // Residual if q != 0, initialized as offset y to avoid one kernel call
    MLCommon::Matrix::BatchedMatrix<double> bm_residual(
      q != 0 ? ls_height : 1, 1, num_batches, cublas_handle, allocator, stream,
      false);
    if (q != 0) {
      MLCommon::copy(bm_residual.raw_data(), bm_ar_fit.raw_data(),
                     ls_height * num_batches, stream);
    }

    // Make a copy of the lagged set because gels modifies it in-place
    MLCommon::Matrix::BatchedMatrix<double> bm_ls_copy(
      ls_height, p_lags, num_batches, cublas_handle, allocator, stream, false);
    MLCommon::copy(bm_ls_copy.raw_data(), bm_ls.raw_data(),
                   ls_height * p_lags * num_batches, stream);

    // Note: this overwrites bm_ls
    int ar_fit_info;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
      cublas_handle, CUBLAS_OP_N, ls_height, p_lags, 1, bm_ls_copy.data(),
      ls_height, bm_ar_fit.data(), ls_height, &ar_fit_info, nullptr,
      num_batches));
    ///@todo raise exception if info < 0? (need to sync stream)

    if (q == 0) {
      // Note: works only for p_lags == 1 ; what to do here when p_lags > 1?
      // Fill AR parameters with ar_fit
      // TODO: matrix-vector op?
      _batched_fill_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ar, bm_ar_fit.raw_data(), p, ls_height);
      CUDA_CHECK(cudaPeekAtLastError());
    } else {
      // Compute residual (technically a gemv but we're missing a col-major
      // batched gemv if I'm correct)
      double alpha = -1.0;
      double beta = 1.0;
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgemmBatched(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, ls_height, 1, p_lags, &alpha,
        bm_ls.data(), ls_height, bm_ar_fit.data(), ls_height, &beta,
        bm_residual.data(), ls_height, num_batches, stream));

      int p_diff = p - p_lags;

      // Create matrices made of the concatenation of lagged sets for ar terms
      // and the residual respectively, side by side
      int arma_fit_offset = std::max(p_lags + q, p);
      int ls_ar_res_height = nobs - arma_fit_offset;
      int ls_res_size = ls_ar_res_height * q;
      int ls_ar_size = ls_ar_res_height * p;
      int ls_ar_res_size = ls_res_size + ls_ar_size;
      int ar_offset = (p < p_lags + q) ? (p_lags + q - p) : 0;
      int res_offset = (p < p_lags + q) ? 0 : p - p_lags - q;
      MLCommon::Matrix::BatchedMatrix<double> bm_ls_ar_res(
        ls_ar_res_height, p + q, num_batches, cublas_handle, allocator, stream,
        false);
      _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_ls_ar_res.raw_data(), d_y, p, ls_ar_res_height, ar_offset, nobs, 0,
        ls_ar_res_size);
      CUDA_CHECK(cudaPeekAtLastError());
      _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_ls_ar_res.raw_data(), bm_residual.raw_data(), q, ls_ar_res_height,
        res_offset, ls_height, ls_ar_size, ls_ar_res_size);
      CUDA_CHECK(cudaPeekAtLastError());

      // ARMA fit (note: larger dimensions because gels works in-place)
      MLCommon::Matrix::BatchedMatrix<double> bm_arma_fit(
        ls_ar_res_height, 1, num_batches, cublas_handle, allocator, stream,
        false);
      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_arma_fit.raw_data(), d_y, arma_fit_offset, nobs, ls_ar_res_height);

      // Note: this overwrites bm_ls_ar_res
      int arma_fit_info;
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
        cublas_handle, CUBLAS_OP_N, ls_ar_res_height, p + q, 1,
        bm_ls_ar_res.data(), ls_ar_res_height, bm_arma_fit.data(),
        ls_ar_res_height, &arma_fit_info, nullptr, num_batches));
      ///@todo raise exception if info < 0? (need to sync stream)

      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ar, bm_arma_fit.raw_data(), 0, ls_ar_res_height, p);
      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ma, bm_arma_fit.raw_data(), p, ls_ar_res_height, q);
    }
  } else {  // p == 0 && q > 0
    ///@todo See how `statsmodels` handles this case

    // Set MA params to -1
    thrust::device_ptr<double> __ma = thrust::device_pointer_cast(d_ma);
    thrust::fill(thrust::cuda::par.on(stream), __ma, __ma + q * num_batches,
                 -1.0);
  }
}

/*
 * TODO: quick doc (internal auxiliary function)
 *
 * @note: The thread id is the starting position and the block id is the
 *        batch id.
 */
static __global__ void _batched_diff_kernel(const double* in, double* out,
                                            int n_elem) {
  const double* batch_in = in + n_elem * blockIdx.x;
  double* batch_out = out + (n_elem - 1) * blockIdx.x;

  for (int i = threadIdx.x; i < n_elem - 1; i += blockDim.x) {
    batch_out[i] = batch_in[i + 1] - batch_in[i];
  }
}

/**
 * @todo: docs
 *
 * @note: if p == 0, we should expect d_ar to be nullptr, and if q == 0 d_ma
 *        to be nullptr (though we don't need to verify it)
 */
void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 const double* d_y, int num_batches, int nobs, int p, int d,
                 int q, int start_ar_lags) {
  auto stream = handle.getStream();
  auto allocator = handle.getDeviceAllocator();

  /* Based on d, differenciate the series or simply copy it
   * Note: the copy is needed because _start_params writes in it */
  double* d_yd;
  int actual_nobs = nobs - d;
  d_yd = (double*)allocator->allocate(
    sizeof(double) * actual_nobs * num_batches, stream);
  if (d == 0) {
    MLCommon::copy(d_yd, d_y, nobs * num_batches, stream);
  } else if (d == 1) {
    const int TPB = nobs > 512 ? 256 : 128;  // Quick heuristics
    _batched_diff_kernel<<<num_batches, TPB, 0, stream>>>(d_y, d_yd, nobs);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }

  // Do the computation of the initial parameters
  _start_params(handle, d_mu, d_ar, d_ma, d_yd, num_batches, actual_nobs, p, d,
                q, start_ar_lags);

  allocator->deallocate(d_yd, sizeof(double) * actual_nobs * num_batches,
                        stream);
}

}  // namespace ML
