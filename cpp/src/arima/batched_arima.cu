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

    allocator->deallocate(y_diff, sizeof(double) * num_batches * (nobs - 1),
                          stream);
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

/**
 * TODO: doc
 *
 * @note: bm_y is mutated!
 *        
 * Determine reasonable starting mu, AR, and MA parameters
 */
static void _start_params(cumlHandle& handle, double* d_mu, double* d_ar,
                          double* d_ma,
                          MLCommon::Matrix::BatchedMatrix<double>& bm_y,
                          int num_batches, int nobs, int p, int d, int q) {
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Initialize params
  cudaMemsetAsync(d_ar, 0, sizeof(double) * p * num_batches, stream);
  cudaMemsetAsync(d_ma, 0, sizeof(double) * q * num_batches, stream);

  if (d > 0) {
    // Compute means and write them in mu
    MLCommon::Stats::mean(d_mu, bm_y.raw_data(), num_batches, nobs, false,
                          false, stream);

    // Center the series around their means in-place
    MLCommon::LinAlg::matrixVectorOp(
      bm_y.raw_data(), bm_y.raw_data(), d_mu, num_batches, nobs, false, true,
      [] __device__(double a, double b) { return a - b; }, stream);
  }

  if (p == 0 && q == 0) {
    return;
  } else if (p != 0) {
    /* Note: p_lags fixed to p to avoid non-full-rank matrix issues */
    int p_lags = p;

    // Create lagged y
    int ls_height = nobs - p_lags;
    MLCommon::Matrix::BatchedMatrix<double> bm_ls =
      MLCommon::Matrix::b_lagged_mat(bm_y, p_lags);

    // Initial AR fit (note: larger dimensions because gels works in-place)
    MLCommon::Matrix::BatchedMatrix<double> bm_ar_fit =
      MLCommon::Matrix::b_2dcopy(bm_y, p_lags, 0, ls_height, 1);

    // Residual if q != 0, initialized as offset y to avoid one kernel call
    MLCommon::Matrix::BatchedMatrix<double> bm_residual(
      q != 0 ? ls_height : 1, 1, num_batches, cublas_handle, allocator, stream,
      false);
    if (q != 0) {
      MLCommon::copy(bm_residual.raw_data(), bm_ar_fit.raw_data(),
                     ls_height * num_batches, stream);
    }

    MLCommon::Matrix::b_gels(bm_ls, bm_ar_fit);

    if (q == 0) {
      // Note: if q == 0, we must always choose p_lags == p!
      MLCommon::Matrix::batched_2dcopy_kernel<<<num_batches, p, 0, stream>>>(
        bm_ar_fit.raw_data(), d_ar, 0, 0, ls_height, 1, p, 1);
      CUDA_CHECK(cudaPeekAtLastError());
    } else {
      // Compute residual (technically a gemv)
      MLCommon::Matrix::b_gemm(false, false, ls_height, 1, p_lags, -1.0, bm_ls,
                               bm_ar_fit, 1.0, bm_residual);

      // Create matrices made of the concatenation of lagged sets of y and the
      // residual respectively, side by side
      int arma_fit_offset = std::max(p_lags + q, p);
      int ls_ar_res_height = nobs - arma_fit_offset;
      int ar_offset = (p < p_lags + q) ? (p_lags + q - p) : 0;
      int res_offset = (p < p_lags + q) ? 0 : p - p_lags - q;
      MLCommon::Matrix::BatchedMatrix<double> bm_ls_ar_res(
        ls_ar_res_height, p + q, num_batches, cublas_handle, allocator, stream,
        false);
      MLCommon::Matrix::b_lagged_mat(bm_y, bm_ls_ar_res, p, ls_ar_res_height,
                                     ar_offset, 0);
      MLCommon::Matrix::b_lagged_mat(bm_residual, bm_ls_ar_res, q,
                                     ls_ar_res_height, res_offset,
                                     ls_ar_res_height * p);

      // ARMA fit (note: larger dimensions because gels works in-place)
      MLCommon::Matrix::BatchedMatrix<double> bm_arma_fit =
        MLCommon::Matrix::b_2dcopy(bm_y, arma_fit_offset, 0, ls_ar_res_height,
                                   1);
      MLCommon::Matrix::b_gels(bm_ls_ar_res, bm_arma_fit);

      // Note: calling directly the kernel as there is not yet a way to wrap
      // existing device pointers in a batched matrix
      MLCommon::Matrix::batched_2dcopy_kernel<<<num_batches, p, 0, stream>>>(
        bm_arma_fit.raw_data(), d_ar, 0, 0, ls_ar_res_height, 1, p, 1);
      CUDA_CHECK(cudaPeekAtLastError());
      MLCommon::Matrix::batched_2dcopy_kernel<<<num_batches, q, 0, stream>>>(
        bm_arma_fit.raw_data(), d_ma, p, 0, ls_ar_res_height, 1, q, 1);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  } else {  // p == 0 && q > 0
    ///@todo See how `statsmodels` handles this case

    // Set MA params to -1
    thrust::device_ptr<double> __ma = thrust::device_pointer_cast(d_ma);
    thrust::fill(thrust::cuda::par.on(stream), __ma, __ma + q * num_batches,
                 -1.0);
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
                 int q) {
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  /* Based on d, differenciate the series or simply copy it
   * Note: the copy is needed because _start_params writes in it */
  int actual_nobs = nobs - d;
  MLCommon::Matrix::BatchedMatrix<double> bm_yd(
    actual_nobs, 1, num_batches, cublas_handle, allocator, stream, false);
  if (d == 0) {
    MLCommon::copy(bm_yd.raw_data(), d_y, nobs * num_batches, stream);
  } else if (d == 1) {
    const int TPB = nobs > 512 ? 256 : 128;  // Quick heuristics
    MLCommon::Matrix::batched_diff_kernel<<<num_batches, TPB, 0, stream>>>(
      d_y, bm_yd.raw_data(), nobs);
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    throw std::runtime_error(
      "Not supported difference parameter. Required: d=0 or 1");
  }

  // Do the computation of the initial parameters
  _start_params(handle, d_mu, d_ar, d_ma, bm_yd, num_batches, actual_nobs, p, d,
                q);
}

}  // namespace ML
