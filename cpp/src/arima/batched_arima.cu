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
                     int p, int d, int q, double* d_params,
                     std::vector<double>& loglike, double* d_vs, bool trans) {
  using std::get;
  using std::vector;

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
  double* d_Tar =
    (double*)allocator->allocate(sizeof(double) * num_batches * p, stream);
  double* d_Tma =
    (double*)allocator->allocate(sizeof(double) * num_batches * q, stream);

  // params -> (mu, ar, ma)
  unpack(d_params, d_mu, d_ar, d_ma, num_batches, p, d, q, handle.getStream());

  CUDA_CHECK(cudaPeekAtLastError());

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

    {
      auto counting = thrust::make_counting_iterator(0);
      // TODO: This for_each should probably go over samples, so batches are in the inner loop.
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
  allocator->deallocate(d_mu, sizeof(double) * num_batches, stream);
  allocator->deallocate(d_ar, sizeof(double) * p * num_batches, stream);
  allocator->deallocate(d_ma, sizeof(double) * q * num_batches, stream);
  allocator->deallocate(d_Tar, sizeof(double) * p * num_batches, stream);
  allocator->deallocate(d_Tma, sizeof(double) * q * num_batches, stream);
  ML::POP_RANGE();
}

/*
 * TODO: quick doc (internal auxiliary function)
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
 * TODO: quick doc (internal auxiliary function)
 * Note: the block id is the batch id and the thread id is the starting index
 */
static __global__ void _batched_ls_set_kernel(double* lagged_series,
                                              const double* data, int ls_width,
                                              int ls_height, int offset, int ld,
                                              int ls_batch_offset,
                                              int ls_batch_stride) {
  const double* batch_in = data + blockIdx.x * ld + offset;
  double* batch_out =
    lagged_series + blockIdx.x * ls_batch_offset + ls_batch_stride;

  for (int lag = 1; lag <= ls_width; lag++) {
    for (int i = threadIdx.x; i < ls_height; i += blockDim.x) {
      batch_out[lag * ls_height + i] = batch_in[i + ls_width - lag];
    }
  }
}

/*
 * TODO: quick doc (internal auxiliary function)
 * Note: the block id is the batch id and the thread id is the starting index
 */
static __global__ void _batched_fill_kernel(double* out, const double* in,
                                            int n_fill) {
  double* batch_out = out + blockIdx.x * n_fill;
  const double batch_in = in[blockIdx.x];
  for (int i = threadIdx.x; i < n_fill; i += blockDim.x) {
    batch_out[i] = batch_in;
  }
}

/**
 * TODO: quick doc (internal auxiliary function)
 *
 * @note: d_yd is mutated!
 *        Also using gelsBatched requires p + q < nobs / 2...
 *        -> if we want to support other cases, we need another batched solver
 *        
 * A quick approach to determine reasonable starting mu, AR, and MA parameters
 */
static void _start_params(double* d_mu, double* d_ar, double* d_ma, double* d_y,
                          int num_batches, int nobs, int p, int d, int q,
                          cublasHandle_t cublas_handle,
                          std::shared_ptr<MLCommon::deviceAllocator> allocator,
                          cudaStream_t stream) {
  const int TPB = nobs > 512 ? 256 : 128;  // Quick heuristics for block size

  // Initialize params
  cudaMemsetAsync(d_mu, 0, sizeof(double) * num_batches, stream);
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
    // do nothing (return later when no deallocation to be done at the end)
  } else if (p != 0) {
    ///@todo statsmodels uses BIC to pick the "best" p for this initial fit.
    // The "best" model is probably p = 1, so we will assume that for now.
    int p_best = 1;

    // Create lagged series set
    int ls_height = nobs - p_best;
    MLCommon::Matrix::BatchedMatrix bm_ls(
      ls_height, p_best, num_batches, cublas_handle, allocator, stream, false);
    _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
      bm_ls.raw_data(), d_y, p_best, ls_height, 0, nobs, 0, p_best * ls_height);
    CUDA_CHECK(cudaPeekAtLastError());

    // Initial AR fit (note: larger dimensions because gels works in-place)
    MLCommon::Matrix::BatchedMatrix bm_ar_fit(
      ls_height, 1, num_batches, cublas_handle, allocator, stream, false);
    batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
      bm_ar_fit.raw_data(), d_y, p_best, nobs, ls_height);

    // Residual if q != 0, initialized as offset y to avoid one kernel call
    MLCommon::Matrix::BatchedMatrix bm_residual(q != 0 ? ls_height : 1, 1,
                                                num_batches, cublas_handle,
                                                allocator, stream, false);
    if (q != 0) {
      MLCommon::copy(bm_residual.raw_data(), bm_ar_fit.raw_data(),
                     ls_height * num_batches, stream);
    }

    ///@todo check that p_best < n_obs / 2
    // Note: this overwrites bm_ls
    int ar_fit_info;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
      cublas_handle, CUBLAS_OP_N, ls_height, p_best, 1, bm_ls.data(), ls_height,
      bm_ar_fit.data(), ls_height, &ar_fit_info, nullptr, num_batches));
    ///@todo raise exception if info < 0? (need to sync stream)

    if (q == 0) {
      // Note: works only for p_best == 1 ; what to do here when p_best > 1?
      // Fill AR parameters with ar_fit
      // TODO: matrix-vector op?
      _batched_fill_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ar, bm_ar_fit.raw_data(), p);
      CUDA_CHECK(cudaPeekAtLastError());
    } else {
      // Compute residual (technically a gemv but we're missing a col-major
      // batched gemv if I'm correct)
      double alpha = -1.0;
      double beta = 1.0;
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgemmBatched(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, ls_height, 1, p_best, &alpha,
        bm_ls.data(), ls_height, bm_ar_fit.data(), p_best, &beta,
        bm_residual.data(), ls_height, num_batches, stream));

      // TODO: useless? At the moment p_best == 1 and later we can fix p_best
      // to the min of p and whetever value works for p_best
      if (p < p_best) {
        throw std::runtime_error("p must be greater than p_best");
      }
      int p_diff = p - p_best;

      // Create matrices made of the concatenation of lagged sets for ar terms
      // and the residual respectively, side by side
      int ls_ar_res_height = ls_height - q - p_diff;
      int ls_res_size = ls_ar_res_height * q;
      int ls_ar_size = ls_ar_res_height * p;
      int ls_ar_res_size = ls_res_size + ls_ar_size;
      MLCommon::Matrix::BatchedMatrix bm_ls_ar_res(ls_ar_res_height, p + q,
                                                   num_batches, cublas_handle,
                                                   allocator, stream, false);
      _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_ls_ar_res.raw_data(), bm_residual.raw_data(), q, ls_ar_res_height,
        p_diff, ls_height, ls_ar_size, ls_ar_res_size);
      CUDA_CHECK(cudaPeekAtLastError());
      _batched_ls_set_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_ls_ar_res.raw_data(), d_y, p, ls_ar_res_height, q, nobs, 0,
        ls_ar_res_size);
      CUDA_CHECK(cudaPeekAtLastError());

      // ARMA fit (note: larger dimensions because gels works in-place)
      MLCommon::Matrix::BatchedMatrix bm_arma_fit(ls_ar_res_height, 1,
                                                  num_batches, cublas_handle,
                                                  allocator, stream, false);
      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        bm_arma_fit.raw_data(), d_y, p_best + q + p_diff, nobs,
        nobs - ls_ar_res_height);

      ///@todo check that p + q < n_obs / 2
      // Note: this overwrites bm_ls
      int arma_fit_info;
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
        cublas_handle, CUBLAS_OP_N, ls_ar_res_height, p + q, 1,
        bm_ls_ar_res.data(), ls_ar_res_height, bm_arma_fit.data(),
        ls_ar_res_height, &arma_fit_info, nullptr, num_batches));
      ///@todo raise exception if info < 0? (need to sync stream)

      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ar, bm_arma_fit.raw_data(), 0, p + q, p);
      batched_offset_copy_kernel<<<num_batches, TPB, 0, stream>>>(
        d_ma, bm_arma_fit.raw_data(), p, p + q, q);
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
                 int q) {
  const auto& handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();
  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

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
  _start_params(d_mu, d_ar, d_ma, d_yd, num_batches, actual_nobs, p, d, q,
                cublas_handle, allocator, stream);

  allocator->deallocate(d_yd, sizeof(double) * actual_nobs * num_batches,
                        stream);
}

}  // namespace ML
