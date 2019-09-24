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

#include <common/nvtx.hpp>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuML.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>

namespace ML {

using std::vector;

void residual(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
              int d, int q, double* d_params, double*& d_vs, bool trans) {
  std::vector<double> loglike;
  batched_loglike(handle, d_y, num_batches, nobs, p, d, q, d_params, loglike,
                  d_vs, trans);
}

void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int batch_size, int nobs, double* d_y_diff, double* d_vs,
              double* d_params, double* d_y_fc) {
  auto alloc = handle.getDeviceAllocator();
  const auto stream = handle.getStream();
  double* d_y_ = (double*)alloc->allocate((p + num_steps) * batch_size, stream);
  double* d_vs_ =
    (double*)alloc->allocate((q + num_steps) * batch_size, stream);
  const auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(counting, counting + batch_size, [=] __device__(int bid) {
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

  thrust::for_each(counting, counting + batch_size, [=] __device__(int bid) {
    int N = p + d + q;
    auto mu_ib = d_params[N * bid];
    double ar_sum = 0.0;
    for (int ip = 0; ip < p; ip++) {
      double ar_i = d_params[N * bid + d + ip];
      ar_sum += ar_i;
    }
    double mu_star = mu_ib * (1 - ar_sum);
    printf("%d: mu_star=%f\n", bid, mu_star);
    for (int i = 0; i < num_steps; i++) {
      auto it = num_steps * bid + i;
      d_y_fc[it] = mu_star;
      if (p > 0) {
        double dot_ar_y = 0.0;
        for (int ip = 0; ip < p; ip++) {
          dot_ar_y +=
            d_params[N * bid + d + ip] * d_y_[(p + num_steps) * bid + ip];
        }
        d_y_fc[it] += dot_ar_y;
        printf("(%d) p:d_y_fc[%d]=%f\n", bid, it, d_y_fc[it]);
      }
      if (q > 0) {
        double dot_ma_y = 0.0;
        for (int iq = 0; iq < q; iq++) {
          dot_ma_y +=
            d_params[N * bid + d + p + iq] * d_vs_[(q + num_steps) * bid + iq];
        }
        d_y_fc[it] += dot_ma_y;
        printf("(%d) q:d_y_fc[%d]=%f\n", bid, it, d_y_fc[it]);
      }
      if (p > 0) {
        d_y_[(p + num_steps) * bid + i + p] = d_y_fc[it];
      }
    }
  });
}

void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                       int nobs, int p, int d, int q, double* d_params,
                       double*& d_vs, double* d_y_p) {
  residual(handle, d_y, num_batches, nobs, p, d, q, d_params, d_vs, false);

  double* d_y_diff;

  if (d == 0) {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(counting, counting + num_batches, [=] __device__(int bid) {
      for (int i = 0; i < nobs; i++) {
        int it = bid * nobs + i;
        d_y_p[it] = d_y[it] - d_vs[it];
      }
    });
  } else {
    d_y_diff = (double*)handle.getDeviceAllocator()->allocate(
      sizeof(double) * num_batches * (nobs - 1), handle.getStream());
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(counting, counting + num_batches, [=] __device__(int bid) {
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
    forecast(handle, 1, p, d, q, num_batches, nobs, d_y_diff, d_vs, d_params,
             d_y_fc);
    // add forecast to original signal (undo-differencing) and append to end of in-sample prediction
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(counting, counting + num_batches, [=] __device__(int bid) {
      d_y_p[bid * nobs + (nobs - 1)] =
        d_y[bid * nobs + (nobs - 1)] + d_y_fc[bid];
    });
  }
}

void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_params,
                     std::vector<double>& loglike, double*& d_vs, bool trans) {
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

  // params -> (mu, ar, ma)
  unpack(d_params, d_mu, d_ar, d_ma, num_batches, p, d, q);

  CUDA_CHECK(cudaPeekAtLastError());

  // Transformed parameters
  double* d_Tar = nullptr;
  double* d_Tma = nullptr;

  if (trans) {
    batched_jones_transform(handle, p, q, num_batches, false, d_ar, d_ma, d_Tar,
                            d_Tma);
  } else {
    // non-transformed case: just use original parameters
    d_Tar = d_ar;
    d_Tma = d_ma;
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
      thrust::for_each(
        counting, counting + num_batches, [=] __device__(int bid) {
          double mu_ib = d_mu[bid];
          for (int i = 0; i < nobs - 1; i++) {
            // diff and center (with `mu` parameter)
            y_diff[bid * (nobs - 1) + i] =
              (d_y[bid * nobs + i + 1] - d_y[bid * nobs + i]) - mu_ib;
          }
        });
    }

    batched_kalman_filter(handle, y_diff, nobs - d, d_Tar, d_Tma, p, q,
                          num_batches, loglike, d_vs);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }

  allocator->deallocate(d_mu, num_batches, stream);
  allocator->deallocate(d_ar, p * num_batches, stream);
  allocator->deallocate(d_ma, q * num_batches, stream);
  if (trans) {
    allocator->deallocate(d_Tar, p * num_batches, stream);
    allocator->deallocate(d_Tma, q * num_batches, stream);
  }

  ML::POP_RANGE();
}

}  // namespace ML
