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

void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                       int nobs, int p, int d, int q, double* d_params,
                       double*& d_y_p) {}

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
  double* d_Tar;
  double* d_Tma;

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
    allocator->deallocate(d_Tma, q * num_batches, stream);
    allocator->deallocate(d_Tar, q * num_batches, stream);
  }

  ML::POP_RANGE();
}

void update_host(cumlHandle& handle, double* d_vs, int N, double* h_vs) {
  printf("Copying %p -> %p\n", d_vs, h_vs);
  MLCommon::updateHost(h_vs, d_vs, N, handle.getStream());
}

}  // namespace ML
