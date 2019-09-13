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

namespace ML {

void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* h_params,
                     std::vector<double>& loglike, bool trans) {
  using std::get;
  using std::vector;

  ML::PUSH_RANGE(__FUNCTION__);

  // unpack parameters
  vector<double> mu(num_batches);
  vector<double> ar(num_batches * p);
  vector<double> ma(num_batches * q);

  int N = (p + d + q);

  for (int ib = 0; ib < num_batches; ib++) {
    if (d > 0) {
      mu.at(ib) = h_params[ib * N];
    }
    for (int ip = 0; ip < p; ip++) {
      ar.at(p * ib + ip) = h_params[ib * N + d + ip];
    }
    for (int iq = 0; iq < q; iq++) {
      ma.at(q * ib + iq) = h_params[ib * N + d + p + iq];
    }
  }

  // Transformed parameters
  vector<double> Tar;
  vector<double> Tma;

  if (trans) {
    batched_jones_transform(handle, p, q, num_batches, false, ar, ma, Tar, Tma);
  } else {
    // non-transformed case: just use original parameters
    Tar = ar;
    Tma = ma;
  }

  std::vector<std::vector<double>> vs_b;
  if (d == 0) {
    // no diff
    batched_kalman_filter(handle, d_y, nobs, Tar, Tma, p, q, num_batches,
                          loglike, vs_b);
  } else if (d == 1) {
    ////////////////////////////////////////////////////////////
    // diff and center (with `mu`):
    ////////////////////////////////////////////////////////////

    // make device array and pointer
    thrust::device_vector<double> y_diff(num_batches * (nobs - 1));
    thrust::device_vector<double> d_mu = mu;
    double* y_diff_ptr = thrust::raw_pointer_cast(y_diff.data());
    double* d_mu_ptr = thrust::raw_pointer_cast(d_mu.data());
    auto counting = thrust::make_counting_iterator(0);
    // TODO: This for_each should probably go over samples, so batches are in the inner loop.
    thrust::for_each(counting, counting + num_batches, [=] __device__(int bid) {
      double mu_ib = d_mu_ptr[bid];
      for (int i = 0; i < nobs - 1; i++) {
        // diff and center (with `mu` parameter)
        y_diff_ptr[bid * (nobs - 1) + i] =
          (d_y[bid * nobs + i + 1] - d_y[bid * nobs + i]) - mu_ib;
      }
    });

    batched_kalman_filter(handle, y_diff_ptr, nobs - d, Tar, Tma, p, q,
                          num_batches, loglike, vs_b);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }
  ML::POP_RANGE();
}
}  // namespace ML
