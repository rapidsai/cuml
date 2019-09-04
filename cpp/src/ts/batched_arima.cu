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

#include "batched_arima.h"
#include "batched_kalman.h"

#include <common/nvtx.hpp>

namespace ML {

void batched_loglike(double* y, int num_batches, int nobs, int p, int d, int q,
                     double* params, std::vector<double>& loglike_b,
                     bool trans) {
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
      mu.at(ib) = params[ib * N];
    }
    for (int ip = 0; ip < p; ip++) {
      ar.at(p * ib + ip) = params[ib * N + d + ip];
    }
    for (int iq = 0; iq < q; iq++) {
      ma.at(q * ib + iq) = params[ib * N + d + p + iq];
    }
  }

  // Transformed parameters
  vector<double> Tar;
  vector<double> Tma;

  if (trans) {
    batched_jones_transform(p, q, num_batches, false, ar, ma, Tar, Tma);
  } else {
    // non-transformed case: just use original parameters
    Tar = ar;
    Tma = ma;
  }

  std::vector<std::vector<double>> vs_b;
  if (d == 0) {
    // no diff
    batched_kalman_filter(y, nobs, Tar, Tma, p, q, num_batches, loglike_b,
                          vs_b);
  } else if (d == 1) {
    // diff and center (with `mu`):
    std::vector<double> y_diff(num_batches * (nobs - 1));
    for (int ib = 0; ib < num_batches; ib++) {
      auto mu_ib = mu[ib];
      for (int i = 0; i < nobs - 1; i++) {
        y_diff[ib * (nobs - 1) + i] =
          (y[ib * nobs + i + 1] - y[ib * nobs + i]) - mu_ib;
      }
    }

    batched_kalman_filter(y_diff.data(), nobs - d, Tar, Tma, p, q, num_batches,
                          loglike_b, vs_b);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }
  ML::POP_RANGE();
}
}  // namespace ML
