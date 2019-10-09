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
/**
* @file stationarity.h
* @brief TODO

*/

// TODO: OpenMP loops?

#include <math.h>
#include <iostream>
#include <stdexcept>
#include <vector>


// TODO: doc
template <typename DataT>
static bool _is_stationary(const DataT* y, size_t i, size_t n_batches,
                           size_t n_samples, DataT pval_threshold) {
  DataT n_samples_f = static_cast<DataT>(n_samples);

  // Compute mean
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t j = 0; j < n_samples; j++) {
    sum += static_cast<double>(y[i * n_samples + j]);
  }
  DataT mean = static_cast<DataT>(sum) / n_samples_f;

  // Null hypothesis: data is stationary around a constant
  std::vector<DataT> y_cent = std::vector<DataT>(n_samples);
#pragma omp parallel for
  for (size_t j = 0; j < n_samples; j++) {
    y_cent[j] = y[i * n_samples + j] - mean;
  }

  std::vector<DataT> csum = std::vector<DataT>(n_samples);
  csum[0] = y_cent[0];
  for (size_t j = 1; j < n_samples; j++) {
    csum[j] = csum[j - 1] + y_cent[j];
  }

  // Eq. 11
  DataT eta = 0.0;
#pragma omp parallel for reduction(+ : eta)
  for (size_t j = 0; j < n_samples; j++) {
    eta += csum[j] * csum[j];
  }
  eta /= n_samples_f * n_samples_f;

  /****** Eq. 10 ******/
  // From Kwiatkowski et al. referencing Schwert (1989)
  DataT lags_f = ceil(12.0 * pow(n_samples_f / 100.0, 0.25));
  int lags = static_cast<int>(lags_f);

  DataT s2_A = 0.0;
#pragma omp parallel for reduction(+ : s2_A)
  for (size_t j = 0; j < n_samples; j++) {
    s2_A += y_cent[j] * y_cent[j];
  }
  s2_A /= n_samples_f;

  DataT s2_B = 0.0;
  for (int k = 1; k < lags + 1; k++) {
    DataT prod = 0;
#pragma omp parallel for reduction(+ : prod)
    for (int j = 0; j < n_samples - k; j++) {
      prod += y_cent[j] * y_cent[j + k];
    }
    s2_B +=
      (2.0 * (1 - static_cast<DataT>(k) / (lags_f + 1.0)) * prod) / n_samples_f;
  }

  DataT s2 = s2_A + s2_B;
  /********************/

  // Table 1, Kwiatkowski 1992
  const DataT crit_vals[4] = {0.347, 0.463, 0.574, 0.739};
  const DataT pvals[4] = {0.10, 0.05, 0.025, 0.01};

  DataT kpss_stat = eta / s2;
  DataT pvalue = pvals[0];
  for (int k = 0; k < 3 && kpss_stat < crit_vals[k + 1]; k++) {
    if (kpss_stat >= crit_vals[k]) {
      pvalue = pvals[k] + (pvals[k + 1] - pvals[k]) *
                            (kpss_stat - crit_vals[k]) /
                            (crit_vals[k + 1] - crit_vals[k]);
    }
  }
  if (kpss_stat >= crit_vals[3]) {
    pvalue = pvals[3];
  }

  return pvalue > pval_threshold;
}

namespace MLCommon {

namespace TimeSeries {

// TODO: doc
// Note: y is column-major
template <typename DataT>
void stationarity(const DataT* y, int* d, size_t n_batches, size_t n_samples,
                  DataT pval_threshold = 0.05) {
  for (size_t i = 0; i < n_batches; i++) {
    /* First the test is performed on the data series
         * No data is copied so the shape and column are passed */
    if (_is_stationary(y, i, n_batches, n_samples, pval_threshold)) {
      d[i] = 0;
      continue;
    }
    /* If the first test fails, the differencial series is constructed
         * The new layout has only one column and its size is one element
         * smaller than the series */
    DataT y_diff[n_samples - 1];
#pragma omp parallel for
    for (int j = 0; j < n_samples - 1; j++) {
      y_diff[j] = y[i * n_samples + j + 1] - y[i * n_samples + j];
    }
    if (_is_stationary(y_diff, 0, 1, n_samples - 1, pval_threshold)) {
      d[i] = 1;
    } else {
      throw std::runtime_error("Stationarity failed for d=0 or 1.");
    }
  }
}

};  //end namespace TimeSeries
};  //end namespace MLCommon