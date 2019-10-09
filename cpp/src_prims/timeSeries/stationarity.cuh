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

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <iostream>

#include "common/cuml_allocator.hpp"
#include "linalg/cublas_wrappers.h"
#include "linalg/subtract.h"
#include "stats/mean.h"
#include "utils.h"

namespace MLCommon {

namespace TimeSeries {

// TODO: doc
template <typename DataT>
static bool _is_stationary(const DataT* y_d, size_t n_samples,
                           cudaStream_t stream, cublasHandle_t cublas_handle,
                           DataT pval_threshold) {
  DataT n_samples_f = static_cast<DataT>(n_samples);

  // Compute mean
  DataT* y_mean_d;
  MLCommon::allocate(y_mean_d, 1);
  MLCommon::Stats::mean(y_mean_d, y_d, static_cast<size_t>(1), n_samples, false,
                        true, stream);
  CUDA_CHECK(cudaPeekAtLastError());
  std::cout << "Mean has been computed" << std::endl;
  DataT *y_mean_h = new DataT();
  MLCommon::updateHost(y_mean_h, y_mean_d, 1,
                       stream);  // TODO: copy later to batch transfers
  std::cout << *y_mean_h << std::endl;

  // Null hypothesis: data is stationary around a constant
  DataT* y_cent_d;
  MLCommon::allocate(y_cent_d, n_samples);
  MLCommon::LinAlg::subtractScalar(y_cent_d, y_d, *y_mean_h, n_samples, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << "Data has been centered" << std::endl;

  // Cumulative sum (inclusive scan with + operator)
  DataT* csum_d;
  MLCommon::allocate(csum_d, n_samples);
  thrust::device_ptr<DataT> __y_cent = thrust::device_pointer_cast(y_cent_d);
  thrust::device_ptr<DataT> __csum = thrust::device_pointer_cast(csum_d);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), __y_cent,
                         __y_cent + n_samples, __csum);
  CUDA_CHECK(cudaPeekAtLastError());
  // TODO: maybe no need to do the verification after each api call

  std::cout << "Cumulative sum has been computed" << std::endl;

  // Eq. 11
  DataT* eta_d;
  MLCommon::allocate(eta_d, 1);
  CUBLAS_CHECK(MLCommon::LinAlg::cublasdot(cublas_handle, n_samples, csum_d, 1,
                                           csum_d, 1, eta_d, stream));
  DataT eta_h;
  MLCommon::updateHost(&eta_h, eta_d, 1, stream);
  eta_h /= n_samples_f * n_samples_f;
  std::cout << eta_h << std::endl;

  /****** Eq. 10 ******/
  // From Kwiatkowski et al. referencing Schwert (1989)
  DataT lags_f = ceil(12.0 * pow(n_samples_f / 100.0, 0.25));
  int lags = static_cast<int>(lags_f);

  DataT* s2A_d;
  MLCommon::allocate(s2A_d, 1);
  CUBLAS_CHECK(MLCommon::LinAlg::cublasdot(cublas_handle, n_samples, y_cent_d,
                                           1, y_cent_d, 1, s2A_d, stream));
  DataT s2A_h;
  MLCommon::updateHost(&s2A_h, s2A_d, 1, stream);
  s2A_h /= n_samples_f;

  std::cout << "s2A: " << s2A_h << std::endl;

  DataT* s2B_partial_d;
  MLCommon::allocate(s2B_partial_d, lags);
  for (int k = 1; k < lags + 1; k++) {
    CUBLAS_CHECK(MLCommon::LinAlg::cublasdot(cublas_handle, n_samples - k,
                                             y_cent_d, 1, y_cent_d + k, 1,
                                             s2B_partial_d + k - 1, stream));
  }
  DataT s2B_partial_h[lags];
  MLCommon::updateHost(s2B_partial_h, s2B_partial_d, lags, stream);
  DataT s2B = 0.0;
  for (int k = 1; k < lags + 1; k++) {
    s2B += (2.0 * (1 - static_cast<DataT>(k) / (lags_f + 1.0)) *
            s2B_partial_h[k - 1]) /
           n_samples_f;
  }

  std::cout << "s2B: " << s2B << std::endl;

  DataT s2 = s2A_h + s2B;
  /********************/

  // Table 1, Kwiatkowski 1992
  const DataT crit_vals[4] = {0.347, 0.463, 0.574, 0.739};
  const DataT pvals[4] = {0.10, 0.05, 0.025, 0.01};

  DataT kpss_stat = eta_h / s2;
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

  CUDA_CHECK(cudaFree(y_mean_d));
  CUDA_CHECK(cudaFree(y_cent_d));
  CUDA_CHECK(cudaFree(csum_d));
  CUDA_CHECK(cudaFree(eta_d));
  CUDA_CHECK(cudaFree(s2A_d));
  CUDA_CHECK(cudaFree(s2B_partial_d));
  delete y_mean_h;

  std::cout << "end of function" << std::endl;

  return pvalue > pval_threshold;
}

// TODO: doc
// TODO: use streams and cuML copy function
template <typename DataT>
void stationarity(const DataT* y, int* d, size_t n_batches, size_t n_samples,
                  cudaStream_t stream, cublasHandle_t cublas_handle,
                  DataT pval_threshold = 0.05) {
  cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);

  // TODO: do this loop in parallel?
  for (size_t i = 0; i < n_batches; i++) {
    std::cout << "Loop #" << i << std::endl;
    DataT* y_d;
    MLCommon::allocate(y_d, n_samples);
    MLCommon::updateDevice(y_d, y + i * n_samples, n_samples, stream);

    /* First the test is performed on the data series */
    if (_is_stationary(y_d, n_samples, stream, cublas_handle, pval_threshold)) {
      d[i] = 0;
    } else {
      std::cout << "Loop #" << i << " diff" << std::endl;

      /* If the first test fails, the differencial series is constructed */
      DataT* ydiff_d;
      MLCommon::allocate(ydiff_d, n_samples - 1);
      MLCommon::LinAlg::subtract(ydiff_d, y_d + 1, y_d, n_samples - 1, stream);
      CUDA_CHECK(cudaPeekAtLastError());

      std::cout<< "ready to launch" << std::endl;

      if (_is_stationary(ydiff_d, n_samples - 1, stream, cublas_handle,
                         pval_threshold)) {
        d[i] = 1;
      } else {
        d[i] = -1;
        // TODO: what to do if non stationary?
      }

      CUDA_CHECK(cudaFree(ydiff_d));
    }

    CUDA_CHECK(cudaFree(y_d));
  }

  for(int i = 0; i < n_batches; i++) {
    std::cout << d[i] << " ";
  }
  std::cout << std::endl;
}

};  //end namespace TimeSeries
};  //end namespace MLCommon