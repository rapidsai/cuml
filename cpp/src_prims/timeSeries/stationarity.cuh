/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
 * @file stationarity.cuh
 * @brief Test a batched times series for stationarity
 * Reference: 'Testing the null hypothesis of stationarity against the
 * alternative of a unit root', Kwiatkowski et al. 1992.
 * See https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html#kpss
 * for additional details.
 */

#pragma once

#include "arima_helpers.cuh"

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cmath>
#include <vector>

namespace MLCommon {

namespace TimeSeries {

/**
 * @brief Auxiliary function to decide the block dimensions
 *
 * @tparam     TPB        Threads per block
 * @tparam     IdxT       Integer type of the indices
 * @param[in]  batch_size Number of batches in the input data
 * @return                The block dimensions
 */
template <int TPB, typename IdxT>
static inline dim3 choose_block_dims(IdxT batch_size)
{
  uint tpb_y = batch_size > 8 ? 4 : 1;
  dim3 block(TPB / tpb_y, tpb_y);
  return block;
}

/**
 * @brief Auxiliary kernel for the computation of s2 (Kwiatkowski 1992 eq.10)
 *
 * @details The kernel computes partial sums for the term of equation 10.
 *          A reduction is performed to get the full sum.
 *          If y is a series and z the accumulator, this kernel computes:
 *          z[t] = w(k) * sum from k=1 to lags of y[t]*y[t+k]
 *          padded with zeros and where w(k)=2/ns*(1-k/(lags+1))
 *
 * @note The accumulator has one extra element per series, which avoids some
 *       index calculations and it has the right size anyway since it is
 *       recycled for another operation.
 *       Performance note: this kernel could use shared memory
 *
 * @tparam      DataT        Scalar type of the data (float or double)
 * @tparam      IdxT         Integer type of the indices
 * @param[out]  accumulator  Output matrix that holds the partial sums
 * @param[in]   data         Source data
 * @param[in]   lags         Number of lags
 * @param[in]   batch_size   Number of columns in the data
 * @param[in]   n_obs        Number of rows in the data
 * @param[in]   coeff_a      Part of the calculation for w(k)=a*k+b
 * @param[in]   coeff_b      Part of the calculation for w(k)=a*k+b
 */
template <typename DataT, typename IdxT>
CUML_KERNEL void s2B_accumulation_kernel(DataT* accumulator,
                                         const DataT* data,
                                         IdxT lags,
                                         IdxT batch_size,
                                         IdxT n_obs,
                                         DataT coeff_a,
                                         DataT coeff_b)
{
  IdxT sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT batch_idx  = blockIdx.y * blockDim.y + threadIdx.y;

  if (sample_idx < n_obs && batch_idx < batch_size) {
    IdxT idx         = batch_idx * n_obs + sample_idx;
    accumulator[idx] = static_cast<DataT>(0.0);
    for (IdxT k = 1; k <= lags && sample_idx < n_obs - k; k++) {
      DataT dp    = data[idx] * data[idx + k];
      DataT coeff = coeff_a * static_cast<DataT>(k) + coeff_b;
      accumulator[idx] += coeff * dp;
    }
  }
}

/**
 * @brief Kernel to decide whether the series are stationary or not
 *
 * @details The kernel uses the results of the different equations to
 *          make the final decision for each series.
 *
 * @tparam      DataT           Scalar type of the data (float or double)
 * @tparam      IdxT            Integer type of the indices
 * @param[out]  results         Boolean array to store the results.
 * @param[in]   s2A             1st component of eq.10 (before division by ns)
 * @param[in]   s2B             2nd component of eq.10
 * @param[in]   eta             Eq.11 (before division by ns^2)
 * @param[in]   batch_size      Number of batches
 * @param[in]   n_obs_f         Number of samples (floating-point number)
 * @param[in]   pval_threshold  P-value threshold above which the series is
 *                              considered stationary
 */
template <typename DataT, typename IdxT>
CUML_KERNEL void kpss_stationarity_check_kernel(bool* results,
                                                const DataT* s2A,
                                                const DataT* s2B,
                                                const DataT* eta,
                                                IdxT batch_size,
                                                DataT n_obs_f,
                                                DataT pval_threshold)
{
  // Table 1, Kwiatkowski 1992
  const DataT crit_vals[4] = {0.347, 0.463, 0.574, 0.739};
  const DataT pvals[4]     = {0.10, 0.05, 0.025, 0.01};

  IdxT i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < batch_size) {
    DataT s2Ai = s2A[i];
    DataT etai = eta[i];
    DataT s2Bi = s2B[i];

    s2Ai /= n_obs_f;
    etai /= n_obs_f * n_obs_f;

    DataT kpss_stat = etai / (s2Ai + s2Bi);

    // Interpolate the pvalue (y) based on the kpss stat (x)
    DataT pvalue = pvals[0];
#pragma unroll
    for (IdxT k = 0; k < 3; k++) {
      if (kpss_stat >= crit_vals[k] && kpss_stat < crit_vals[k + 1]) {
        pvalue = pvals[k] + (pvals[k + 1] - pvals[k]) * (kpss_stat - crit_vals[k]) /
                              (crit_vals[k + 1] - crit_vals[k]);
      }
    }
    if (kpss_stat >= crit_vals[3]) { pvalue = pvals[3]; }

    // A higher pvalue means a higher chance that the data is stationary
    results[i] = (pvalue > pval_threshold);
  }
}

/* A structure that defines a function to get the column of an element of
 * a matrix from its index. This makes possible a 2d scan with thrust.
 * Found in thrust/examples/scan_matrix_by_rows.cu
 */
template <typename IdxT>
struct which_col {
  IdxT col_length;
  __host__ __device__ which_col(IdxT col_length_) : col_length(col_length_) {}
  __host__ __device__ IdxT operator()(IdxT idx) const { return idx / col_length; }
};

/**
 * @brief Applies the KPSS stationarity test to the differenced series
 *
 * @details The following algorithm is based on Kwiatkowski 1992:
 *          - Center each series around its mean
 *          - Calculate s^2 (eq. 10) and eta (eq. 11)
 *          - Deduce the p-value and compare against the threshold
 *
 * @tparam      DataT           Scalar type of the data (float or double)
 * @tparam      IdxT            Integer type of the indices
 * @param[in]   d_y             Input data
 * @param[out]  results         Boolean array to store the results of the test
 * @param[in]   batch_size      Batch size
 * @param[in]   n_obs           Number of observations
 * @param[in]   stream          CUDA stream
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary
 */
template <typename DataT, typename IdxT>
static void _kpss_test(const DataT* d_y,
                       bool* results,
                       IdxT batch_size,
                       IdxT n_obs,
                       cudaStream_t stream,
                       DataT pval_threshold)
{
  constexpr int TPB = 256;
  dim3 block        = choose_block_dims<TPB>(batch_size);
  dim3 grid(raft::ceildiv<IdxT>(n_obs, block.x), raft::ceildiv<IdxT>(batch_size, block.y));

  DataT n_obs_f = static_cast<DataT>(n_obs);

  // Compute mean
  rmm::device_uvector<DataT> y_means(batch_size, stream);
  raft::stats::mean<false>(y_means.data(), d_y, batch_size, n_obs, false, stream);

  // Center the data around its mean
  rmm::device_uvector<DataT> y_cent(batch_size * n_obs, stream);
  raft::linalg::matrixVectorOp<false, true>(
    y_cent.data(),
    d_y,
    y_means.data(),
    batch_size,
    n_obs,
    [] __device__(DataT a, DataT b) { return a - b; },
    stream);

  // This calculates the first sum in eq. 10 (first part of s^2)
  rmm::device_uvector<DataT> s2A(batch_size, stream);
  raft::linalg::reduce<false, false>(s2A.data(),
                                     y_cent.data(),
                                     batch_size,
                                     n_obs,
                                     static_cast<DataT>(0.0),
                                     stream,
                                     false,
                                     raft::L2Op<DataT>(),
                                     raft::add_op());

  // From Kwiatkowski et al. referencing Schwert (1989)
  DataT lags_f = ceil(12.0 * pow(n_obs_f / 100.0, 0.25));
  IdxT lags    = static_cast<IdxT>(lags_f);

  /* This accumulator will be used for both the calculation of s2B, and later
   * the cumulative sum or y centered */
  rmm::device_uvector<DataT> accumulator(batch_size * n_obs, stream);

  // This calculates the second sum in eq. 10 (second part of s^2)
  DataT coeff_base = static_cast<DataT>(2.0) / n_obs_f;
  s2B_accumulation_kernel<<<grid, block, 0, stream>>>(
    accumulator.data(),
    y_cent.data(),
    lags,
    batch_size,
    n_obs,
    -coeff_base / (lags_f + static_cast<DataT>(1.0)),
    coeff_base);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  rmm::device_uvector<DataT> s2B(batch_size, stream);
  raft::linalg::reduce<false, false>(
    s2B.data(), accumulator.data(), batch_size, n_obs, static_cast<DataT>(0.0), stream, false);

  // Cumulative sum (inclusive scan with + operator)
  thrust::counting_iterator<IdxT> c_first(0);
  thrust::transform_iterator<which_col<IdxT>, thrust::counting_iterator<IdxT>> t_first(
    c_first, which_col<IdxT>(n_obs));
  thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                                t_first,
                                t_first + batch_size * n_obs,
                                y_cent.data(),
                                accumulator.data());

  // Eq. 11 (eta)
  rmm::device_uvector<DataT> eta(batch_size, stream);
  raft::linalg::reduce<false, false>(eta.data(),
                                     accumulator.data(),
                                     batch_size,
                                     n_obs,
                                     static_cast<DataT>(0.0),
                                     stream,
                                     false,
                                     raft::L2Op<DataT>(),
                                     raft::add_op());

  /* The following kernel will decide whether each series is stationary based on
   * s^2 and eta */
  kpss_stationarity_check_kernel<<<raft::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
    results, s2A.data(), s2B.data(), eta.data(), batch_size, n_obs_f, pval_threshold);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Perform the KPSS stationarity test on the data differenced according
 *        to the given order
 *
 * @tparam      DataT           Scalar type of the data (float or double)
 * @tparam      IdxT            Integer type of the indices
 * @param[in]   d_y             Input data
 * @param[out]  results         Boolean device array to store the results
 * @param[in]   batch_size      Batch size
 * @param[in]   n_obs           Number of observations
 * @param[in]   d               Order of simple differencing
 * @param[out]  D               Order of seasonal differencing
 * @param[in]   s               Seasonal period if D > 0 (else unused)
 * @param[in]   stream          CUDA stream
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary
 */
template <typename DataT, typename IdxT>
void kpss_test(const DataT* d_y,
               bool* results,
               IdxT batch_size,
               IdxT n_obs,
               int d,
               int D,
               int s,
               cudaStream_t stream,
               DataT pval_threshold = 0.05)
{
  const DataT* d_y_diff;

  int n_obs_diff = n_obs - d - s * D;

  // Compute differenced series
  rmm::device_uvector<DataT> diff_buffer(0, stream);
  if (d == 0 && D == 0) {
    d_y_diff = d_y;
  } else {
    diff_buffer.resize(batch_size * n_obs_diff, stream);
    prepare_data(diff_buffer.data(), d_y, batch_size, n_obs, d, D, s, stream);
    d_y_diff = diff_buffer.data();
  }

  // KPSS test
  _kpss_test(d_y_diff, results, batch_size, n_obs_diff, stream, pval_threshold);
}

};  // end namespace TimeSeries
};  // end namespace MLCommon
