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
* @brief Compute the recommended trend parameter for a batched series.
* Reference: 'Testing the null hypothesis of stationarity against the
* alternative of a unit root', Kwiatkowski et al. 1992.
* See https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html#kpss
* for additional details.
*/

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <vector>

#include <common/cudart_utils.h>
#include "cuml/common/cuml_allocator.hpp"
#include "linalg/cublas_wrappers.h"
#include "linalg/matrix_vector_op.h"
#include "linalg/reduce.h"
#include "stats/mean.h"

namespace MLCommon {

namespace TimeSeries {

/**
* @brief Auxiliary function to decide the block dimensions
*
* @tparam     TPB        Threads per block
* @tparam     IdxT       Integer type of the indices
* @param[in]  n_batches  Number of batches in the input data
* @return                The block dimensions
*/
template <int TPB, typename IdxT>
static inline dim3 choose_block_dims(IdxT n_batches) {
  uint tpb_y = n_batches > 8 ? 4 : 1;
  dim3 block(TPB / tpb_y, tpb_y);
  return block;
}

/**
* @brief Kernel to batch the first differences of a selection of series
*
* @details The kernel combines 2 operations: selecting a number of series from
*          the original data and derivating them (calculating the difference of
*          consecutive terms)
*
* @note The number of batches in the input matrix is not known by this function
*       which trusts the gather_map array to hold correct column numbers. The
*       number of samples in the input matrix is one more than in the output.
*
* @tparam      DataT           Scalar type of the data (float or double)
* @tparam      IdxT            Integer type of the indices
* @param[out]  diff            Output matrix
* @param[in]   data            Input matrix
* @param[in]   gather_map      Array that indicates the source column in the
                               input matrix for each column of the output matrix
* @param[in]   n_diff_batches  Number of columns in the output matrix
* @param[in]   n_diff_samples  Number of rows in the output matrix
*/
template <typename DataT, typename IdxT>
static __global__ void gather_diff_kernel(DataT* diff, const DataT* data,
                                          IdxT* gather_map, IdxT n_diff_batches,
                                          IdxT n_diff_samples) {
  IdxT sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (batch_idx < n_diff_batches && sample_idx < n_diff_samples) {
    IdxT source_batch_idx = gather_map[batch_idx];
    IdxT source_location = source_batch_idx * (n_diff_samples + 1) + sample_idx;
    diff[batch_idx * n_diff_samples + sample_idx] =
      data[source_location + 1] - data[source_location];
  }
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
 * @param[in]   n_batches    Number of columns in the data
 * @param[in]   n_samples    Number of rows in the data
 * @param[in]   coeff_a      Part of the calculation for w(k)=a*k+b
 * @param[in]   coeff_b      Part of the calculation for w(k)=a*k+b
*/
template <typename DataT, typename IdxT>
static __global__ void s2B_accumulation_kernel(DataT* accumulator,
                                               const DataT* data, IdxT lags,
                                               IdxT n_batches, IdxT n_samples,
                                               DataT coeff_a, DataT coeff_b) {
  IdxT sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (sample_idx < n_samples && batch_idx < n_batches) {
    IdxT idx = batch_idx * n_samples + sample_idx;
    accumulator[idx] = static_cast<DataT>(0.0);
    for (IdxT k = 1; k <= lags && sample_idx < n_samples - k; k++) {
      DataT dp = data[idx] * data[idx + k];
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
 * @param[in]   n_batches       Number of batches
 * @param[in]   n_samples_f     Number of samples (floating-point number)
 * @param[in]   pval_threshold  P-value threshold above which the series is
 *                              considered stationary
*/
template <typename DataT, typename IdxT>
static __global__ void stationarity_check_kernel(
  bool* results, const DataT* s2A, const DataT* s2B, const DataT* eta,
  IdxT n_batches, DataT n_samples_f, DataT pval_threshold) {
  // Table 1, Kwiatkowski 1992
  const DataT crit_vals[4] = {0.347, 0.463, 0.574, 0.739};
  const DataT pvals[4] = {0.10, 0.05, 0.025, 0.01};

  IdxT i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_batches) {
    DataT s2Ai = s2A[i];
    DataT etai = eta[i];
    DataT s2Bi = s2B[i];

    s2Ai /= n_samples_f;
    etai /= n_samples_f * n_samples_f;

    DataT kpss_stat = etai / (s2Ai + s2Bi);

    // Interpolate the pvalue (y) based on the kpss stat (x)
    DataT pvalue = pvals[0];
#pragma unroll
    for (IdxT k = 0; k < 3; k++) {
      if (kpss_stat >= crit_vals[k] && kpss_stat < crit_vals[k + 1]) {
        pvalue = pvals[k] + (pvals[k + 1] - pvals[k]) *
                              (kpss_stat - crit_vals[k]) /
                              (crit_vals[k + 1] - crit_vals[k]);
      }
    }
    if (kpss_stat >= crit_vals[3]) {
      pvalue = pvals[3];
    }

    // A higher pvalue means a higher chance that the data is stationary
    results[i] = (pvalue > pval_threshold);
  }
}

/* A structure that defines a function to get the column of an element of
 * a matrix from its index. This makes possible a 2d scan with thrust.
 * Found in thrust/examples/scan_matrix_by_rows.cu
 */
template <typename IdxT>
struct which_col : thrust::unary_function<IdxT, IdxT> {
  IdxT col_length;
  __host__ __device__ which_col(IdxT col_length_) : col_length(col_length_) {}
  __host__ __device__ IdxT operator()(IdxT idx) const {
    return idx / col_length;
  }
};

/**
 * @brief Applies the stationarity test to the given series
 * 
 * @details The following algorithm is based on Kwiatkowski 1992:
 *          - Center each series around its mean
 *          - Calculate s^2 (eq. 10) and eta (eq. 11)
 *          - Deduce the p-value and compare against the threshold
 * 
 * @note The data is a column-major matrix where the series are columns.
 *       This function will be called at most twice by `stationarity`
 *
 * @tparam      DataT           Scalar type of the data (float or double)
 * @tparam      IdxT            Integer type of the indices
 * @param[in]   y_d             Input data
 * @param[out]  results         Boolean array to store the results of the test
 * @param[in]   n_batches       Number of batches
 * @param[in]   n_samples       Number of samples
 * @param[in]   allocator       cuML device memory allocator
 * @param[in]   stream          CUDA stream
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary 
 */
template <typename DataT, typename IdxT>
static void _is_stationary(const DataT* y_d, bool* results, IdxT n_batches,
                           IdxT n_samples,
                           std::shared_ptr<MLCommon::deviceAllocator> allocator,
                           cudaStream_t stream, DataT pval_threshold) {
  constexpr int TPB = 256;
  dim3 block = choose_block_dims<TPB>(n_batches);
  dim3 grid(ceildiv<IdxT>(n_samples, block.x),
            ceildiv<IdxT>(n_batches, block.y));

  DataT n_samples_f = static_cast<DataT>(n_samples);

  // Compute mean
  DataT* y_means_d =
    (DataT*)allocator->allocate(n_batches * sizeof(DataT), stream);
  MLCommon::Stats::mean(y_means_d, y_d, n_batches, n_samples, false, false,
                        stream);

  // Center the data around its mean
  DataT* y_cent_d =
    (DataT*)allocator->allocate(n_batches * n_samples * sizeof(DataT), stream);
  MLCommon::LinAlg::matrixVectorOp(
    y_cent_d, y_d, y_means_d, n_batches, n_samples, false, true,
    [] __device__(DataT a, DataT b) { return a - b; }, stream);

  // This calculates the first sum in eq. 10 (first part of s^2)
  DataT* s2A_d = (DataT*)allocator->allocate(n_batches * sizeof(DataT), stream);
  MLCommon::LinAlg::reduce(s2A_d, y_cent_d, n_batches, n_samples,
                           static_cast<DataT>(0.0), false, false, stream, false,
                           L2Op<DataT>(), Sum<DataT>());

  // From Kwiatkowski et al. referencing Schwert (1989)
  DataT lags_f = ceil(12.0 * pow(n_samples_f / 100.0, 0.25));
  IdxT lags = static_cast<IdxT>(lags_f);

  /* This accumulator will be used for both the calculation of s2B, and later
   * the cumulative sum or y_cent_d */
  DataT* accumulator_d =
    (DataT*)allocator->allocate(n_batches * n_samples * sizeof(DataT), stream);

  // This calculates the second sum in eq. 10 (second part of s^2)
  DataT coeff_base = static_cast<DataT>(2.0) / n_samples_f;
  s2B_accumulation_kernel<<<grid, block, 0, stream>>>(
    accumulator_d, y_cent_d, lags, n_batches, n_samples,
    -coeff_base / (lags_f + static_cast<DataT>(1.0)), coeff_base);
  CUDA_CHECK(cudaPeekAtLastError());
  DataT* s2B_d = (DataT*)allocator->allocate(n_batches * sizeof(DataT), stream);
  MLCommon::LinAlg::reduce(s2B_d, accumulator_d, n_batches, n_samples,
                           static_cast<DataT>(0.0), false, false, stream,
                           false);

  // Cumulative sum (inclusive scan with + operator)
  thrust::counting_iterator<IdxT> c_first(0);
  thrust::transform_iterator<which_col<IdxT>, thrust::counting_iterator<IdxT>>
    t_first(c_first, which_col<IdxT>(n_samples));
  thrust::device_ptr<DataT> __y_cent = thrust::device_pointer_cast(y_cent_d);
  thrust::device_ptr<DataT> __csum = thrust::device_pointer_cast(accumulator_d);
  thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream), t_first,
                                t_first + n_batches * n_samples, __y_cent,
                                __csum);

  // Eq. 11 (eta)
  DataT* eta_d = (DataT*)allocator->allocate(n_batches * sizeof(DataT), stream);
  MLCommon::LinAlg::reduce(eta_d, accumulator_d, n_batches, n_samples,
                           static_cast<DataT>(0.0), false, false, stream, false,
                           L2Op<DataT>(), Sum<DataT>());

  /* The following kernel will decide whether each series is stationary based on
   * s^2 and eta */
  bool* results_d =
    (bool*)allocator->allocate(n_batches * sizeof(bool), stream);
  stationarity_check_kernel<<<ceildiv<int>(n_batches, TPB), TPB, 0, stream>>>(
    results_d, s2A_d, s2B_d, eta_d, n_batches, n_samples_f, pval_threshold);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::updateHost(results, results_d, n_batches, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* Free device memory */
  allocator->deallocate(y_means_d, n_batches * sizeof(DataT), stream);
  allocator->deallocate(y_cent_d, n_batches * n_samples * sizeof(DataT),
                        stream);
  allocator->deallocate(accumulator_d, n_batches * n_samples * sizeof(DataT),
                        stream);
  allocator->deallocate(eta_d, n_batches * sizeof(DataT), stream);
  allocator->deallocate(s2A_d, n_batches * sizeof(DataT), stream);
  allocator->deallocate(s2B_d, n_batches * sizeof(DataT), stream);
  allocator->deallocate(results_d, n_batches * sizeof(bool), stream);
}

/**
 * @brief Compute recommended trend parameter (d=0 or 1) for a batched series
 * 
 * @details This function operates a stationarity test on the given series
 *          and for the series that fails the test, differenciates them
 *          and runs the test again on the first difference.
 * 
 * @note The data is a column-major matrix where the series are columns.
 *       The output is an array of size n_batches.
 * 
 * @tparam      DataT           Scalar type of the data (float or double)
 * @tparam      IdxT            Integer type of the indices
 * @param[in]   y_d             Input data
 * @param[out]  d               Integer array to store the trends
 * @param[in]   n_batches       Number of batches
 * @param[in]   n_samples       Number of samples
 * @param[in]   allocator       cuML device memory allocator
 * @param[in]   stream          CUDA stream
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary
 * 
 * @return      An integer to track if some series failed the test
 * @retval  -1  Some series failed the test
 * @retval   0  All series passed the test for d=0
 * @retval   1  Some series passed for d=0, the others for d=1
 */
template <typename DataT, typename IdxT>
int stationarity(const DataT* y_d, int* d, IdxT n_batches, IdxT n_samples,
                 std::shared_ptr<MLCommon::deviceAllocator> allocator,
                 cudaStream_t stream, DataT pval_threshold = 0.05) {
  // Run the test for d=0
  bool is_statio[n_batches];
  _is_stationary(y_d, is_statio, n_batches, n_samples, allocator, stream,
                 pval_threshold);

  // Check the results
  std::vector<int> gather_map_h;
  for (IdxT i = 0; i < n_batches; i++) {
    if (is_statio[i]) {
      d[i] = 0;
    } else {
      gather_map_h.push_back(i);
    }
  }

  IdxT n_diff_batches = gather_map_h.size();
  if (n_diff_batches == 0) return 0;  // All series are stationary with d=0

  /* Construct a matrix of the first difference of the series that failed
       the test for d=0 */
  IdxT n_diff_samples = n_samples - 1;
  IdxT* gather_map_d =
    (IdxT*)allocator->allocate(n_diff_batches * sizeof(int), stream);
  MLCommon::updateDevice(gather_map_d, gather_map_h.data(), n_diff_batches,
                         stream);
  DataT* y_diff_d = (DataT*)allocator->allocate(
    n_diff_batches * n_diff_samples * sizeof(DataT), stream);

  constexpr int TPB = 256;
  dim3 block = choose_block_dims<TPB>(n_diff_batches);
  dim3 grid(ceildiv<IdxT>(n_diff_samples, block.x),
            ceildiv<IdxT>(n_diff_batches, block.y));

  gather_diff_kernel<<<grid, block, 0, stream>>>(
    y_diff_d, y_d, gather_map_d, n_diff_batches, n_diff_samples);
  CUDA_CHECK(cudaPeekAtLastError());

  // Test these series with d=1
  _is_stationary(y_diff_d, is_statio, n_diff_batches, n_diff_samples, allocator,
                 stream, pval_threshold);

  // Check the results
  int ret_value = 1;
  for (int i = 0; i < n_diff_batches; i++) {
    if (is_statio[i]) {
      d[gather_map_h[i]] = 1;
    } else {
      d[gather_map_h[i]] = -1;  // Invalid value to indicate failure
      ret_value = -1;
    }
  }

  allocator->deallocate(gather_map_d, n_diff_batches * sizeof(IdxT), stream);
  allocator->deallocate(
    y_diff_d, n_diff_batches * n_diff_samples * sizeof(DataT), stream);

  return ret_value;
}

};  //end namespace TimeSeries
};  //end namespace MLCommon
