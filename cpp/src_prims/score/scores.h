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

#pragma once

#include "linalg/eltwise.h"
#include "stats/mean.h"
#include "linalg/subtract.h"
#include "linalg/power.h"

#include "common/cuml_allocator.hpp"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

namespace MLCommon {
    namespace Score {
        template<typename math_t>

        /**
         * Calculates the "Coefficient of Determination" (R-Squared) score
         * normalizing the sum of squared errors by the total sum of squares.
         *
         * This score indicates the proportionate amount of variation in an
         * expected response variable is explained by the independent variables
         * in a linear regression model. The larger the R-squared value, the
         * more variability is explained by the linear regression model.
         *
         * @param y: Array of ground-truth response variables
         * @param y_hat: Array of predicted response variables
         * @param n: Number of elements in y and y_hat
         * @return: The R-squared value.
         */
        math_t r2_score(math_t *y, math_t *y_hat, int n, cudaStream_t stream) {

            math_t *y_bar;
            MLCommon::allocate(y_bar, 1);

            MLCommon::Stats::mean(y_bar, y, 1, n, false, false, stream);
            CUDA_CHECK(cudaPeekAtLastError());

            math_t *sse_arr;
            MLCommon::allocate(sse_arr, n);

            MLCommon::LinAlg::eltwiseSub(sse_arr, y, y_hat, n, stream);
            MLCommon::LinAlg::powerScalar(sse_arr, sse_arr, math_t(2.0), n, stream);
            CUDA_CHECK(cudaPeekAtLastError());

            math_t *ssto_arr;
            MLCommon::allocate(ssto_arr, n);

            MLCommon::LinAlg::subtractDevScalar(ssto_arr, y, y_bar, n, stream);
            MLCommon::LinAlg::powerScalar(ssto_arr, ssto_arr, math_t(2.0), n, stream);
            CUDA_CHECK(cudaPeekAtLastError());

            thrust::device_ptr<math_t> d_sse = thrust::device_pointer_cast(sse_arr);
            thrust::device_ptr<math_t> d_ssto = thrust::device_pointer_cast(ssto_arr);

            math_t sse = thrust::reduce(thrust::cuda::par.on(stream), d_sse, d_sse+n);
            math_t ssto = thrust::reduce(thrust::cuda::par.on(stream), d_ssto, d_ssto+n);

            CUDA_CHECK(cudaFree(y_bar));
            CUDA_CHECK(cudaFree(sse_arr));
            CUDA_CHECK(cudaFree(ssto_arr));

            return 1.0 - sse/ssto;
        }

        /**
          * @brief Compute accuracy of predictions. Useful for classification.
          * @tparam math_t: data type for predictions (e.g., int for classification)
          * @param[in] predictions: array of predictions (GPU pointer).
          * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
          * @param[in] n: number of elements in each of predictions, ref_predictions.
          * @param[in] d_alloc: device allocator.
          * @param[in] stream: cuda stream.
          * @return: Accuracy score in [0, 1]; higher is better.
          */
        template<typename math_t>
        float accuracy_score(const math_t * predictions, const math_t * ref_predictions, int n,
                            std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream) {

            unsigned long long correctly_predicted = 0ULL;
            math_t * diffs_array = (math_t *)d_alloc->allocate(n * sizeof(math_t), stream);

            //TODO could write a kernel instead
            MLCommon::LinAlg::eltwiseSub(diffs_array, predictions, ref_predictions, n, stream);
            CUDA_CHECK(cudaGetLastError());
            correctly_predicted = thrust::count(thrust::cuda::par.on(stream), diffs_array, diffs_array + n, 0);
            d_alloc->deallocate(diffs_array, n * sizeof(math_t), stream);

            float accuracy = correctly_predicted * 1.0f/n;
            return accuracy;
        }

        template<typename T>
        __global__ void reg_metrics_kernel(const T * predictions, const T * ref_predictions, int n,
                                        double * abs_diffs, double * tmp_sums) {

            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            __shared__ double shmem[2]; // {abs_difference_sum, squared difference sum}

            for (int i = threadIdx.x; i < 2; i += blockDim.x) {
                shmem[i] = 0;
            }
            __syncthreads();

            for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
                double diff = predictions[i] - ref_predictions[i];
                double abs_diff = abs(diff);
                atomicAdd(&shmem[0], abs_diff);
                atomicAdd(&shmem[1], diff * diff);

                // update absolute difference in global memory for subsequent abs. median computation
                abs_diffs[i] = abs_diff;
            }
            __syncthreads();

            // Update tmp_sum w/ total abs_difference_sum and squared difference sum.
            for (int i = threadIdx.x; i < 2; i += blockDim.x) {
                atomicAdd(&tmp_sums[i], shmem[i]);
            }
        }

        /**
          * @brief Compute regression metrics mean absolute error, mean squared error, median absolute error
          * @tparam T: data type for predictions (e.g., float or double for regression).
          * @param[in] predictions: array of predictions (GPU pointer).
          * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
          * @param[in] n: number of elements in each of predictions, ref_predictions. Should be > 0.
          * @param[in] d_alloc: device allocator.
          * @param[in] stream: cuda stream.
          * @param[out] mean_abs_error: Mean Absolute Error. Sum over n of (|predictions[i] - ref_predictions[i]|) / n.
          * @param[out] mean_squared_error: Mean Squared Error. Sum over n of ((predictions[i] - ref_predictions[i])^2) / n.
          * @param[out] median_abs_error: Median Absolute Error. Median of |predictions[i] - ref_predictions[i]| for i in [0, n).
          */
        template<typename T>
        void regression_metrics(const T * predictions, const T * ref_predictions, int n, std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream,
                                double & mean_abs_error, double & mean_squared_error, double & median_abs_error) {

            std::vector<double> mean_errors(2);
            std::vector<double> h_sorted_abs_diffs(n);
            int thread_cnt = 256;
            int block_cnt = ceildiv(n, thread_cnt);

            int array_size = n * sizeof(double);
            double * abs_diffs_array = (double *)d_alloc->allocate(array_size, stream);
            double * sorted_abs_diffs = (double *)d_alloc->allocate(array_size, stream);
            double * tmp_sums = (double *)d_alloc->allocate(2 * sizeof(double), stream);
            CUDA_CHECK(cudaMemsetAsync(tmp_sums, 0, 2 * sizeof(double), stream));

            reg_metrics_kernel<T><<<block_cnt, thread_cnt, 0, stream>>>(predictions, ref_predictions, n, abs_diffs_array, tmp_sums);
            CUDA_CHECK(cudaGetLastError());
            MLCommon::updateHost(&mean_errors[0], tmp_sums, 2, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            mean_abs_error = mean_errors[0] / n;
            mean_squared_error = mean_errors[1] / n;

            // Compute median error. Sort diffs_array and pick median value
            char * temp_storage = nullptr;
            size_t temp_storage_bytes;
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void *)temp_storage, temp_storage_bytes, abs_diffs_array, sorted_abs_diffs,
                       n, 0, 8*sizeof(double), stream));
            temp_storage = (char *)d_alloc->allocate(temp_storage_bytes, stream);
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void *)temp_storage, temp_storage_bytes, abs_diffs_array, sorted_abs_diffs,
                       n, 0, 8*sizeof(double), stream));

            MLCommon::updateHost(h_sorted_abs_diffs.data(), sorted_abs_diffs, n, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int middle =  n / 2;
            if (n % 2 == 1) {
                median_abs_error = h_sorted_abs_diffs[middle];
            } else {
                median_abs_error = (h_sorted_abs_diffs[middle] + h_sorted_abs_diffs[middle - 1]) / 2;
            }

            d_alloc->deallocate(abs_diffs_array, array_size, stream);
            d_alloc->deallocate(sorted_abs_diffs, array_size, stream);
            d_alloc->deallocate(temp_storage, temp_storage_bytes, stream);
            d_alloc->deallocate(tmp_sums, 2 * sizeof(double), stream);
        }
    }
}

