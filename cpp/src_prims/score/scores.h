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
    }
}

