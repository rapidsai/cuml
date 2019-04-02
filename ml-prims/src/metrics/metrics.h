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

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

namespace MLCommon {
    namespace Metrics {
        template<typename math_t>
        math_t r_squared(math_t *y, math_t *y_hat, int n) {

            math_t *y_bar;
            MLCommon::allocate(y_bar, 1);

            MLCommon::Stats::mean(y_bar, y, 1, n, false, false);
            CUDA_CHECK(cudaPeekAtLastError());

            math_t *ssr_arr;
            MLCommon::allocate(ssr_arr, n);

            MLCommon::LinAlg::eltwiseSub(ssr_arr, y, y_hat, n);
            MLCommon::LinAlg::powerScalar(ssr_arr, ssr_arr, 2.0f, n);
            CUDA_CHECK(cudaPeekAtLastError());

            math_t *ssto_arr;
            MLCommon::allocate(ssto_arr, n);

            MLCommon::LinAlg::subtractDevScalar(ssto_arr, y, y_bar, n);
            MLCommon::LinAlg::powerScalar(ssto_arr, ssto_arr, 2.0f, n);
            CUDA_CHECK(cudaPeekAtLastError());

            thrust::device_ptr<math_t> d_ssr = thrust::device_pointer_cast(ssr_arr);
            thrust::device_ptr<math_t> d_ssto = thrust::device_pointer_cast(ssto_arr);

            math_t ssr = thrust::reduce(d_ssr, d_ssr+n);
            math_t ssto = thrust::reduce(d_ssto, d_ssto+n);

            CUDA_CHECK(cudaFree(y_bar));
            CUDA_CHECK(cudaFree(ssr_arr));
            CUDA_CHECK(cudaFree(ssto_arr));

            return 1.0 - ssr/ssto;
        }



    }
}

