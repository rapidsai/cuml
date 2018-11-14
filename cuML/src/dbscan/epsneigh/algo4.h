/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include "pack.h"
#include "../common.h"

namespace Dbscan {
namespace EpsNeigh {
namespace Algo4 {


//// Used by epsneigh to compute the final result C <= (accumulator < eps * eps)
template <typename accum_t, typename output_t, typename scalar_t>
class epsneigh_epilogue {
private:
    scalar_t eps2;

public:
    inline __device__ __host__ epsneigh_epilogue(scalar_t eps):
        eps2(eps * eps) {
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   output_t c,
                                                   size_t x_idx,
                                                   size_t y_idx) const {
        return output_t(accumulator <= eps2);
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   size_t x_idx,
                                                   size_t y_idx) const {
        return output_t(accumulator <= eps2);
    }

    /**
     * Configure epilogue as to whether the thread block cheme  is a secondary
     * accumulator in an inter-block k-splitting scheme
     * @note: this is a no-op here!
     */
    inline __device__ void set_secondary_accumulator() {
    }

    /// Return whether the beta-scaled addend needs initialization
    /// @note: we don't need initialization
    inline __device__ bool must_init_addend() {
        return false;
    }
};


template <typename value_t, cutlass::gemm::tiling_strategy::kind_t TilingStrategy>
void launcher(Pack<value_t> data, cudaStream_t stream) {
    using epilogue_op_t = epsneigh_epilogue<value_t, char, value_t>;
    epilogue_op_t epilogue(data.eps);
    epsneigh_dispatch<TilingStrategy, value_t, char, cutlass::math_operation_scalar,
                      epilogue_op_t, ds_accummulate<value_t, value_t> > edis;
    auto res = edis(data.N, data.N, data.D, data.x, data.x, data.adj,
                    epilogue, stream, false, false);
    CUDA_CHECK(res.result);
}

} // namespace Algo4
} // namespace EpsNeigh
} // namespace Dbscan
