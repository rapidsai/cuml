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
namespace VertexDeg {
namespace Algo4 {

template <typename accum_t, typename output_t, typename scalar_t>
class vertexdeg_epilogue {
private:
    scalar_t eps2;
    int* vdeg;
    int dim_m;

public:
    inline __device__ __host__ vertexdeg_epilogue(scalar_t eps, int* vd, int m):
        eps2(eps * eps), vdeg(vd), dim_m(m) {
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   output_t c,
                                                   size_t x_idx,
						   size_t y_idx) const {
        output_t res = output_t(accumulator <= eps2);
        atomicAdd(vdeg+x_idx, res);
        atomicAdd(vdeg+dim_m, res);
        return res;
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   size_t x_idx,
	                                           size_t y_idx) const {
        output_t res = output_t(accumulator <= eps2);
        atomicAdd(vdeg+x_idx, res);
        atomicAdd(vdeg+dim_m, res);
	return res;
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
void launcher(Pack<value_t> data, cudaStream_t stream, int startVertexId, int batchSize) {
    using epilogue_op_t = vertexdeg_epilogue<value_t, bool, value_t>;
    data.resetArray(stream, batchSize+1);
    epilogue_op_t epilogue(data.eps, data.vd, batchSize);
    epsneigh_dispatch<TilingStrategy, value_t, bool, cutlass::math_operation_scalar,
                      epilogue_op_t, ds_accummulate<value_t, value_t> > edis;
    int m = data.N;
    int n = min(data.N - startVertexId, batchSize);
    int k = data.D;
    auto res = edis(m, n, k, data.x, data.x + startVertexId*data.D, data.adj,
                    epilogue, stream, false, false);
    CUDA_CHECK(res.result);
}

} // namespace Algo4
} // namespace VertexDeg
} // namespace Dbscan
