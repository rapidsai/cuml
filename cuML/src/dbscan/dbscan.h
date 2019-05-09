/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "runner.h"
#include <common/device_buffer.hpp>

namespace ML {

using namespace Dbscan;

// Default max mem set to a reasonable value for a 16gb card.
static const size_t DEFAULT_MAX_MEM_BYTES = 13e9;

template<typename T>
int computeBatchCount(int n_rows, size_t max_bytes_per_batch) {

    int n_batches = 1;
    // There seems to be a weird overflow bug with cutlass gemm kernels
    // hence, artifically limiting to a smaller batchsize!
    ///TODO: in future, when we bump up the underlying cutlass version, this should go away
    // paving way to cudaMemGetInfo based workspace allocation
    while(true) {
        size_t batchSize = ceildiv<size_t>(n_rows, n_batches);
        if(batchSize * n_rows * sizeof(T) < max_bytes_per_batch || batchSize == 1)
            break;
        ++n_batches;
    }
    return n_batches;
}

template<typename T>
void dbscanFitImpl(const ML::cumlHandle_impl& handle, T *input,
        int n_rows, int n_cols,
        T eps, int min_pts,
        int *labels,
        size_t max_bytes_per_batch,
        cudaStream_t stream,
        bool verbose) {
    int algoVd = 1;
    int algoAdj = 1;
    int algoCcl = 2;

    if(max_bytes_per_batch <= 0)
        // @todo: Query device for remaining memory
        max_bytes_per_batch = DEFAULT_MAX_MEM_BYTES;

    int n_batches = computeBatchCount<T>(n_rows, max_bytes_per_batch);

    if(verbose) {
        size_t batchSize = ceildiv<size_t>(n_rows, n_batches);
        if(n_batches > 1) {
            std::cout << "Running batched training on " << n_batches << " batches w/ ";
            std::cout << batchSize * n_rows * sizeof(T) << " bytes." << std::endl;
        }
    }

    size_t workspaceSize = Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts,
                                       labels, algoVd, algoAdj, algoCcl, NULL,
                                       n_batches, stream);

    MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream, workspaceSize);
    Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels, algoVd, algoAdj,
                algoCcl, workspace.data(), n_batches, stream);
}

/** @} */

}
;
// end namespace ML
