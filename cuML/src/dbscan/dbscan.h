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

#include "runner.h"

namespace ML {

using namespace Dbscan;

int computeBatchCount(int n_rows) {
    int n_batches = 1;
    // HACK HACK HACK: there seems to be a weird overflow bug with cutlass gemm kernels
    // hence, artifically limiting to a smaller batchsize!
    ///@todo: in future, when we bump up the underlying cutlass version, this should go away
    // paving way to cudaMemGetInfo based workspace allocation
    static const size_t MaxElems = (size_t)1024 * 1024 * 1024 * 2;  // 2e9
    while(true) {

	// @todo: Shouldn't this be dependent on the columns and not just the rows?
        size_t batchSize = ceildiv<size_t>(n_rows, n_batches);
        if(batchSize * n_rows < MaxElems)
            break;
        ++n_batches;
    }

    return n_batches;
}

template<typename T>
void dbscanFitImpl(T *input, int n_rows, int n_cols, T eps, int min_pts, int *labels) {
    int algoVd = 1;
    int algoAdj = 1;
    int algoCcl = 2;
    int n_batches = computeBatchCount(n_rows);
    size_t workspaceSize = Dbscan::run(input, n_rows, n_cols, eps, min_pts,
                                       labels, algoVd, algoAdj, algoCcl, NULL,
                                       n_batches, 0);

    char* workspace;
    CUDA_CHECK(cudaMalloc((void** )&workspace, workspaceSize));

    Dbscan::run(input, n_rows, n_cols, eps, min_pts, labels, algoVd, algoAdj,
                algoCcl, workspace, n_batches, 0);

    CUDA_CHECK(cudaFree(workspace));
}

/** @} */

}
;
// end namespace ML
