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

#include "common/device_buffer.hpp"
#include "common/nvtx.hpp"
#include "runner.h"

namespace ML {

using namespace Dbscan;
static const size_t DEFAULT_MAX_MEM_MBYTES = 13e3;

// Default max mem set to a reasonable value for a 16gb card.
template <typename T, typename Index_ = int>
Index_ computeBatchCount(Index_ n_rows, size_t max_mbytes_per_batch) {
  Index_ n_batches = 1;
  // There seems to be a weird overflow bug with cutlass gemm kernels
  // hence, artifically limiting to a smaller batchsize!
  ///TODO: in future, when we bump up the underlying cutlass version, this should go away
  // paving way to cudaMemGetInfo based workspace allocation

  if (max_mbytes_per_batch <= 0) max_mbytes_per_batch = DEFAULT_MAX_MEM_MBYTES;

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  while (true) {
    size_t batchSize = ceildiv<size_t>(n_rows, n_batches);
    if (((batchSize * n_rows * sizeof(T) * 1e-6 < max_mbytes_per_batch) &&
         /**
          * Though single precision can be faster per execution of each kernel,
          * there's a trade-off to be made between using single precision with
          * many more batches (which become smaller as n_rows grows) and using
          * double precision, which will have less batches but could become 8-10x
          * slower per batch.
          */
         (batchSize * n_rows < MAX_LABEL)) ||
        batchSize == 1)
      break;
    ++n_batches;
  }
  return n_batches;
}

template <typename T, typename Index_ = int>
void dbscanFitImpl(const ML::cumlHandle_impl &handle, T *input, Index_ n_rows,
                   Index_ n_cols, T eps, int min_pts, Index_ *labels,
                   size_t max_mbytes_per_batch, cudaStream_t stream,
                   bool verbose) {
  ML::PUSH_RANGE("ML::Dbscan::Fit");
  int algoVd = 1;
  int algoAdj = 1;
  int algoCcl = 2;

  // @todo: Query device for remaining memory
  Index_ n_batches = computeBatchCount<T, Index_>(n_rows, max_mbytes_per_batch);

  if (verbose) {
    Index_ batchSize = ceildiv<Index_>(n_rows, n_batches);
    if (n_batches > 1) {
      std::cout << "Running batched training on " << n_batches
                << " batches w/ ";
      std::cout << batchSize * n_rows * sizeof(T) * 1e-6 << " megabytes."
                << std::endl;
    }
  }

  size_t workspaceSize =
    Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels, algoVd,
                algoAdj, algoCcl, NULL, n_batches, stream);

  MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream,
                                          workspaceSize);
  Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels, algoVd,
              algoAdj, algoCcl, workspace.data(), n_batches, stream, verbose);
  ML::POP_RANGE();
}

};  // namespace ML
