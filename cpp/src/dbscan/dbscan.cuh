/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include "common/device_buffer.hpp"
#include "common/nvtx.hpp"
#include "runner.cuh"

#include <algorithm>

namespace ML {

using namespace Dbscan;
// Default max mem set to a reasonable value for a 16gb card.
static const size_t DEFAULT_MAX_MEM_MBYTES = 13e3;

template <typename Index_ = int>
Index_ computeBatchCount(size_t &estimated_memory, Index_ n_rows,
    size_t max_mbytes_per_batch = 0, Index_ neigh_per_row = 0) {
  // In real applications, it's unlikely that the sparse adjacency matrix
  // comes even close to the worst-case memory usage, because if epsilon
  // is so large that all points are connected to 10% or even more of other
  // points, the clusters would probably not be interesting/relevant anymore
  ///@todo: expose `neigh_per_row` to the user

  if (neigh_per_row <= 0) neigh_per_row = n_rows;

  // we'll estimate the memory consumption per row.
  // First the dense adjacency matrix
  estimated_memory = n_rows * sizeof(bool);
  // sparse adjacency matrix
  estimated_memory += neigh_per_row * sizeof(Index_);
  // core points and two indicator variables
  estimated_memory += 3 * sizeof(bool);
  // the rest will be so small that it should fit into what we have left over
  // from the over-estimation of the sparse adjacency matrix
  estimated_memory *= n_rows;

  if (max_mbytes_per_batch <= 0) {
    /* using default here as in decision tree, waiting for mem info from device allocator
    size_t total_mem;
	  CUDA_CHECK(cudaMemGetInfo(&max_mbytes_per_batch, &total_mem));
    */
    max_mbytes_per_batch = DEFAULT_MAX_MEM_MBYTES;
  }

  Index_ nBatches = (Index_)ceildiv<size_t>(
    estimated_memory, max_mbytes_per_batch * 1000000);
  size_t MAX_LABEL = (size_t)std::numeric_limits<Index_>::max();
  // n_rows * n_rows_per_batch < MAX_LABEL => n_rows * (n_rows / nBatches) < MAX_LABEL
  // => nBatches >= n_rows * n_rows / MAX_LABEL
  Index_ nBatchesPrec = (Index_)ceildiv<size_t>((size_t)n_rows * n_rows, MAX_LABEL);
  if (nBatchesPrec >= 4 * nBatches) {
    CUML_LOG_WARN(
      "Due to precision limitations of the index type (%d bytes) "
      "we need to use %ld batches, but you have memory for %ld batches. "
      "Consider upgrading the index type (output label type).",
      (int)sizeof(Index_), (size_t)nBatchesPrec, (size_t)nBatches
    );
  }
  if (sizeof(Index_) > sizeof(int) && 
        (size_t)n_rows * ceildiv<Index_>(n_rows, nBatches) <
          std::numeric_limits<int>::max()) {
    CUML_LOG_WARN(
      "You are using an index type of size (%d bytes) but a smaller index "
      "type (%d bytes) would be sufficient. Consider using the smaller "
      "index type for better performance.",
      (int)sizeof(Index_), (int)sizeof(int)
    );
  }
  return std::max({ (Index_)1, nBatchesPrec, nBatches });
}

template <typename T, typename Index_ = int>
void dbscanFitImpl(const ML::cumlHandle_impl &handle, T *input, Index_ n_rows,
                   Index_ n_cols, T eps, Index_ min_pts, Index_ *labels,
                   size_t max_mbytes_per_batch, cudaStream_t stream,
                   int verbosity) {
  ML::PUSH_RANGE("ML::Dbscan::Fit");
  ML::Logger::get().setLevel(verbosity);
  int algoVd = 1;
  int algoAdj = 1;
  int algoCcl = 2;

  ///@todo: Query device for remaining memory
  size_t estimated_memory;
  Index_ n_batches = computeBatchCount<Index_>(
    estimated_memory, n_rows, max_mbytes_per_batch);

  if (n_batches > 1) {
    CUML_LOG_DEBUG("Running batched training on %ld batches w/ %lf MB",
                   (unsigned long)n_batches,
                   (double)estimated_memory * 1e-6 / n_batches);
  }

  size_t workspaceSize =
    Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels, algoVd,
                algoAdj, algoCcl, NULL, n_batches, stream);

  MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream,
                                          workspaceSize);
  Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels, algoVd,
              algoAdj, algoCcl, workspace.data(), n_batches, stream);
  ML::POP_RANGE();
}

};  // namespace ML
