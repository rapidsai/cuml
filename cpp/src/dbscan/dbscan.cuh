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

#include <common/device_buffer.hpp>
#include <common/nvtx.hpp>
#include <cuml/common/logger.hpp>
#include "runner.cuh"

#include <algorithm>

namespace ML {

using namespace Dbscan;
// Default max mem set to a reasonable value for a 16gb card.
static const size_t DEFAULT_MAX_MEM_MBYTES = 13e3;

template <typename Index_ = int>
Index_ computeBatchCount(size_t &estimated_memory, Index_ n_rows,
                         size_t max_mbytes_per_batch = 0,
                         Index_ neigh_per_row = 0) {
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

  Index_ nBatches =
    (Index_)ceildiv<size_t>(estimated_memory, max_mbytes_per_batch * 1000000);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  // to avoid overflow, we need: batch_size <= MAX_LABEL / n_rows (floor div)
  // -> num_batches >= ceildiv(n_rows / (MAX_LABEL / n_rows))
  Index_ nBatchesPrec = ceildiv(n_rows, MAX_LABEL / n_rows);
  // at some point, if nBatchesPrec is larger than nBatches
  // (or larger by a given factor) and we know that there are clear
  // performance benefits of using a smaller number of batches,
  // we should probably warn the user.
  // In the latest benchmarks, it seems like using int64 indexing and batches
  // that are much larger than 2.10^9 points (the limit for int32), doesn't
  // actually improve performance, even when using >16.10^9 points per batch.
  // Much larger batches than 16.10^9 do not currently fit on GPU architectures
  if (sizeof(Index_) > sizeof(int) &&
      (size_t)n_rows * ceildiv<Index_>(n_rows, nBatches) <
        std::numeric_limits<int>::max()) {
    CUML_LOG_WARN(
      "You are using an index type of size (%d bytes) but a smaller index "
      "type (%d bytes) would be sufficient. Consider using the smaller "
      "index type for better performance.",
      (int)sizeof(Index_), (int)sizeof(int));
  }
  if (nBatchesPrec > nBatches) {
    nBatches = nBatchesPrec;
    // we have to re-adjust memory estimation here
    estimated_memory = nBatches * (estimated_memory / n_rows);
  }
  return max((Index_)1, nBatches);
}

template <typename T, typename Index_ = int>
void dbscanFitImpl(const raft::handle_t &handle, T *input, Index_ n_rows,
                   Index_ n_cols, T eps, Index_ min_pts, Index_ *labels,
                   Index_ *core_sample_indices, size_t max_mbytes_per_batch,
                   cudaStream_t stream, int verbosity) {
  ML::PUSH_RANGE("ML::Dbscan::Fit");
  ML::Logger::get().setLevel(verbosity);
  int algoVd = 1;
  int algoAdj = 1;
  int algoCcl = 2;

  ///@todo: Query device for remaining memory
  size_t estimated_memory;
  Index_ n_batches =
    computeBatchCount<Index_>(estimated_memory, n_rows, max_mbytes_per_batch);

  if (n_batches > 1) {
    CUML_LOG_DEBUG("Running batched training on %ld batches w/ %lf MB",
                   (unsigned long)n_batches,
                   (double)estimated_memory * 1e-6 / n_batches);
  }

  size_t workspaceSize = Dbscan::run(
    handle, input, n_rows, n_cols, eps, min_pts, labels, core_sample_indices,
    algoVd, algoAdj, algoCcl, NULL, n_batches, stream);

  MLCommon::device_buffer<char> workspace(handle.get_device_allocator(), stream,
                                          workspaceSize);
  Dbscan::run(handle, input, n_rows, n_cols, eps, min_pts, labels,
              core_sample_indices, algoVd, algoAdj, algoCcl, workspace.data(),
              n_batches, stream);
  ML::POP_RANGE();
}

};  // namespace ML
