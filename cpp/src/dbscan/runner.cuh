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

#include <common/cudart_utils.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <common/nvtx.hpp>
#include <cuda_utils.cuh>
#include <label/classlabels.cuh>
#include <sparse/csr.cuh>
#include "adjgraph/runner.cuh"
#include "corepoints/runner.cuh"
#include "vertexdeg/runner.cuh"

#include <sys/time.h>
#include <cuml/common/logger.hpp>

namespace ML {
namespace Dbscan {

using namespace MLCommon;

static const int TPB = 256;

/**
 * Adjust labels from weak_cc primitive to match sklearn:
 * 1. Turn any labels matching MAX_LABEL into -1
 * 2. Subtract 1 from all other labels.
 */
template <typename Index_ = int>
__global__ void relabelForSkl(Index_* labels, Index_ N, Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    if (labels[tid] == MAX_LABEL) {
      labels[tid] = -1;
    } else {
      --labels[tid];
    }
  }
}

/**
 * Turn the non-monotonic labels from weak_cc primitive into
 * an array of labels drawn from a monotonically increasing set.
 */
template <typename Index_ = int>
void final_relabel(Index_* db_cluster, Index_ N, cudaStream_t stream,
                   std::shared_ptr<deviceAllocator> allocator) {
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  MLCommon::Label::make_monotonic(
    db_cluster, db_cluster, N, stream,
    [MAX_LABEL] __device__(Index_ val) { return val == MAX_LABEL; }, allocator);
}

/**
 * @tparam opg Whether we are running in a multi-node multi-GPU context
 * @param N number of points
 * @param D dimensionality of the points
 * @param eps epsilon neighborhood criterion
 * @param minPts core points criterion
 * @param labels the output labels (should be of size N)
 * @param ....
 * @param temp temporary global memory buffer used to store intermediate computations
 *             If this is a null pointer, then this function will return the workspace size needed.
 *             It is the responsibility of the user to cudaMalloc and cudaFree this buffer!
 * @param stream the cudaStream where to launch the kernels
 * @return in case the temp buffer is null, this returns the size needed.
 */
template <typename Type_f, typename Index_ = int, bool opg = false>
size_t run(const raft::handle_t& handle, Type_f* x, Index_ N, Index_ D,
           Index_ start_row, Index_ n_owned_rows, Type_f eps, Index_ minPts,
           Index_* labels, Index_* core_sample_indices, int algoVd, int algoAdj,
           int algoCcl, void* workspace, Index_ nBatches, cudaStream_t stream) {
  const size_t align = 256;
  size_t batchSize = raft::ceildiv<size_t>(n_owned_rows, nBatches);

  /// TODO: unify naming convention? Check if there is a cuML conv for the case

  /**
   * Note on coupling between data types:
   * - adjacency graph has a worst case size of N * batchSize elements. Thus,
   * if N is very close to being greater than the maximum 32-bit IdxType type used, a
   * 64-bit IdxType should probably be used instead.
   * - exclusive scan is the CSR row index for the adjacency graph and its values have a
   * risk of overflowing when N * batchSize becomes larger what can be stored in IdxType
   * - the vertex degree array has a worst case of each element having all other
   * elements in their neighborhood, so any IdxType can be safely used, so long as N doesn't
   * overflow.
   */
  size_t adjSize = raft::alignTo<size_t>(sizeof(bool) * N * batchSize, align);
  size_t corePtsSize = raft::alignTo<size_t>(sizeof(bool) * N, align);
  size_t xaSize = raft::alignTo<size_t>(sizeof(bool) * N, align);
  size_t mSize = raft::alignTo<size_t>(sizeof(bool), align);
  size_t vdSize =
    raft::alignTo<size_t>(sizeof(Index_) * (batchSize + 1), align);
  size_t exScanSize = raft::alignTo<size_t>(sizeof(Index_) * batchSize, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(
    N * batchSize < MAX_LABEL,
    "An overflow occurred with the current choice of precision "
    "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
    "Consider using double precision for the output labels.",
    (unsigned long)(MAX_LABEL / N), (unsigned long)batchSize);

  if (workspace == NULL) {
    auto size =
      adjSize + corePtsSize + 2 * xaSize + mSize + vdSize + exScanSize;
    return size;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  Index_ maxadjlen = 0;
  Index_ curradjlen = 0;
  char* temp = (char*)workspace;
  bool* adj = (bool*)temp;
  temp += adjSize;
  bool* core_pts = (bool*)temp;
  temp += corePtsSize;
  bool* xa = (bool*)temp;
  temp += xaSize;
  bool* fa = (bool*)temp;
  temp += xaSize;
  bool* m = (bool*)temp;
  temp += mSize;
  Index_* vd = (Index_*)temp;
  temp += vdSize;
  Index_* ex_scan = (Index_*)temp;
  temp += exScanSize;

  /// TODO: do logs and timings make sense for asynchronous operations?

  // Compute the mask
  // 1. Compute the part owned by this worker (reversed order of batches to
  // keep the batch 0 in memory)
  for (int i = nBatches - 1; i >= 0; i--) {
    Index_ startVertexId = start_row + i * batchSize;
    Index_ nPoints = min(size_t(n_owned_rows - i * batchSize), batchSize);
    if (nPoints <= 0) break;

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)nBatches, (unsigned long)nPoints);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
    int64_t start_time = raft::curTimeMillis();
    VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algoVd,
                                   startVertexId, nPoints, stream);
    // raft::update_host(&curradjlen, vd + nPoints, 1, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    int64_t cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));

    CUML_LOG_DEBUG("--> Computing core point mask");
    ML::PUSH_RANGE("Trace::Dbscan::CorePoints");
    start_time = raft::curTimeMillis();
    CorePoints::run<Type_f, Index_>(handle, vd, core_pts, minPts, startVertexId,
                                    nPoints, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));
  }
  // 2. Exchange with the other workers
  /// TODO: allgatherv

  // Compute the labelling for the owned part of the graph
  MLCommon::Sparse::WeakCCState state(xa, fa, m);
  MLCommon::device_buffer<Index_> adj_graph(handle.get_device_allocator(),
                                            stream);

  for (int i = 0; i < nBatches; i++) {
    Index_ startVertexId = start_row + i * batchSize;
    Index_ nPoints = min(size_t(n_owned_rows - i * batchSize), batchSize);
    if (nPoints <= 0) break;

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)nBatches, (unsigned long)nPoints);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
    int64_t start_time = raft::curTimeMillis();
    VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algoVd,
                                   startVertexId, nPoints, stream);
    raft::update_host(&curradjlen, vd + nPoints, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int64_t cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));

    CUML_LOG_DEBUG("--> Computing adjacency graph of size %ld samples.",
                   (unsigned long)curradjlen);
    ML::PUSH_RANGE("Trace::Dbscan::AdjGraph");
    start_time = raft::curTimeMillis();
    if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
      maxadjlen = curradjlen;
      adj_graph.resize(maxadjlen, stream);
    }
    AdjGraph::run<Index_>(handle, adj, vd, adj_graph.data(), curradjlen,
                          ex_scan, N, minPts, core_pts + startVertexId, algoAdj,
                          nPoints, stream);
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    CUML_LOG_DEBUG("    |-> Took %ld ms.", (cur_time - start_time));

    CUML_LOG_DEBUG("--> Computing connected components");
    ML::PUSH_RANGE("Trace::Dbscan::WeakCC");
    start_time = raft::curTimeMillis();
    MLCommon::Sparse::weak_cc_batched<Index_, 1024>(
      labels, ex_scan, adj_graph.data(), curradjlen, N, startVertexId, nPoints,
      &state, stream,
      [core_pts, startVertexId, nPoints] __device__(Index_ global_id) {
        return global_id < startVertexId + nPoints ? core_pts[global_id]
                                                   : false;
      });
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    CUML_LOG_DEBUG("    |-> Took %ld ms.", (cur_time - start_time));
  }

  // Combine the results in the multi-node multi-GPU case

  if (opg) {
    // Temporary: naive merge
    const auto& comm = handle.get_comms();
    comm.allreduce(labels, labels, N, raft::comms::op_t::MIN, stream);
    ASSERT(
      comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
      "An error occurred in the distributed operation. This can result from "
      "a failed rank");
  }

  // Final relabel

  ML::PUSH_RANGE("Trace::Dbscan::FinalRelabel");
  if (algoCcl == 2)
    final_relabel(labels, N, stream, handle.get_device_allocator());
  size_t nblks = raft::ceildiv<size_t>(N, TPB);
  relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
  ML::POP_RANGE();

  // Calculate the core_sample_indices only if an array was passed in
  /// TODO: MNMG version
  if (core_sample_indices != nullptr) {
    ML::PUSH_RANGE("Trace::Dbscan::CoreSampleIndices");

    // Create the execution policy
    ML::thrustAllocatorAdapter alloc(handle.get_device_allocator(), stream);
    auto thrust_exec_policy = thrust::cuda::par(alloc).on(stream);

    // Get wrappers for the device ptrs
    thrust::device_ptr<bool> dev_core_pts =
      thrust::device_pointer_cast(core_pts);
    thrust::device_ptr<Index_> dev_core_sample_indices =
      thrust::device_pointer_cast(core_sample_indices);

    // First fill the core_sample_indices with -1 which will be used if core_point_count < N
    thrust::fill_n(thrust_exec_policy, dev_core_sample_indices, N, (Index_)-1);

    auto index_iterator = thrust::counting_iterator<int>(0);

    //Perform stream reduction on the core points. The core_pts acts as the stencil and we use thrust::counting_iterator to return the index
    auto core_point_count = thrust::copy_if(
      thrust_exec_policy, index_iterator, index_iterator + N, dev_core_pts,
      dev_core_sample_indices,
      [=] __device__(const bool is_core_point) { return is_core_point; });

    ML::POP_RANGE();
  }

  CUML_LOG_DEBUG("Done.");
  return (size_t)0;
}
}  // namespace Dbscan
}  // namespace ML