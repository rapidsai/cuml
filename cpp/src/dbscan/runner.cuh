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

#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <common/nvtx.hpp>
#include <label/classlabels.cuh>
#include <raft/cuda_utils.cuh>
#include <sparse/csr.cuh>
#include "adjgraph/runner.cuh"
#include "corepoints/runner.cuh"
#include "mergelabels/runner.cuh"
#include "vertexdeg/runner.cuh"

#include <sys/time.h>
#include <cuml/common/logger.hpp>

#include <sstream>  // TODO: remove

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

  int my_rank, n_rank;
  if (opg) {
    const auto& comm = handle.get_comms();
    my_rank = comm.get_rank();
    n_rank = comm.get_size();
  } else {
    my_rank = 0;
    n_rank = 1;
  }

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
  size_t mSize = raft::alignTo<size_t>(sizeof(bool), align);
  size_t vdSize =
    raft::alignTo<size_t>(sizeof(Index_) * (batchSize + 1), align);
  size_t exScanSize = raft::alignTo<size_t>(sizeof(Index_) * batchSize, align);
  size_t labelsSize = raft::alignTo<size_t>(sizeof(Index_) * N, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(
    N * batchSize < MAX_LABEL,
    "An overflow occurred with the current choice of precision "
    "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
    "Consider using double precision for the output labels.",
    (unsigned long)(MAX_LABEL / N), (unsigned long)batchSize);

  if (workspace == NULL) {
    auto size =
      adjSize + corePtsSize + mSize + vdSize + exScanSize + 2 * labelsSize;
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
  bool* m = (bool*)temp;
  temp += mSize;
  Index_* vd = (Index_*)temp;
  temp += vdSize;
  Index_* ex_scan = (Index_*)temp;
  temp += exScanSize;
  Index_* labelsTemp = (Index_*)temp;
  temp += labelsSize;
  Index_* workBuffer = (Index_*)temp;
  temp += labelsSize;

  int64_t start_time, cur_time;

  /// TODO: do logs and timings make sense for asynchronous operations?

  // Compute the mask
  // 1. Compute the part owned by this worker (reversed order of batches to
  // keep the batch 0 in memory)
  for (int i = nBatches - 1; i >= 0; i--) {
    Index_ startVertexId = start_row + i * batchSize;
    Index_ nPoints = min(size_t(n_owned_rows - i * batchSize), batchSize);
    if (nPoints <= 0) break;
    /// TODO: can this happen? If yes, is the rest of the code correct?

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)nBatches, (unsigned long)nPoints);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
    start_time = raft::curTimeMillis();
    VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algoVd,
                                   startVertexId, nPoints, stream);
    // raft::update_host(&curradjlen, vd + nPoints, 1, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    // CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));

    CUML_LOG_DEBUG("--> Computing core point mask");
    ML::PUSH_RANGE("Trace::Dbscan::CorePoints");
    start_time = raft::curTimeMillis();
    CorePoints::run<Index_>(handle, vd, core_pts, minPts, startVertexId,
                            nPoints, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    // CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));
  }
  // 2. Exchange with the other workers
  if (opg) {
    const auto& comm = handle.get_comms();

    // Array with the size of the contribution of each worker
    Index_ rows_per_rank = raft::ceildiv<Index_>(N, n_rank);
    std::vector<size_t> recvcounts = std::vector<size_t>(n_rank, rows_per_rank);
    recvcounts[n_rank - 1] = N - (n_rank - 1) * rows_per_rank;

    // Array with the displacement of each part
    std::vector<size_t> displs = std::vector<size_t>(n_rank);
    for (int i = 0; i < n_rank; i++) displs[i] = i * rows_per_rank;

    // All-gather operation with variable contribution length
    comm.allgatherv<char>((char*)core_pts + start_row, (char*)core_pts,
                          recvcounts.data(), displs.data(), stream);
    /// TODO: is it ok to use char datatype for bool?
    ASSERT(
      comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
      "An error occurred in the distributed operation. This can result from "
      "a failed rank");
  }

  // Compute the labelling for the owned part of the graph
  raft::sparse::WeakCCState state(m);
  MLCommon::device_buffer<Index_> adj_graph(handle.get_device_allocator(),
                                            stream);

  for (int i = 0; i < nBatches; i++) {
    Index_ startVertexId = start_row + i * batchSize;
    Index_ nPoints = min(size_t(n_owned_rows - i * batchSize), batchSize);
    if (nPoints <= 0) break;

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)nBatches, (unsigned long)nPoints);

    // i==0 -> adj and vd for batch 0 already in memory
    if (i > 0) {
      CUML_LOG_DEBUG("--> Computing vertex degrees");
      ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
      start_time = raft::curTimeMillis();
      VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algoVd,
                                     startVertexId, nPoints, stream);
      cur_time = raft::curTimeMillis();
      ML::POP_RANGE();
      // CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));
    }
    raft::update_host(&curradjlen, vd + nPoints, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUML_LOG_DEBUG("--> Computing adjacency graph of size %ld samples.",
                   (unsigned long)curradjlen);
    ML::PUSH_RANGE("Trace::Dbscan::AdjGraph");
    start_time = raft::curTimeMillis();
    if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
      maxadjlen = curradjlen;
      adj_graph.resize(maxadjlen, stream);
    }
    AdjGraph::run<Index_>(handle, adj, vd, adj_graph.data(), curradjlen,
                          ex_scan, N, algoAdj, nPoints, stream);
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    // CUML_LOG_DEBUG("    |-> Took %ld ms.", (cur_time - start_time));

    CUML_LOG_DEBUG("--> Computing connected components");
    ML::PUSH_RANGE("Trace::Dbscan::WeakCC");
    start_time = raft::curTimeMillis();
    raft::sparse::weak_cc_batched<Index_, 1024>(
      i == 0 ? labels : labelsTemp, ex_scan, adj_graph.data(), curradjlen, N,
      startVertexId, nPoints, &state, stream,
      [core_pts, N] __device__(Index_ global_id) {
        return global_id < N ? __ldg((char*)core_pts + global_id) : 0;
      });
    cur_time = raft::curTimeMillis();
    ML::POP_RANGE();
    // CUML_LOG_DEBUG("    |-> Took %ld ms.", (cur_time - start_time));

    if (i > 0) {
      CUML_LOG_DEBUG("--> Accumulating labels");
      ML::PUSH_RANGE("Trace::Dbscan::MergeLabels");
      start_time = raft::curTimeMillis();
      MergeLabels::run<Index_>(handle, labels, labelsTemp, core_pts, workBuffer,
                               m, N, 0, stream);
      cur_time = raft::curTimeMillis();
      ML::POP_RANGE();
      // CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));
    }
  }

  // Combine the results in the multi-node multi-GPU case

  if (opg) {
    const auto& comm = handle.get_comms();
    raft::comms::request_t request;

    int s = 1;
    while (s < n_rank) {
      CUML_LOG_DEBUG("Tree reduction, s=", s);

      // Find out whether the node is a receiver / sender / passive
      bool receiver = my_rank % (2 * s) == 0 && my_rank + s < n_rank;
      bool sender = my_rank % (2 * s) == s;

      if (receiver) {
        CUML_LOG_DEBUG("--> Receive labels (from %d)", my_rank + s);
        comm.irecv(labelsTemp, N, my_rank + s, 0, &request);
      } else if (sender) {
        CUML_LOG_DEBUG("--> Send labels (from %d)", my_rank - s);
        comm.isend(labels, N, my_rank - s, 0, &request);
      }

      try {
        comm.waitall(1, &request);
      } catch (raft::exception& e) {
        CUML_LOG_DEBUG("Communication failure");
      }

      if (receiver) {
        CUML_LOG_DEBUG("--> Merge labels");
        ML::PUSH_RANGE("Trace::Dbscan::MergeLabels");
        start_time = raft::curTimeMillis();
        MergeLabels::run<Index_>(handle, labels, labelsTemp, core_pts,
                                 workBuffer, m, N, 0, stream);
        cur_time = raft::curTimeMillis();
        ML::POP_RANGE();
        // CUML_LOG_DEBUG("    |-> Took %ld ms", (cur_time - start_time));
      }

      s *= 2;
    }
  }

  /// TODO: optional minimalization step for border points

  // Final relabel
  /// TODO: only rank 0

  ML::PUSH_RANGE("Trace::Dbscan::FinalRelabel");
  if (algoCcl == 2)
    final_relabel(labels, N, stream, handle.get_device_allocator());
  size_t nblks = raft::ceildiv<size_t>(N, TPB);
  relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
  ML::POP_RANGE();

  // Calculate the core_sample_indices only if an array was passed in
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