/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <common/nvtx.hpp>
#include <cuml/common/device_buffer.hpp>
#include <label/classlabels.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/sparse/csr.cuh>
#include "adjgraph/runner.cuh"
#include "corepoints/compute.cuh"
#include "corepoints/exchange.cuh"
#include "mergelabels/runner.cuh"
#include "mergelabels/tree_reduction.cuh"
#include "vertexdeg/runner.cuh"

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
 * Run the DBSCAN algorithm (common code for single-GPU and multi-GPU)
 * @tparam opg Whether we are running in a multi-node multi-GPU context
 * @param[in]  handle       raft handle
 * @param[in]  x            Input data (N*D row-major device array)
 * @param[in]  N            Number of points
 * @param[in]  D            Dimensionality of the points
 * @param[in]  start_row    Index of the offset for this node
 * @param[in]  n_owned_rows Number of rows (points) owned by this node
 * @param[in]  eps          Epsilon neighborhood criterion
 * @param[in]  min_pts      Core points criterion
 * @param[out] labels       Output labels (device array of length N)
 * @param[out] core_indices If not nullptr, the indices of core points are written in this array
 * @param[in]  algo_vd      Algorithm used for the vertex degrees
 * @param[in]  algo_adj     Algorithm used for the adjacency graph
 * @param[in]  algo_ccl     Algorithm used for the final relabel
 * @param[in]  workspace    Temporary global memory buffer used to store intermediate computations
 *                          If nullptr, then this function will return the workspace size needed.
 *                          It is the responsibility of the user to allocate and free this buffer!
 * @param[in]  batch_size   Batch size
 * @param[in]  stream       The CUDA stream where to launch the kernels
 * @return In case the workspace pointer is null, this returns the size needed.
 */
template <typename Type_f, typename Index_ = int, bool opg = false>
size_t run(const raft::handle_t& handle, const Type_f* x, Index_ N, Index_ D,
           Index_ start_row, Index_ n_owned_rows, Type_f eps, Index_ min_pts,
           Index_* labels, Index_* core_indices, int algo_vd, int algo_adj,
           int algo_ccl, void* workspace, Index_ batch_size,
           cudaStream_t stream) {
  const size_t align = 256;
  Index_ n_batches = raft::ceildiv<Index_>(n_owned_rows, batch_size);

  int my_rank;
  if (opg) {
    const auto& comm = handle.get_comms();
    my_rank = comm.get_rank();
  } else
    my_rank = 0;

  /**
   * Note on coupling between data types:
   * - adjacency graph has a worst case size of N * batch_size elements. Thus,
   * if N is very close to being greater than the maximum 32-bit IdxType type used, a
   * 64-bit IdxType should probably be used instead.
   * - exclusive scan is the CSR row index for the adjacency graph and its values have a
   * risk of overflowing when N * batch_size becomes larger what can be stored in IdxType
   * - the vertex degree array has a worst case of each element having all other
   * elements in their neighborhood, so any IdxType can be safely used, so long as N doesn't
   * overflow.
   */
  size_t adj_size = raft::alignTo<size_t>(sizeof(bool) * N * batch_size, align);
  size_t core_pts_size = raft::alignTo<size_t>(sizeof(bool) * N, align);
  size_t m_size = raft::alignTo<size_t>(sizeof(bool), align);
  size_t vd_size =
    raft::alignTo<size_t>(sizeof(Index_) * (batch_size + 1), align);
  size_t ex_scan_size =
    raft::alignTo<size_t>(sizeof(Index_) * batch_size, align);
  size_t labels_size = raft::alignTo<size_t>(sizeof(Index_) * N, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(
    N * batch_size < MAX_LABEL,
    "An overflow occurred with the current choice of precision "
    "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
    "Consider using double precision for the output labels.",
    (unsigned long)(MAX_LABEL / N), (unsigned long)batch_size);

  if (workspace == NULL) {
    auto size = adj_size + core_pts_size + m_size + vd_size + ex_scan_size +
                2 * labels_size;
    return size;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  Index_ maxadjlen = 0;
  Index_ curradjlen = 0;
  char* temp = (char*)workspace;
  bool* adj = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  // Compute the mask
  // 1. Compute the part owned by this worker (reversed order of batches to
  // keep the batch 0 in memory)
  for (int i = n_batches - 1; i >= 0; i--) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points = min(n_owned_rows - i * batch_size, batch_size);

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)n_batches, (unsigned long)n_points);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
    VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algo_vd,
                                   start_vertex_id, n_points, stream);
    ML::POP_RANGE();

    CUML_LOG_DEBUG("--> Computing core point mask");
    ML::PUSH_RANGE("Trace::Dbscan::CorePoints");
    CorePoints::compute<Index_>(handle, vd, core_pts, min_pts, start_vertex_id,
                                n_points, stream);
    ML::POP_RANGE();
  }
  // 2. Exchange with the other workers
  if (opg) CorePoints::exchange(handle, core_pts, N, start_row, stream);

  // Compute the labelling for the owned part of the graph
  raft::sparse::WeakCCState state(m);
  MLCommon::device_buffer<Index_> adj_graph(handle.get_device_allocator(),
                                            stream);

  for (int i = 0; i < n_batches; i++) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points = min(n_owned_rows - i * batch_size, batch_size);
    if (n_points <= 0) break;

    CUML_LOG_DEBUG("- Batch %d / %ld (%ld samples)", i + 1,
                   (unsigned long)n_batches, (unsigned long)n_points);

    // i==0 -> adj and vd for batch 0 already in memory
    if (i > 0) {
      CUML_LOG_DEBUG("--> Computing vertex degrees");
      ML::PUSH_RANGE("Trace::Dbscan::VertexDeg");
      VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algo_vd,
                                     start_vertex_id, n_points, stream);
      ML::POP_RANGE();
    }
    raft::update_host(&curradjlen, vd + n_points, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUML_LOG_DEBUG("--> Computing adjacency graph of size %ld samples.",
                   (unsigned long)curradjlen);
    ML::PUSH_RANGE("Trace::Dbscan::AdjGraph");
    if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
      maxadjlen = curradjlen;
      adj_graph.resize(maxadjlen, stream);
    }
    AdjGraph::run<Index_>(handle, adj, vd, adj_graph.data(), curradjlen,
                          ex_scan, N, algo_adj, n_points, stream);
    ML::POP_RANGE();

    CUML_LOG_DEBUG("--> Computing connected components");
    ML::PUSH_RANGE("Trace::Dbscan::WeakCC");
    raft::sparse::weak_cc_batched<Index_, 1024>(
      i == 0 ? labels : labels_temp, ex_scan, adj_graph.data(), curradjlen, N,
      start_vertex_id, n_points, &state, stream,
      [core_pts, N] __device__(Index_ global_id) {
        return global_id < N ? __ldg((char*)core_pts + global_id) : 0;
      });
    ML::POP_RANGE();

    if (i > 0) {
      // The labels_temp array contains the labelling for the neighborhood
      // graph of the current batch. This needs to be merged with the labelling
      // created by the previous batches.
      // Using the labelling from the previous batches as initial value for
      // weak_cc_batched and skipping the merge step would lead to incorrect
      // results as described in #3094.
      CUML_LOG_DEBUG("--> Accumulating labels");
      ML::PUSH_RANGE("Trace::Dbscan::MergeLabels");
      MergeLabels::run<Index_>(handle, labels, labels_temp, core_pts,
                               work_buffer, m, N, stream);
      ML::POP_RANGE();
    }
  }

  // Combine the results in the multi-node multi-GPU case
  if (opg)
    MergeLabels::tree_reduction(handle, labels, labels_temp, core_pts,
                                work_buffer, m, N, stream);

  /// TODO: optional minimalization step for border points

  // Final relabel
  if (my_rank == 0) {
    ML::PUSH_RANGE("Trace::Dbscan::FinalRelabel");
    if (algo_ccl == 2)
      final_relabel(labels, N, stream, handle.get_device_allocator());
    size_t nblks = raft::ceildiv<size_t>(N, TPB);
    relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
    CUDA_CHECK(cudaPeekAtLastError());
    ML::POP_RANGE();

    // Calculate the core_indices only if an array was passed in
    if (core_indices != nullptr) {
      ML::PUSH_RANGE("Trace::Dbscan::CoreSampleIndices");

      // Create the execution policy
      ML::thrustAllocatorAdapter alloc(handle.get_device_allocator(), stream);
      auto thrust_exec_policy = thrust::cuda::par(alloc).on(stream);

      // Get wrappers for the device ptrs
      thrust::device_ptr<bool> dev_core_pts =
        thrust::device_pointer_cast(core_pts);
      thrust::device_ptr<Index_> dev_core_indices =
        thrust::device_pointer_cast(core_indices);

      // First fill the core_indices with -1 which will be used if core_point_count < N
      thrust::fill_n(thrust_exec_policy, dev_core_indices, N, (Index_)-1);

      auto index_iterator = thrust::counting_iterator<Index_>(0);

      //Perform stream reduction on the core points. The core_pts acts as the stencil and we use thrust::counting_iterator to return the index
      auto core_point_count = thrust::copy_if(
        thrust_exec_policy, index_iterator, index_iterator + N, dev_core_pts,
        dev_core_indices,
        [=] __device__(const bool is_core_point) { return is_core_point; });

      ML::POP_RANGE();
    }
  }

  CUML_LOG_DEBUG("Done.");
  return (size_t)0;
}
}  // namespace Dbscan
}  // namespace ML