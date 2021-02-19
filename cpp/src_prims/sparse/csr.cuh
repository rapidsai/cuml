/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <specializations/raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>

#include <label/classlabels.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "utils.h"

namespace raft {
namespace sparse {

struct WeakCCState {
 public:
  bool *m;
  WeakCCState(bool *m) : m(m) {}
};

template <typename Index_, int TPB_X = 256, typename Lambda>
__global__ void weak_cc_label_device(Index_ *__restrict__ labels,
                                     const Index_ *__restrict__ row_ind,
                                     const Index_ *__restrict__ row_ind_ptr,
                                     Index_ nnz, bool *__restrict__ m,
                                     Index_ start_vertex_id, Index_ batch_size,
                                     Index_ N, Lambda filter_op) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  Index_ global_id = tid + start_vertex_id;
  if (tid < batch_size && global_id < N) {
    Index_ start = __ldg(row_ind + tid);

    Index_ ci, cj;
    bool ci_mod = false;
    ci = labels[global_id];
    bool ci_allow_prop = filter_op(global_id);

    Index_ end = get_stop_idx(tid, batch_size, nnz, row_ind);
    /// TODO: add one element to row_ind and avoid get_stop_idx
    for (Index_ j = start; j < end; j++) {
      Index_ j_ind = __ldg(row_ind_ptr + j);
      cj = labels[j_ind];
      bool cj_allow_prop = filter_op(j_ind);
      if (ci < cj && ci_allow_prop) {
        if (sizeof(Index_) == 4)
          atomicMin((int *)(labels + j_ind), ci);
        else if (sizeof(Index_) == 8)
          atomicMin((long long int *)(labels + j_ind), ci);
        if (cj_allow_prop) *m = true;
      } else if (ci > cj && cj_allow_prop) {
        ci = cj;
        ci_mod = true;
      }
    }
    if (ci_mod) {
      if (sizeof(Index_) == 4)
        atomicMin((int *)(labels + global_id), ci);
      else if (sizeof(Index_) == 8)
        atomicMin((long long int *)(labels + global_id), ci);
      if (ci_allow_prop) *m = true;
    }
  }
}

template <typename Index_, int TPB_X = 256, typename Lambda>
__global__ void weak_cc_init_all_kernel(Index_ *labels, Index_ N,
                                        Index_ MAX_LABEL, Lambda filter_op) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (filter_op(tid))
      labels[tid] = tid + 1;
    else
      labels[tid] = MAX_LABEL;
  }
}  // namespace sparse

/**
 * @brief Partial calculation of the weakly connected components in the
 * context of a batched algorithm: the labels are computed wrt the sub-graph
 * represented by the given CSR matrix of dimensions batch_size * N.
 * Note that this overwrites the labels array and it is the responsibility of
 * the caller to combine the results from different batches
 * (cf label/merge_labels.cuh)
 *
 * @tparam Index_ the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param start_vertex_id the starting vertex index for the current batch
 * @param batch_size number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_, int TPB_X = 256,
          typename Lambda = auto(Index_)->bool>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ start_vertex_id, Index_ batch_size,
                     WeakCCState *state, cudaStream_t stream,
                     Lambda filter_op) {
  ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8,
         "Index_ should be 4 or 8 bytes");

  bool host_m;

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  weak_cc_init_all_kernel<Index_, TPB_X>
    <<<raft::ceildiv(N, Index_(TPB_X)), TPB_X, 0, stream>>>(
      labels, N, MAX_LABEL, filter_op);
  CUDA_CHECK(cudaPeekAtLastError());

  int n_iters = 0;
  do {
    CUDA_CHECK(cudaMemsetAsync(state->m, false, sizeof(bool), stream));

    weak_cc_label_device<Index_, TPB_X>
      <<<raft::ceildiv(batch_size, Index_(TPB_X)), TPB_X, 0, stream>>>(
        labels, row_ind, row_ind_ptr, nnz, state->m, start_vertex_id,
        batch_size, N, filter_op);
    CUDA_CHECK(cudaPeekAtLastError());

    //** Updating m *
    raft::update_host(&host_m, state->m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    n_iters++;
  } while (host_m);
}

/**
 * @brief Partial calculation of the weakly connected components in the
 * context of a batched algorithm: the labels are computed wrt the sub-graph
 * represented by the given CSR matrix of dimensions batch_size * N.
 * Note that this overwrites the labels array and it is the responsibility of
 * the caller to combine the results from different batches
 * (cf label/merge_labels.cuh)
 *
 * @tparam Index_ the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param start_vertex_id the starting vertex index for the current batch
 * @param batch_size number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 256>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ start_vertex_id, Index_ batch_size,
                     WeakCCState *state, cudaStream_t stream) {
  weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, start_vertex_id,
                  batch_size, state, stream,
                  [] __device__(Index_ tid) { return true; });
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param d_alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_ = int, int TPB_X = 256,
          typename Lambda = auto(Index_)->bool>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N,
             std::shared_ptr<raft::mr::device::allocator> d_alloc,
             cudaStream_t stream, Lambda filter_op) {
  MLCommon::device_buffer<bool> m(d_alloc, stream, 1);

  WeakCCState state(m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param d_alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 256>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N,
             std::shared_ptr<raft::mr::device::allocator> d_alloc,
             cudaStream_t stream) {
  MLCommon::device_buffer<bool> m(d_alloc, stream, 1);
  WeakCCState state(m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, [](Index_) { return true; });
}

};  // namespace sparse
};  // namespace raft
