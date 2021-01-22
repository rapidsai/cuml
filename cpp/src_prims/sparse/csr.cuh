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
#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>

#include <label/classlabels.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>

namespace raft {
namespace sparse {

struct WeakCCState {
 public:
  bool *xa;
  bool *fa;
  bool *m;

  WeakCCState(bool *xa, bool *fa, bool *m) : xa(xa), fa(fa), m(m) {}
};

template <typename Index_, int TPB_X = 256, typename Lambda>
__global__ void weak_cc_label_device(Index_ *labels, const Index_ *row_ind,
                                     const Index_ *row_ind_ptr, Index_ nnz,
                                     bool *fa, bool *xa, bool *m,
                                     Index_ startVertexId, Index_ batchSize,
                                     Index_ N, Lambda filter_op) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  Index_ global_id = tid + startVertexId;
  if (tid < batchSize && global_id < N) {
    if (fa[global_id]) {
      fa[global_id] = false;
      Index_ row_ind_val = row_ind[tid];

      Index_ start = row_ind_val;
      Index_ ci, cj;
      bool ci_mod = false;
      ci = labels[global_id];
      bool ci_allow_prop = filter_op(global_id);

      Index_ degree = get_stop_idx(tid, batchSize, nnz, row_ind) - row_ind_val;
      for (Index_ j = 0; j < degree;
           j++) {  // TODO: Can't this be calculated from the ex_scan?
        Index_ j_ind = row_ind_ptr[start + j];
        cj = labels[j_ind];
        bool cj_allow_prop = filter_op(j_ind);
        if (ci < cj && ci_allow_prop) {
          if (sizeof(Index_) == 4)
            atomicMin((int *)(labels + j_ind), ci);
          else if (sizeof(Index_) == 8)
            atomicMin((long long int *)(labels + j_ind), ci);
          ///@todo see https://github.com/rapidsai/cuml/issues/2306.
          // It may be worth it to use an atomic op here such as
          // atomicLogicalOr(xa + j_ind, cj_allow_prop);
          // Same can be done for m : atomicLogicalOr(m, cj_allow_prop);
          // Both can be done below for xa[global_id] with ci_allow_prop, too.
          xa[j_ind] = true;
          m[0] = true;
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
        xa[global_id] = true;
        m[0] = true;
      }
    }
  }
}

template <typename Index_, int TPB_X = 256, typename Lambda>
__global__ void weak_cc_init_label_kernel(Index_ *labels, Index_ startVertexId,
                                          Index_ batchSize, Index_ MAX_LABEL,
                                          Lambda filter_op) {
  /** F1 and F2 in the paper correspond to fa and xa */
  /** Cd in paper corresponds to db_cluster */
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    Index_ global_id = tid + startVertexId;
    if (filter_op(global_id) && labels[global_id] == MAX_LABEL)
      labels[global_id] = global_id + 1;
  }
}

template <typename Index_, int TPB_X = 256>
__global__ void weak_cc_init_all_kernel(Index_ *labels, bool *fa, bool *xa,
                                        Index_ N, Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    labels[tid] = MAX_LABEL;
    fa[tid] = true;
    xa[tid] = false;
  }
}

template <typename Index_, int TPB_X = 256, typename Lambda>
void weak_cc_label_batched(Index_ *labels, const Index_ *row_ind,
                           const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                           WeakCCState *state, Index_ startVertexId,
                           Index_ batchSize, cudaStream_t stream,
                           Lambda filter_op) {
  ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8,
         "Index_ should be 4 or 8 bytes");

  bool host_m;

  dim3 blocks(raft::ceildiv(batchSize, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  weak_cc_init_label_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
    labels, startVertexId, batchSize, MAX_LABEL, filter_op);
  CUDA_CHECK(cudaPeekAtLastError());

  int n_iters = 0;
  do {
    CUDA_CHECK(cudaMemsetAsync(state->m, false, sizeof(bool), stream));

    weak_cc_label_device<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, row_ind, row_ind_ptr, nnz, state->fa, state->xa, state->m,
      startVertexId, batchSize, N, filter_op);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //** swapping F1 and F2
    std::swap(state->fa, state->xa);

    //** Updating m *
    raft::update_host(&host_m, state->m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    n_iters++;
  } while (host_m);
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
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_, int TPB_X = 256,
          typename Lambda = auto(Index_)->bool>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream, Lambda filter_op) {
  dim3 blocks(raft::ceildiv(N, Index_(TPB_X)));
  dim3 threads(TPB_X);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  if (startVertexId == 0) {
    weak_cc_init_all_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, state->fa, state->xa, N, MAX_LABEL);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  weak_cc_label_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N,
                                       state, startVertexId, batchSize, stream,
                                       filter_op);
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
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 256>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream) {
  weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, startVertexId,
                  batchSize, state, stream,
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
  MLCommon::device_buffer<bool> xa(d_alloc, stream, N);
  MLCommon::device_buffer<bool> fa(d_alloc, stream, N);
  MLCommon::device_buffer<bool> m(d_alloc, stream, 1);

  WeakCCState state(xa.data(), fa.data(), m.data());
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
  MLCommon::device_buffer<bool> xa(d_alloc, stream, N);
  MLCommon::device_buffer<bool> fa(d_alloc, stream, N);
  MLCommon::device_buffer<bool> m(d_alloc, stream, 1);
  WeakCCState state(xa.data(), fa.data(), m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, [](Index_) { return true; });
}

};  // namespace sparse
};  // namespace raft
