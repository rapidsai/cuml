/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "haversine_knn.cuh"
#include "processing.cuh"

#include <label/classlabels.cuh>

#include <cuml/neighbors/knn.hpp>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/utils/Heap.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>
#include <iostream>
#include <set>

namespace MLCommon {
namespace Selection {

template <bool precomp_lbls, typename T>
inline __device__ T get_lbls(const T* labels, const int64_t* knn_indices, int64_t idx)
{
  if (precomp_lbls) {
    return labels[idx];
  } else {
    int64_t neighbor_idx = knn_indices[idx];
    return labels[neighbor_idx];
  }
}

template <typename OutType = float, bool precomp_lbls = false>
__global__ void class_probs_kernel(OutType* out,
                                   const int64_t* knn_indices,
                                   const int* labels,
                                   int n_uniq_labels,
                                   std::size_t n_samples,
                                   int n_neighbors)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i   = row * n_neighbors;

  float n_neigh_inv = 1.0f / n_neighbors;

  if (row >= n_samples) return;

  for (int j = 0; j < n_neighbors; j++) {
    int out_label = get_lbls<precomp_lbls>(labels, knn_indices, i + j);
    int out_idx   = row * n_uniq_labels + out_label;
    out[out_idx] += n_neigh_inv;
  }
}

template <typename OutType = int>
__global__ void class_vote_kernel(OutType* out,
                                  const float* class_proba,
                                  int* unique_labels,
                                  int n_uniq_labels,
                                  std::size_t n_samples,
                                  int n_outputs,
                                  int output_offset,
                                  bool use_shared_mem)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i   = row * n_uniq_labels;

  extern __shared__ int label_cache[];
  if (use_shared_mem) {
    for (int j = threadIdx.x; j < n_uniq_labels; j += blockDim.x) {
      label_cache[j] = unique_labels[j];
    }

    __syncthreads();
  }

  if (row >= n_samples) return;
  float cur_max = -1.0;
  int cur_label = -1;
  for (int j = 0; j < n_uniq_labels; j++) {
    float cur_proba = class_proba[i + j];
    if (cur_proba > cur_max) {
      cur_max   = cur_proba;
      cur_label = j;
    }
  }

  int val = use_shared_mem ? label_cache[cur_label] : unique_labels[cur_label];

  out[row * n_outputs + output_offset] = val;
}

template <typename LabelType, bool precomp_lbls = false>
__global__ void regress_avg_kernel(LabelType* out,
                                   const int64_t* knn_indices,
                                   const LabelType* labels,
                                   std::size_t n_samples,
                                   int n_neighbors,
                                   int n_outputs,
                                   int output_offset)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i   = row * n_neighbors;

  if (row >= n_samples) return;

  LabelType pred = 0;
  for (int j = 0; j < n_neighbors; j++) {
    pred += get_lbls<precomp_lbls>(labels, knn_indices, i + j);
  }

  out[row * n_outputs + output_offset] = pred / (LabelType)n_neighbors;
}

/**
 * A naive knn classifier to predict probabilities
 * @tparam TPB_X number of threads per block to use. each thread
 *               will process a single row of knn_indices
 * @tparam precomp_lbls is set to true for the reduction step of MNMG KNN Classifier. In this case,
 *         the knn_indices array is not used as the y arrays already store the labels for each row.
 *         This makes it possible to compute the reduction step without holding all the data on a
 * single machine.
 * @param[out] out vector of output class probabilities of the same size as y.
 *            each element should be of size size (n_samples * n_classes[i])
 * @param[in] knn_indices the index array resulting from a knn search
 * @param[in] y vector of label arrays. for multulabel classification,
 *          each output in the vector is a different array of labels
 *          corresponding to the i'th output.
 * @param[in] n_index_rows number of vertices in index (eg. size of each y array)
 * @param[in] n_query_rows number of rows in knn_indices
 * @param[in] k number of neighbors in knn_indices
 * @param[in] uniq_labels vector of the sorted unique labels for each array in y
 * @param[in] n_unique vector of sizes for each array in uniq_labels
 * @param[in] user_stream main stream to use for queuing isolated CUDA events
 * @param[in] int_streams internal streams to use for parallelizing independent CUDA events.
 * @param[in] n_int_streams number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32, bool precomp_lbls = false>
void class_probs(const raft::handle_t& handle,
                 std::vector<float*>& out,
                 const int64_t* knn_indices,
                 std::vector<int*>& y,
                 std::size_t n_index_rows,
                 std::size_t n_query_rows,
                 int k,
                 std::vector<int*>& uniq_labels,
                 std::vector<int>& n_unique)
{
  for (std::size_t i = 0; i < y.size(); i++) {
    cudaStream_t stream = handle.get_next_usable_stream();

    int n_unique_labels = n_unique[i];
    size_t cur_size     = n_query_rows * n_unique_labels;

    RAFT_CUDA_TRY(cudaMemsetAsync(out[i], 0, cur_size * sizeof(float), stream));

    dim3 grid(raft::ceildiv(n_query_rows, static_cast<std::size_t>(TPB_X)), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    /**
     * Build array of class probability arrays from
     * knn_indices and labels
     */
    rmm::device_uvector<int> y_normalized(n_index_rows + n_unique_labels, stream);

    /*
     * Appending the array of unique labels to the original labels array
     * to prevent make_monotonic function from producing misleading results
     * due to the absence of some of the unique labels in the labels array
     */
    rmm::device_uvector<int> y_tmp(n_index_rows + n_unique_labels, stream);
    raft::update_device(y_tmp.data(), y[i], n_index_rows, stream);
    raft::update_device(y_tmp.data() + n_index_rows, uniq_labels[i], n_unique_labels, stream);

    MLCommon::Label::make_monotonic(y_normalized.data(), y_tmp.data(), y_tmp.size(), stream);
    raft::linalg::unaryOp<int>(
      y_normalized.data(),
      y_normalized.data(),
      n_index_rows,
      [] __device__(int input) { return input - 1; },
      stream);
    class_probs_kernel<float, precomp_lbls><<<grid, blk, 0, stream>>>(
      out[i], knn_indices, y_normalized.data(), n_unique_labels, n_query_rows, k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

/**
 * KNN classifier using voting based on the statistical mode of classes.
 * In the event of a tie, the class with the lowest index in the sorted
 * array of unique monotonically increasing labels will be used.
 *
 * @tparam TPB_X the number of threads per block to use
 * @tparam precomp_lbls is set to true for the reduction step of MNMG KNN Classifier. In this case,
 * the knn_indices array is not used as the y arrays already store the labels for each row.
 * This makes it possible to compute the reduction step without holding all the data on a single
 * machine.
 * @param[out] out output array of size (n_samples * y.size())
 * @param[in] knn_indices index array from knn search
 * @param[in] y vector of label arrays. for multilabel classification, each
 *          element in the vector is a different "output" array of labels corresponding
 *          to the i'th output.
 * @param[in] n_index_rows number of vertices in index (eg. size of each y array)
 * @param[in] n_query_rows number of rows in knn_indices
 * @param[in] k number of neighbors in knn_indices
 * @param[in] uniq_labels vector of the sorted unique labels for each array in y
 * @param[in] n_unique vector of sizes for each array in uniq_labels
 * @param[in] user_stream main stream to use for queuing isolated CUDA events
 * @param[in] int_streams internal streams to use for parallelizing independent CUDA events.
 * @param[in] n_int_streams number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32, bool precomp_lbls = false>
void knn_classify(const raft::handle_t& handle,
                  int* out,
                  const int64_t* knn_indices,
                  std::vector<int*>& y,
                  std::size_t n_index_rows,
                  std::size_t n_query_rows,
                  int k,
                  std::vector<int*>& uniq_labels,
                  std::vector<int>& n_unique)
{
  std::vector<float*> probs;
  std::vector<rmm::device_uvector<float>> tmp_probs;

  // allocate temporary memory
  for (std::size_t i = 0; i < n_unique.size(); i++) {
    int size = n_unique[i];

    cudaStream_t stream = handle.get_next_usable_stream(i);

    tmp_probs.emplace_back(n_query_rows * size, stream);
    probs.push_back(tmp_probs.back().data());
  }

  /**
   * Compute class probabilities
   *
   * Note: Since class_probs will use the same round robin strategy for distributing
   * work to the streams, we don't need to explicitly synchronize the streams here.
   */
  class_probs<32, precomp_lbls>(
    handle, probs, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique);

  dim3 grid(raft::ceildiv(n_query_rows, static_cast<std::size_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  for (std::size_t i = 0; i < y.size(); i++) {
    cudaStream_t stream = handle.get_next_usable_stream(i);

    int n_unique_labels = n_unique[i];

    /**
     * Choose max probability
     */
    // Use shared memory for label lookups if the number of classes is small enough
    int smem            = sizeof(int) * n_unique_labels;
    bool use_shared_mem = smem < raft::getSharedMemPerBlock();

    class_vote_kernel<<<grid, blk, use_shared_mem ? smem : 0, stream>>>(
      out, probs[i], uniq_labels[i], n_unique_labels, n_query_rows, y.size(), i, use_shared_mem);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

/**
 * KNN regression using voting based on the mean of the labels for the
 * nearest neighbors.
 * @tparam ValType data type of the labels
 * @tparam TPB_X the number of threads per block to use
 * @tparam precomp_lbls is set to true for the reduction step of MNMG KNN Regressor. In this case,
 * the knn_indices array is not used as the y arrays already store the output for each row.
 * This makes it possible to compute the reduction step without holding all the data on a single
 * machine.
 * @param[out] out output array of size (n_samples * y.size())
 * @param[in] knn_indices index array from knn search
 * @param[in] y vector of label arrays. for multilabel classification, each
 *          element in the vector is a different "output" array of labels corresponding
 *          to the i'th output.
 * @param[in] n_index_rows number of vertices in index (eg. size of each y array)
 * @param[in] n_query_rows number of rows in knn_indices
 * @param[in] k number of neighbors in knn_indices
 * @param[in] user_stream main stream to use for queuing isolated CUDA events
 * @param[in] int_streams internal streams to use for parallelizing independent CUDA events.
 * @param[in] n_int_streams number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */

template <typename ValType, int TPB_X = 32, bool precomp_lbls = false>
void knn_regress(const raft::handle_t& handle,
                 ValType* out,
                 const int64_t* knn_indices,
                 const std::vector<ValType*>& y,
                 size_t n_index_rows,
                 size_t n_query_rows,
                 int k)
{
  /**
   * Vote average regression value
   */
  for (std::size_t i = 0; i < y.size(); i++) {
    cudaStream_t stream = handle.get_next_usable_stream();

    regress_avg_kernel<ValType, precomp_lbls>
      <<<raft::ceildiv(n_query_rows, static_cast<std::size_t>(TPB_X)), TPB_X, 0, stream>>>(
        out, knn_indices, y[i], n_query_rows, k, y.size(), i);

    handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

};  // namespace Selection
};  // namespace MLCommon
