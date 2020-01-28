/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "cuda_utils.h"

#include "distance/distance.h"

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"

#include <iostream>

namespace MLCommon {
namespace Selection {

/**
 * @brief Simple utility function to determine whether user_stream or one of the
 * internal streams should be used.
 * @param user_stream main user stream
 * @param int_streams array of internal streams
 * @param n_int_streams number of internal streams
 * @param idx the index for which to query the stream
 */
inline cudaStream_t select_stream(cudaStream_t user_stream,
                                  cudaStream_t *int_streams, int n_int_streams,
                                  int idx) {
  return n_int_streams > 0 ? int_streams[idx % n_int_streams] : user_stream;
}

template <int warp_q, int thread_q, int tpb>
__global__ void knn_merge_parts_kernel(float *inK, int64_t *inV, float *outK,
                                       int64_t *outV, size_t n_samples,
                                       int n_parts, float initK, int64_t initV,
                                       int k, int64_t *translations) {
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ float smemK[kNumWarps * warp_q];
  __shared__ int64_t smemV[kNumWarps * warp_q];

  /**
   * Uses shared memory
   */
  faiss::gpu::BlockSelect<float, int64_t, false, faiss::gpu::Comparator<float>,
                          warp_q, thread_q, tpb>
    heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;
  int total_k = k * n_parts;

  int i = threadIdx.x;

  // Get starting pointers for cols in current thread
  int part = i / k;
  size_t row_idx = (row * k) + (part * n_samples * k);

  int col = i % k;

  float *inKStart = inK + (row_idx + col);
  int64_t *inVStart = inV + (row_idx + col);

  int limit = faiss::gpu::utils::roundDown(total_k, faiss::gpu::kWarpSize);
  int64_t translation = 0;

  for (; i < limit; i += tpb) {
    translation = translations[part];
    heap.add(*inKStart, (*inVStart) + translation);

    part = (i + tpb) / k;
    row_idx = (row * k) + (part * n_samples * k);

    col = (i + tpb) % k;

    inKStart = inK + (row_idx + col);
    inVStart = inV + (row_idx + col);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < total_k) {
    translation = translations[part];
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

template <int warp_q, int thread_q>
inline void knn_merge_parts_impl(float *inK, int64_t *inV, float *outK,
                                 int64_t *outV, size_t n_samples, int n_parts,
                                 int k, cudaStream_t stream,
                                 int64_t *translations) {
  auto grid = dim3(n_samples);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block = dim3(n_threads);

  auto kInit = faiss::gpu::Limits<float>::getMax();
  auto vInit = -1;
  knn_merge_parts_kernel<warp_q, thread_q, n_threads>
    <<<grid, block, 0, stream>>>(inK, inV, outK, outV, n_samples, n_parts,
                                 kInit, vInit, k, translations);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
inline void knn_merge_parts(float *inK, int64_t *inV, float *outK,
                            int64_t *outV, size_t n_samples, int n_parts, int k,
                            cudaStream_t stream, int64_t *translations) {
  if (k == 1)
    knn_merge_parts_impl<1, 1>(inK, inV, outK, outV, n_samples, n_parts, k,
                               stream, translations);
  else if (k <= 32)
    knn_merge_parts_impl<32, 2>(inK, inV, outK, outV, n_samples, n_parts, k,
                                stream, translations);
  else if (k <= 64)
    knn_merge_parts_impl<64, 3>(inK, inV, outK, outV, n_samples, n_parts, k,
                                stream, translations);
  else if (k <= 128)
    knn_merge_parts_impl<128, 3>(inK, inV, outK, outV, n_samples, n_parts, k,
                                 stream, translations);
  else if (k <= 256)
    knn_merge_parts_impl<256, 4>(inK, inV, outK, outV, n_samples, n_parts, k,
                                 stream, translations);
  else if (k <= 512)
    knn_merge_parts_impl<512, 8>(inK, inV, outK, outV, n_samples, n_parts, k,
                                 stream, translations);
  else if (k <= 1024)
    knn_merge_parts_impl<1024, 8>(inK, inV, outK, outV, n_samples, n_parts, k,
                                  stream, translations);
}

/**
   * Search the kNN for the k-nearest neighbors of a set of query vectors
   * @param input vector of device device memory array pointers to search
   * @param sizes vector of memory sizes for each device array pointer in input
   * @param D number of cols in input and search_items
   * @param search_items set of vectors to query for neighbors
   * @param n        number of items in search_items
   * @param res_I    pointer to device memory for returning k nearest indices
   * @param res_D    pointer to device memory for returning k nearest distances
   * @param k        number of neighbors to query
   * @param allocator the device memory allocator to use for temporary scratch memory
   * @param userStream the main cuda stream to use
   * @param internalStreams optional when n_params > 0, the index partitions can be
   *        queried in parallel using these streams. Note that n_int_streams also
   *        has to be > 0 for these to be used and their cardinality does not need
   *        to correspond to n_parts.
   * @param n_int_streams size of internalStreams. When this is <= 0, only the
   *        user stream will be used.
   * @param rowMajorIndex are the index arrays in row-major layout?
   * @param rowMajorQuery are the query array in row-major layout?
   * @param translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   */
template <typename IntType = int,
          Distance::DistanceType DistanceType = Distance::EucUnexpandedL2>
void brute_force_knn(std::vector<float *> &input, std::vector<int> &sizes,
                     IntType D, float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
                     cudaStream_t *internalStreams = nullptr,
                     int n_int_streams = 0, bool rowMajorIndex = true,
                     bool rowMajorQuery = true,
                     std::vector<int64_t> *translations = nullptr) {
  ASSERT(DistanceType == Distance::EucUnexpandedL2 ||
           DistanceType == Distance::EucUnexpandedL2Sqrt,
         "Only EucUnexpandedL2Sqrt and EucUnexpandedL2 metrics are supported "
         "currently.");

  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors should be the same size");

  std::vector<int64_t> *id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (int i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  device_buffer<int64_t> trans(allocator, userStream, id_ranges->size());
  updateDevice(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  device_buffer<float> all_D(allocator, userStream, input.size() * k * n);
  device_buffer<int64_t> all_I(allocator, userStream, input.size() * k * n);

  // Sync user stream only if using other streams to parallelize query
  if (n_int_streams > 0) CUDA_CHECK(cudaStreamSynchronize(userStream));

  for (int i = 0; i < input.size(); i++) {
    faiss::gpu::StandardGpuResources gpu_res;

    cudaStream_t stream =
      select_stream(userStream, internalStreams, n_int_streams, i);

    gpu_res.noTempMemory();
    gpu_res.setCudaMallocWarning(false);
    gpu_res.setDefaultStream(device, stream);

    faiss::gpu::bruteForceKnn(
      &gpu_res, faiss::METRIC_L2, input[i], rowMajorIndex, sizes[i],
      search_items, rowMajorQuery, n, D, k, all_D.data() + (i * k * n),
      all_I.data() + (i * k * n));

    CUDA_CHECK(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  for (int i = 0; i < n_int_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(internalStreams[i]));
  }

  knn_merge_parts(all_D.data(), all_I.data(), res_D, res_I, n, input.size(), k,
                  userStream, trans.data());

  MLCommon::LinAlg::unaryOp<float>(
    res_D, res_D, n * k, [] __device__(float input) { return sqrt(input); },
    userStream);

  if (translations == nullptr) delete id_ranges;
};

template <typename IntType = int,
          Distance::DistanceType DistanceType = Distance::EucUnexpandedL2>
void brute_force_knn(float **input, int *sizes, int n_params, IntType D,
                     float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
                     cudaStream_t *internalStreams = nullptr,
                     int n_int_streams = 0, bool rowMajorIndex = true,
                     bool rowMajorQuery = true,
                     std::vector<int64_t> *translations = nullptr) {
  std::vector<float *> input_vec(n_params);
  std::vector<int> sizes_vec(n_params);

  for (int i = 0; i < n_params; i++) {
    input_vec.push_back(input[i]);
    sizes_vec.push_back(sizes[i]);
  }

  brute_force_knn<IntType, DistanceType>(
    input_vec, sizes_vec, D, search_items, n, res_I, res_D, k, allocator,
    userStream, internalStreams, n_int_streams, rowMajorIndex, rowMajorQuery,
    translations);
}

/**
 * @brief Binary tree recursion for finding a label in the unique_labels array.
 * This provides a good middle-ground between having to create a new
 * labels array just to map non-monotonically increasing labels, or
 * the alternative, which is having to search over O(n) space for the labels
 * array in each thread. This is going to cause warp divergence of log(n)
 * per iteration.
 * @param unique_labels array of unique labels
 * @param n_labels number of unique labels
 * @param target_val the label value to search for in unique_labels
 */
template <typename IdxType = int>
__device__ int label_binary_search(IdxType *unique_labels, IdxType n_labels,
                                   IdxType target_val) {
  int out_label_idx = -1;

  int level = 1;
  int cur_break_idx = round(n_labels / (2.0 * level));
  while (out_label_idx == -1) {
    int cur_cached_label = unique_labels[cur_break_idx];

    // If we found our label, terminate
    if (cur_cached_label == target_val) {
      return cur_break_idx;

      // check left neighbor
    } else if (cur_break_idx > 0 &&
               unique_labels[cur_break_idx - 1] == target_val) {
      return cur_break_idx - 1;

      // check right neighbor
    } else if (cur_break_idx < n_labels - 1 &&
               unique_labels[cur_break_idx + 1] == target_val) {
      return cur_break_idx + 1;

      // traverse
    } else {
      level += 1;

      int subtree = round(n_labels / (2.0 * level));
      if (target_val < cur_cached_label) {
        // take left subtree
        cur_break_idx -= subtree;
      } else {
        // take right subtree
        cur_break_idx += subtree;
      }
    }
  }
  return -1;
}

template <typename OutType = float>
__global__ void class_probs_kernel(OutType *out, const int64_t *knn_indices,
                                   const int *labels, int *unique_labels,
                                   int n_uniq_labels, size_t n_samples,
                                   int n_neighbors) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_neighbors;

  float n_neigh_inv = 1.0f / n_neighbors;

  extern __shared__ int label_cache[];
  for (int j = threadIdx.x; j < n_uniq_labels; j += blockDim.x) {
    label_cache[j] = unique_labels[j];
  }

  __syncthreads();

  if (row >= n_samples) return;

  for (int j = 0; j < n_neighbors; j++) {
    int64_t neighbor_idx = knn_indices[i + j];
    int out_label = labels[neighbor_idx];

    // Trading off warp divergence in the outputs so that we don't
    // need to copy / modify the label memory to do these mappings.
    // Found a middle-ground between between using shared memory
    // for the mappings.
    int out_label_idx =
      label_binary_search(label_cache, n_uniq_labels, out_label);

    int out_idx = row * n_uniq_labels + out_label_idx;
    out[out_idx] += n_neigh_inv;
  }
}

template <typename OutType = int>
__global__ void class_vote_kernel(OutType *out, const float *class_proba,
                                  int *unique_labels, int n_uniq_labels,
                                  size_t n_samples, int n_outputs,
                                  int output_offset) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_uniq_labels;

  extern __shared__ int label_cache[];
  for (int j = threadIdx.x; j < n_uniq_labels; j += blockDim.x) {
    label_cache[j] = unique_labels[j];
  }

  __syncthreads();

  if (row >= n_samples) return;
  float cur_max = -1.0;
  int cur_label = -1;
  for (int j = 0; j < n_uniq_labels; j++) {
    float cur_count = class_proba[i + j];
    if (cur_count > cur_max) {
      cur_max = cur_count;
      cur_label = j;
    }
  }
  out[row * n_outputs + output_offset] = label_cache[cur_label];
}

template <typename LabelType>
__global__ void regress_avg_kernel(LabelType *out, const int64_t *knn_indices,
                                   const LabelType *labels, size_t n_samples,
                                   int n_neighbors, int n_outputs,
                                   int output_offset) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_neighbors;

  if (row >= n_samples) return;

  // should work for moderately small number of classes
  LabelType pred = 0;
  for (int j = 0; j < n_neighbors; j++) {
    int64_t neighbor_idx = knn_indices[i + j];
    pred += labels[neighbor_idx];
  }

  out[row * n_outputs + output_offset] = pred / (LabelType)n_neighbors;
}

/**
 * A naive knn classifier to predict probabilities
 * @tparam TPB_X number of threads per block to use. each thread
 *               will process a single row of knn_indices
 *
 * @param out vector of output class probabilities of the same size as y.
 *            each element should be of size size (n_samples * n_classes[i])
 * @param knn_indices the index array resulting from a knn search
 * @param y vector of label arrays. for multulabel classification,
 *          each output in the vector is a different array of labels
 *          corresponding to the i'th output.
 * @param n_rows number of rows in knn_indices
 * @param k number of neighbors in knn_indices
 * @param uniq_labels vector of the sorted unique labels for each array in y
 * @param n_unique vector of sizes for each array in uniq_labels
 * @param allocator device allocator to use for temporary workspace
 * @param user_stream main stream to use for queuing isolated CUDA events
 * @param int_streams internal streams to use for parallelizing independent CUDA events.
 * @param n_int_stream number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32>
void class_probs(std::vector<float *> &out, const int64_t *knn_indices,
                 std::vector<int *> &y, size_t n_rows, int k,
                 std::vector<int *> &uniq_labels, std::vector<int> &n_unique,
                 std::shared_ptr<deviceAllocator> allocator,
                 cudaStream_t user_stream, cudaStream_t *int_streams = nullptr,
                 int n_int_streams = 0) {
  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      select_stream(user_stream, int_streams, n_int_streams, i);

    int n_labels = n_unique[i];
    int cur_size = n_rows * n_labels;

    CUDA_CHECK(cudaMemsetAsync(out[i], 0, cur_size * sizeof(float), stream));

    dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    /**
     * Build array of class probability arrays from
     * knn_indices and labels
     */
    int smem = sizeof(int) * n_labels;
    class_probs_kernel<<<grid, blk, smem, stream>>>(
      out[i], knn_indices, y[i], uniq_labels[i], n_labels, n_rows, k);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

/**
 * KNN classifier using voting based on the statistical mode of classes.
 * In the event of a tie, the class with the lowest index in the sorted
 * array of unique monotonically increasing labels will be used.
 *
 * @tparam TPB_X the number of threads per block to use
 * @param out output array of size (n_samples * y.size())
 * @param knn_indices index array from knn search
 * @param y vector of label arrays. for multilabel classification, each
 *          element in the vector is a different "output" array of labels corresponding
 *          to the i'th output.
 * @param n_rows number of rows in knn_indices
 * @param k number of neighbors in knn_indices
 * @param uniq_labels vector of the sorted unique labels for each array in y
 * @param n_unique vector of sizes for each array in uniq_labels
 * @param allocator device allocator to use for temporary workspace
 * @param user_stream main stream to use for queuing isolated CUDA events
 * @param int_streams internal streams to use for parallelizing independent CUDA events.
 * @param n_int_stream number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32>
void knn_classify(int *out, const int64_t *knn_indices, std::vector<int *> &y,
                  size_t n_rows, int k, std::vector<int *> &uniq_labels,
                  std::vector<int> &n_unique,
                  std::shared_ptr<deviceAllocator> &allocator,
                  cudaStream_t user_stream, cudaStream_t *int_streams = nullptr,
                  int n_int_streams = 0) {
  std::vector<float *> probs;
  std::vector<device_buffer<float> *> tmp_probs;

  // allocate temporary memory
  for (int i = 0; i < n_unique.size(); i++) {
    int size = n_unique[i];

    cudaStream_t stream =
      select_stream(user_stream, int_streams, n_int_streams, i);

    device_buffer<float> *probs_buff =
      new device_buffer<float>(allocator, stream, n_rows * size);

    tmp_probs.push_back(probs_buff);
    probs.push_back(probs_buff->data());
  }

  /**
   * Compute class probabilities
   *
   * Note: Since class_probs will use the same round robin strategy for distributing
   * work to the streams, we don't need to explicitly synchronize the streams here.
   */
  class_probs(probs, knn_indices, y, n_rows, k, uniq_labels, n_unique,
              allocator, user_stream, int_streams, n_int_streams);

  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      select_stream(user_stream, int_streams, n_int_streams, i);

    int n_labels = n_unique[i];

    /**
     * Choose max probability
     */
    int smem = sizeof(int) * n_labels;
    class_vote_kernel<<<grid, blk, smem, stream>>>(
      out, probs[i], uniq_labels[i], n_labels, n_rows, y.size(), i);
    CUDA_CHECK(cudaPeekAtLastError());

    delete tmp_probs[i];
  }
}

/**
 * KNN regression using voting based on the mean of the labels for the
 * nearest neighbors.
 * @tparam ValType data type of the labels
 * @tparam TPB_X the number of threads per block to use
 * @param out output array of size (n_samples * y.size())
 * @param knn_indices index array from knn search
 * @param y vector of label arrays. for multilabel classification, each
 *          element in the vector is a different "output" array of labels corresponding
 *          to the i'th output.
 * @param n_rows number of rows in knn_indices
 * @param k number of neighbors in knn_indices
 * @param user_stream main stream to use for queuing isolated CUDA events
 * @param int_streams internal streams to use for parallelizing independent CUDA events.
 * @param n_int_stream number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */

template <typename ValType, int TPB_X = 32>
void knn_regress(ValType *out, const int64_t *knn_indices,
                 const std::vector<ValType *> &y, size_t n_rows, int k,
                 cudaStream_t user_stream, cudaStream_t *int_streams = nullptr,
                 int n_int_streams = 0) {
  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Vote average regression value
   */
  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      select_stream(user_stream, int_streams, n_int_streams, i);
    regress_avg_kernel<<<grid, blk, 0, stream>>>(out, knn_indices, y[i], n_rows,
                                                 k, y.size(), i);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

};  // namespace Selection
};  // namespace MLCommon
