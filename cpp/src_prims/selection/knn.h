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

#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"

#include <iostream>

namespace MLCommon {
namespace Selection {

template <bool Dir, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void blockSelectPairKernel(float *inK, int64_t *inV, float *outK,
                                      int64_t *outV, size_t n_samples,
                                      int n_parts, float initK, int64_t initV,
                                      int k, int64_t *translations) {
  constexpr int kNumWarps = ThreadsPerBlock / faiss::gpu::kWarpSize;

  __shared__ float smemK[kNumWarps * NumWarpQ];
  __shared__ int64_t smemV[kNumWarps * NumWarpQ];

  /**
   * Uses shared memory
   */
  faiss::gpu::BlockSelect<float, int64_t, Dir, faiss::gpu::Comparator<float>,
                          NumWarpQ, NumThreadQ, ThreadsPerBlock>
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

  for (; i < limit; i += ThreadsPerBlock) {
    translation = translations[part];
    heap.add(*inKStart, (*inVStart) + translation);

    part = (i + ThreadsPerBlock) / k;
    row_idx = (row * k) + (part * n_samples * k);

    col = (i + ThreadsPerBlock) % k;

    inKStart = inK + (row_idx + col);
    inVStart = inV + (row_idx + col);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < total_k) {
    translation = translations[part];
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

#define BLOCK_SELECT_IMPL(DIR, WARP_Q, THREAD_Q)                             \
  inline void runBlockSelectPair_##DIR##_##WARP_Q##_(                        \
    float *inK, int64_t *inV, float *outK, int64_t *outV, size_t n_samples,  \
    int n_parts, bool dir, int k, cudaStream_t stream,                       \
    int64_t *translations) {                                                 \
    auto grid = dim3(n_samples);                                             \
                                                                             \
    constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;      \
    auto block = dim3(kBlockSelectNumThreads);                               \
                                                                             \
    auto kInit = dir ? faiss::gpu::Limits<float>::getMin()                   \
                     : faiss::gpu::Limits<float>::getMax();                  \
    auto vInit = -1;                                                         \
    blockSelectPairKernel<DIR, WARP_Q, THREAD_Q, kBlockSelectNumThreads>     \
      <<<grid, block, 0, stream>>>(inK, inV, outK, outV, n_samples, n_parts, \
                                   kInit, vInit, k, translations);           \
    CUDA_CHECK(cudaPeekAtLastError());                                       \
  }

#define BLOCK_SELECT_DECL(DIR, WARP_Q)                                      \
  extern void runBlockSelectPair_##DIR##_##WARP_Q##_(                       \
    float *inK, int64_t *inV, float *outK, int64_t *outV, size_t n_samples, \
    int n_parts, bool dir, int k, cudaStream_t stream, int64_t *translations);

//BLOCK_SELECT_DECL(true, 1);
//BLOCK_SELECT_DECL(true, 32);
//BLOCK_SELECT_DECL(true, 64);
//BLOCK_SELECT_DECL(true, 128);
//BLOCK_SELECT_DECL(true, 256);
//BLOCK_SELECT_DECL(true, 512);
//BLOCK_SELECT_DECL(true, 1024);

BLOCK_SELECT_DECL(false, 1);
BLOCK_SELECT_DECL(false, 32);
BLOCK_SELECT_DECL(false, 64);

BLOCK_SELECT_IMPL(false, 1, 1);
BLOCK_SELECT_IMPL(false, 32, 2);
BLOCK_SELECT_IMPL(false, 64, 3);
//BLOCK_SELECT_IMPL(false, 128, 3);
//BLOCK_SELECT_IMPL(false, 256, 4);
//BLOCK_SELECT_IMPL(false, 512, 8);
//BLOCK_SELECT_IMPL(false, 1024, 8);

#define BLOCK_SELECT_PAIR_CALL(DIR, WARP_Q) \
  runBlockSelectPair_##DIR##_##WARP_Q##_(   \
    inK, inV, outK, outV, n_samples, n_parts, dir, k, stream, translations)

inline void runBlockSelectPair(float *inK, int64_t *inV, float *outK,
                               int64_t *outV, size_t n_samples, int n_parts,
                               bool dir, int k, cudaStream_t stream,
                               int64_t *translations) {
  if (k == 1) {
    BLOCK_SELECT_PAIR_CALL(false, 1);
  } else if (k <= 32) {
    BLOCK_SELECT_PAIR_CALL(false, 32);
  } else if (k <= 64) {
    BLOCK_SELECT_PAIR_CALL(false, 64);
  }
}

/**
   * Search the kNN for the k-nearest neighbors of a set of query vectors
   * @param input device memory to search as an array of device pointers
   * @param sizes array of memory sizes
   * @param n_params size of input and sizes arrays
   * @param D number of cols in input and search_items
   * @param search_items set of vectors to query for neighbors
   * @param n        number of items in search_items
   * @param res_I      pointer to device memory for returning k nearest indices
   * @param res_D      pointer to device memory for returning k nearest distances
   * @param k        number of neighbors to query
   * @param allocator the device memory allocator to use for temporary scratch memory
   * @param s the cuda stream to use
   * @param internalStreams optional when n_params > 0, the index partitions can be
   *        queried in parallel using these streams. Note that n_int_streams also
   *        has to be > 0 for these to be used and their cardinality does not need
   *        to correspond to n_parts.
   * @param n_int_streams size of internalStreams. When this is <= 0, only the
   *        user stream will be used.
   * @param translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   */
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
  ASSERT(DistanceType == Distance::EucUnexpandedL2 ||
           DistanceType == Distance::EucUnexpandedL2Sqrt,
         "Only EucUnexpandedL2Sqrt and EucUnexpandedL2 metrics are supported "
         "currently.");

  std::vector<int64_t> *id_ranges;
  if (translations == nullptr) {
    id_ranges = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (int i = 0; i < n_params; i++) {
      if (i < n_params) {
        id_ranges->push_back(total_n);
      }
      total_n += sizes[i];
    }
  } else {
    id_ranges = translations;
  }

  device_buffer<int64_t> trans(allocator, userStream, id_ranges->size());
  updateDevice(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  CUDA_CHECK(cudaStreamSynchronize(userStream));

  ASSERT_DEVICE_MEM(search_items, "search items");
  ASSERT_DEVICE_MEM(res_I, "output index array");
  ASSERT_DEVICE_MEM(res_D, "output distance array");

  device_buffer<float> all_D(allocator, userStream, n_params * k * n);
  device_buffer<int64_t> all_I(allocator, userStream, n_params * k * n);

  if (n_int_streams > 0) CUDA_CHECK(cudaStreamSynchronize(userStream));

  for (int i = 0; i < n_params; i++) {
    const float *ptr = input[i];
    IntType size = sizes[i];

    cudaPointerAttributes att;
    cudaError_t err = cudaPointerGetAttributes(&att, ptr);

    if (err == 0 && att.device > -1) {
      CUDA_CHECK(cudaSetDevice(att.device));
      CUDA_CHECK(cudaPeekAtLastError());

      try {
        faiss::gpu::StandardGpuResources gpu_res;

        cudaStream_t stream =
          n_int_streams > 0 ? internalStreams[i % n_int_streams] : userStream;

        gpu_res.noTempMemory();
        gpu_res.setCudaMallocWarning(false);
        gpu_res.setDefaultStream(att.device, stream);

        faiss::gpu::bruteForceKnn(
          &gpu_res, faiss::METRIC_L2, ptr, rowMajorIndex, size, search_items,
          rowMajorQuery, n, D, k, all_D.data() + (i * k * n),
          all_I.data() + (i * k * n));

        CUDA_CHECK(cudaPeekAtLastError());

      } catch (const std::exception &e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
      }

    } else {
      std::stringstream ss;
      ss << "Input memory for " << ptr
         << " failed. isDevice?=" << att.devicePointer << ", N=" << sizes[i];
      std::cout << "Exception: " << ss.str() << std::endl;
    }
  }

  for (int i = 0; i < n_int_streams; i++) {
    //TODO: Use cudaStreamWaitEvent() because it's very likely / possible we could
    // be waiting on more streams than were used
    CUDA_CHECK(cudaStreamSynchronize(internalStreams[i]));
  }

/*  if (n_params > 1) {*/
    runBlockSelectPair(all_D.data(), all_I.data(), res_D, res_I, n, n_params,
                       false, k, userStream, trans.data());
//  } else {
//    copy(res_D, all_D.data(), n * k, userStream);
//    copy(res_I, all_I.data(), n * k, userStream);
//  }

  MLCommon::LinAlg::unaryOp<float>(
    res_D, res_D, n * k,
    [] __device__(float input) { return sqrt(input); },
    userStream);


  if (translations == nullptr) delete id_ranges;

};

/**
 * Binary tree recursion for finding a label in the unique_labels array.
 * This provides a good middle-ground between having to create a new
 * labels array just to map non-monotonically increasing labels, or
 * the alternative, which is having to search over O(n) space for the labels
 * array in each thread. This is going to cause warp divergence of log(n)
 * per iteration.
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

  extern __shared__ int label_cache[];

  if (threadIdx.x == 0) {
    for (int j = 0; j < n_uniq_labels; j++) label_cache[j] = unique_labels[j];
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
    out[out_idx] += 1.0f;
  }
}

template <typename OutType = int, typename ProbaType = float>
__global__ void class_vote_kernel(OutType *out, const ProbaType *class_proba,
                                  int *unique_labels, int n_uniq_labels,
                                  size_t n_samples, int n_outputs,
                                  int output_offset) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_uniq_labels;

  extern __shared__ int label_cache[];
  if (threadIdx.x == 0) {
    for (int j = 0; j < n_uniq_labels; j++) {
      label_cache[j] = unique_labels[j];
    }
  }

  __syncthreads();

  if (row >= n_samples) return;
  int cur_max = -1;
  int cur_label = -1;
  for (int j = 0; j < n_uniq_labels; j++) {
    int cur_count = class_proba[i + j];
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
 * @param stream stream to use for queuing isolated CUDA events
 */
template <int TPB_X = 32>
void class_probs(std::vector<float *> &out, const int64_t *knn_indices,
                 std::vector<int *> &y, size_t n_rows, int k,
                 std::vector<int *> &uniq_labels, std::vector<int> &n_unique,
                 std::shared_ptr<deviceAllocator> allocator,
                 cudaStream_t stream) {
  // todo: Use separate streams
  for (int i = 0; i < y.size(); i++) {
    int n_labels = n_unique[i];
    int cur_size = n_rows * n_labels;

    CUDA_CHECK(cudaMemsetAsync(out[i], 0, cur_size * sizeof(int), stream));

    dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    /**
     * Build array of class probability arrays from
     * knn_indices and labels
     */
    int smem = sizeof(int) * n_labels;
    class_probs_kernel<<<grid, blk, smem, stream>>>(
      out[i], knn_indices, y[i], uniq_labels[i], n_labels, n_rows, k);

    LinAlg::unaryOp(
      out[i], out[i], cur_size,
      [=] __device__(float input) {
        float n_neighbors = k;
        return input / n_neighbors;
      },
      stream);
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
 *          output in the vector is a different array of labels corresponding
 *          to the i'th output.
 * @param n_rows number of rows in knn_indices
 * @param k number of neighbors in knn_indices
 * @param uniq_labels vector of the sorted unique labels for each array in y
 * @param n_unique vector of sizes for each array in uniq_labels
 * @param allocator device allocator to use for temporary workspace
 * @param stream stream to use for queuing isolated CUDA events
 */
template <int TPB_X = 32>
void knn_classify(int *out, const int64_t *knn_indices, std::vector<int *> &y,
                  size_t n_rows, int k, std::vector<int *> &uniq_labels,
                  std::vector<int> &n_unique,
                  std::shared_ptr<deviceAllocator> &allocator,
                  cudaStream_t stream) {
  std::vector<float *> probs;
  std::vector<device_buffer<float> *> tmp_probs;

  // allocate temporary memory
  for (int size : n_unique) {
    device_buffer<float> *probs_buff =
      new device_buffer<float>(allocator, stream, n_rows * size);

    tmp_probs.push_back(probs_buff);
    probs.push_back(probs_buff->data());
  }

  /**
   * Compute class probabilities
   */
  class_probs(probs, knn_indices, y, n_rows, k, uniq_labels, n_unique,
              allocator, stream);

  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  // todo: Use separate streams
  for (int i = 0; i < y.size(); i++) {
    int n_labels = n_unique[i];

    /**
     * Choose max probability
     */
    int smem = sizeof(int) * n_labels;
    class_vote_kernel<<<grid, blk, smem, stream>>>(
      out, probs[i], uniq_labels[i], n_labels, n_rows, y.size(), i);

    delete tmp_probs[i];
  }
}

template <typename ValType, int TPB_X = 32>
void knn_regress(ValType *out, const int64_t *knn_indices,
                 const std::vector<ValType *> &y, size_t n_rows, int k,
                 cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Vote average regression value
   */

  // TODO: Use separate streams
  for (int i = 0; i < y.size(); i++) {
    regress_avg_kernel<<<grid, blk, 0, stream>>>(out, knn_indices, y[i], n_rows,
                                                 k, y.size(), i);
  }
}

};  // namespace Selection
};  // namespace MLCommon
