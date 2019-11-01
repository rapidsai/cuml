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

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"

#include <iostream>

namespace MLCommon {
namespace Selection {

/** Merge results from several shards into a single result set.
   * @param n number of elements in search array
   * @param k number of neighbors returned
   * @param distances output distance array
   * @param labels output index array
   * @param all_distances  row-wise stacked array of intermediary knn output distances size nshard * n * k
   * @param all_labels     row-wise stacked array of intermediary knn output indices size nshard * n * k
   * @param translations  label translations to apply, size nshard
   */
template <class C>
void merge_tables(int64_t n, int k, int nshard, float *distances,
                  int64_t *labels, float *all_distances, int64_t *all_labels,
                  int64_t *shard_offsets) {
  if (k == 0) {
    return;
  }

  size_t stride = n * k;
#pragma omp parallel
  {
    std::vector<int> buf(2 * nshard);
    int *pointer = buf.data();
    int *shard_ids = pointer + nshard;
    std::vector<float> buf2(nshard);
    float *heap_vals = buf2.data();
#pragma omp for
    for (int64_t i = 0; i < n; i++) {
      // the heap maps values to the shard where they are
      // produced.
      const float *D_in = all_distances + i * k;
      const int64_t *I_in = all_labels + i * k;
      int heap_size = 0;

      for (int s = 0; s < nshard; s++) {
        pointer[s] = 0;
        if (I_in[stride * s] >= 0)
          faiss::heap_push<C>(++heap_size, heap_vals, shard_ids,
                              D_in[stride * s], s);
      }

      float *D = distances + i * k;
      int64_t *I = labels + i * k;

      for (int j = 0; j < k; j++) {
        if (heap_size == 0) {
          I[j] = -1;
          D[j] = C::neutral();
        } else {
          // pop best element
          int s = shard_ids[0];
          int &p = pointer[s];
          D[j] = heap_vals[0];
          I[j] = I_in[stride * s + p] + shard_offsets[s];

          faiss::heap_pop<C>(heap_size--, heap_vals, shard_ids);
          p++;
          if (p < k && I_in[stride * s + p] >= 0)
            faiss::heap_push<C>(++heap_size, heap_vals, shard_ids,
                                D_in[stride * s + p], s);
        }
      }
    }
  }
};

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
   * @param s the cuda stream to use
   * @param translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   */
template <typename IntType = int,
          Distance::DistanceType DistanceType = Distance::EucUnexpandedL2>
void brute_force_knn(float **input, int *sizes, int n_params, IntType D,
                     float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k, cudaStream_t s,
                     bool rowMajorIndex = true, bool rowMajorQuery = true,
                     std::vector<int64_t> *translations = nullptr) {
  // TODO: Also pass internal streams down from handle.

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

  float *result_D = new float[k * n];
  long *result_I = new int64_t[k * n];

  float *all_D = new float[n_params * k * n];
  long *all_I = new int64_t[n_params * k * n];

  ASSERT_DEVICE_MEM(search_items, "search items");
  ASSERT_DEVICE_MEM(res_I, "output index array");
  ASSERT_DEVICE_MEM(res_D, "output distance array");

  CUDA_CHECK(cudaStreamSynchronize(s));

#pragma omp parallel
  {
#pragma omp for
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

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));

          gpu_res.noTempMemory();
          gpu_res.setCudaMallocWarning(false);
          gpu_res.setDefaultStream(att.device, stream);

          faiss::gpu::bruteForceKnn(
            &gpu_res, faiss::METRIC_L2, ptr, rowMajorIndex, size, search_items,
            rowMajorQuery, n, D, k, all_D + (i * k * n), all_I + (i * k * n));

          CUDA_CHECK(cudaPeekAtLastError());
          CUDA_CHECK(cudaStreamSynchronize(stream));

          CUDA_CHECK(cudaStreamDestroy(stream));

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
  }

  merge_tables<faiss::CMin<float, IntType>>(n, k, n_params, result_D, result_I,
                                            all_D, all_I, (*id_ranges).data());

  if (DistanceType == Distance::EucUnexpandedL2Sqrt) {
    MLCommon::LinAlg::unaryOp<float>(
      res_D, res_D, n * k, [] __device__(float input) { return sqrt(input); },
      s);
  }

  MLCommon::updateDevice(res_D, result_D, k * n, s);
  MLCommon::updateDevice(res_I, result_I, k * n, s);

  delete all_D;
  delete all_I;

  delete result_D;
  delete result_I;

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
__device__ int find_label(IdxType *unique_labels, IdxType n_labels,
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

template <typename OutType>
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
    int out_label_idx = find_label(label_cache, n_uniq_labels, out_label);
    out[(row * n_uniq_labels) + out_label_idx] += 1.0;
  }
}

template <typename OutType, typename ProbaType>
__global__ void class_vote_kernel(OutType *out, const ProbaType *class_proba,
                                  int *unique_labels, int n_uniq_labels,
                                  size_t n_samples) {
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
  out[row] = label_cache[cur_label];
}

template <typename LabelType>
__global__ void regress_avg_kernel(LabelType *out, const int64_t *knn_indices,
                                   const LabelType *labels, size_t n_samples,
                                   int n_neighbors) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_neighbors;

  if (row >= n_samples) return;

  // should work for moderately small number of classes
  LabelType pred = 0;
  for (int j = 0; j < n_neighbors; j++) {
    int64_t neighbor_idx = knn_indices[i + j];
    pred += labels[neighbor_idx];
  }

  out[row] = pred / (LabelType)n_neighbors;
}

/**
 * A naive knn classifier to predict probabilities
 *
 * @param out output array of size (n_samples * n_classes)
 */
template <int TPB_X = 32>
void class_probs(float *out, const int64_t *knn_indices, const int *y,
                 size_t n_rows, int k, int *uniq_labels, int n_unique,
                 std::shared_ptr<deviceAllocator> allocator,
                 cudaStream_t stream) {
  // need to get unique classes
  CUDA_CHECK(cudaMemsetAsync(out, 0, n_rows * n_unique, stream));

  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Build array class probability arrays from
   * knn_indices and labels
   */
  int smem = sizeof(int) * n_unique;
  class_probs_kernel<<<grid, blk, smem, stream>>>(
    out, knn_indices, y, uniq_labels, n_unique, n_rows, k);

  /**
   * Normalize numbers between 0 and 1
   */
  LinAlg::unaryOp(
    out, out, n_rows * n_unique,
    [=] __device__(float input) {
      float n_neighbors = k;
      return input / n_neighbors;
    },
    stream);
}

template <int TPB_X = 32>
void knn_classify(int *out, const int64_t *knn_indices, const int *y,
                  size_t n_rows, int k, int *uniq_labels, int n_unique,
                  std::shared_ptr<deviceAllocator> &allocator,
                  cudaStream_t stream) {
  device_buffer<float> probs(allocator, stream, n_rows * n_unique);
  /**
   * Compute class probabilities
   */

  class_probs(probs.data(), knn_indices, y, n_rows, k, uniq_labels, n_unique,
              allocator, stream);

  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Choose max probability
   */
  int smem = sizeof(int) * n_unique;
  class_vote_kernel<<<grid, blk, smem, stream>>>(out, probs.data(), uniq_labels,
                                                 n_unique, n_rows);
}

template <typename ValType, int TPB_X = 32>
void knn_regress(ValType *out, const int64_t *knn_indices, const ValType *y,
                 size_t n_rows, int k, cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(n_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Vote average regression value
   */
  regress_avg_kernel<<<grid, blk, 0, stream>>>(out, knn_indices, y, n_rows, k);
}

};  // namespace Selection
};  // namespace MLCommon
