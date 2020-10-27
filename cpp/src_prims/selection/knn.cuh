/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>

#include <distance/distance.cuh>
#include <label/classlabels.cuh>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <selection/processing.cuh>

#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/neighbors/knn.hpp>

#include <iostream>

namespace MLCommon {
namespace Selection {

template <bool precomp_lbls, typename T>
inline __device__ T get_lbls(const T *labels, const int64_t *knn_indices,
                             int64_t idx) {
  if (precomp_lbls) {
    return labels[idx];
  } else {
    int64_t neighbor_idx = knn_indices[idx];
    return labels[neighbor_idx];
  }
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

inline faiss::MetricType build_faiss_metric(ML::MetricType metric) {
  switch (metric) {
    case ML::MetricType::METRIC_Cosine:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case ML::MetricType::METRIC_Correlation:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    default:
      return (faiss::MetricType)metric;
  }
}

/**
   * Search the kNN for the k-nearest neighbors of a set of query vectors
   * @param[in] input vector of device device memory array pointers to search
   * @param[in] sizes vector of memory sizes for each device array pointer in input
   * @param[in] D number of cols in input and search_items
   * @param[in] search_items set of vectors to query for neighbors
   * @param[in] n        number of items in search_items
   * @param[out] res_I    pointer to device memory for returning k nearest indices
   * @param[out] res_D    pointer to device memory for returning k nearest distances
   * @param[in] k        number of neighbors to query
   * @param[in] allocator the device memory allocator to use for temporary scratch memory
   * @param[in] userStream the main cuda stream to use
   * @param[in] internalStreams optional when n_params > 0, the index partitions can be
   *        queried in parallel using these streams. Note that n_int_streams also
   *        has to be > 0 for these to be used and their cardinality does not need
   *        to correspond to n_parts.
   * @param[in] n_int_streams size of internalStreams. When this is <= 0, only the
   *        user stream will be used.
   * @param[in] rowMajorIndex are the index arrays in row-major layout?
   * @param[in] rowMajorQuery are the query array in row-major layout?
   * @param[in] translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   * @param[in] metric corresponds to the FAISS::metricType enum (default is euclidean)
   * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
   * @param[in] expanded_form whether or not lp variants should be reduced w/ lp-root
   */
template <typename IntType = int>
void brute_force_knn(std::vector<float *> &input, std::vector<int> &sizes,
                     IntType D, float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
                     cudaStream_t *internalStreams = nullptr,
                     int n_int_streams = 0, bool rowMajorIndex = true,
                     bool rowMajorQuery = true,
                     std::vector<int64_t> *translations = nullptr,
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors should be the same size");

  faiss::MetricType m = build_faiss_metric(metric);

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

  // perform preprocessing
  std::unique_ptr<MetricProcessor<float>> query_metric_processor =
    create_processor<float>(metric, n, D, k, rowMajorQuery, userStream,
                            allocator);
  query_metric_processor->preprocess(search_items);

  std::vector<std::unique_ptr<MetricProcessor<float>>> metric_processors(
    input.size());
  for (int i = 0; i < input.size(); i++) {
    metric_processors[i] = create_processor<float>(
      metric, sizes[i], D, k, rowMajorQuery, userStream, allocator);
    metric_processors[i]->preprocess(input[i]);
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  device_buffer<int64_t> trans(allocator, userStream, id_ranges->size());
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(),
                      userStream);

  device_buffer<float> all_D(allocator, userStream, 0);
  device_buffer<int64_t> all_I(allocator, userStream, 0);

  float *out_D = res_D;
  int64_t *out_I = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // Sync user stream only if using other streams to parallelize query
  if (n_int_streams > 0) CUDA_CHECK(cudaStreamSynchronize(userStream));

  for (int i = 0; i < input.size(); i++) {
    faiss::gpu::StandardGpuResources gpu_res;

    cudaStream_t stream =
      raft::select_stream(userStream, internalStreams, n_int_streams, i);

    gpu_res.noTempMemory();
    gpu_res.setCudaMallocWarning(false);
    gpu_res.setDefaultStream(device, stream);

    faiss::gpu::GpuDistanceParams args;
    args.metric = m;
    args.metricArg = metricArg;
    args.k = k;
    args.dims = D;
    args.vectors = input[i];
    args.vectorsRowMajor = rowMajorIndex;
    args.numVectors = sizes[i];
    args.queries = search_items;
    args.queriesRowMajor = rowMajorQuery;
    args.numQueries = n;
    args.outDistances = out_D + (i * k * n);
    args.outIndices = out_I + (i * k * n);

    /**
     * @todo: Until FAISS supports pluggable allocation strategies,
     * we will not reap the benefits of the pool allocator for
     * avoiding device-wide synchronizations from cudaMalloc/cudaFree
     */
    bfKnn(&gpu_res, args);

    CUDA_CHECK(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  for (int i = 0; i < n_int_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(internalStreams[i]));
  }

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    knn_merge_parts(out_D, out_I, res_D, res_I, n, input.size(), k, userStream,
                    trans.data());
  }

  // Perform necessary post-processing
  if ((m == faiss::MetricType::METRIC_L2 ||
       m == faiss::MetricType::METRIC_Lp) &&
      !expanded_form) {
    /**
	* post-processing
	*/
    float p = 0.5;  // standard l2
    if (m == faiss::MetricType::METRIC_Lp) p = 1.0 / metricArg;
    raft::linalg::unaryOp<float>(
      res_D, res_D, n * k,
      [p] __device__(float input) { return powf(input, p); }, userStream);
  }

  query_metric_processor->revert(search_items);
  query_metric_processor->postprocess(out_D);
  for (int i = 0; i < input.size(); i++) {
    metric_processors[i]->revert(input[i]);
  }

  if (translations == nullptr) delete id_ranges;
};

template <typename OutType = float, bool precomp_lbls = false>
__global__ void class_probs_kernel(OutType *out, const int64_t *knn_indices,
                                   const int *labels, int n_uniq_labels,
                                   size_t n_samples, int n_neighbors) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_neighbors;

  float n_neigh_inv = 1.0f / n_neighbors;

  if (row >= n_samples) return;

  for (int j = 0; j < n_neighbors; j++) {
    int out_label = get_lbls<precomp_lbls>(labels, knn_indices, i + j);
    int out_idx = row * n_uniq_labels + out_label;
    out[out_idx] += n_neigh_inv;
  }
}

template <typename OutType = int>
__global__ void class_vote_kernel(OutType *out, const float *class_proba,
                                  int *unique_labels, int n_uniq_labels,
                                  size_t n_samples, int n_outputs,
                                  int output_offset, bool use_shared_mem) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_uniq_labels;

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
      cur_max = cur_proba;
      cur_label = j;
    }
  }

  int val = use_shared_mem ? label_cache[cur_label] : unique_labels[cur_label];

  out[row * n_outputs + output_offset] = val;
}

template <typename LabelType, bool precomp_lbls = false>
__global__ void regress_avg_kernel(LabelType *out, const int64_t *knn_indices,
                                   const LabelType *labels, size_t n_samples,
                                   int n_neighbors, int n_outputs,
                                   int output_offset) {
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int i = row * n_neighbors;

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
 *         This makes it possible to compute the reduction step without holding all the data on a single machine.
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
 * @param[in] allocator device allocator to use for temporary workspace
 * @param[in] user_stream main stream to use for queuing isolated CUDA events
 * @param[in] int_streams internal streams to use for parallelizing independent CUDA events.
 * @param[in] n_int_streams number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32, bool precomp_lbls = false>
void class_probs(std::vector<float *> &out, const int64_t *knn_indices,
                 std::vector<int *> &y, size_t n_index_rows,
                 size_t n_query_rows, int k, std::vector<int *> &uniq_labels,
                 std::vector<int> &n_unique,
                 const std::shared_ptr<deviceAllocator> allocator,
                 cudaStream_t user_stream, cudaStream_t *int_streams = nullptr,
                 int n_int_streams = 0) {
  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      raft::select_stream(user_stream, int_streams, n_int_streams, i);

    int n_unique_labels = n_unique[i];
    int cur_size = n_query_rows * n_unique_labels;

    CUDA_CHECK(cudaMemsetAsync(out[i], 0, cur_size * sizeof(float), stream));

    dim3 grid(raft::ceildiv(n_query_rows, (size_t)TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    /**
     * Build array of class probability arrays from
     * knn_indices and labels
     */
    device_buffer<int> y_normalized(allocator, stream, n_index_rows);

    /*
     * Appending the array of unique labels to the original labels array
     * to prevent make_monotonic function from producing misleading results
     * due to the absence of some of the unique labels in the labels array
     */
    device_buffer<int> y_tmp(allocator, stream, n_index_rows + n_unique_labels);
    raft::update_device(y_tmp.data(), y[i], n_index_rows, stream);
    raft::update_device(y_tmp.data() + n_index_rows, uniq_labels[i],
                        n_unique_labels, stream);

    MLCommon::Label::make_monotonic(y_normalized.data(), y_tmp.data(),
                                    y_tmp.size(), stream, allocator);
    raft::linalg::unaryOp<int>(
      y_normalized.data(), y_normalized.data(), n_index_rows,
      [] __device__(int input) { return input - 1; }, stream);
    class_probs_kernel<float, precomp_lbls>
      <<<grid, blk, 0, stream>>>(out[i], knn_indices, y_normalized.data(),
                                 n_unique_labels, n_query_rows, k);
    CUDA_CHECK(cudaPeekAtLastError());
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
 * This makes it possible to compute the reduction step without holding all the data on a single machine.
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
 * @param[in] allocator device allocator to use for temporary workspace
 * @param[in] user_stream main stream to use for queuing isolated CUDA events
 * @param[in] int_streams internal streams to use for parallelizing independent CUDA events.
 * @param[in] n_int_streams number of elements in int_streams array. If this is less than 1,
 *        the user_stream is used.
 */
template <int TPB_X = 32, bool precomp_lbls = false>
void knn_classify(int *out, const int64_t *knn_indices, std::vector<int *> &y,
                  size_t n_index_rows, size_t n_query_rows, int k,
                  std::vector<int *> &uniq_labels, std::vector<int> &n_unique,
                  const std::shared_ptr<deviceAllocator> &allocator,
                  cudaStream_t user_stream, cudaStream_t *int_streams = nullptr,
                  int n_int_streams = 0) {
  std::vector<float *> probs;
  std::vector<device_buffer<float> *> tmp_probs;

  // allocate temporary memory
  for (int i = 0; i < n_unique.size(); i++) {
    int size = n_unique[i];

    cudaStream_t stream =
      raft::select_stream(user_stream, int_streams, n_int_streams, i);

    device_buffer<float> *probs_buff =
      new device_buffer<float>(allocator, stream, n_query_rows * size);

    tmp_probs.push_back(probs_buff);
    probs.push_back(probs_buff->data());
  }

  /**
   * Compute class probabilities
   *
   * Note: Since class_probs will use the same round robin strategy for distributing
   * work to the streams, we don't need to explicitly synchronize the streams here.
   */
  class_probs<32, precomp_lbls>(
    probs, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique,
    allocator, user_stream, int_streams, n_int_streams);

  dim3 grid(raft::ceildiv(n_query_rows, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      raft::select_stream(user_stream, int_streams, n_int_streams, i);

    int n_unique_labels = n_unique[i];

    /**
     * Choose max probability
     */
    // Use shared memory for label lookups if the number of classes is small enough
    int smem = sizeof(int) * n_unique_labels;
    bool use_shared_mem = smem < raft::getSharedMemPerBlock();

    class_vote_kernel<<<grid, blk, use_shared_mem ? smem : 0, stream>>>(
      out, probs[i], uniq_labels[i], n_unique_labels, n_query_rows, y.size(), i,
      use_shared_mem);
    CUDA_CHECK(cudaPeekAtLastError());

    delete tmp_probs[i];
  }
}

/**
 * KNN regression using voting based on the mean of the labels for the
 * nearest neighbors.
 * @tparam ValType data type of the labels
 * @tparam TPB_X the number of threads per block to use
 * @tparam precomp_lbls is set to true for the reduction step of MNMG KNN Regressor. In this case,
 * the knn_indices array is not used as the y arrays already store the output for each row.
 * This makes it possible to compute the reduction step without holding all the data on a single machine.
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
void knn_regress(ValType *out, const int64_t *knn_indices,
                 const std::vector<ValType *> &y, size_t n_index_rows,
                 size_t n_query_rows, int k, cudaStream_t user_stream,
                 cudaStream_t *int_streams = nullptr, int n_int_streams = 0) {
  /**
   * Vote average regression value
   */
  for (int i = 0; i < y.size(); i++) {
    cudaStream_t stream =
      raft::select_stream(user_stream, int_streams, n_int_streams, i);

    regress_avg_kernel<ValType, precomp_lbls>
      <<<raft::ceildiv(n_query_rows, (size_t)TPB_X), TPB_X, 0, stream>>>(
        out, knn_indices, y[i], n_query_rows, k, y.size(), i);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

};  // namespace Selection
};  // namespace MLCommon
