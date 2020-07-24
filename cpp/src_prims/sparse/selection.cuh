/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "cusparse_wrappers.h"
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>


namespace MLCommon {
namespace Sparse {
namespace Selection {

/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
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
   * @param metric corresponds to the FAISS::metricType enum (default is euclidean)
   * @param metricArg metric argument to use. Corresponds to the p arg for lp norm
   * @param expanded_form whether or not lp variants should be reduced w/ lp-root
   */
template <typename value_idx = int, typename value_t>
void brute_force_knn(value_idx idxIndptr, value_idx idxIndices, value_t idxData, value_idx idxNNZ,
					 value_idx queryIndptr, value_idx queryIndices, value_t queryData, value_idx queryNNZ,
					 value_idx output_indices, value_t output_dists, int k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
                     cudaStream_t *internalStreams = nullptr,
                     int n_int_streams = 0,
                     std::vector<int64_t> *translations = nullptr,
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {

}
}
} // END namespace MLCommon
