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

#include <selection/columnWiseSort.cuh>
#include <selection/knn.cuh>

#include "cusparse_wrappers.h"
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>


namespace MLCommon {
namespace Sparse {
namespace Selection {

/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
   * @param allocator the device memory allocator to use for temporary scratch memory
   * @param userStream the main cuda stream to use
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
					 int batch_size,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
                     std::vector<int64_t> *translations = nullptr,
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {
	for(int i = 0; i < batch_size; i++) {

		for(int j = 0; j < batch_size; j++) {

			// cusparseSpGEMM_workEstimation
			// cusparseSpGEMM_compute
			// cusparseSpGEMM_compute
			// cusparseSpGEMM_copy
			// cusparseScsr2dense

			// sortColumnsPerRow

			// knn_merge_parts
		}

		// Copy final merged batch to output array
	}
}
}
} // END namespace MLCommon
