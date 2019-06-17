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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <limits>
#include "classlabels.h"
#include "cuda_utils.h"

namespace MLCommon {
namespace Label {

template <typename Type, int TPB_X, typename Lambda>
__global__ void map_label_kernel(Type *map_ids, size_t N_labels, Type *in,
                                 Type *out, size_t N, Lambda filter_op) {
  int tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (!filter_op(in[tid])) {
      for (size_t i = 0; i < N_labels; i++) {
        if (in[tid] == map_ids[i]) {
          out[tid] = i + 1;
          break;
        }
      }
    }
  }
}

/**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 * @param filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 */
template <typename Type, typename Lambda>
void make_monotonic(Type *out, Type *in, size_t N, cudaStream_t stream,
                    Lambda filter_op) {
  static const size_t TPB_X = 256;

  dim3 blocks(ceildiv(N, TPB_X));
  dim3 threads(TPB_X);

  std::shared_ptr<deviceAllocator> allocator(new defaultDeviceAllocator);

  Type *map_ids;
  int num_clusters;
  getUniqueLabels(in, N, &map_ids, &num_clusters, stream, allocator);

  map_label_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(
    map_ids, num_clusters, in, out, N, filter_op);

  allocator->deallocate(map_ids, num_clusters * sizeof(Type), stream);
}

/**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 */
template <typename Type>
void make_monotonic(Type *out, Type *in, size_t N, cudaStream_t stream) {
  make_monotonic<Type>(out, in, N, stream,
                       [] __device__(Type val) { return false; });
}
};  // namespace Label
};  // namespace MLCommon
