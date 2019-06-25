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

#include <cub/cub.cuh>

#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/unary_op.h"
#include "ml_utils.h"

namespace MLCommon {
namespace Label {

using namespace MLCommon;

/**
 * Get uniuque class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * \tparam math_t numeric type of the arrays with class labels
 * \param [in] handle cuML handle
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [out] y_unique device array of unique labels, unallocated on entry,
 *   on exit it has size [n_unique]
 * \param [out] n_unique number of unique labels
 * \param stream
 */
template <typename math_t>
void getUniqueLabels(math_t *y, size_t n, math_t **y_unique, int *n_unique,
                     cudaStream_t stream,
                     std::shared_ptr<deviceAllocator> allocator) {
  device_buffer<math_t> y2(allocator, stream, n);
  device_buffer<math_t> y3(allocator, stream, n);
  device_buffer<int> d_num_selected(allocator, stream, 1);
  size_t bytes = 0;
  size_t bytes2 = 0;

  // Query how much temporary storage we will need for cub operations
  // and allocate it
  cub::DeviceRadixSort::SortKeys(NULL, bytes, y, y2.data(), n);
  cub::DeviceSelect::Unique(NULL, bytes2, y2.data(), y3.data(),
                            d_num_selected.data(), n);
  bytes = max(bytes, bytes2);
  device_buffer<char> cub_storage(allocator, stream, bytes);

  // Select Unique classes
  cub::DeviceRadixSort::SortKeys(cub_storage.data(), bytes, y, y2.data(), n);
  cub::DeviceSelect::Unique(cub_storage.data(), bytes, y2.data(), y3.data(),
                            d_num_selected.data(), n);
  updateHost(n_unique, d_num_selected.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy unique classes to output
  *y_unique = (math_t *)allocator->allocate(*n_unique * sizeof(math_t), stream);
  copy(*y_unique, y3.data(), *n_unique, stream);
}

/**
 * Assign one versus rest labels.
 *
 * The output labels will have values +/-1:
 * y_out = (y == y_unique[idx]) ? +1 : -1;
 *
 * The output type currently is set to math_t, but for SVM in principle we are
 * free to choose other type for y_out (it should represent +/-1, and it is used
 * in floating point arithmetics).
 *
 * \param [in] y device array if input labels, size [n]
 * \param [in] n number of labels
 * \param [in] y_unique device array of unique labels, size [n_classes]
 * \param [in] n_classes number of unique labels
 * \param [out] y_out device array of output labels
 * \param [in] idx index of unique label that should be labeled as 1
 */
template <typename math_t>
void getOvrLabels(math_t *y, int n, math_t *y_unique, int n_classes,
                  math_t *y_out, int idx, cudaStream_t stream) {
  ASSERT(idx < n_classes,
         "Parameter idx should not be larger than the number "
         "of classes");
  LinAlg::unaryOp(
    y_out, y, n,
    [idx, y_unique] __device__(math_t y) {
      return y == y_unique[idx] ? +1 : -1;
    },
    stream);
  CUDA_CHECK(cudaPeekAtLastError());
}

// TODO: add one-versus-one selection: select two classes, relabel them to
// +/-1, return array with the new class labels and corresponding indices.

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
};  // end namespace MLCommon
