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

#include <cub/cub.cuh>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/label/classlabels.hpp>
#include <raft/linalg/unary_op.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

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
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [out] unique device array of unique labels, needs to be pre-allocated
 * \param [in] stream cuda stream
 */
template <typename math_t>
int getUniqueLabels(math_t* y, size_t n, math_t* unique, cudaStream_t stream)
{
  rmm::device_uvector<math_t> unique_v(0, stream);
  auto n_unique = raft::label::getUniquelabels(unique_v, y, n, stream);
  raft::copy(unique, unique_v.data(), n_unique, stream);
  raft::interruptible::synchronize(stream);
  return n_unique;
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
 * \param [in] stream cuda stream
 */
template <typename math_t>
void getOvrLabels(
  math_t* y, int n, math_t* y_unique, int n_classes, math_t* y_out, int idx, cudaStream_t stream)
{
  ASSERT(idx < n_classes,
         "Parameter idx should not be larger than the number "
         "of classes");
  raft::linalg::unaryOp(
    y_out,
    y,
    n,
    [idx, y_unique] __device__(math_t y) { return y == y_unique[idx] ? +1 : -1; },
    stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// TODO: add one-versus-one selection: select two classes, relabel them to
// +/-1, return array with the new class labels and corresponding indices.

template <typename Type, int TPB_X, typename Lambda>
__global__ void map_label_kernel(
  Type* map_ids, size_t N_labels, Type* in, Type* out, size_t N, Lambda filter_op)
{
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
 * @param out the output monotonic array
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 * @param filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 */
template <typename Type, typename Lambda>
int make_monotonic(Type* out, Type* in, size_t N, cudaStream_t stream, Lambda filter_op)
{
  static const size_t TPB_X = 256;
  dim3 blocks(raft::ceildiv(N, TPB_X));
  dim3 threads(TPB_X);

  rmm::device_uvector<Type> unique(0, stream);
  int n_unique = raft::label::getUniquelabels(unique, in, N, stream);
  unique.resize(n_unique, stream);

  map_label_kernel<Type, TPB_X>
    <<<blocks, threads, 0, stream>>>(unique.data(), n_unique, in, out, N, filter_op);

  return n_unique;
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
 * @param out output label array with labels assigned monotonically
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 */
template <typename Type>
void make_monotonic(Type* out, Type* in, size_t N, cudaStream_t stream)
{
  make_monotonic<Type>(out, in, N, stream, [] __device__(Type val) { return false; });
}

template <typename Type>
int make_monotonic(const raft::handle_t& handle, Type* out, Type* in, size_t N)
{
  return make_monotonic<Type>(
    out, in, N, handle.get_stream(), [] __device__(Type val) { return false; });
}
};  // namespace Label
};  // end namespace MLCommon
