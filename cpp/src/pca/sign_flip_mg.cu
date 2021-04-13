/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <common/allocatorAdapter.hpp>
#include <cuml/common/device_buffer.hpp>
#include <cuml/decomposition/sign_flip_mg.hpp>
#include <raft/comms/comms.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/mr/device/allocator.hpp>

using namespace MLCommon;

namespace ML {
namespace PCA {
namespace opg {

// TODO: replace these thrust code with cuda kernels or prims
template <typename T>
void findMaxAbsOfColumns(T *input, int n_rows, int n_cols, T *max_vals,
                         std::shared_ptr<raft::mr::device::allocator> allocator,
                         cudaStream_t stream, bool row_major = false) {
  auto counting = thrust::make_counting_iterator(0);
  auto m = n_rows;
  auto n = n_cols;

  ML::thrustAllocatorAdapter alloc(allocator, stream);
  auto execution_policy = thrust::cuda::par(alloc).on(stream);

  if (row_major) {
    thrust::for_each(execution_policy, counting, counting + n_rows,
                     [=] __device__(int idx) {
                       T max = 0.0;
                       int max_index = 0;
                       int d_i = idx;
                       int end = d_i + (m * n);

                       for (int i = d_i; i < end; i = i + m) {
                         T val = input[i];
                         if (val < 0.0) {
                           val = -val;
                         }
                         if (val > max) {
                           max = val;
                           max_index = i;
                         }
                       }
                       max_vals[idx] = input[max_index];
                     });
  } else {
    thrust::for_each(execution_policy, counting, counting + n_cols,
                     [=] __device__(int idx) {
                       T max = 0.0;
                       int max_index = 0;
                       int d_i = idx * m;
                       int end = d_i + m;

                       for (int i = d_i; i < end; i++) {
                         T val = input[i];
                         if (val < 0.0) {
                           val = -val;
                         }
                         if (val > max) {
                           max = val;
                           max_index = i;
                         }
                       }
                       max_vals[idx] = input[max_index];
                     });
  }
}

// TODO: replace these thrust code with cuda kernels or prims
template <typename T>
void flip(T *input, int n_rows, int n_cols, T *max_vals,
          std::shared_ptr<raft::mr::device::allocator> allocator,
          cudaStream_t stream) {
  auto counting = thrust::make_counting_iterator(0);
  auto m = n_rows;

  ML::thrustAllocatorAdapter alloc(allocator, stream);
  auto execution_policy = thrust::cuda::par(alloc).on(stream);
  thrust::for_each(execution_policy, counting, counting + n_cols,
                   [=] __device__(int idx) {
                     int d_i = idx * m;
                     int end = d_i + m;

                     if (max_vals[idx] < 0.0) {
                       for (int i = d_i; i < end; i++) {
                         input[i] = -input[i];
                       }
                     }
                   });
}

/**
 * @brief sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen vectors
 * @input param handle: the internal cuml handle object
 * @input/output param input param input: input matrix that will be used to determine the sign.
 * @input param input_desc: MNMG description of the input
 * @input/output param  components: components matrix.
 * @input param n_components: number of columns of components matrix
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 * @{
 */
template <typename T>
void sign_flip_imp(raft::handle_t &handle,
                   std::vector<Matrix::Data<T> *> &input,
                   Matrix::PartDescriptor &input_desc, T *components,
                   int n_components, cudaStream_t *streams, int n_stream) {
  int rank = handle.get_comms().get_rank();

  const auto &comm = handle.get_comms();
  const auto allocator = handle.get_device_allocator();

  std::vector<Matrix::RankSizePair *> local_blocks =
    input_desc.blocksOwnedBy(rank);
  device_buffer<T> max_vals(
    allocator, streams[0],
    std::max(size_t(comm.get_size()), local_blocks.size()) * n_components);

  for (int i = 0; i < input.size(); i++) {
    T *mv_loc = max_vals.data() + (i * n_components);
    findMaxAbsOfColumns(input[i]->ptr, local_blocks[i]->size, n_components,
                        mv_loc, allocator, streams[i % n_stream]);
  }

  for (int i = 0; i < n_stream; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  findMaxAbsOfColumns(max_vals.data(), n_components, local_blocks.size(),
                      max_vals.data(), allocator, streams[0], true);

  comm.allgather(max_vals.data(), max_vals.data(), n_components, streams[0]);
  comm.sync_stream(streams[0]);

  findMaxAbsOfColumns(max_vals.data(), n_components, comm.get_size(),
                      max_vals.data(), allocator, streams[0], true);

  for (int i = 0; i < local_blocks.size(); i++) {
    flip(input[i]->ptr, local_blocks[i]->size, n_components, max_vals.data(),
         allocator, streams[i % n_stream]);
  }

  for (int i = 0; i < n_stream; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  flip(components, input_desc.N, n_components, max_vals.data(), allocator,
       streams[0]);
}

void sign_flip(raft::handle_t &handle,
               std::vector<Matrix::Data<float> *> &input_data,
               Matrix::PartDescriptor &input_desc, float *components,
               int n_components, cudaStream_t *streams, int n_stream) {
  sign_flip_imp(handle, input_data, input_desc, components, n_components,
                streams, n_stream);
}

void sign_flip(raft::handle_t &handle,
               std::vector<Matrix::Data<double> *> &input_data,
               Matrix::PartDescriptor &input_desc, double *components,
               int n_components, cudaStream_t *streams, int n_stream) {
  sign_flip_imp(handle, input_data, input_desc, components, n_components,
                streams, n_stream);
}

}  // namespace opg
}  // namespace PCA
}  // namespace ML
