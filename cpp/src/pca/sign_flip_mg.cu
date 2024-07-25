/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cuml/decomposition/sign_flip_mg.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>

using namespace MLCommon;

namespace ML {
namespace PCA {
namespace opg {

// TODO: replace these thrust code with cuda kernels or prims
template <typename T>
void findMaxAbsOfColumns(T* input,
                         std::size_t n_rows,
                         std::size_t n_cols,
                         T* max_vals,
                         cudaStream_t stream,
                         bool row_major = false)
{
  auto counting = thrust::make_counting_iterator(0);
  auto m        = n_rows;
  auto n        = n_cols;

  auto execution_policy = rmm::exec_policy(stream);

  if (row_major) {
    thrust::for_each(
      execution_policy, counting, counting + n_rows, [=] __device__(std::size_t idx) {
        T max                 = 0.0;
        std::size_t max_index = 0;
        std::size_t d_i       = idx;
        std::size_t end       = d_i + (m * n);

        for (auto i = d_i; i < end; i = i + m) {
          T val = input[i];
          if (val < 0.0) { val = -val; }
          if (val > max) {
            max       = val;
            max_index = i;
          }
        }
        max_vals[idx] = input[max_index];
      });
  } else {
    thrust::for_each(
      execution_policy, counting, counting + n_cols, [=] __device__(std::size_t idx) {
        T max                 = 0.0;
        std::size_t max_index = 0;
        std::size_t d_i       = idx * m;
        std::size_t end       = d_i + m;

        for (auto i = d_i; i < end; i++) {
          T val = input[i];
          if (val < 0.0) { val = -val; }
          if (val > max) {
            max       = val;
            max_index = i;
          }
        }
        max_vals[idx] = input[max_index];
      });
  }
}

// TODO: replace these thrust code with cuda kernels or prims
template <typename T>
void flip(T* input, std::size_t n_rows, std::size_t n_cols, T* max_vals, cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator(0);
  auto m        = n_rows;

  thrust::for_each(
    rmm::exec_policy(stream), counting, counting + n_cols, [=] __device__(std::size_t idx) {
      auto d_i = idx * m;
      auto end = d_i + m;

      if (max_vals[idx] < 0.0) {
        for (auto i = d_i; i < end; i++) {
          input[i] = -input[i];
        }
      }
    });
}

/**
 * @brief sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
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
void sign_flip_imp(raft::handle_t& handle,
                   std::vector<Matrix::Data<T>*>& input,
                   Matrix::PartDescriptor& input_desc,
                   T* components,
                   std::size_t n_components,
                   cudaStream_t* streams,
                   std::uint32_t n_stream)
{
  int rank = handle.get_comms().get_rank();

  const auto& comm = handle.get_comms();

  std::vector<Matrix::RankSizePair*> local_blocks = input_desc.blocksOwnedBy(rank);
  rmm::device_uvector<T> max_vals(
    std::max(size_t(comm.get_size()), local_blocks.size()) * n_components, streams[0]);

  for (std::size_t i = 0; i < input.size(); i++) {
    T* mv_loc = max_vals.data() + (i * n_components);
    findMaxAbsOfColumns(
      input[i]->ptr, local_blocks[i]->size, n_components, mv_loc, streams[i % n_stream]);
  }

  for (std::uint32_t i = 0; i < n_stream; i++) {
    handle.sync_stream(streams[i]);
  }

  findMaxAbsOfColumns(
    max_vals.data(), n_components, local_blocks.size(), max_vals.data(), streams[0], true);

  comm.allgather(max_vals.data(), max_vals.data(), n_components, streams[0]);
  comm.sync_stream(streams[0]);

  findMaxAbsOfColumns(
    max_vals.data(), n_components, comm.get_size(), max_vals.data(), streams[0], true);

  for (std::size_t i = 0; i < local_blocks.size(); i++) {
    flip(
      input[i]->ptr, local_blocks[i]->size, n_components, max_vals.data(), streams[i % n_stream]);
  }

  for (std::uint32_t i = 0; i < n_stream; i++) {
    handle.sync_stream(streams[i]);
  }

  flip(components, input_desc.N, n_components, max_vals.data(), streams[0]);
}

void sign_flip(raft::handle_t& handle,
               std::vector<Matrix::Data<float>*>& input_data,
               Matrix::PartDescriptor& input_desc,
               float* components,
               std::size_t n_components,
               cudaStream_t* streams,
               std::uint32_t n_stream)
{
  sign_flip_imp(handle, input_data, input_desc, components, n_components, streams, n_stream);
}

void sign_flip(raft::handle_t& handle,
               std::vector<Matrix::Data<double>*>& input_data,
               Matrix::PartDescriptor& input_desc,
               double* components,
               std::size_t n_components,
               cudaStream_t* streams,
               std::uint32_t n_stream)
{
  sign_flip_imp(handle, input_data, input_desc, components, n_components, streams, n_stream);
}

}  // namespace opg
}  // namespace PCA
}  // namespace ML
