/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/sign_flip_mg.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/sum.cuh>
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

template <typename T>
void col_means_mg(const raft::handle_t& handle,
                  std::vector<Matrix::Data<T>*>& input,
                  Matrix::PartDescriptor& input_desc,
                  T* dots,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  cudaStream_t* streams,
                  std::uint32_t n_stream)
{
  const auto& comm                                = handle.get_comms();
  int rank                                        = comm.get_rank();
  std::vector<Matrix::RankSizePair*> local_blocks = input_desc.blocksOwnedBy(rank);
  std::size_t local_block_size = std::max(size_t(comm.get_size()), local_blocks.size());
  rmm::device_uvector<T> col_means_raw(local_block_size * n_cols, streams[0]);
  for (std::size_t i = 0; i < input.size(); i++) {
    T* input_chunk           = input[i]->ptr;
    std::size_t n_rows_chunk = local_blocks[i]->size;
    // Compute col_mean_chunk = sum(col_values) / n_rows
    raft::linalg::reduce<false, false>(col_means_raw.data() + (i * n_cols),
                                       input_chunk,
                                       n_cols,
                                       n_rows_chunk,
                                       T(0),
                                       streams[i],
                                       false,
                                       raft::identity_op(),
                                       raft::add_op(),
                                       raft::div_const_op<T>(n_rows));
  }
  for (std::uint32_t i = 0; i < n_stream; i++) {
    handle.sync_stream(streams[i]);
  }
  // Compute sum(col_mean_chunk)
  raft::stats::sum<true>(dots, col_means_raw.data(), n_cols, local_block_size, streams[0]);
}

template <typename T>
void sign_flip_components_u_imp(raft::handle_t& handle,
                                std::vector<Matrix::Data<T>*>& input,
                                Matrix::PartDescriptor& input_desc,
                                T* components,
                                std::size_t n_samples,
                                std::size_t n_features,
                                std::size_t n_components,
                                cudaStream_t* streams,
                                std::uint32_t n_stream,
                                bool center)
{
  const auto& comm = handle.get_comms();
  int rank         = comm.get_rank();

  std::vector<Matrix::RankSizePair*> local_blocks = input_desc.blocksOwnedBy(rank);

  auto components_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
    components, n_components, n_features);
  std::size_t local_block_size = std::max(size_t(comm.get_size()), local_blocks.size());
  rmm::device_uvector<T> max_vals_raw(local_block_size * n_components, streams[0]);
  rmm::device_uvector<T> max_vals(n_components, streams[0]);
  auto max_vals_view = raft::make_device_vector_view<T, std::size_t>(max_vals.data(), n_components);

  auto max_abs_op = [] __device__(T a, T b) {
    T abs_a = a >= T(0) ? a : -a;
    T abs_b = b >= T(0) ? b : -b;
    return abs_a >= abs_b ? a : b;
  };

  // Step 1: compute column means from input and center input
  rmm::device_uvector<T> col_means(n_features, streams[0]);
  col_means_mg<T>(
    handle, input, input_desc, col_means.data(), n_samples, n_features, streams, n_stream);
  handle.sync_stream(streams[0]);

  // Step 2: compute col-wise max abs per chunk
  for (std::size_t i = 0; i < input.size(); i++) {
    T* input_chunk              = input[i]->ptr;
    std::size_t n_samples_chunk = local_blocks[i]->size;
    if (center) {
      raft::stats::meanCenter<false, true>(
        input_chunk, input_chunk, col_means.data(), n_features, n_samples_chunk, streams[i]);
    }
    rmm::device_uvector<T> US_chunk(n_samples_chunk * n_components, streams[i]);
    raft::linalg::gemm(handle,
                       input_chunk,
                       n_samples_chunk,
                       n_features,
                       components,
                       US_chunk.data(),
                       n_samples_chunk,
                       n_components,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       T(1),
                       T(0),
                       streams[i]);
    raft::linalg::reduce<false, false>(max_vals_raw.data() + i * n_components,
                                       US_chunk.data(),
                                       n_components,
                                       n_samples_chunk,
                                       T(0),
                                       streams[i],
                                       false,
                                       raft::identity_op(),
                                       max_abs_op,
                                       raft::identity_op());
  }
  for (std::uint32_t i = 0; i < n_stream; i++) {
    handle.sync_stream(streams[i]);
  }
  raft::linalg::reduce<true, false>(max_vals.data(),
                                    max_vals_raw.data(),
                                    n_components,
                                    local_block_size,
                                    T(0),
                                    streams[0],
                                    false,
                                    raft::identity_op(),
                                    max_abs_op,
                                    raft::identity_op());
  handle.sync_stream(streams[0]);

  // Step 3: flip rows where needed
  raft::linalg::map_offset(
    handle,
    components_view,
    [components_view, max_vals_view, n_components, n_features] __device__(auto idx) {
      std::size_t row    = idx % n_components;
      std::size_t column = idx / n_components;
      return (max_vals_view(row) < T(0)) ? (-components_view(row, column))
                                         : components_view(row, column);
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

void sign_flip_components_u(raft::handle_t& handle,
                            std::vector<Matrix::Data<float>*>& input_data,
                            Matrix::PartDescriptor& input_desc,
                            float* components,
                            std::size_t n_samples,
                            std::size_t n_features,
                            std::size_t n_components,
                            cudaStream_t* streams,
                            std::uint32_t n_stream,
                            bool center)
{
  sign_flip_components_u_imp(handle,
                             input_data,
                             input_desc,
                             components,
                             n_samples,
                             n_features,
                             n_components,
                             streams,
                             n_stream,
                             center);
}

void sign_flip_components_u(raft::handle_t& handle,
                            std::vector<Matrix::Data<double>*>& input_data,
                            Matrix::PartDescriptor& input_desc,
                            double* components,
                            std::size_t n_samples,
                            std::size_t n_features,
                            std::size_t n_components,
                            cudaStream_t* streams,
                            std::uint32_t n_stream,
                            bool center)
{
  sign_flip_components_u_imp(handle,
                             input_data,
                             input_desc,
                             components,
                             n_samples,
                             n_features,
                             n_components,
                             streams,
                             n_stream,
                             center);
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
