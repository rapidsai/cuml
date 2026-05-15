/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "quantiles.h"
#include "random_utils.cuh"

#include <cuml/common/export.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/fill.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace ML {
namespace DT {

namespace detail {

inline std::vector<std::size_t> proportionalSampleCounts(
  std::vector<std::uint64_t> const& row_counts, std::uint64_t global_rows, int sample_count)
{
  std::vector<std::size_t> result(row_counts.size());
  std::uint64_t prefix = 0;
  for (std::size_t i = 0; i < row_counts.size(); ++i) {
    auto begin = (static_cast<std::uint64_t>(sample_count) * prefix) / global_rows;
    prefix += row_counts[i];
    auto end  = (static_cast<std::uint64_t>(sample_count) * prefix) / global_rows;
    result[i] = static_cast<std::size_t>(end - begin);
  }
  return result;
}

template <typename T>
static __global__ void gatherUniformSampledColumnKernel(
  T* out, const T* data, int sample_count, int n_rows, int col, uint64_t seed)
{
  int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  auto col_seed = fnv1a32_basis;
  col_seed      = fnv1a32(col_seed, static_cast<uint32_t>(seed));
  col_seed      = fnv1a32(col_seed, static_cast<uint32_t>(seed >> 32));
  col_seed      = fnv1a32(col_seed, static_cast<uint32_t>(col));
  // Sampling is with replacement. Duplicate values from sample collisions are
  // removed later when quantile candidates are compacted with thrust::unique.
  for (int sample_idx = tid; sample_idx < sample_count; sample_idx += blockDim.x * gridDim.x) {
    // Use sample_idx as the generator subsequence so each output position is
    // deterministic and independent of the CUDA block/thread layout.
    raft::random::PCGenerator gen(col_seed, static_cast<uint64_t>(sample_idx), uint64_t(0));
    raft::random::UniformIntDistParams<int, uint64_t> uniform_int_dist_params;
    uniform_int_dist_params.start = 0;
    uniform_int_dist_params.end   = n_rows;
    uniform_int_dist_params.diff  = static_cast<uint64_t>(n_rows);
    int row;
    raft::random::custom_next(gen, &row, uniform_int_dist_params, int(0), int(0));
    out[sample_idx] = data[static_cast<int64_t>(col) * n_rows + row];
  }
}

}  // namespace detail

template <typename T>
static __global__ void computeQuantilesKernel(
  T* quantiles, int* n_bins, const T* sorted_data, const int max_n_bins, const int n_rows)
{
  double bin_width = static_cast<double>(n_rows) / max_n_bins;

  for (int bin = threadIdx.x; bin < max_n_bins; bin += blockDim.x) {
    // get index by interpolation
    int idx        = int(round((bin + 1) * bin_width)) - 1;
    idx            = min(max(0, idx), n_rows - 1);
    quantiles[bin] = sorted_data[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // make quantiles unique, in-place
    // thrust::seq to explicitly disable cuda dynamic parallelism here
    auto new_last = thrust::unique(thrust::seq, quantiles, quantiles + max_n_bins);
    // get the unique count
    *n_bins = new_last - quantiles;
  }

  __syncthreads();
  return;
}

template <typename T>
using QuantileReturnValue = std::tuple<ML::DT::Quantiles<T, int>,
                                       std::shared_ptr<rmm::device_uvector<T>>,
                                       std::shared_ptr<rmm::device_uvector<int>>>;

/**
 * @brief Compute per-feature quantile split candidates from uniformly sampled rows.
 *
 * Each feature column is sampled independently with replacement using a deterministic
 * seed derived from `seed`, the feature index, and the output sample index. When the
 * requested sample budget is at least the local row count, the full column is used.
 *
 * @tparam T Floating-point input type.
 * @param handle RAFT handle used for stream and resource access.
 * @param data Column-major input matrix with shape `[n_cols, n_rows]`.
 * @param max_n_bins Maximum number of quantile candidates to retain per feature.
 * @param n_rows Number of rows in `data`.
 * @param n_cols Number of columns in `data`.
 * @param oversampling_factor Multiplier applied to `max_n_bins` to choose the
 * sampled row budget per feature before sorting and quantile extraction. The
 * default of 4 is a conservative choice while still bounding memory; for fixed
 * `max_n_bins`, rank error decreases like O(1 / sqrt(oversampling_factor)), so
 * returns from increasing this are strongly diminishing.
 * @param seed User seed for deterministic sampling.
 * @return Quantile metadata and owning device buffers for quantile values and bin counts.
 */
template <typename T>
CUML_EXPORT QuantileReturnValue<T> computeQuantiles(const raft::handle_t& handle,
                                                    const T* data,
                                                    int max_n_bins,
                                                    int n_rows,
                                                    int n_cols,
                                                    int oversampling_factor = 4,
                                                    uint64_t seed           = uint64_t{0})
{
  raft::common::nvtx::push_range("computeQuantiles");
  RAFT_EXPECTS(data != nullptr, "data pointer must not be null");
  RAFT_EXPECTS(max_n_bins > 0, "max_n_bins must be positive");
  RAFT_EXPECTS(n_rows > 0, "n_rows must be positive");
  RAFT_EXPECTS(n_cols > 0, "n_cols must be positive");
  RAFT_EXPECTS(oversampling_factor > 0, "oversampling_factor must be positive");

  auto stream      = handle.get_stream();
  bool distributed = raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1;
  int rank         = distributed ? handle.get_comms().get_rank() : 0;
  int comm_size    = distributed ? handle.get_comms().get_size() : 1;
  int64_t size     = static_cast<int64_t>(max_n_bins) * oversampling_factor;
  auto target_sample = std::max<int64_t>(1, size);

  std::uint64_t global_rows = static_cast<std::uint64_t>(n_rows);
  std::vector<std::uint64_t> rank_rows(1, global_rows);
  std::vector<std::size_t> rank_sample_counts(1, 0);
  std::vector<std::size_t> rank_sample_displs(1, 0);

  if (distributed) {
    rmm::device_uvector<std::uint64_t> row_counts(comm_size, stream);
    auto local_rows = static_cast<std::uint64_t>(n_rows);
    raft::update_device(row_counts.data(), &local_rows, 1, stream);
    handle.get_comms().allgather(row_counts.data(), row_counts.data(), 1, stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF quantile row-count all-gather.");
    rank_rows.resize(comm_size);
    raft::update_host(rank_rows.data(), row_counts.data(), comm_size, stream);
    handle.sync_stream(stream);
    global_rows = std::accumulate(rank_rows.begin(), rank_rows.end(), std::uint64_t{0});
  }
  RAFT_EXPECTS(global_rows > 0, "global row count must be positive");

  int sample_count = static_cast<int>(
    std::min<std::uint64_t>(global_rows, static_cast<std::uint64_t>(target_sample)));
  rank_sample_counts = detail::proportionalSampleCounts(rank_rows, global_rows, sample_count);
  rank_sample_displs.resize(comm_size);
  for (int i = 1; i < comm_size; ++i) {
    rank_sample_displs[i] = rank_sample_displs[i - 1] + rank_sample_counts[i - 1];
  }
  int local_sample_count = static_cast<int>(rank_sample_counts[rank]);

  rmm::device_uvector<T> sampled_column(sample_count, stream);
  rmm::device_uvector<T> sorted_sample(sample_count, stream);
  rmm::device_uvector<T> local_sampled_column(
    distributed ? std::max(1, local_sample_count) : sample_count, stream);
  auto quantiles_array = std::make_shared<rmm::device_uvector<T>>(n_cols * max_n_bins, stream);
  auto n_bins_array    = std::make_shared<rmm::device_uvector<int>>(n_cols, stream);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr,
                                               temp_storage_bytes,
                                               sampled_column.data(),
                                               sorted_sample.data(),
                                               sample_count,
                                               0,
                                               8 * sizeof(T),
                                               stream));
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, stream);

  int n_threads = 256;
  int n_blocks  = raft::ceildiv(std::max(1, local_sample_count), n_threads);
  n_blocks      = std::min(n_blocks, 1024);

  for (int col = 0; col < n_cols; col++) {
    raft::common::nvtx::push_range("sample quantile column");
    T* sort_input = sampled_column.data();
    if (distributed) {
      if (local_sample_count == n_rows) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(local_sampled_column.data(),
                                      data + static_cast<int64_t>(col) * n_rows,
                                      sizeof(T) * n_rows,
                                      cudaMemcpyDeviceToDevice,
                                      stream));
      } else if (local_sample_count > 0) {
        detail::gatherUniformSampledColumnKernel<<<n_blocks, n_threads, 0, stream>>>(
          local_sampled_column.data(), data, local_sample_count, n_rows, col, seed);
        RAFT_CUDA_TRY(cudaGetLastError());
      }
      handle.get_comms().allgatherv(local_sampled_column.data(),
                                    sampled_column.data(),
                                    rank_sample_counts.data(),
                                    rank_sample_displs.data(),
                                    stream);
      ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed RF quantile sample all-gather.");
    } else {
      if (sample_count == n_rows) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(sampled_column.data(),
                                      data + static_cast<int64_t>(col) * n_rows,
                                      sizeof(T) * n_rows,
                                      cudaMemcpyDeviceToDevice,
                                      stream));
      } else {
        detail::gatherUniformSampledColumnKernel<<<n_blocks, n_threads, 0, stream>>>(
          sampled_column.data(), data, sample_count, n_rows, col, seed);
        RAFT_CUDA_TRY(cudaGetLastError());
      }
    }
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("sort sampled quantile column");
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys((void*)(d_temp_storage.data()),
                                                 temp_storage_bytes,
                                                 sort_input,
                                                 sorted_sample.data(),
                                                 sample_count,
                                                 0,
                                                 8 * sizeof(T),
                                                 stream));
    raft::common::nvtx::pop_range();

    int quantile_offset = col * max_n_bins;
    int bins_offset     = col;
    raft::common::nvtx::push_range("computeQuantilesKernel @quantiles.cuh");
    computeQuantilesKernel<<<1, std::min(1024, max_n_bins), 0, stream>>>(
      quantiles_array->data() + quantile_offset,
      n_bins_array->data() + bins_offset,
      sorted_sample.data(),
      max_n_bins,
      sample_count);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::common::nvtx::pop_range();
  }

  handle.sync_stream(stream);

  Quantiles<T, int> quantiles;
  quantiles.quantiles_array = quantiles_array->data();
  quantiles.n_bins_array    = n_bins_array->data();
  raft::common::nvtx::pop_range();
  return std::make_tuple(quantiles, quantiles_array, n_bins_array);
}

}  // namespace DT
}  // namespace ML
