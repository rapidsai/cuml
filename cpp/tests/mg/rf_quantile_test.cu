/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_opg_utils.h"

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

void initialize_mpi_once()
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) { MPI_Init(nullptr, nullptr); }
}

template <typename T>
void hash_value(std::uint64_t& hash, const T& value)
{
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
}

template <typename T>
std::uint64_t hash_quantiles(std::vector<T> const& quantiles, int n_bins)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, n_bins);
  for (int i = 0; i < n_bins; ++i) {
    hash_value(hash, quantiles[i]);
  }
  return hash;
}

template <typename T>
class RfMgQuantileTest : public ::testing::Test {
 public:
  void SetUp() override
  {
    initialize_mpi_once();
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_gpus = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&n_gpus));
    ASSERT_GE(n_gpus, size);
    RAFT_CUDA_TRY(cudaSetDevice(rank));

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    constexpr int n_cols         = 3;
    constexpr int max_n_bins     = 16;
    constexpr int oversample     = 4;
    constexpr std::uint64_t seed = 123456789ULL;
    int n_rows                   = 257 + 17 * rank;

    std::vector<T> h_data(static_cast<std::size_t>(n_cols) * n_rows);
    for (int col = 0; col < n_cols; ++col) {
      for (int row = 0; row < n_rows; ++row) {
        h_data[static_cast<std::size_t>(col) * n_rows + row] =
          static_cast<T>(col * 100000 + rank * 1000 + row);
      }
    }

    rmm::device_uvector<T> data(h_data.size(), handle.get_stream());
    raft::update_device(data.data(), h_data.data(), h_data.size(), handle.get_stream());

    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, data.data(), max_n_bins, n_rows, n_cols, oversample, seed);

    std::vector<int> h_n_bins(n_cols);
    std::vector<T> h_quantiles(static_cast<std::size_t>(n_cols) * max_n_bins);
    raft::update_host(h_n_bins.data(), n_bins_array->data(), n_cols, handle.get_stream());
    raft::update_host(
      h_quantiles.data(), quantiles_array->data(), h_quantiles.size(), handle.get_stream());
    handle.sync_stream();

    std::uint64_t local_hash = 1469598103934665603ULL;
    for (int col = 0; col < n_cols; ++col) {
      ASSERT_GT(h_n_bins[col], 1);
      ASSERT_LE(h_n_bins[col], max_n_bins);
      std::vector<T> col_quantiles(h_quantiles.begin() + col * max_n_bins,
                                   h_quantiles.begin() + (col + 1) * max_n_bins);
      hash_value(local_hash, hash_quantiles(col_quantiles, h_n_bins[col]));
    }

    std::vector<std::uint64_t> rank_hashes(size);
    MPI_Allgather(
      &local_hash, 1, MPI_UINT64_T, rank_hashes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    for (auto hash : rank_hashes) {
      EXPECT_EQ(hash, local_hash);
    }
  }
};

using RfMgQuantileTestF = RfMgQuantileTest<float>;
TEST_F(RfMgQuantileTestF, SharedAcrossRanks) {}

using RfMgQuantileTestD = RfMgQuantileTest<double>;
TEST_F(RfMgQuantileTestD, SharedAcrossRanks) {}

}  // namespace opg
}  // namespace Test
}  // namespace ML
