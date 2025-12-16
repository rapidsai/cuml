/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>
#include <mpi.h>

namespace MLCommon {
namespace Test {
namespace opg {

/**
 *
 * @brief Testing environment to handle googletest runs
 * @note Inspired from:
 * http://www.parresianz.com/mpi/c++/mpi-unit-testing-googletests-cmake/
 */
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp()
  {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nGpus;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&nGpus));

    ASSERT(
      nGpus >= size, "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d", nGpus, size);

    RAFT_CUDA_TRY(cudaSetDevice(rank));
  }

  void TearDown() { MPI_Finalize(); }
};

};  // end namespace opg
};  // end namespace Test
};  // end namespace MLCommon
