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

#include <gtest/gtest.h>
#include <mpi.h>
#include <cuda_utils.cuh>
#include <opg/matrix/data.hpp>
#include <opg/matrix/descriptor.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>
#include <cuML_comms.hpp>

namespace MLCommon {
namespace Test {
namespace opg {
/**
   * @brief A naive attempt at creating different inputs in each of the ranks
   * @param seed the input seed (typically as defined in the test params)
   * @return a unique rank across all ranks
   */
inline unsigned long long rankBasedSeed(const cumlCommunicator &comm,
                                        unsigned long long seed) {
  int myRank = comm.getRank();
  return seed + myRank;
}

/** checks whether the current process is the root or not */
inline bool amIroot(const cumlCommunicator &comm, int rootRank = 0) {
  int myRank = comm.getRank();
  return myRank == rootRank;
}

/**
     *
     * @brief Testing environment to handle googletest runs
     * @note Inspired from:
     * http://www.parresianz.com/mpi/c++/mpi-unit-testing-googletests-cmake/
     */
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nGpus;
    CUDA_CHECK(cudaGetDeviceCount(&nGpus));

    ASSERT(nGpus >= size,
           "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d",
           nGpus, size);

    CUDA_CHECK(cudaSetDevice(rank));
  }

  void TearDown() { MPI_Finalize(); }
};

template <typename Type>
size_t computeTotalSize(const Matrix::Descriptor &desc, int rank) {
  auto myBlocks = desc.totalBlocksOwnedBy(rank);
  ///@todo: handle ragged cases!?
  return sizeof(Type) * myBlocks * desc.MB * desc.NB;
}

/*
    * @brief allocate the buffer for this worker
    * @param desc the descriptor showing the data allocation
    */
template <typename T>
void data_alloc(Matrix::Data<T> &data, const Matrix::Descriptor &desc, int rank,
                bool setZero = false) {
  ASSERT(data.ptr == nullptr, "Data seems to have already been allocated!");
  data.totalSize = computeTotalSize<T>(desc, rank);
  CUDA_CHECK(cudaMalloc((void **)&data.ptr, data.totalSize));
  if (setZero) CUDA_CHECK(cudaMemset(data.ptr, 0, data.totalSize));
}

/**
   * @brief Convert input data distribution to be one as requested by the caller
   * @param out the output converted data
   * @param outDesc the distribution descriptor for the output
   * @param in the input data
   * @param inDesc the distribution descriptor for the input
   * @note This currently only supports the special case of MB,NB being equal both
   *  for input and output buffers.
   */
template <typename Type>
void redistribute(Matrix::Data<Type> &out, const Matrix::Descriptor &outDesc,
                  const Matrix::Data<Type> &in,
                  const Matrix::Descriptor &inDesc,
                  const cumlCommunicator &comm) {
  ASSERT(outDesc.M == inDesc.M && outDesc.N == inDesc.N &&
           outDesc.MB == inDesc.MB && outDesc.NB == inDesc.NB &&
           outDesc.intraBlockLayout == inDesc.intraBlockLayout,
         "redistribute: currently only supports same values of M,N,MB,NB! "
         "You've passed in=%d,%d,%d,%d and out=%d,%d,%d,%d.",
         inDesc.M, inDesc.N, inDesc.MB, inDesc.NB, outDesc.M, outDesc.N,
         outDesc.MB, outDesc.NB);
  auto nBlocks = outDesc.blocks2device.size();
  int myRank = comm.getRank();

  auto myInBlocks = inDesc.totalBlocksOwnedBy(myRank);
  auto myOutBlocks = outDesc.totalBlocksOwnedBy(myRank);
  auto totalBlocks = myInBlocks + myOutBlocks;

  std::vector<MLCommon::cumlCommunicator::request_t> requests;
  requests.resize(totalBlocks);

  auto blockLen = inDesc.MB * inDesc.NB;
  // all the messages that I need to send
  int reqIdx = 0;
  for (size_t i = 0; i < nBlocks; ++i) {
    if (inDesc.blocks2device[i] == myRank) {
      auto dstRank = outDesc.blocks2device[i];
      comm.isend(in.ptr + reqIdx * blockLen, blockLen, dstRank, 0,
                 requests.data() + reqIdx);
      ++reqIdx;
    }
  }
  // all the messages that I need to receive
  reqIdx = 0;
  for (size_t i = 0; i < nBlocks; ++i) {
    if (outDesc.blocks2device[i] == myRank) {
      auto srcRank = inDesc.blocks2device[i];
      comm.irecv(out.ptr + reqIdx * blockLen, blockLen, srcRank, 0,
                 requests.data() + reqIdx);
      ++reqIdx;
    }
  }

  comm.waitall(requests.size(), requests.data());
  comm.barrier();
}

};  // end namespace opg
};  // end namespace Test
};  // end namespace MLCommon
