/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/linalg/gemm.hpp>
#include <cuml/prims/opg/matrix/matrix_utils.hpp>

#include <raft/linalg/gemm.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

template <typename T>
void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<T>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<T>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<T>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream)
{
  ASSERT(inXDesc.N == inYDesc.M,
         "MLCommon::LinAlg::opg: Number of rows in X "
         "and number of columns in Y while performing Z = X * Y GEMM "
         "operation, can not be different");
  ASSERT(outZDesc.M == inXDesc.M,
         "MLCommon::LinAlg::opg: Number of rows in Z "
         "and number of rows in X while performing Z = X * Y GEMM "
         "operation, can not be different");
  ASSERT(outZDesc.N == inYDesc.N,
         "MLCommon::LinAlg::opg: Number of columns "
         "in X and number of columns in Y while performing Z = X * Y GEMM "
         "operation, can not be different");

  ASSERT(outZDesc.partsToRanks.size() == inXDesc.partsToRanks.size(),
         "MLCommon::LinAlg::opg: Distribution of parts of Z and X while "
         "performing Z = X * Y GEMM operation, can not be different");

  for (size_t i = 0; i < inXDesc.partsToRanks.size(); i++) {
    ASSERT(outZDesc.partsToRanks[i]->size == inXDesc.partsToRanks[i]->size,
           "MLCommon::LinAlg::opg: Distribution of parts of Z and X while "
           "performing Z = X * Y GEMM operation, can not be different");
    ASSERT(outZDesc.partsToRanks[i]->rank == inXDesc.partsToRanks[i]->rank,
           "MLCommon::LinAlg::opg: Distribution of parts of Z and X while "
           "performing Z = X * Y GEMM operation, can not be different");
  }

  int totalSizeY = inYDesc.M * inYDesc.N;

  rmm::device_uvector<T> gatheredY(totalSizeY, stream);

  Matrix::opg::allGather(h, gatheredY.data(), inYParts, inYDesc, myRank, stream);
  // gathered Y is always going to be row major format
  for (size_t i = 0, localIndex = 0; i < outZDesc.partsToRanks.size(); i++) {
    if (myRank == outZDesc.partsToRanks[i]->rank) {
      raft::linalg::gemm(h,
                         outZParts[localIndex]->ptr,
                         inXParts[localIndex]->ptr,
                         gatheredY.data(),
                         outZDesc.partsToRanks[i]->size,
                         outZDesc.N,
                         inXDesc.N,
                         outZDesc.layout == Matrix::LayoutColMajor,
                         inXDesc.layout == Matrix::LayoutColMajor,
                         false,
                         stream);
      localIndex++;
    }
  }
}

// Instantiations

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<float>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<float>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<float>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream)
{
  gemm<float>(h, outZParts, outZDesc, inXParts, inXDesc, inYParts, inYDesc, myRank, stream);
}

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<double>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<double>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<double>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream)
{
  gemm<double>(h, outZParts, outZDesc, inXParts, inXDesc, inYParts, inYDesc, myRank, stream);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
