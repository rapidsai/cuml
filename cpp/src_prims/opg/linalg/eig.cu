/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/matrix/matrix_utils.hpp>

#include <raft/linalg/eig.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

template <typename T>
void eigDC(const raft::handle_t& h,
           T* eigenValues,
           T* eigenVectors,
           std::vector<Matrix::Data<T>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  ASSERT(desc.N == desc.M,
         "MLCommon::LinAlg::opg:Eig: Matrix needs to be square for Eigen"
         " computation");

  const auto& comm = h.get_comms();

  rmm::device_uvector<T> inGathered(0, stream);
  if (myRank == 0) { inGathered.resize(desc.M * desc.N, stream); }
  Matrix::opg::gather(h, inGathered.data(), inParts, desc, 0, myRank, stream);

  if (myRank == 0) {
    raft::linalg::eigDC(h, inGathered.data(), desc.N, desc.N, eigenVectors, eigenValues, stream);
  }

  comm.bcast(eigenVectors, desc.N * desc.N, 0, stream);
  comm.bcast(eigenValues, desc.N, 0, stream);
}

template <typename T>
void eigJacobi(const raft::handle_t& h,
               T* eigenValues,
               T* eigenVectors,
               std::vector<Matrix::Data<T>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  const auto& comm = h.get_comms();

  rmm::device_uvector<T> inGathered(0, stream);
  if (myRank == 0) { inGathered.resize(desc.N * desc.N, stream); }
  Matrix::opg::gather(h, inGathered.data(), inParts, desc, 0, myRank, stream);

  if (myRank == 0) {
    raft::linalg::eigJacobi(
      h, inGathered.data(), desc.N, desc.N, eigenVectors, eigenValues, stream);
  }

  comm.bcast(eigenVectors, desc.N * desc.N, 0, stream);
  comm.bcast(eigenValues, desc.N, 0, stream);
}

// Instantiations
void eigDC(const raft::handle_t& h,
           float* eigenValues,
           float* eigenVectors,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  eigDC<float>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigDC(const raft::handle_t& h,
           double* eigenValues,
           double* eigenVectors,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  eigDC<double>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigJacobi(const raft::handle_t& h,
               float* eigenValues,
               float* eigenVectors,
               std::vector<Matrix::Data<float>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  eigJacobi<float>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigJacobi(const raft::handle_t& h,
               double* eigenValues,
               double* eigenVectors,
               std::vector<Matrix::Data<double>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  eigJacobi<double>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
