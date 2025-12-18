/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/linalg/gemm.hpp>
#include <cuml/prims/opg/linalg/qr.hpp>
#include <cuml/prims/opg/linalg/qr_based_svd.hpp>
#include <cuml/prims/opg/matrix/matrix_utils.hpp>

#include <raft/linalg/svd.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

template <typename T>
void svdQR(const raft::handle_t& h,
           T* sVector,
           std::vector<Matrix::Data<T>*>& uParts,
           T* vMatrix,
           bool genUMatrix,
           bool genVMatrix,
           T tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<T>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  const auto& comm = h.get_comms();

  cudaStream_t userStream = h.get_stream();

  size_t N     = desc.N;
  int numParts = desc.partsToRanks.size();

  size_t minPartSize = desc.M;
  for (int i = 0; i < numParts; i++) {
    if (minPartSize > desc.partsToRanks[i]->size) { minPartSize = desc.partsToRanks[i]->size; }
  }

  ASSERT(desc.M >= desc.N,
         "MLCommon::LinAlg::opg::SVD: Number of rows of"
         " input matrix can not be less than number of columns");
  ASSERT(minPartSize >= desc.N,
         "MLCommon::LinAlg::opg::SVD: Number of rows of "
         " any input matrix block can not be less than number of columns in"
         " the block");
  ASSERT(desc.layout == Matrix::Layout::LayoutColMajor,
         "MLCommon::LinAlg::opg::SVD: Intra block layout other than column"
         " major is not supported.");

  std::vector<Matrix::Data<T>*> qParts;
  Matrix::opg::allocate(h, qParts, desc, myRank, userStream);
  rmm::device_uvector<T> rMatrix(N * N, userStream);
  rmm::device_uvector<T> uOfR(N * N, userStream);

  RAFT_CUDA_TRY(cudaMemsetAsync(rMatrix.data(), 0, N * N * sizeof(T), userStream));
  qrDecomp(h, qParts, rMatrix.data(), inParts, desc, myRank);

  if (myRank == 0) {
    raft::linalg::svdJacobi(h,
                            rMatrix.data(),
                            N,
                            N,
                            sVector,
                            uOfR.data(),
                            vMatrix,
                            genUMatrix,
                            genVMatrix,
                            tolerance,
                            maxSweeps,
                            userStream);
  }
  comm.bcast(sVector, N, 0, userStream);
  comm.bcast(uOfR.data(), N * N, 0, userStream);
  comm.bcast(vMatrix, N * N, 0, userStream);
  comm.sync_stream(userStream);

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::gemm(h,
                         uParts[localId]->ptr,
                         qParts[localId]->ptr,
                         uOfR.data(),
                         desc.partsToRanks[i]->size,
                         N,
                         N,
                         true,
                         true,
                         true,
                         stream);
      localId++;
    }
  }
  h.sync_stream_pool();

  Matrix::opg::deallocate(h, qParts, desc, myRank, userStream);
}

// Instantiations
void svdQR(const raft::handle_t& h,
           float* sVector,
           std::vector<Matrix::Data<float>*>& uMatrixParts,
           float* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           float tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  svdQR<float>(h,
               sVector,
               uMatrixParts,
               vMatrixParts,
               genUMatrix,
               genVMatrix,
               tolerance,
               maxSweeps,
               inParts,
               desc,
               myRank);
}

void svdQR(const raft::handle_t& h,
           double* sVector,
           std::vector<Matrix::Data<double>*>& uMatrixParts,
           double* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           double tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  svdQR<double>(h,
                sVector,
                uMatrixParts,
                vMatrixParts,
                genUMatrix,
                genVMatrix,
                tolerance,
                maxSweeps,
                inParts,
                desc,
                myRank);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
