/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../matrix/matrix_utils.hpp"
#include "gemm.hpp"
#include "qr.hpp"

#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/transpose.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * \brief This function repacks multiple blocks of matrix in to a single blocks.
 *  Input and output blocks are assumed to be in column major order.
 */
template <typename T>
static __global__ void r1Repack(
  T* inMatNR, T* inMatNRCorrected, int totalElements, int M, int totalBlocks)
{
  int inElemId = threadIdx.x + blockDim.x * blockIdx.x;

  if (inElemId < totalElements) {
    int partId                  = inElemId / (M * M);
    int blockOffset             = inElemId % (M * M);
    int inRow                   = blockOffset % M;
    int inCol                   = blockOffset / M;
    int outRow                  = partId * M + inRow;
    int outCol                  = inCol;
    int outElemId               = outCol * totalBlocks * M + outRow;
    inMatNRCorrected[outElemId] = inMatNR[inElemId];
  }
}

/**
 * \brief This function copies upper triangular part of in matrix to out matrix
 */
template <typename T>
static __global__ void copyUpperTriangle(T* out, T* in, int N)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = threadIdx.y + blockDim.y * blockIdx.y;
  if (row < N && col < N && row <= col) { out[col * N + row] = in[col * N + row]; }
}

template <typename T>
void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<T>*>& outQParts,
              T* outR,
              std::vector<Matrix::Data<T>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  const auto& comm = h.get_comms();

  cudaStream_t userStream = h.get_stream();

  size_t N          = desc.N;
  int numParts      = desc.partsToRanks.size();
  int numLocalParts = desc.totalBlocksOwnedBy(myRank);

  std::vector<size_t> partSizes;
  size_t maxPartSize = 0;
  size_t minPartSize = desc.M;
  for (int i = 0; i < numParts; i++) {
    partSizes.push_back(desc.partsToRanks[i]->size);
    if (maxPartSize < desc.partsToRanks[i]->size) { maxPartSize = desc.partsToRanks[i]->size; }
    if (minPartSize > desc.partsToRanks[i]->size) { minPartSize = desc.partsToRanks[i]->size; }
  }

  ASSERT(desc.M >= desc.N,
         "MLCommon::LinAlg::opg Number of rows of input"
         " matrix can not be less than number of columns");
  ASSERT(minPartSize >= desc.N,
         "MLCommon::LinAlg::opg Number of rows in "
         " any part of matrix can not be less than number of columns in the "
         "matrix");
  ASSERT(desc.layout == Matrix::Layout::LayoutColMajor,
         "Multi::QRDecomp: Intra block layout other than column major is not"
         " supported.");

  T* r1Parts     = nullptr;
  T* r1Collected = nullptr;
  T* r1Corrected = nullptr;
  T* q2          = nullptr;
  T* r2          = nullptr;
  T* q2RowMajor  = nullptr;
  T* q2Parts     = nullptr;

  rmm::device_uvector<T> aBuffer(numParts * N * N, userStream);
  rmm::device_uvector<T> bBuffer(numParts * N * N, userStream);
  rmm::device_uvector<T> gemmBuffer(maxPartSize * N, userStream);

  r1Parts = aBuffer.data();
  q2Parts = bBuffer.data();

  if (myRank == 0) {
    r1Collected = bBuffer.data();
    r1Corrected = aBuffer.data();
    q2          = bBuffer.data();
    q2RowMajor  = aBuffer.data();
  }

  if (myRank == 0) {
    r2 = outR;
  } else {
    r2 = aBuffer.data();
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(r1Parts, 0, numLocalParts * N * N * sizeof(T), userStream));

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::qrGetQR(h,
                            inParts[localId]->ptr,
                            outQParts[localId]->ptr,
                            r1Parts + localId * N * N,
                            partSizes[i],
                            N,
                            stream);
      localId++;
    }
  }
  h.sync_stream_pool();
  comm.sync_stream(userStream);

  std::vector<raft::comms::request_t> requests;
  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      requests.resize(requests.size() + 1);
      comm.isend(r1Parts + localId * N * N, N * N, 0, 0, &requests.back());
      localId++;
    }
    if (myRank == 0) {
      requests.resize(requests.size() + 1);
      comm.irecv(r1Collected + i * N * N, N * N, desc.partsToRanks[i]->rank, 0, &requests.back());
    }
  }
  comm.waitall(requests.size(), requests.data());
  requests.clear();
  if (myRank == 0) {
    dim3 block(256);
    dim3 grid((numParts * N * N + block.x - 1) / block.x);
    r1Repack<<<grid, block, 0, userStream>>>(
      r1Collected, r1Corrected, numParts * N * N, N, numParts);
    raft::linalg::qrGetQR(h, r1Corrected, q2, r2, numParts * N, N, userStream);
    raft::linalg::transpose(h, q2, q2RowMajor, numParts * N, N, userStream);
  }
  comm.sync_stream(userStream);

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (myRank == 0) {
      requests.resize(requests.size() + 1);
      comm.isend(q2RowMajor + i * N * N, N * N, desc.partsToRanks[i]->rank, 0, &requests.back());
    }
    if (desc.partsToRanks[i]->rank == myRank) {
      requests.resize(requests.size() + 1);
      comm.irecv(q2Parts + localId * N * N, N * N, 0, 0, &requests.back());
      localId++;
    }
  }
  comm.waitall(requests.size(), requests.data());
  requests.clear();

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::gemm(h,
                         gemmBuffer.data(),
                         outQParts[localId]->ptr,
                         q2Parts + localId * N * N,
                         partSizes[i],
                         N,
                         N,
                         true,
                         true,
                         false,
                         stream);
      RAFT_CUDA_TRY(cudaMemcpyAsync(outQParts[localId]->ptr,
                                    gemmBuffer.data(),
                                    partSizes[i] * N * sizeof(T),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
      localId++;
    }
  }
  h.sync_stream_pool();

  comm.bcast(r2, N * N, 0, userStream);
  comm.sync_stream(userStream);

  dim3 block(128, 4, 1);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y, 1);
  copyUpperTriangle<<<1, 10, 0, userStream>>>(outR, r2, N);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Instantiations
void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& outQParts,
              float* outR,
              std::vector<Matrix::Data<float>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  qrDecomp<float>(h, outQParts, outR, inParts, desc, myRank);
}

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& outQParts,
              double* outR,
              std::vector<Matrix::Data<double>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  qrDecomp<double>(h, outQParts, outR, inParts, desc, myRank);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
