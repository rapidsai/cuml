/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "math.hpp"

#include <raft/linalg/divide.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>

#include <opg/comm_utils.h>

namespace MLCommon {
namespace Matrix {
namespace opg {

template <bool rowMajor, bool bcastAlongRows, typename T, int TPB = 256>
void matrixVectorBinaryDivSkipZero_impl(std::vector<Matrix::Data<T>*>& data,
                                        const Matrix::PartDescriptor& dataDesc,
                                        const Matrix::Data<T>& vec,
                                        bool return_zero,
                                        const raft::comms::comms_t& comm,
                                        cudaStream_t* streams,
                                        int n_streams)
{
  int rank = comm.get_rank();

  std::vector<Matrix::RankSizePair*> localBlocks = dataDesc.blocksOwnedBy(rank);

  for (size_t i = 0; i < localBlocks.size(); i++) {
    raft::matrix::matrixVectorBinaryDivSkipZero<rowMajor, bcastAlongRows>(
      data[i]->ptr, vec.ptr, localBlocks[i]->size, dataDesc.N, streams[i], return_zero);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }
}

template <bool rowMajor, bool bcastAlongRows, typename T, int TPB = 256>
void matrixVectorBinaryMult_impl(std::vector<Matrix::Data<T>*>& data,
                                 const Matrix::PartDescriptor& dataDesc,
                                 const Matrix::Data<T>& vec,
                                 const raft::comms::comms_t& comm,
                                 cudaStream_t* streams,
                                 int n_streams)
{
  int rank = comm.get_rank();

  std::vector<Matrix::RankSizePair*> localBlocks = dataDesc.blocksOwnedBy(rank);

  for (size_t i = 0; i < localBlocks.size(); i++) {
    raft::matrix::matrixVectorBinaryMult<rowMajor, bcastAlongRows>(
      data[i]->ptr, vec.ptr, localBlocks[i]->size, dataDesc.N, streams[i]);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<double>*>& data,
                                   const Matrix::PartDescriptor& dataDesc,
                                   const Matrix::Data<double>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams)
{
  matrixVectorBinaryDivSkipZero_impl<rowMajor, bcastAlongRows, double>(
    data, dataDesc, vec, return_zero, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<float>*>& data,
                                   const Matrix::PartDescriptor& dataDesc,
                                   const Matrix::Data<float>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams)
{
  matrixVectorBinaryDivSkipZero_impl<rowMajor, bcastAlongRows, float>(
    data, dataDesc, vec, return_zero, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<double>*>& data,
                            const Matrix::PartDescriptor& dataDesc,
                            const Matrix::Data<double>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams)
{
  matrixVectorBinaryMult_impl<rowMajor, bcastAlongRows, double>(
    data, dataDesc, vec, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<float>*>& data,
                            const Matrix::PartDescriptor& dataDesc,
                            const Matrix::Data<float>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams)
{
  matrixVectorBinaryMult_impl<rowMajor, bcastAlongRows, float>(
    data, dataDesc, vec, comm, streams, n_streams);
}

template void matrixVectorBinaryDivSkipZero<false, true>(std::vector<Matrix::Data<double>*>&,
                                                         const Matrix::PartDescriptor&,
                                                         const Matrix::Data<double>&,
                                                         bool,
                                                         const raft::comms::comms_t&,
                                                         cudaStream_t*,
                                                         int);

template void matrixVectorBinaryDivSkipZero<false, true>(std::vector<Matrix::Data<float>*>&,
                                                         const Matrix::PartDescriptor&,
                                                         const Matrix::Data<float>&,
                                                         bool,
                                                         const raft::comms::comms_t&,
                                                         cudaStream_t*,
                                                         int);

template void matrixVectorBinaryMult<false, true>(std::vector<Matrix::Data<double>*>&,
                                                  const Matrix::PartDescriptor&,
                                                  const Matrix::Data<double>&,
                                                  const raft::comms::comms_t&,
                                                  cudaStream_t*,
                                                  int);

template void matrixVectorBinaryMult<false, true>(std::vector<Matrix::Data<float>*>&,
                                                  const Matrix::PartDescriptor&,
                                                  const Matrix::Data<float>&,
                                                  const raft::comms::comms_t&,
                                                  cudaStream_t*,
                                                  int);

};  // namespace opg
};  // namespace Matrix
};  // namespace MLCommon
