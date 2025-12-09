/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <opg/linalg/mm_aTa.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <rmm/device_uvector.hpp>

#include "opg/comm_utils.h"

namespace ML {
namespace LinAlg {
namespace opg {

template <typename math_t, int TPB = 256>
void mm_aTa_impl(const raft::handle_t& handle,
                 Matrix::Data<math_t>& out,
                 const std::vector<Matrix::Data<math_t>*>& A,
                 const Matrix::PartDescriptor& ADesc,
                 cudaStream_t* streams,
                 int n_streams)
{
  auto& comm = handle.get_comms();
  int rank   = comm.get_rank();

  std::vector<Matrix::RankSizePair*> local_blocks = ADesc.blocksOwnedBy(rank);

  rmm::device_uvector<math_t> local_mm_tmp(ADesc.N * ADesc.N, streams[0]);

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  int TN       = ADesc.N * ADesc.N;

  raft::linalg::gemm(handle,
                     A[0]->ptr,
                     local_blocks[0]->size,
                     ADesc.N,
                     A[0]->ptr,
                     out.ptr,
                     ADesc.N,
                     ADesc.N,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     streams[0]);

  for (int i = 1; i < A.size(); i++) {
    raft::linalg::gemm(handle,
                       A[i]->ptr,
                       local_blocks[i]->size,
                       ADesc.N,
                       A[i]->ptr,
                       local_mm_tmp.data(),
                       ADesc.N,
                       ADesc.N,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       streams[0]);

    raft::linalg::add(out.ptr, local_mm_tmp.data(), out.ptr, TN, streams[0]);
  }

  comm.allreduce(out.ptr, out.ptr, ADesc.N * ADesc.N, raft::comms::op_t::SUM, streams[0]);

  comm.sync_stream(streams[0]);
}

void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<double>& out,
            const std::vector<Matrix::Data<double>*>& A,
            const Matrix::PartDescriptor& ADesc,
            cudaStream_t* streams,
            int n_streams)
{
  mm_aTa_impl<double>(handle, out, A, ADesc, streams, n_streams);
}

void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<float>& out,
            const std::vector<Matrix::Data<float>*>& A,
            const Matrix::PartDescriptor& ADesc,
            cudaStream_t* streams,
            int n_streams)
{
  mm_aTa_impl<float>(handle, out, A, ADesc, streams, n_streams);
}

};  // namespace opg
};  // namespace LinAlg
};  // namespace ML

