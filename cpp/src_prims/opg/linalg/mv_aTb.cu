/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/linalg/mv_aTb.hpp>

#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/matrix/matrix.cuh>

#include <rmm/device_uvector.hpp>

#include <cuml/prims/opg/comm_utils.h>

namespace MLCommon {
namespace LinAlg {
namespace opg {

template <typename T, int TPB = 256>
void mv_aTb_impl(const raft::handle_t& handle,
                 Matrix::Data<T>& out,
                 const std::vector<Matrix::Data<T>*>& A,
                 const Matrix::PartDescriptor& ADesc,
                 const std::vector<Matrix::Data<T>*>& b,
                 cudaStream_t* streams,
                 int n_streams)
{
  auto& comm = handle.get_comms();

  int rank = comm.get_rank();

  std::vector<Matrix::RankSizePair*> local_blocks = ADesc.blocksOwnedBy(rank);

  rmm::device_uvector<T> local_mm_tmp(ADesc.N, streams[0]);

  T alpha = T(1);
  T beta  = T(0);

  raft::linalg::gemm(handle,
                     A[0]->ptr,
                     local_blocks[0]->size,
                     ADesc.N,
                     b[0]->ptr,
                     out.ptr,
                     ADesc.N,
                     1,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     streams[0]);

  for (size_t i = 1; i < A.size(); i++) {
    raft::linalg::gemm(handle,
                       A[i]->ptr,
                       local_blocks[i]->size,
                       ADesc.N,
                       b[i]->ptr,
                       local_mm_tmp.data(),
                       ADesc.N,
                       1,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       streams[0]);

    raft::linalg::add(out.ptr, local_mm_tmp.data(), out.ptr, ADesc.N, streams[0]);
  }

  comm.allreduce(out.ptr, out.ptr, ADesc.N, raft::comms::op_t::SUM, streams[0]);

  comm.sync_stream(streams[0]);
}

void mv_aTb(const raft::handle_t& handle,
            Matrix::Data<double>& out,
            const std::vector<Matrix::Data<double>*>& A,
            const Matrix::PartDescriptor& ADesc,
            const std::vector<Matrix::Data<double>*>& b,
            cudaStream_t* streams,
            int n_streams)
{
  mv_aTb_impl(handle, out, A, ADesc, b, streams, n_streams);
}

void mv_aTb(const raft::handle_t& handle,
            Matrix::Data<float>& out,
            const std::vector<Matrix::Data<float>*>& A,
            const Matrix::PartDescriptor& ADesc,
            const std::vector<Matrix::Data<float>*>& b,
            cudaStream_t* streams,
            int n_streams)
{
  mv_aTb_impl(handle, out, A, ADesc, b, streams, n_streams);
}

};  // namespace opg
};  // namespace LinAlg
};  // namespace MLCommon
