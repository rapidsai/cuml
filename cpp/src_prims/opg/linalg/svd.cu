/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/linalg/mm_aTa.hpp>
#include <cuml/prims/opg/linalg/svd.hpp>

#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>

#include <rmm/device_uvector.hpp>

#include <cuml/prims/opg/comm_utils.h>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG SVD calculation.
 */
template <typename T>
void svdEig_impl(const raft::handle_t& handle,
                 const std::vector<Matrix::Data<T>*>& A,
                 const Matrix::PartDescriptor& ADesc,
                 std::vector<Matrix::Data<T>*>& U,
                 T* S,
                 T* V,
                 cudaStream_t* streams,
                 int n_streams)
{
  auto& comm = handle.get_comms();

  bool gen_left_vec = true;
  int len           = ADesc.N * ADesc.N;

  rmm::device_uvector<T> cov_data(len, streams[0]);
  size_t cov_data_size = cov_data.size();
  Matrix::Data<T> cov{cov_data.data(), cov_data_size};

  LinAlg::opg::mm_aTa(handle, cov, A, ADesc, streams, n_streams);

  raft::linalg::eigDC(handle, cov.ptr, ADesc.N, ADesc.N, V, S, streams[0]);

  raft::matrix::colReverse(V, ADesc.N, ADesc.N, streams[0]);
  raft::matrix::rowReverse(S, ADesc.N, (size_t)1, streams[0]);

  T alpha = T(1);
  T beta  = T(0);
  raft::matrix::seqRoot(S, S, alpha, ADesc.N, streams[0], true);

  if (gen_left_vec) {
    std::vector<Matrix::RankSizePair*> partsToRanks = ADesc.blocksOwnedBy(comm.get_rank());
    for (size_t i = 0; i < partsToRanks.size(); i++) {
      raft::linalg::gemm(handle,
                         A[i]->ptr,
                         partsToRanks[i]->size,
                         ADesc.N,
                         V,
                         U[i]->ptr,
                         partsToRanks[i]->size,
                         ADesc.N,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         streams[i]);
      raft::matrix::matrixVectorBinaryDivSkipZero<false, true>(
        U[i]->ptr, S, partsToRanks[i]->size, ADesc.N, streams[i]);
    }

    // Wait for every partition to be completed
    for (int i = 0; i < n_streams; i++) {
      RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }
  }
}

void svdEig(const raft::handle_t& handle,
            const std::vector<Matrix::Data<float>*>& A,
            const Matrix::PartDescriptor& ADesc,
            std::vector<Matrix::Data<float>*>& U,
            float* S,
            float* V,
            cudaStream_t* streams,
            int n_streams)
{
  svdEig_impl(handle, A, ADesc, U, S, V, streams, n_streams);
}

void svdEig(const raft::handle_t& handle,
            const std::vector<Matrix::Data<double>*>& A,
            const Matrix::PartDescriptor& ADesc,
            std::vector<Matrix::Data<double>*>& U,
            double* S,
            double* V,
            cudaStream_t* streams,
            int n_streams)
{
  svdEig_impl(handle, A, ADesc, U, S, V, streams, n_streams);
}

};  // namespace opg
};  // namespace LinAlg
};  // namespace MLCommon
