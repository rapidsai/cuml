/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/comm_utils.h>
#include <cuml/prims/opg/linalg/mm_aTa.hpp>
#include <cuml/prims/opg/linalg/svd.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/reverse.cuh>
#include <raft/matrix/sqrt.cuh>

#include <rmm/device_uvector.hpp>

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

  T alpha          = T(1);
  T beta           = T(0);
  auto orig_stream = handle.get_stream();
  raft::resource::set_cuda_stream(handle, streams[0]);
  raft::matrix::col_reverse(
    handle, raft::make_device_matrix_view<T, std::size_t, raft::col_major>(V, ADesc.N, ADesc.N));
  raft::matrix::row_reverse(
    handle,
    raft::make_device_matrix_view<T, std::size_t, raft::row_major>(S, ADesc.N, std::size_t(1)));

  raft::matrix::weighted_sqrt(
    handle,
    raft::make_device_matrix_view<const T, std::size_t, raft::row_major>(
      S, std::size_t(1), ADesc.N),
    raft::make_device_matrix_view<T, std::size_t, raft::row_major>(S, std::size_t(1), ADesc.N),
    raft::make_host_scalar_view(&alpha),
    true);
  raft::resource::set_cuda_stream(handle, orig_stream);

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
      auto orig_stream_i = handle.get_stream();
      raft::resource::set_cuda_stream(handle, streams[i]);
      raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
          U[i]->ptr, partsToRanks[i]->size, ADesc.N),
        raft::make_device_vector_view<const T, std::size_t>(S, ADesc.N));
      raft::resource::set_cuda_stream(handle, orig_stream_i);
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
