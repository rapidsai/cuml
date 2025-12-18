/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/linalg/mm_aTa.hpp>
#include <cuml/prims/opg/stats/cov.hpp>
#include <cuml/prims/opg/stats/mean_center.hpp>

#include <raft/linalg/divide.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>

#include <cuml/prims/opg/comm_utils.h>

namespace MLCommon {
namespace Stats {
namespace opg {

template <typename math_t, int TPB = 256>
void cov_impl(const raft::handle_t& handle,
              Matrix::Data<math_t>& covar,
              const std::vector<Matrix::Data<math_t>*>& data,
              const Matrix::PartDescriptor& dataDesc,
              const Matrix::Data<math_t>& mu,
              bool sample,
              cudaStream_t* streams,
              int n_streams)
{
  auto& comm = handle.get_comms();
  // Subtract the mean
  Stats::opg::mean_center(data, dataDesc, mu, comm, streams, n_streams);

  // Wait for every partition to be completed
  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }

  // Perform matrix multiplication to get the covariance matrix
  LinAlg::opg::mm_aTa(handle, covar, data, dataDesc, streams, n_streams);

  raft::linalg::divideScalar(
    covar.ptr, covar.ptr, math_t(dataDesc.M - 1), dataDesc.N * dataDesc.N, streams[0]);
}

/// Instantiations
void cov(const raft::handle_t& handle,
         Matrix::Data<double>& covar,
         const std::vector<Matrix::Data<double>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<double>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams)
{
  cov_impl<double>(handle, covar, data, dataDesc, mu, sample, streams, n_streams);
}

void cov(const raft::handle_t& handle,
         Matrix::Data<float>& covar,
         const std::vector<Matrix::Data<float>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<float>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams)
{
  cov_impl<float>(handle, covar, data, dataDesc, mu, sample, streams, n_streams);
}

};  // namespace opg
};  // namespace Stats
};  // namespace MLCommon
