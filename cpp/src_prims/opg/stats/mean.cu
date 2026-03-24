/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/comm_utils.h>
#include <cuml/prims/opg/stats/mean.hpp>

#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Stats {
namespace opg {

template <typename T, int TPB = 256>
void mean_impl(const raft::handle_t& handle,
               Matrix::Data<T>& out,
               const std::vector<Matrix::Data<T>*>& in,
               const Matrix::PartDescriptor& inDesc,
               cudaStream_t* streams,
               int n_streams)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<T> local_means_tmp(in.size() * inDesc.N, streams[0]);
  rmm::device_uvector<T> local_means_tmp_t(in.size() * inDesc.N, streams[0]);

  std::vector<Matrix::RankSizePair*> localBlocks = inDesc.blocksOwnedBy(comm.get_rank());
  T weight                                       = T(1) / T(inDesc.M);

  for (size_t i = 0; i < localBlocks.size(); i++) {
    T* loc = local_means_tmp.data() + (i * inDesc.N);
    raft::stats::sum<false>(
      loc, in[i]->ptr, inDesc.N, localBlocks[i]->size, streams[i % n_streams]);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }

  raft::linalg::multiplyScalar(
    local_means_tmp.data(), local_means_tmp.data(), weight, in.size() * inDesc.N, streams[0]);

  raft::linalg::transpose(handle,
                          local_means_tmp.data(),
                          local_means_tmp_t.data(),
                          inDesc.N,
                          localBlocks.size(),
                          streams[0]);

  raft::stats::sum<false>(
    local_means_tmp.data(), local_means_tmp_t.data(), inDesc.N, localBlocks.size(), streams[0]);

  comm.allreduce(local_means_tmp.data(), out.ptr, inDesc.N, raft::comms::op_t::SUM, streams[0]);

  comm.sync_stream(streams[0]);
}

void mean(const raft::handle_t& handle,
          Matrix::Data<double>& out,
          const std::vector<Matrix::Data<double>*>& in,
          const Matrix::PartDescriptor& inDesc,
          cudaStream_t* streams,
          int n_streams)
{
  mean_impl<double>(handle, out, in, inDesc, streams, n_streams);
}

void mean(const raft::handle_t& handle,
          Matrix::Data<float>& out,
          const std::vector<Matrix::Data<float>*>& in,
          const Matrix::PartDescriptor& inDesc,
          cudaStream_t* streams,
          int n_streams)
{
  mean_impl<float>(handle, out, in, inDesc, streams, n_streams);
}

};  // namespace opg
};  // namespace Stats
};  // namespace MLCommon
