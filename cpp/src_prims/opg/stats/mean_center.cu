/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mean_center.hpp"

#include <raft/stats/mean_center.cuh>

#include <opg/comm_utils.h>

namespace MLCommon {
namespace Stats {
namespace opg {

template <typename math_t, int TPB = 256>
void mean_center_impl(const std::vector<Matrix::Data<math_t>*>& data,
                      const Matrix::PartDescriptor& dataDesc,
                      const Matrix::Data<math_t>& mu,
                      const raft::comms::comms_t& comm,
                      cudaStream_t* streams,
                      int n_streams)
{
  std::vector<Matrix::RankSizePair*> local_blocks = dataDesc.blocksOwnedBy(comm.get_rank());

  for (size_t i = 0; i < data.size(); i++) {
    raft::stats::meanCenter<false, true>(data[i]->ptr,
                                         data[i]->ptr,
                                         mu.ptr,
                                         int(dataDesc.N),
                                         int(local_blocks[i]->size),
                                         streams[i % n_streams]);
  }
}

template <typename math_t, int TPB = 256>
void mean_add_impl(const std::vector<Matrix::Data<math_t>*>& data,
                   const Matrix::PartDescriptor& dataDesc,
                   const Matrix::Data<math_t>& mu,
                   const raft::comms::comms_t& comm,
                   cudaStream_t* streams,
                   int n_streams)
{
  std::vector<Matrix::RankSizePair*> local_blocks = dataDesc.blocksOwnedBy(comm.get_rank());

  for (size_t i = 0; i < data.size(); i++) {
    raft::stats::meanAdd<false, true>(data[i]->ptr,
                                      data[i]->ptr,
                                      mu.ptr,
                                      int(dataDesc.N),
                                      int(local_blocks[i]->size),
                                      streams[i % n_streams]);
  }
}

void mean_center(const std::vector<Matrix::Data<double>*>& data,
                 const Matrix::PartDescriptor& dataDesc,
                 const Matrix::Data<double>& mu,
                 const raft::comms::comms_t& comm,
                 cudaStream_t* streams,
                 int n_streams)
{
  mean_center_impl<double>(data, dataDesc, mu, comm, streams, n_streams);
}

void mean_center(const std::vector<Matrix::Data<float>*>& data,
                 const Matrix::PartDescriptor& dataDesc,
                 const Matrix::Data<float>& mu,
                 const raft::comms::comms_t& comm,
                 cudaStream_t* streams,
                 int n_streams)
{
  mean_center_impl<float>(data, dataDesc, mu, comm, streams, n_streams);
}

void mean_add(const std::vector<Matrix::Data<double>*>& data,
              const Matrix::PartDescriptor& dataDesc,
              const Matrix::Data<double>& mu,
              const raft::comms::comms_t& comm,
              cudaStream_t* streams,
              int n_streams)
{
  mean_add_impl<double>(data, dataDesc, mu, comm, streams, n_streams);
}

void mean_add(const std::vector<Matrix::Data<float>*>& data,
              const Matrix::PartDescriptor& dataDesc,
              const Matrix::Data<float>& mu,
              const raft::comms::comms_t& comm,
              cudaStream_t* streams,
              int n_streams)
{
  mean_add_impl<float>(data, dataDesc, mu, comm, streams, n_streams);
}

};  // end namespace opg
};  // namespace Stats
};  // end namespace MLCommon
