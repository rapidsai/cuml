/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace MLCommon {
namespace Stats {
namespace opg {

/**
 * @brief performs MNMG mean calculation.
 * @param[out] out mean of every column of the input
 * @param[in] in all the data partitions
 * @param[in] inDesc MNMG description of the input data
 * @param[in] comm communicator
 * @param[in] allocator allocator
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
 */
void mean(const raft::handle_t& handle,
          Matrix::Data<float>& out,
          const std::vector<Matrix::Data<float>*>& in,
          const Matrix::PartDescriptor& inDesc,
          cudaStream_t* streams,
          int n_streams);

void mean(const raft::handle_t& handle,
          Matrix::Data<double>& out,
          const std::vector<Matrix::Data<double>*>& in,
          const Matrix::PartDescriptor& inDesc,
          cudaStream_t* streams,
          int n_streams);

}  // end namespace opg
}  // end namespace Stats
}  // end namespace MLCommon
