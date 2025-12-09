/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace ML {
namespace Stats {
namespace opg {

/**
 * @brief performs MNMG var calculation.
 * @output param Matrix: var of every column of the "in"
 * @input param in: all the data partitions
 * @input param inDesc: MNMG description of the input data
 * @input param comm: communicator
 * @input param allocator: allocator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 */
void var(const raft::handle_t& handle,
         Matrix::Data<float>& out,
         const std::vector<Matrix::Data<float>*>& in,
         const Matrix::PartDescriptor& inDesc,
         const float* mu,
         cudaStream_t* streams,
         int n_streams);

void var(const raft::handle_t& handle,
         Matrix::Data<double>& out,
         const std::vector<Matrix::Data<double>*>& in,
         const Matrix::PartDescriptor& inDesc,
         const double* mu,
         cudaStream_t* streams,
         int n_streams);
/** @} */

}  // end namespace opg
}  // end namespace Stats
}  // end namespace ML

