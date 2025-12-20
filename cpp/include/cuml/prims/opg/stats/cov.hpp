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
 * @brief performs MNMG covariance calculation.
 * @output param covar: resulting covariance matrix
 * @input param data: the data that cov matrix is calculated for
 * @input param dataDesc: MNMG description of the input data
 * @input param mu: mean of every column in data
 * @input param comm: communicator
 * @input param allocator: data allocator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 * @input param handle: cublas handle
 */
void cov(const raft::handle_t& handle,
         Matrix::Data<float>& covar,
         const std::vector<Matrix::Data<float>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<float>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams);

void cov(const raft::handle_t& handle,
         Matrix::Data<double>& covar,
         const std::vector<Matrix::Data<double>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<double>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams);

}  // end namespace opg
}  // end namespace Stats
}  // end namespace MLCommon
