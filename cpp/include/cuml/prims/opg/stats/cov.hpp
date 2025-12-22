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
 * @param[in] handle cuML handle object
 * @param[out] covar resulting covariance matrix
 * @param[in] data the data that cov matrix is calculated for
 * @param[in] dataDesc MNMG description of the input data
 * @param[in] mu mean of every column in data
 * @param[in] sample whether to compute sample covariance
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
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
