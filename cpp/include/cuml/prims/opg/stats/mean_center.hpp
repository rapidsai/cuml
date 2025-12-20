/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <raft/core/comms.hpp>

namespace MLCommon {
namespace Stats {
namespace opg {

/**
 * @brief performs MNMG mean subtraction calculation.
 * @param[in,out] data the data that mean of every column is added
 * @param[in] dataDesc MNMG description of the input data
 * @param[in] mu mean of every column in data
 * @param[in] comm communicator
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
 */
void mean_center(const std::vector<Matrix::Data<double>*>& data,
                 const Matrix::PartDescriptor& dataDesc,
                 const Matrix::Data<double>& mu,
                 const raft::comms::comms_t& comm,
                 cudaStream_t* streams,
                 int n_streams);

void mean_center(const std::vector<Matrix::Data<float>*>& data,
                 const Matrix::PartDescriptor& dataDesc,
                 const Matrix::Data<float>& mu,
                 const raft::comms::comms_t& comm,
                 cudaStream_t* streams,
                 int n_streams);

/**
 * @brief performs MNMG mean add calculation.
 * @param[in,out] data the data that mean of every column is added
 * @param[in] dataDesc MNMG description of the input data
 * @param[in] mu mean of every column in data
 * @param[in] comm communicator
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
 */
void mean_add(const std::vector<Matrix::Data<double>*>& data,
              const Matrix::PartDescriptor& dataDesc,
              const Matrix::Data<double>& mu,
              const raft::comms::comms_t& comm,
              cudaStream_t* streams,
              int n_streams);

void mean_add(const std::vector<Matrix::Data<float>*>& data,
              const Matrix::PartDescriptor& dataDesc,
              const Matrix::Data<float>& mu,
              const raft::comms::comms_t& comm,
              cudaStream_t* streams,
              int n_streams);

}  // end namespace opg
}  // end namespace Stats
}  // end namespace MLCommon
