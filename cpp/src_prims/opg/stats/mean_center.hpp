/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>

namespace ML {
namespace Stats {
namespace opg {

/**
 * @brief performs MNMG mean subtraction calculation.
 * @input/output param data: the data that mean of every column is added
 * @input param dataDesc: MNMG description of the input data
 * @input param mu: mean of every column in data
 * @input param comm: communicator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
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
 * @input/output param data: the data that mean of every column is added
 * @input param dataDesc: MNMG description of the input data
 * @input param mu: mean of every column in data
 * @input param comm: communicator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
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
/** @} */

}  // end namespace opg
}  // end namespace Stats
}  // end namespace ML

