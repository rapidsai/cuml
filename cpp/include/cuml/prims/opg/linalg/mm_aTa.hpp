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
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG A^T x A calculation.
 * @param[in] handle cuML handle object
 * @param[out] out resulting A^T x A matrix
 * @param[in] in1 input data
 * @param[in] in1Desc MNMG description of the input
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
 */
void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<double>& out,
            const std::vector<Matrix::Data<double>*>& in1,
            const Matrix::PartDescriptor& in1Desc,
            cudaStream_t* streams,
            int n_streams);

void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<float>& out,
            const std::vector<Matrix::Data<float>*>& in1,
            const Matrix::PartDescriptor& in1Desc,
            cudaStream_t* streams,
            int n_streams);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
