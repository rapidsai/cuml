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
 * @brief performs MNMG A^T x b calculation.
 * @param[in] handle cuML handle object
 * @param[out] out resulting A^T x b matrix
 * @param[in] A input data matrix
 * @param[in] ADesc MNMG description of A
 * @param[in] b input vector
 * @param[in] streams cuda streams
 * @param[in] n_streams number of streams
 */
void mv_aTb(const raft::handle_t& handle,
            Matrix::Data<double>& out,
            const std::vector<Matrix::Data<double>*>& A,
            const Matrix::PartDescriptor& ADesc,
            const std::vector<Matrix::Data<double>*>& b,
            cudaStream_t* streams,
            int n_streams);

void mv_aTb(const raft::handle_t& handle,
            Matrix::Data<float>& out,
            const std::vector<Matrix::Data<float>*>& A,
            const Matrix::PartDescriptor& ADesc,
            const std::vector<Matrix::Data<float>*>& b,
            cudaStream_t* streams,
            int n_streams);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
