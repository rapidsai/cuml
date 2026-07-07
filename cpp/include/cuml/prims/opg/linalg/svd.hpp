/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <cuml/common/export.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace CUML_EXPORT MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG Least squares calculation.
 */
void svdEig(const raft::handle_t& handle,
            const std::vector<Matrix::Data<float>*>& A,
            const Matrix::PartDescriptor& ADesc,
            std::vector<Matrix::Data<float>*>& U,
            float* S,
            float* V,
            cudaStream_t* streams,
            int n_streams);

void svdEig(const raft::handle_t& handle,
            const std::vector<Matrix::Data<double>*>& A,
            const Matrix::PartDescriptor& ADesc,
            std::vector<Matrix::Data<double>*>& U,
            double* S,
            double* V,
            cudaStream_t* streams,
            int n_streams);

}  // end namespace opg
}  // end namespace LinAlg
}  // namespace CUML_EXPORT MLCommon
