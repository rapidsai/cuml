/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <raft/core/handle.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG Least squares calculation.
 */
void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<float>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<float>*>& b,
              float* w,
              cudaStream_t* streams,
              int n_streams);

void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<double>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<double>*>& b,
              double* w,
              cudaStream_t* streams,
              int n_streams);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon

