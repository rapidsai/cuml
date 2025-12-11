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
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG Least squares calculation.
 */
void colNorm2(const raft::handle_t& handle,
              Matrix::Data<double>& out,
              const std::vector<Matrix::Data<double>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams);

void colNorm2(const raft::handle_t& handle,
              Matrix::Data<float>& out,
              const std::vector<Matrix::Data<float>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams);

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<double>& out,
                   const std::vector<Matrix::Data<double>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams);

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<float>& out,
                   const std::vector<Matrix::Data<float>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams);

/** @} */

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace ML

