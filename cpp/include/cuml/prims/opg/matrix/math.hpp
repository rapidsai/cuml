/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "data.hpp"
#include "part_descriptor.hpp"

#include <cuml/common/export.hpp>

#include <raft/core/comms.hpp>

namespace CUML_EXPORT MLCommon {
namespace Matrix {
namespace opg {

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<double>*>& data,
                                   const Matrix::PartDescriptor& inDesc,
                                   const Matrix::Data<double>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams);

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<float>*>& data,
                                   const Matrix::PartDescriptor& inDesc,
                                   const Matrix::Data<float>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams);

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<double>*>& data,
                            const Matrix::PartDescriptor& inDesc,
                            const Matrix::Data<double>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams);

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<float>*>& data,
                            const Matrix::PartDescriptor& inDesc,
                            const Matrix::Data<float>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams);

};  // namespace opg
};  // end namespace Matrix
};  // namespace CUML_EXPORT MLCommon
