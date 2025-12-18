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
 * @brief A multi gpu generalized matrix multiplication function.
 * This function performs
 * Z = X * Y
 * The X and Y matrix are distributed in blocks on different ranks.
 * First Y matrix is duplicated at each rank. It is multiplied with blocks of X
 * local to the rank.
 * \param      h          cuML handle object.
 * \param[out] outZParts  Result of the multiplication with size M x N.
 *                        Distributed across ranks according to inXDesc/outZDesc
 *                        descriptor.
 * \param[out] outZDesc   Descriptor for outZParts matrix. It has to be
 *                        same as inXDesc.
 * \param[in]  inX        Input matrix X with dimensions M x K. Distributed
 *                        across ranks according to inXDesc descriptor.
 * \param[in]  inXDesc    Descriptor for X matrix.
 * \param[in]  inY        Input matrix Y with dimensions K x N. Distributed
 *                        across ranks according to inYDesc descriptor.
 * \param[in]  inYDesc    Descriptor for Y matrix.
 * \param[in]  myRank     Rank of calling process.
 * \param[in]  stream     cuda stream on which work is launched.
 */

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<float>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<float>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<float>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream);

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<double>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<double>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<double>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon

