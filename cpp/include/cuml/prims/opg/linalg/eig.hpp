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
 * \brief    Multi-GPU version of Eigen decomposition. This function works for
 *           symmetric matrices only. Whole input matrix is gathered at rank 0
 *           and Eigen decomposition is carried out sequentially.
 *
 * \param       h               cuML handle object.
 * \param[out]  eigenValues     Output N Eigen values.
 * \param[out]  eigenVectors    Output N Eigen vectors of size N x 1.
 * \param[in]   inParts         Input symmetric matrix of size N x N.
 * \param[in]   desc            Descriptor of input matrix.
 * \param[in]   myRank          MPI rank of the process.
 * \param[in]   stream          CUDA stream.
 */

void eigDC(const raft::handle_t& h,
           float* eigenValues,
           float* eigenVectors,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void eigDC(const raft::handle_t& h,
           double* eigenValues,
           double* eigenVectors,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void eigJacobi(const raft::handle_t& h,
               float* eigenValues,
               float* eigenVectors,
               std::vector<Matrix::Data<float>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void eigJacobi(const raft::handle_t& h,
               double* eigenValues,
               double* eigenVectors,
               std::vector<Matrix::Data<double>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
