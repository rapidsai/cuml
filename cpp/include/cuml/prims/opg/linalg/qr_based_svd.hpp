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
 * \brief       Multi-GPU SVD decomposition for tall (rows >= columns) matrices
 * \param       h             cuML handle object.
 * \param[out]  sVector       A vector of size 1 x N with eigen values of
                              in matrix.
 * \param[out]  uParts        Parts of output U matrix from SVD decomposition
                              with size M x N. It is distributed among ranks.
                              Descriptor desc describes the matrix.
 * \param[out]  vMatrix       Full output V matrix from SVD decomposition with
                              size N x N. It is duplicated on all ranks.
 * \param[in]   genUMatrix    Currently ignored.
                              U matrix is generated only if this is true.
 * \param[in]   genVMatrix    Currently ignored.
                              V matrix is generated only if this is true.
 * \param[in]   tolerance     Error tolerance used for single GPU SVD.
                              Algorithm stops when the error is below
                              tolerance
 * \param[in]   maxSweeps     Number of sweeps in the single GPU SVD using
                              Jacobi algorithm. More sweeps provide better
                              accuracy.
 * \param[in]   inParts       Parts of the tall input matrix, distributed among
                              ranks. The size of in matrix is M x N,
                              where M >> N.
 * \param[in]   desc          Descriptor of in matrix (inParts) and U matrix
                              (uParts).
 * \param       myRank        MPI rank of the process
 */

void svdQR(const raft::handle_t& h,
           float* sVector,
           std::vector<Matrix::Data<float>*>& uMatrixParts,
           float* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           float tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank);

void svdQR(const raft::handle_t& h,
           double* sVector,
           std::vector<Matrix::Data<double>*>& uMatrixParts,
           double* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           double tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon

