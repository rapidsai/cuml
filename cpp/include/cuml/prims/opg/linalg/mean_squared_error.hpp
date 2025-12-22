/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <raft/core/comms.hpp>
#include <raft/core/device_mdspan.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief multi-gpu mean squared error
 * @param[out] out the output data (device pointer)
 * @param[in] in1 the first input data
 * @param[in] in1Desc descriptor for the first input data
 * @param[in] in2 the second input data
 * @param[in] in2Desc descriptor for the second input data
 * @param[in] comm the communicator object
 * @param[in] stream cuda stream where to launch work
 * @param[in] root worker ID which is supposed to be considered as root
 * @param[in] broadcastResult if false, only root process will have the result,
 *            else all ranks
 */
void meanSquaredError(double* out,
                      const Matrix::Data<double>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<double>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root             = 0,
                      bool broadcastResult = true);
void meanSquaredError(float* out,
                      const Matrix::Data<float>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<float>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root             = 0,
                      bool broadcastResult = true);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
