/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace ML {
namespace GLM {
namespace opg {

void preProcessData(raft::handle_t& handle,
                    std::vector<MLCommon::Matrix::Data<float>*>& input_data,
                    MLCommon::Matrix::PartDescriptor& input_desc,
                    std::vector<MLCommon::Matrix::Data<float>*>& labels,
                    float* mu_input,
                    float* mu_labels,
                    float* norm2_input,
                    bool fit_intercept,
                    bool normalize,
                    cudaStream_t* streams,
                    int n_streams,
                    bool verbose);

void preProcessData(raft::handle_t& handle,
                    std::vector<MLCommon::Matrix::Data<double>*>& input_data,
                    MLCommon::Matrix::PartDescriptor& input_desc,
                    std::vector<MLCommon::Matrix::Data<double>*>& labels,
                    double* mu_input,
                    double* mu_labels,
                    double* norm2_input,
                    bool fit_intercept,
                    bool normalize,
                    cudaStream_t* streams,
                    int n_streams,
                    bool verbose);

void postProcessData(raft::handle_t& handle,
                     std::vector<MLCommon::Matrix::Data<float>*>& input_data,
                     MLCommon::Matrix::PartDescriptor& input_desc,
                     std::vector<MLCommon::Matrix::Data<float>*>& labels,
                     float* coef,
                     float* intercept,
                     float* mu_input,
                     float* mu_labels,
                     float* norm2_input,
                     bool fit_intercept,
                     bool normalize,
                     cudaStream_t* streams,
                     int n_streams,
                     bool verbose);

void postProcessData(raft::handle_t& handle,
                     std::vector<MLCommon::Matrix::Data<double>*>& input_data,
                     MLCommon::Matrix::PartDescriptor& input_desc,
                     std::vector<MLCommon::Matrix::Data<double>*>& labels,
                     double* coef,
                     double* intercept,
                     double* mu_input,
                     double* mu_labels,
                     double* norm2_input,
                     bool fit_intercept,
                     bool normalize,
                     cudaStream_t* streams,
                     int n_streams,
                     bool verbose);

};  // end namespace opg
};  // namespace GLM
};  // end namespace ML
