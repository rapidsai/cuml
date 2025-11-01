/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/handle.hpp>

namespace ML {
namespace PCA {
namespace opg {

/**
 * @brief sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_data: input matrix that will be used to determine the sign.
 * @param[in] input_desc: MNMG description of the input
 * @param[out]  components: components matrix.
 * @param[in] n_samples: number of rows of input matrix
 * @param[in] n_features: number of columns of input/components matrix
 * @param[in] n_components: number of rows of components matrix
 * @param[in] streams: cuda streams
 * @param[in] n_stream: number of streams
 * @{
 */
void sign_flip_components_u(raft::handle_t& handle,
                            std::vector<MLCommon::Matrix::Data<float>*>& input_data,
                            MLCommon::Matrix::PartDescriptor& input_desc,
                            float* components,
                            std::size_t n_samples,
                            std::size_t n_features,
                            std::size_t n_components,
                            cudaStream_t* streams,
                            std::uint32_t n_stream,
                            bool center);

void sign_flip_components_u(raft::handle_t& handle,
                            std::vector<MLCommon::Matrix::Data<double>*>& input_data,
                            MLCommon::Matrix::PartDescriptor& input_desc,
                            double* components,
                            std::size_t n_samples,
                            std::size_t n_features,
                            std::size_t n_components,
                            cudaStream_t* streams,
                            std::uint32_t n_stream,
                            bool center);

/**
 * @brief sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_data: input matrix that will be used to determine the sign.
 * @param[in] input_desc: MNMG description of the input
 * @param[out]  components: components matrix.
 * @param[in] n_components: number of columns of components matrix
 * @param[in] streams: cuda streams
 * @param[in] n_stream: number of streams
 * @{
 */
void sign_flip(raft::handle_t& handle,
               std::vector<MLCommon::Matrix::Data<float>*>& input_data,
               MLCommon::Matrix::PartDescriptor& input_desc,
               float* components,
               std::size_t n_components,
               cudaStream_t* streams,
               std::uint32_t n_stream);

void sign_flip(raft::handle_t& handle,
               std::vector<MLCommon::Matrix::Data<double>*>& input_data,
               MLCommon::Matrix::PartDescriptor& input_desc,
               double* components,
               std::size_t n_components,
               cudaStream_t* streams,
               std::uint32_t n_stream);

};  // end namespace opg
};  // end namespace PCA
};  // end namespace ML
