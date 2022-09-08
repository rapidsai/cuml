/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
