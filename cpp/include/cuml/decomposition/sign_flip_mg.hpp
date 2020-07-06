/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <common/cumlHandle.hpp>
#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>

namespace ML {
namespace PCA {
namespace opg {

/**
 * @brief sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen vectors
 * @input param handle: the internal cuml handle object
 * @input/output param input param input: input matrix that will be used to determine the sign.
 * @input param input_desc: MNMG description of the input
 * @input/output param  components: components matrix.
 * @input param n_components: number of columns of components matrix
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 * @{
 */
void sign_flip(cumlHandle &handle,
               std::vector<MLCommon::Matrix::Data<float> *> &input_data,
               MLCommon::Matrix::PartDescriptor &input_desc, float *components,
               int n_components, cudaStream_t *streams, int n_stream);

void sign_flip(cumlHandle &handle,
               std::vector<MLCommon::Matrix::Data<double> *> &input_data,
               MLCommon::Matrix::PartDescriptor &input_desc, double *components,
               int n_components, cudaStream_t *streams, int n_stream);

};  // end namespace opg
};  // end namespace PCA
};  // end namespace ML
