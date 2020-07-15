/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cuml/random_projection/rproj_c.h>
#include "rproj.cuh"

namespace ML {

using namespace MLCommon;

template void RPROJfit(const cumlHandle& handle, rand_mat<float>* random_matrix,
                       paramsRPROJ* params);
template void RPROJfit(const cumlHandle& handle,
                       rand_mat<double>* random_matrix, paramsRPROJ* params);
template void RPROJtransform(const cumlHandle& handle, float* input,
                             rand_mat<float>* random_matrix, float* output,
                             paramsRPROJ* params);
template void RPROJtransform(const cumlHandle& handle, double* input,
                             rand_mat<double>* random_matrix, double* output,
                             paramsRPROJ* params);

};  // namespace ML
