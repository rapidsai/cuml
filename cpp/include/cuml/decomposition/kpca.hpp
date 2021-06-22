/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "params.hpp"

namespace raft {
class handle_t;
}

namespace ML {

void kpcaFit(raft::handle_t &handle, float *input, float *alphas, float *lambdas, const paramsKPCA &prms);

void kpcaFit(raft::handle_t &handle, double *input, double *alphas,
             double *lambdas, const paramsKPCA &prms);

void kpcaTransform(raft::handle_t &handle, float *input, float *alphas, float *lambdas,
                   float *trans_input, const paramsKPCA &prms);

void kpcaTransform(raft::handle_t &handle, double *input, double *alphas, double *lambdas,
                   double *trans_input, const paramsKPCA &prms);


};  // end namespace ML
