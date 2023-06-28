/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/comms.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> 
#include <cuda_runtime.h>

namespace ML {
namespace GLM {
namespace opg {

void qnFit(const raft::handle_t &handle, 
           const qn_params& pams,
           float* X,
           bool X_col_major,
           float *y,
           int N,
           int D,
           int C,
           float* w0,
           float* f,
           int* num_iters,
           int n_samples,
           int rank,
           int n_ranks);

};  // namespace opg
};  // namespace GLM
};  // namespace ML