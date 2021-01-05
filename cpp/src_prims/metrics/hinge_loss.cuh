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

#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <distance/distance.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <cuml/metrics/penalty_type.hpp>
#include "functions/hinge.cuh"


namespace MLCommon {
namespace Metrics {

template <typename math_t>
double hinge_loss(const raft::handle_t &handle, math_t *input, int n_rows,
               int n_cols, const math_t *labels, const math_t *coef,
               MLCommon::Functions::penalty pen,
               math_t alpha, math_t l1_ratio)  {

  double loss;
  //Call the distance function
  Functions::hingeLoss(
  handle,
    input, n_rows, n_cols, labels, coef, &loss, pen, alpha, l1_ratio,
    handle.get_stream());
  return loss;
}

};  // namespace Metrics
};  // namespace MLCommon
