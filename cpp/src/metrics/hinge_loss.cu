
/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include <cuml/metrics/metrics.hpp>
#include <metrics/hinge_loss.cuh>

namespace ML {

namespace Metrics {
void hinge_loss(const raft::handle_t &handle, math_t *input, int n_rows,
               int n_cols, const math_t *labels, const math_t *coef,
               math_t *loss, penalty pen, math_t alpha, math_t l1_ratio,
               cudaStream_t stream) {
  return MLCommon::Metrics::hinge_loss(
    input, n_rows, n_cols, labels, coef, loss, pen, alpha, l1_ratio,
    handle.get_device_allocator(),
    handle.get_stream());
}
}  // namespace Metrics
}  // namespace ML
