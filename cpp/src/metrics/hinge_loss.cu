
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
#include <cuml/metrics/penalty_type.hpp>

namespace ML {

namespace Metrics {
double hinge_loss(const raft::handle_t &handle, double *input, int n_rows,
               int n_cols, const double *labels, const double *coef,
                  MLCommon::Functions::penalty pen, double alpha, double l1_ratio) {
  return MLCommon::Metrics::hinge_loss(
    handle, input, n_rows, n_cols, labels, coef, pen, alpha, l1_ratio);
}
}  // namespace Metrics
}  // namespace ML
