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

#pragma once

#include <cuml/cuml.hpp>

#include <cusparse_v2.h>

#include <cuml/neighbors/knn.hpp>

namespace ML {
namespace Sparse {

void brute_force_knn(cumlHandle &handle, const int *idxIndptr,
                     const int *idxIndices, const float *idxData, int idxNNZ,
                     int n_idx_rows, size_t n_idx_cols, const int *queryIndptr,
                     const int *queryIndices, const float *queryData,
                     int queryNNZ, int n_query_rows, int n_query_cols,
                     int *output_indices, float *output_dists, int k,
                     size_t batch_size = 2 << 20,  // approx 1M
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false);
};  // end namespace Sparse
};  // end namespace ML
