/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

// #include "metrics.h"
#include "cuda_utils.h"
#include "metrics.hpp"

#include "metrics/adjustedRandIndex.h"
#include "metrics/randIndex.h"
#include "metrics/silhouetteScore.h"
#include "score/scores.h"

namespace ML {

namespace Metrics {

float r2_score_py(const cumlHandle &handle, float *y, float *y_hat, int n) {
  return MLCommon::Score::r2_score(y, y_hat, n, handle.getStream());
}

double r2_score_py(const cumlHandle &handle, double *y, double *y_hat, int n) {
  return MLCommon::Score::r2_score(y, y_hat, n, handle.getStream());
}

double randIndex(const cumlHandle &handle, const double *y, const double *y_hat,
                 int n) {
  return MLCommon::Metrics::computeRandIndex(
    y, y_hat, (uint64_t)n, handle.getDeviceAllocator(), handle.getStream());
}

double silhouetteScore(const cumlHandle &handle, double *y, int nRows,
                       int nCols, int *labels, int nLabels, double *silScores,
                       int metric) {
  return MLCommon::Metrics::silhouetteScore<double, int>(
    y, nRows, nCols, labels, nLabels, silScores, handle.getDeviceAllocator(),
    handle.getStream(), metric);
}

double adjustedRandIndex(const cumlHandle &handle, const int *y,
                         const int *y_hat, const int n,
                         const int lower_class_range,
                         const int upper_class_range) {
  return MLCommon::Metrics::computeAdjustedRandIndex(
    y, y_hat, n, lower_class_range, upper_class_range,
    handle.getDeviceAllocator(), handle.getStream());
}

}  // namespace Metrics
}  // namespace ML