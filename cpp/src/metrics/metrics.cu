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

#include <raft/cuda_utils.cuh>
#include <cuml/metrics/metrics.hpp>
#include <metrics/adjustedRandIndex.cuh>
#include <metrics/klDivergence.cuh>
#include <metrics/pairwiseDistance.cuh>
#include <metrics/randIndex.cuh>
#include <metrics/silhouetteScore.cuh>
#include <metrics/vMeasure.cuh>
#include <score/scores.cuh>

namespace ML {

namespace Metrics {

float r2_score_py(const raft::handle_t &handle, float *y, float *y_hat, int n) {
  return MLCommon::Score::r2_score(y, y_hat, n, handle.get_stream());
}

double r2_score_py(const raft::handle_t &handle, double *y, double *y_hat,
                   int n) {
  return MLCommon::Score::r2_score(y, y_hat, n, handle.get_stream());
}

double randIndex(const raft::handle_t &handle, const double *y,
                 const double *y_hat, int n) {
  return MLCommon::Metrics::computeRandIndex(
    y, y_hat, (uint64_t)n, handle.get_device_allocator(), handle.get_stream());
}

double silhouetteScore(const raft::handle_t &handle, double *y, int nRows,
                       int nCols, int *labels, int nLabels, double *silScores,
                       int metric) {
  return MLCommon::Metrics::silhouetteScore<double, int>(
    y, nRows, nCols, labels, nLabels, silScores, handle.get_device_allocator(),
    handle.get_stream(), metric);
}

double adjustedRandIndex(const raft::handle_t &handle, const int64_t *y,
                         const int64_t *y_hat, const int64_t n) {
  return MLCommon::Metrics::computeAdjustedRandIndex<int64_t,
                                                     unsigned long long>(
    y, y_hat, n, handle.get_device_allocator(), handle.get_stream());
}

double adjustedRandIndex(const raft::handle_t &handle, const int *y,
                         const int *y_hat, const int n) {
  return MLCommon::Metrics::computeAdjustedRandIndex<int, unsigned long long>(
    y, y_hat, n, handle.get_device_allocator(), handle.get_stream());
}

double klDivergence(const raft::handle_t &handle, const double *y,
                    const double *y_hat, int n) {
  return MLCommon::Metrics::klDivergence(
    y, y_hat, n, handle.get_device_allocator(), handle.get_stream());
}

float klDivergence(const raft::handle_t &handle, const float *y,
                   const float *y_hat, int n) {
  return MLCommon::Metrics::klDivergence(
    y, y_hat, n, handle.get_device_allocator(), handle.get_stream());
}

double entropy(const raft::handle_t &handle, const int *y, const int n,
               const int lower_class_range, const int upper_class_range) {
  return MLCommon::Metrics::entropy(y, n, lower_class_range, upper_class_range,
                                    handle.get_device_allocator(),
                                    handle.get_stream());
}

double mutualInfoScore(const raft::handle_t &handle, const int *y,
                       const int *y_hat, const int n,
                       const int lower_class_range,
                       const int upper_class_range) {
  return MLCommon::Metrics::mutualInfoScore(
    y, y_hat, n, lower_class_range, upper_class_range,
    handle.get_device_allocator(), handle.get_stream());
}

double homogeneityScore(const raft::handle_t &handle, const int *y,
                        const int *y_hat, const int n,
                        const int lower_class_range,
                        const int upper_class_range) {
  return MLCommon::Metrics::homogeneityScore(
    y, y_hat, n, lower_class_range, upper_class_range,
    handle.get_device_allocator(), handle.get_stream());
}

double completenessScore(const raft::handle_t &handle, const int *y,
                         const int *y_hat, const int n,
                         const int lower_class_range,
                         const int upper_class_range) {
  return MLCommon::Metrics::homogeneityScore(
    y_hat, y, n, lower_class_range, upper_class_range,
    handle.get_device_allocator(), handle.get_stream());
}

double vMeasure(const raft::handle_t &handle, const int *y, const int *y_hat,
                const int n, const int lower_class_range,
                const int upper_class_range) {
  return MLCommon::Metrics::vMeasure(
    y, y_hat, n, lower_class_range, upper_class_range,
    handle.get_device_allocator(), handle.get_stream());
}

float accuracy_score_py(const raft::handle_t &handle, const int *predictions,
                        const int *ref_predictions, int n) {
  return MLCommon::Score::accuracy_score(predictions, ref_predictions, n,
                                         handle.get_device_allocator(),
                                         handle.get_stream());
}

void pairwiseDistance(const raft::handle_t &handle, const double *x,
                      const double *y, double *dist, int m, int n, int k,
                      ML::Distance::DistanceType metric, bool isRowMajor) {
  MLCommon::Metrics::pairwiseDistance(x, y, dist, m, n, k, metric,
                                      handle.get_device_allocator(),
                                      handle.get_stream(), isRowMajor);
}

void pairwiseDistance(const raft::handle_t &handle, const float *x,
                      const float *y, float *dist, int m, int n, int k,
                      ML::Distance::DistanceType metric, bool isRowMajor) {
  MLCommon::Metrics::pairwiseDistance(x, y, dist, m, n, k, metric,
                                      handle.get_device_allocator(),
                                      handle.get_stream(), isRowMajor);
}

}  // namespace Metrics
}  // namespace ML
