#pragma once
#include <selection/knn.cuh>

namespace MLCommon {
namespace Selection {

extern template class MetricProcessor<float>;
extern template class CosineMetricProcessor<float>;
extern template class CorrelationMetricProcessor<float>;
extern template class DefaultMetricProcessor<float>;

}  // namespace Selection
}  // namespace MLCommon
