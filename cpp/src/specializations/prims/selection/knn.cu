#include <specializations/prims/selection/knn.cuh>

namespace MLCommon {
namespace Selection {
    template class MetricProcessor<float>;
    template class CosineMetricProcessor<float>;
    template class CorrelationMetricProcessor<float>;
    template class DefaultMetricProcessor<float>;
}
}
