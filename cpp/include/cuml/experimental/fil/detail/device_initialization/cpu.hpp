#pragma once
#include <type_traits>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace device_initialization {

/* Specialization for any initialization required for CPUs
 *
 * This specialization will also be used for non-GPU-enabled builds
 * (as a GPU no-op).
 */
template<typename forest_t, kayak::device_type D>
std::enable_if_t<!kayak::GPU_ENABLED || D == kayak::device_type::cpu, void> initialize_device(kayak::device_id<D> device) {}

}
}
}
}
}
