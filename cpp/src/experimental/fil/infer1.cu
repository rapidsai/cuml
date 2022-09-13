#include <cuml/experimental/fil/detail/device_initialization/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/experimental/fil/detail/infer/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/infer_macros.hpp>
namespace herring {
namespace detail {
namespace inference {
HERRING_INFER_ALL(template, kayak::device_type::gpu, 1)
}
namespace device_initialization {
HERRING_INITIALIZE_DEVICE(template, 1)
}
}
}
