#include <cuml/experimental/fil/detail/device_initialization/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/experimental/fil/detail/infer/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/infer_macros.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, kayak::device_type::gpu, 4)
}
namespace device_initialization {
CUML_FIL_INITIALIZE_DEVICE(template, 4)
}
}
}
}
}
