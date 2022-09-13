#include <herring3/detail/device_initialization/gpu.cuh>
#include <herring3/specializations/device_initialization_macros.hpp>
#include <herring3/detail/infer/gpu.cuh>
#include <herring3/specializations/infer_macros.hpp>
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
