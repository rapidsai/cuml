#include <cuml/experimental/fil/detail/infer/cpu.hpp>
#include <cuml/experimental/fil/specializations/infer_macros.hpp>
namespace herring {
namespace detail {
namespace inference {
HERRING_INFER_ALL(template, kayak::device_type::cpu, 2)
}
}
}
