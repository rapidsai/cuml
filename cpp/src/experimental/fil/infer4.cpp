#include <cuml/experimental/fil/detail/infer/cpu.hpp>
#include <cuml/experimental/fil/detail/specializations/infer_macros.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, kayak::device_type::cpu, 4)
}
}
}
}
}
