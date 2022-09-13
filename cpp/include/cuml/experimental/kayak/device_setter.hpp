#pragma once
#include <cuml/experimental/kayak/detail/device_setter/base.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/kayak/detail/device_setter/gpu.hpp>
#endif
#include <cuml/experimental/kayak/device_type.hpp>

namespace kayak {

using device_setter = detail::device_setter<device_type::gpu>;

}
