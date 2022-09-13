#pragma once
#include <cuml/experimental/kayak/detail/host_only_throw/base.hpp>
#include <cuml/experimental/kayak/detail/host_only_throw/cpu.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {
template<typename T, bool host=!GPU_COMPILATION>
using host_only_throw = detail::host_only_throw<T, host>;
}
