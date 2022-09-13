#pragma once
#include <kayak/detail/host_only_throw/base.hpp>
#include <kayak/detail/host_only_throw/cpu.hpp>
#include <kayak/gpu_support.hpp>

namespace kayak {
template<typename T, bool host=!GPU_COMPILATION>
using host_only_throw = detail::host_only_throw<T, host>;
}
