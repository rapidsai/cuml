#pragma once
#include <kayak/device_type.hpp>
#include <kayak/detail/owning_buffer/cpu.hpp>
#ifdef ENABLE_GPU
#include <kayak/detail/owning_buffer/gpu.hpp>
#endif
namespace kayak {
template<device_type D, typename T>
using owning_buffer = detail::owning_buffer<D, T>;
}
