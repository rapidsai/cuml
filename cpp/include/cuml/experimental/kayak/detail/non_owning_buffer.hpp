#pragma once
#include <kayak/device_type.hpp>
#include <kayak/detail/non_owning_buffer/base.hpp>

namespace kayak {
template<device_type D, typename T>
using non_owning_buffer = detail::non_owning_buffer<D, T>;
}
