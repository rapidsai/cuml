#pragma once

#include <cuml/experimental/kayak/detail/device_id/base.hpp>
#include <cuml/experimental/kayak/detail/device_id/cpu.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/kayak/detail/device_id/gpu.hpp>
#endif
#include <cuml/experimental/kayak/device_type.hpp>
#include <variant>

namespace kayak {
template <device_type D>
using device_id = detail::device_id<D>;

using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}
