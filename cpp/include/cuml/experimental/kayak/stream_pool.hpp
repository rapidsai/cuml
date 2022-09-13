#pragma once
#include <cuml/experimental/kayak/detail/stream_pool/base.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/kayak/detail/stream_pool/gpu.hpp>
#endif
#include <cuml/experimental/kayak/device_type.hpp>

namespace kayak {
template<device_type D>
using stream_pool = detail::stream_pool<D>;
}
