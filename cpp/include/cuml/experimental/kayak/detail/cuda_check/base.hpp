#pragma once
#include <cuml/experimental/kayak/device_type.hpp>

namespace kayak {
namespace detail {

template <device_type D, typename error_t>
void cuda_check(error_t const& err) {
}

}
}
