#pragma once
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {
template <typename T, typename U>
HOST DEVICE auto constexpr ceildiv(T dividend, U divisor) {
  return (dividend + divisor - T{1}) / divisor;
}
}
