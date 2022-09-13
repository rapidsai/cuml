#pragma once
#include <kayak/gpu_support.hpp>

namespace kayak {
namespace detail {
template<typename T, bool host>
struct host_only_throw {
  template <typename... Args>
  host_only_throw(Args&&... args) {
    static_assert(host);  // Do not allow constexpr branch to compile if !host
  }
};
}
}
