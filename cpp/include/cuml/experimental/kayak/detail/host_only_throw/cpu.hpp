#pragma once
#include <cuml/experimental/kayak/detail/host_only_throw/base.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {
namespace detail {
template<typename T>
struct host_only_throw<T, true>{
  template <typename... Args>
  host_only_throw(Args&&... args) noexcept(false)  {
    throw T{std::forward<Args>(args)...};
  }
};
}
}
