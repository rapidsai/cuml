#pragma once
#include <cuml/experimental/kayak/device_type.hpp>
#include <memory>
#include <type_traits>

namespace kayak {
namespace detail {
template<device_type D, typename T>
struct non_owning_buffer {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;
  non_owning_buffer() : data_{nullptr} { }

  non_owning_buffer(T* ptr) : data_{ptr} { }

  auto* get() const { return data_; }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  T* data_;
};
}
}

