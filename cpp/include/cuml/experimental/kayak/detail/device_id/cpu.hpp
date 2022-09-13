#pragma once
#include <cuml/experimental/kayak/detail/device_id/base.hpp>
#include <cuml/experimental/kayak/device_type.hpp>

namespace kayak {
namespace detail {
template <>
struct device_id<device_type::cpu> {
  using value_type = int;
  device_id() : id_{value_type{}} {};
  device_id(value_type dev_id) : id_{dev_id} {};

  auto value() const noexcept { return id_; }
 private:
  value_type id_;
};
}
}
