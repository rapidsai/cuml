#pragma once
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/device_id.hpp>

namespace kayak {
namespace detail {

/** Struct for setting current device within a code block */
template <device_type D>
struct device_setter {
  device_setter(device_id<D> device) {}
};

}
}
