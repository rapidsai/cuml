#pragma once
#include <cstddef>
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {
template<typename T, std::size_t N>
struct raw_array {
  HOST DEVICE raw_array() {}
  HOST DEVICE auto& operator[](std::size_t index) { return data[index]; }
  HOST DEVICE auto const& operator[](std::size_t index) const { return data[index]; }
 private:
  T data[N];
};
}
