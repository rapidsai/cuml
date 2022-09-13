#pragma once
#include <cuml/experimental/kayak/gpu_support.hpp>

namespace kayak {

/* Return the value that must be added to val to equal the next multiple of
 * alignment greater than or equal to val */
template <typename T, typename U>
HOST DEVICE auto padding_size(T val, U alignment) {
  auto result = val;
  if (alignment != 0) {
    auto remainder = val % alignment;
    result = alignment - remainder;
    result *= (remainder != 0);
  }
  return result;
}

/* Return the next multiple of alignment >= val */
template <typename T, typename U>
HOST DEVICE auto padded_size(T val, U alignment) {
  return val + padding_size(val, alignment);
}

/* Return the value that must be added to val to equal the next multiple of
 * alignment less than or equal to val */
template <typename T, typename U>
HOST DEVICE auto downpadding_size(T val, U alignment) {
  auto result = val;
  if (alignment != 0) {
    result = val % alignment;
  }
  return result;
}

/* Return the next multiple of alignment <= val */
template <typename T, typename U>
HOST DEVICE auto downpadded_size(T val, U alignment) {
  return val - downpadding_size(val, alignment);
}

}
