#pragma once
#include <type_traits>

namespace kayak {
template <typename T, typename U, typename V = void>
using const_agnostic_same_t =
  std::enable_if_t<std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>, V>;

template <typename T, typename U>
inline constexpr auto const_agnostic_same_v =
  std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>;
}
