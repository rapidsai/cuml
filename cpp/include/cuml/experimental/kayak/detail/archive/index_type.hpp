#pragma once
#include <limits.h>
#include <stdint.h>
#include <stddef.h>

#include <exception>
#include <cuml/experimental/kayak/detail/host_only_throw.hpp>
#include <cuml/experimental/kayak/detail/universal_cmp.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
#include <type_traits>

namespace kayak {
using raw_index_t = uint32_t;
using raw_diff_t = int32_t;

namespace detail {

auto constexpr static const MAX_INDEX = UINT_MAX;
auto constexpr static const MAX_DIFF = INT_MAX;
auto constexpr static const MIN_DIFF = INT_MIN;

struct bad_index : std::exception {
  bad_index() : bad_index("Invalid index") {}
  bad_index(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template <bool bounds_check>
struct index_type {
  using value_type = raw_index_t;
  HOST DEVICE constexpr index_type() : val{} {};
  HOST DEVICE constexpr index_type(index_type<!bounds_check> index): val{index.value()} {}
  HOST DEVICE constexpr index_type(value_type index): val{index} {}
  HOST DEVICE constexpr index_type(size_t index) noexcept(!bounds_check) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index > MAX_INDEX) {
        kayak::host_only_throw<bad_index>("Index exceeds maximum allowed value");
      }
    }
    result = universal_min(index, MAX_INDEX);
    return result;
  }()} {}
  HOST DEVICE index_type(int index) noexcept(!bounds_check) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index < 0 || index > MAX_INDEX) {
        kayak::host_only_throw<bad_index>("Invalid value for index");
      }
    }
    result = universal_min(static_cast<raw_index_t>(universal_max(0, index)), MAX_INDEX);
    return result;
  }()} {}
  HOST DEVICE constexpr operator value_type&() noexcept { return val; }
  HOST DEVICE constexpr operator value_type() const noexcept { return val; }
  HOST DEVICE constexpr operator size_t() const noexcept { return static_cast<size_t>(val); }
  HOST DEVICE constexpr auto value() const { return val; }

 private:
  value_type val;
};

template <bool bounds_check>
struct diff_type {
  using value_type = raw_diff_t;
  HOST DEVICE diff_type() : val{} {};
  HOST DEVICE diff_type(diff_type<!bounds_check> index): val{index.value()} {}
  HOST DEVICE diff_type(value_type index): val{index} {}
  HOST DEVICE diff_type(ptrdiff_t index) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index < MIN_DIFF || index > MAX_DIFF) {
        kayak::host_only_throw<bad_index>("Invalid value for diff");
      }
    }
    result = universal_min(universal_max(index, MIN_DIFF), MAX_DIFF);
    return result;
  }()} {}
  HOST DEVICE operator value_type&() noexcept { return val; }
  HOST DEVICE operator value_type() const noexcept { return val; }
  HOST DEVICE operator ptrdiff_t() const noexcept { return static_cast<ptrdiff_t>(val); }
  HOST DEVICE auto value() const { return val; }

 private:
  value_type val;
};

template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator==(T const& lhs, U const& rhs) {
  return static_cast<index_type<false>>(lhs).value() == static_cast<index_type<false>>(rhs).value();
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator<(T const& lhs, U const& rhs) {
  return static_cast<index_type<false>>(lhs).value() < static_cast<index_type<false>>(rhs).value();
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator!=(T const& lhs, U const& rhs) {
  return !(lhs == rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator>=(T const& lhs, U const& rhs) {
  return !(lhs < rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator>(T const& lhs, U const& rhs) {
  return (lhs >= rhs) && !(lhs == rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, index_type<false>> ||
    std::is_same_v<T, index_type<true>> ||
    std::is_same_v<U, index_type<false>> ||
    std::is_same_v<U, index_type<true>>
  ) && (
    std::is_convertible_v<T, index_type<false>> &&
    std::is_convertible_v<U, index_type<false>>
  ), bool
> operator<=(T const& lhs, U const& rhs) {
  return (lhs < rhs) && (lhs == rhs);
}

template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator==(T const& lhs, U const& rhs) {
  return static_cast<diff_type<false>>(lhs).value() == static_cast<diff_type<false>>(rhs).value();
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator<(T const& lhs, U const& rhs) {
  return static_cast<diff_type<false>>(lhs).value() < static_cast<diff_type<false>>(rhs).value();
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator!=(T const& lhs, U const& rhs) {
  return !(lhs == rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator>=(T const& lhs, U const& rhs) {
  return !(lhs < rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator>(T const& lhs, U const& rhs) {
  return (lhs >= rhs) && !(lhs == rhs);
}
template <typename T, typename U>
HOST DEVICE std::enable_if_t<
  (
    std::is_same_v<T, diff_type<false>> ||
    std::is_same_v<T, diff_type<true>> ||
    std::is_same_v<U, diff_type<false>> ||
    std::is_same_v<U, diff_type<true>>
  ) && (
    std::is_convertible_v<T, diff_type<false>> &&
    std::is_convertible_v<U, diff_type<false>>
  ), bool
> operator<=(T const& lhs, U const& rhs) {
  return (lhs < rhs) && (lhs == rhs);
}
}
}
