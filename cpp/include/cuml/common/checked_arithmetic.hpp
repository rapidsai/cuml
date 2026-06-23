/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace ML {

/**
 * @brief Integer type expected by CUDA launch configuration (`dim3` components,
 * shared-mem size, etc.).
 *
 * Prefer `ML::narrow_cast<ML::cuda_launch_t>(...)` over a bare
 * `narrow_cast<unsigned int>(...)` when the value is destined for a `<<<>>>`
 * grid/block dimension. This keeps call sites self-documenting and lets us
 * adapt to future CUDA API changes in one place.
 */
using cuda_launch_t = unsigned int;

/**
 * @file checked_arithmetic.hpp
 *
 * Host-side integer arithmetic helpers that trap on overflow, underflow, or
 * divide-by-zero. Use these whenever an integer product, sum, difference, or
 * quotient flows into:
 *   - an allocation size (`rmm::device_uvector` ctor / `.resize(...)`,
 *     `cudaMalloc*`, `*allocator*.allocate(...)`, `rmm::device_buffer`),
 *   - a CUDA launch dimension (`dim3(...)` arguments, ceil-div block counts),
 *   - a span constructor (`raft::span`, `cuda::std::span`),
 *   - or a host pointer offset / index that could exceed the operand width.
 *
 * Each helper widens to the target type @c T (typically `std::size_t` or
 * `std::int64_t`), performs the operation with overflow detection, and calls
 * `RAFT_FAIL` with a diagnostic message on failure. All checks are host-side
 * and cost only a handful of instructions per call site.
 *
 * **Do not use these in __device__ / __global__ code.** Kernel-side arithmetic
 * is intentionally out of scope; checks belong at the host-side computation
 * site so device code stays branch-free.
 *
 * Example:
 * @code
 *   // Before (silent overflow on large batch workloads):
 *   rmm::device_uvector<T> buf(n * batch_size, stream);
 *
 *   // After:
 *   rmm::device_uvector<T> buf(ML::checked_mul<std::size_t>(n, batch_size),
 *                              stream);
 * @endcode
 */

/**
 * @brief A type usable as the target of a `checked_*` helper.
 *
 * Restricted to integral types other than `bool` so the resulting helpers
 * make sense as size/index arithmetic.
 */
template <typename T>
concept checked_target = std::integral<T> && !std::same_as<T, bool>;

/**
 * @brief A type usable as a source operand of a `checked_*` helper.
 */
template <typename T>
concept checked_source = std::integral<T> && !std::same_as<T, bool>;

namespace detail {

/**
 * @brief Widen @p value to @c T, trapping if the source value cannot be
 * represented in @c T.
 */
template <checked_target T, checked_source U>
constexpr T widen_or_fail(U value, char const* op)
{
  if (!std::in_range<T>(value)) {
    if constexpr (std::unsigned_integral<U>) {
      RAFT_FAIL("checked_arithmetic: operand %llu does not fit target type in %s",
                static_cast<unsigned long long>(value),
                op);
    } else {
      RAFT_FAIL("checked_arithmetic: operand %lld does not fit target type in %s",
                static_cast<long long>(value),
                op);
    }
  }
  return static_cast<T>(value);
}

template <checked_target T>
constexpr T checked_mul_pair(T a, T b)
{
  T result{};
  if (__builtin_mul_overflow(a, b, &result)) {
    RAFT_FAIL("checked_mul overflow: operands do not fit target type");
  }
  return result;
}

template <checked_target T>
constexpr T checked_add_pair(T a, T b)
{
  T result{};
  if (__builtin_add_overflow(a, b, &result)) {
    RAFT_FAIL("checked_add overflow: operands do not fit target type");
  }
  return result;
}

}  // namespace detail

/**
 * @brief Convert @p value to target type @c T, trapping if the value does not
 * fit (negative source with unsigned target, or magnitude exceeding @c T's
 * range).
 *
 * Use at sites where an existing API forces a narrowing — e.g. passing a
 * `std::size_t` size to a function that takes `int`, or storing a `pair::first`
 * into an `int` variable. The cast itself is preserved; this helper only
 * ensures it doesn't silently corrupt the value.
 *
 * When @c T is strictly wider than @c U the magnitude check is skipped (no
 * bit-level loss is possible), so a misplaced `narrow_cast<std::size_t>(int)`
 * on a non-negative value is a free `static_cast`. Sign-loss is still flagged:
 * a negative source with an unsigned target traps regardless of widths, since
 * sign loss is a real correctness bug even on a "widening" conversion.
 *
 * Example:
 * @code
 *   // Was: int n = m_shape.first;             // silent narrow
 *   int n = ML::narrow_cast<int>(m_shape.first);  // traps if first > INT_MAX
 * @endcode
 *
 * @tparam T  target integral type
 * @tparam U  source integral type (deduced)
 * @return    @p value as @c T
 *
 * @throws raft::exception if @p value cannot be represented in @c T
 */
template <checked_target T, checked_source U>
constexpr T narrow_cast(U value)
{
  return detail::widen_or_fail<T>(value, "narrow_cast");
}

/**
 * @brief Multiply two or more integers in target type @c T, trapping on
 * overflow or on operands that cannot be represented in @c T.
 *
 * @tparam T   target integral type (typically `std::size_t` or `std::int64_t`)
 * @tparam U1  type of the first factor (deduced)
 * @tparam U2  type of the second factor (deduced)
 * @tparam Us  types of additional factors (deduced)
 * @return the product as @c T
 *
 * @throws raft::exception on overflow or unrepresentable operand
 */
template <checked_target T, checked_source U1, checked_source U2, checked_source... Us>
constexpr T checked_mul(U1 a, U2 b, Us... rest)
{
  T acc = detail::checked_mul_pair<T>(detail::widen_or_fail<T>(a, "checked_mul"),
                                      detail::widen_or_fail<T>(b, "checked_mul"));
  ((acc = detail::checked_mul_pair<T>(acc, detail::widen_or_fail<T>(rest, "checked_mul"))), ...);
  return acc;
}

/**
 * @brief Add two or more integers in target type @c T, trapping on overflow
 * or on operands that cannot be represented in @c T.
 */
template <checked_target T, checked_source U1, checked_source U2, checked_source... Us>
constexpr T checked_add(U1 a, U2 b, Us... rest)
{
  T acc = detail::checked_add_pair<T>(detail::widen_or_fail<T>(a, "checked_add"),
                                      detail::widen_or_fail<T>(b, "checked_add"));
  ((acc = detail::checked_add_pair<T>(acc, detail::widen_or_fail<T>(rest, "checked_add"))), ...);
  return acc;
}

/**
 * @brief Compute @p a - @p b in target type @c T, trapping on underflow or on
 * operands that cannot be represented in @c T.
 *
 * For unsigned @c T this enforces @p a >= @p b.
 */
template <checked_target T, checked_source U1, checked_source U2>
constexpr T checked_sub(U1 a, U2 b)
{
  T const wa = detail::widen_or_fail<T>(a, "checked_sub");
  T const wb = detail::widen_or_fail<T>(b, "checked_sub");
  T result{};
  if (__builtin_sub_overflow(wa, wb, &result)) {
    if constexpr (std::unsigned_integral<T>) {
      RAFT_FAIL("checked_sub underflow: %llu - %llu would be negative",
                static_cast<unsigned long long>(wa),
                static_cast<unsigned long long>(wb));
    } else {
      RAFT_FAIL("checked_sub overflow: %lld - %lld does not fit target type",
                static_cast<long long>(wa),
                static_cast<long long>(wb));
    }
  }
  return result;
}

/**
 * @brief Compute @p a / @p b in target type @c T, trapping on divide-by-zero
 * (and on signed `INT_MIN / -1` overflow for signed @c T) or on operands that
 * cannot be represented in @c T.
 */
template <checked_target T, checked_source U1, checked_source U2>
constexpr T checked_div(U1 a, U2 b)
{
  T const wa = detail::widen_or_fail<T>(a, "checked_div");
  T const wb = detail::widen_or_fail<T>(b, "checked_div");
  if (wb == 0) { RAFT_FAIL("checked_div: divide by zero"); }
  if constexpr (std::signed_integral<T>) {
    if (wa == std::numeric_limits<T>::min() && wb == static_cast<T>(-1)) {
      RAFT_FAIL("checked_div overflow: INT_MIN / -1 does not fit target type");
    }
  }
  return static_cast<T>(wa / wb);
}

}  // namespace ML
