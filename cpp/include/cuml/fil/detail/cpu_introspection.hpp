/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <new>

namespace ML {
namespace fil {
namespace detail {
#ifdef __cpplib_hardware_interference_size
using std::hardware_constructive_interference_size;
#else
auto constexpr static const hardware_constructive_interference_size = std::size_t{64};
#endif
}  // namespace detail
}  // namespace fil
}  // namespace ML
