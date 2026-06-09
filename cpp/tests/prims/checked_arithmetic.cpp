/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/checked_arithmetic.hpp>

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace ML {

TEST(CheckedArithmetic, NarrowCastAcceptsRepresentableSignedMinimum)
{
  auto const value = static_cast<long long>(std::numeric_limits<int>::min());

  EXPECT_EQ(narrow_cast<int>(value), std::numeric_limits<int>::min());
}

TEST(CheckedArithmetic, NarrowCastRejectsOutOfRangeValues)
{
  auto const too_large = static_cast<long long>(std::numeric_limits<int>::max()) + 1;

  EXPECT_THROW(narrow_cast<int>(too_large), raft::exception);
  EXPECT_THROW(narrow_cast<std::size_t>(-1), raft::exception);
}

TEST(CheckedArithmetic, CheckedMulRejectsOutOfRangeOperandsAndOverflow)
{
  auto const too_large = static_cast<long long>(std::numeric_limits<int>::max()) + 1;

  EXPECT_THROW(checked_mul<int>(too_large, 1), raft::exception);
  EXPECT_THROW(checked_mul<int>(std::numeric_limits<int>::max(), 2), raft::exception);
  EXPECT_EQ(checked_mul<std::size_t>(2, 3, 4), std::size_t{24});
}

TEST(CheckedArithmetic, CheckedAddSubAndDivTrapInvalidOperations)
{
  EXPECT_THROW(checked_add<int>(std::numeric_limits<int>::max(), 1), raft::exception);
  EXPECT_THROW(checked_sub<std::size_t>(0, 1), raft::exception);
  EXPECT_THROW(checked_div<int>(1, 0), raft::exception);
  EXPECT_THROW(checked_div<int>(std::numeric_limits<int>::min(), -1), raft::exception);

  EXPECT_EQ(checked_add<int>(1, 2, 3), 6);
  EXPECT_EQ(checked_sub<int>(7, 3), 4);
  EXPECT_EQ(checked_div<int>(8, 2), 4);
}

}  // namespace ML
