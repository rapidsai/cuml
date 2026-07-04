/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../bench/common/ml_benchmark.hpp"

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace MLCommon {
namespace Bench {

TEST(MlBenchmarkFixtureAllocator, ThrowOnHugeAllocationOrDeallocationLength)
{
  Fixture fixture("MLBenchmarkAllocatorTest");

  int* ptr      = nullptr;
  auto const len = std::numeric_limits<size_t>::max() / sizeof(int) + 1;

  EXPECT_THROW(fixture.alloc(ptr, len), raft::exception);
  EXPECT_THROW(fixture.dealloc(ptr, len), raft::exception);
}

}  // namespace Bench
}  // namespace MLCommon
