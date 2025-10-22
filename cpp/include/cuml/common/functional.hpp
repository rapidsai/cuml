/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/thread/thread_operators.cuh>
#include <cuda/functional>

namespace ML::detail {
#if CCCL_MAJOR_VERSION >= 3
using maximum = cuda::maximum<void>;
using minimum = cuda::minimum<void>;
#else
using maximum = cub::Max;
using minimum = cub::Min;
#endif
}  // namespace ML::detail
