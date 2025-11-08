/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <execinfo.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#define CUML_KERNEL __global__ static
#else
#define CUML_KERNEL static
#endif
