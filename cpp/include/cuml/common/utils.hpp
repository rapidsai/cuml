/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
