/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <math.h>
#include <utils.h>
#include <vector>
#include "../memory.h"
#include "cuda_utils.h"

struct SquareFunctor {
  template <typename T>
  static DI T exec(T x);
};

struct AbsFunctor {
  template <typename T>
  static DI T exec(T x);
};

struct GiniFunctor {
  static float exec(std::vector<int>& hist, int nrows);
  static float max_val(int nclass);
};

struct EntropyFunctor {
  static float exec(std::vector<int>& hist, int nrows);
  static float max_val(int nclass);
};
