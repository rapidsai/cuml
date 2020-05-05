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
#include "cuda_utils.h"
#include "metric_def.cuh"

namespace ML {
namespace DecisionTree {

template <class T>
DI T SquareFunctor::exec(T x) {
  return MLCommon::myPow(x, (T)2);
}

template <class T>
DI T AbsFunctor::exec(T x) {
  return MLCommon::myAbs(x);
}

float GiniFunctor::max_val(int nclass) { return 1.0; }

float EntropyFunctor::max_val(int nclass) {
  float prob = 1.0 / nclass;
  return (-1.0 * nclass * prob * logf(prob));
}
float GiniFunctor::exec(std::vector<unsigned int> &hist, int nrows) {
  float gval = 1.0;
  for (int i = 0; i < hist.size(); i++) {
    float prob = ((float)hist[i]) / nrows;
    gval -= prob * prob;
  }
  return gval;
}

float EntropyFunctor::exec(std::vector<unsigned int> &hist, int nrows) {
  float eval = 0.0;
  for (int i = 0; i < hist.size(); i++) {
    if (hist[i] != 0) {
      float prob = ((float)hist[i]) / nrows;
      eval += prob * logf(prob);
    }
  }
  return (-1 * eval);
}

}  // namespace DecisionTree
}  // namespace ML
