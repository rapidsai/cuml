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
#include <treelite/c_api.h>

namespace ML {
enum SPLIT_ALGO {
  HIST,
  GLOBAL_QUANTILE,
  SPLIT_ALGO_END,
};

enum CRITERION {
  GINI,
  ENTROPY,
  MSE,
  MAE,
  CRITERION_END,
};

/** check for treelite runtime API errors and assert accordingly */
#define TREELITE_CHECK(call)                                            \
  do {                                                                  \
    int status = call;                                                  \
    ASSERT(status >= 0, "TREELITE FAIL: call='%s'. Reason:%s\n", #call, \
           TreeliteGetLastError());                                     \
  } while (0)

};  // namespace ML
