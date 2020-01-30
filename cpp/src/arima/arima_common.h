/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithm>

namespace ML {

/**
 * Structure to hold the ARIMA order (makes it easier to pass as an argument)
 */
struct ARIMAOrder {
  int p;  // Basic order
  int d;
  int q;
  int P;  // Seasonal order
  int D;
  int Q;
  int s;  // Seasonal period
  int k;  // Fit intercept?

  inline int r() const { return std::max(p + s * P, q + s * Q + 1); }
  inline int complexity() const { return p + P + q + Q + k + 1; }
  inline int lost_in_diff() const { return d + s * D; }

  inline bool need_prep() const { return static_cast<bool>(d + D + k); }
};

/**
 * Structure to hold the parameters (makes it easier to pass as an argument)
 * @note: a const structure doesn't mean that the arrays can't be modified,
 *        only the pointers can't!
 */
template <typename T>
struct ARIMAParams {
  T* mu = nullptr;
  T* ar = nullptr;
  T* ma = nullptr;
  T* sar = nullptr;
  T* sma = nullptr;
  T* sigma2 = nullptr;
};

typedef ARIMAParams<double> ARIMAParamsD;

}  // namespace ML
