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

#include <cuda_utils.h>
#include <stdint.h>

namespace MLCommon {

// Hacker's Delight, Second Edition, Chapter 10, Integer Division By Constants
struct FastIntDiv {
  FastIntDiv(int _d) : d(_d) {
    computeScalars();
  }
  FastIntDiv& operator=(int _d) {
    d = _d;
    computeScalars();
    return *this;
  }

  HDI FastIntDiv(const FastIntDiv& other) : d(other.d), m(other.m), p(other.p) {
  }
  HDI FastIntDiv& operator=(const FastIntDiv& other) {
    d = other.d;
    m = other.m;
    p = other.p;
    return *this;
  }

  int d;
  unsigned m;
  int p;

 private:
  void computeScalars() {
    if (d == 1) {
      m = 0;
      p = 1;
      return;
    } else if (d < 0) {
      ASSERT(false, "FastIntDiv: division by negative numbers not supported!");
    } else if (d == 0) {
      ASSERT(false, "FastIntDiv: got division by zero!");
    }
    int64_t nc = ((1LL << 31) / d) * d - 1;
    p = 31;
    int64_t twoP, rhs;
    do {
      ++p;
      twoP = 1LL << p;
      rhs = nc * (d - twoP % d);
    } while (twoP <= rhs);
    m = (twoP + d - twoP % d) / d;
  }
};  // struct FastIntDiv

HDI int operator/(int n, const FastIntDiv& divisor) {
  if (divisor.d == 1) return n;
  return (int64_t(divisor.m) * int64_t(n)) >> divisor.p;
}

HDI int operator%(int n, const FastIntDiv& divisor) {
  int quotient = n / divisor;
  int remainder = n - quotient * divisor.d;
  return remainder;
}

};  // namespace MLCommon
