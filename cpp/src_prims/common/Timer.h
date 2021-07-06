/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <chrono>

namespace MLCommon {
class TimerCPU {
 public:
  TimerCPU() { this->reset(); }

  void reset() { this->time = std::chrono::high_resolution_clock::now(); }

  double getElapsedSeconds() const
  {
    return 1.0e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - this->time)
                      .count();
  }

  double getElapsedMilliseconds() const
  {
    return 1.0e-3 * std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - this->time)
                      .count();
  }

 private:
  std::chrono::high_resolution_clock::time_point time;
};
}  // End namespace MLCommon
