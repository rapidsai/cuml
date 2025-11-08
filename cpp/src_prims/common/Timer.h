/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
