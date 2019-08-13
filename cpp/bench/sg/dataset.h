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

#include <cuda_runtime.h>
#include <common/device_buffer.hpp>
#include <cuML.hpp>
#include <string>

namespace ML {
namespace Bench {

/** A simple object to hold the loaded dataset for benchmarking */
struct Dataset {
  int nrows, ncols;
  // This is useless in the case of regression
  int nclasses;
  float* X;
  int* y;

  void allocate(const cumlHandle& handle);
  void deallocate(const cumlHandle& handle);
};

std::string allGeneratorNames();
int findGeneratorStart(int argc, char** argv);
bool loadDataset(Dataset& ret, const cumlHandle& handle, int argc, char** argv);

}  // end namespace Bench
}  // end namespace ML
