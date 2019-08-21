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

#include "harness.h"


int main(int argc, char **argv) {
  try {
    using namespace ML::Bench;
    Harness::Init(argc, argv);
    Harness::RunAll();
    return 0;
  } catch (const std::runtime_error &re) {
    printf("Benchmarking failed! Reason: %s\n", re.what());
    return 1;
  } catch (...) {
    printf("Benchmarking failed!\n");
    return 2;
  }
}
