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
#include <cuML.hpp>
#include "dataset.h"

namespace ML {
namespace Bench {

std::string allAlgoNames();
int findAlgoStart(int argc, char** argv);
bool runAlgo(const Dataset& ret, const cumlHandle& handle, int argc,
             char** argv);

}  // end namespace Bench
}  // end namespace ML
