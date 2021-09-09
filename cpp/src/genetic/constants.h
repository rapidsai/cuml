/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/** @file constants.h Common GPU functionality + constants for all operations */

#pragma once

namespace cuml {
namespace genetic {

// Max number of threads per block to use with tournament and evaluation kernels
const int GENE_TPB = 256;

// Max size of stack used for AST evaluation
const int MAX_STACK_SIZE = 20;

}  // namespace genetic
}  // namespace cuml