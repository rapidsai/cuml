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

namespace MLCommon {
namespace GramMatrix {

enum KernelType { LINEAR, POLYNOMIAL, RBF, TANH };

class KernelParams {
 public:
  // Kernel function parameters
  KernelType kernel;  //!< Type of the kernel function
  int degree;         //!< Degree of polynomial kernel (ignored by others)
  double gamma;
  double coef0;
  KernelParams(KernelType kernel=RBF, int degree=3, double gamma=1,
    double coef0=0)
    : kernel(kernel), degree(degree), gamma(gamma), coef0(coef0) {}
};

};  //end namespace GramMatrix
};  //end namespace MLCommon
