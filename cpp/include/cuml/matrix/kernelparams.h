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

namespace MLCommon {
namespace Matrix {

enum KernelType { LINEAR, POLYNOMIAL, RBF, TANH };

/**
 * Parameters for kernel matrices.
 * The following kernels are implemented:
 * - LINEAR \f[ K(x_1,x_2) = <x_1,x_2>, \f] where \f$< , >\f$ is the dot product
 * - POLYNOMIAL \f[ K(x_1, x_2) = (\gamma <x_1,x_2> + \mathrm{coef0})^\mathrm{degree} \f]
 * - RBF \f[ K(x_1, x_2) = \exp(- \gamma |x_1-x_2|^2) \f]
 * - TANH \f[ K(x_1, x_2) = \tanh(\gamma <x_1,x_2> + \mathrm{coef0}) \f]
 */
struct KernelParams {
  // Kernel function parameters
  KernelType kernel;  //!< Type of the kernel function
  int degree;         //!< Degree of polynomial kernel (ignored by others)
  double gamma;       //!< multiplier in the
  double coef0;       //!< additive constant in poly and tanh kernels
};

};  // end namespace Matrix
};  // end namespace MLCommon
