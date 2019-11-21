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

#include <cuml/cuml.hpp>

namespace ML {

void SparseSVD(const cumlHandle &handle,
               const float *__restrict X,
               float *__restrict U,
               float *__restrict S,
               float *__restrict VT,
               const int n_components = 2,
               const int n_oversamples = 10,
               const int max_iter = 3);

void SparseSVD(const cumlHandle &handle,
               const double *__restrict X,
               double *__restrict U,
               double *__restrict S,
               double *__restrict VT,
               const int n_components = 2,
               const int n_oversamples = 10,
               const int max_iter = 3);

}  // namespace ML
