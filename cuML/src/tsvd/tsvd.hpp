/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "ml_utils.h"

namespace ML{

void tsvdFit(cumlHandle& handle, float *input, float *components, float *singular_vals, paramsTSVD prms);
void tsvdFit(cumlHandle& handle, double *input, double *components, double *singular_vals, paramsTSVD prms);
void tsvdInverseTransform(cumlHandle& handle, float *trans_input, float *components,float *input, paramsTSVD prms);
void tsvdInverseTransform(cumlHandle& handle, double *trans_input, double *components,double *input, paramsTSVD prms);
void tsvdTransform(cumlHandle& handle, float *input, float *components, float *trans_input, paramsTSVD prms);
void tsvdTransform(cumlHandle& handle, double *input, double *components, double *trans_input, paramsTSVD prms);
void tsvdFitTransform(cumlHandle& handle, float *input, float *trans_input, float *components, float *explained_var,
                    float *explained_var_ratio, float *singular_vals, paramsTSVD prms);
void tsvdFitTransform(cumlHandle& handle, double *input, double *trans_input, double *components, double *explained_var,
                    double *explained_var_ratio, double *singular_vals, paramsTSVD prms);


}

