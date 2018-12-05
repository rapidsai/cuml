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

void tsvdFit(float *input, float *components, float *singular_vals, paramsTSVD prms);
void tsvdFit(double *input, double *components, double *singular_vals, paramsTSVD prms);
void tsvdInverseTransform(float *trans_input, float *components,float *input, paramsTSVD prms);
void tsvdInverseTransform(double *trans_input, double *components,double *input, paramsTSVD prms);
void tsvdTransform(float *input, float *components, float *trans_input, paramsTSVD prms);
void tsvdTransform(double *input, double *components, double *trans_input, paramsTSVD prms);
void tsvdFitTransform(float *input, float *trans_input, float *components, float *explained_var,
                    float *explained_var_ratio, float *singular_vals, paramsTSVD prms);
void tsvdFitTransform(double *input, double *trans_input, double *components, double *explained_var,
                    double *explained_var_ratio, double *singular_vals, paramsTSVD prms);


}

