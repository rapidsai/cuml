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

#include "ml_utils.h"

namespace ML {

void tsvdFitSPMG(float *h_input, float *h_components, float *h_singular_vals,
                 paramsTSVD prms, int *gpu_ids, int n_gpus);
void tsvdFitSPMG(double *h_input, double *h_components, double *h_singular_vals,
                 paramsTSVD prms, int *gpu_ids, int n_gpus);
void tsvdInverseTransformSPMG(float *h_trans_input, float *h_components,
                              bool trans_comp, float *input, paramsTSVD prms,
                              int *gpu_ids, int n_gpus);
void tsvdInverseTransformSPMG(double *h_trans_input, double *h_components,
                              bool trans_comp, double *input, paramsTSVD prms,
                              int *gpu_ids, int n_gpus);
void tsvdTransformSPMG(float *h_input, float *h_components, bool trans_comp,
                       float *h_trans_input, paramsTSVD prms, int *gpu_ids,
                       int n_gpus);
void tsvdTransformSPMG(double *h_input, double *h_components, bool trans_comp,
                       double *h_trans_input, paramsTSVD prms, int *gpu_ids,
                       int n_gpus);
void tsvdFitTransformSPMG(float *h_input, float *h_trans_input,
                          float *h_components, float *h_explained_var,
                          float *h_explained_var_ratio, float *h_singular_vals,
                          paramsTSVD prms, int *gpu_ids, int n_gpus);
void tsvdFitTransformSPMG(double *h_input, double *h_trans_input,
                          double *h_components, double *h_explained_var,
                          double *h_explained_var_ratio,
                          double *h_singular_vals, paramsTSVD prms,
                          int *gpu_ids, int n_gpus);

}  // namespace ML
