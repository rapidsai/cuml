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
#include "gmm/gmm_variables.h"

namespace gmm {

void init_f32(GMM<float> &gmm,
              float *dmu,  float *dsigma,
              float *dPis,  float *dPis_inv,  float *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full,
              int lddPis, int lddLlhd,
              float *cur_llhd, float reg_covar,
              int nCl, int nDim, int nObs);

void compute_lbow_f32(GMM<float> &gmm);

void update_llhd_f32(float* dX, GMM<float>& gmm);

void update_rhos_f32(GMM<float>& gmm, float* dX);

void update_mus_f32(float* dX, GMM<float>& gmm);

void update_sigmas_f32(float* dX, GMM<float>& gmm);

void update_pis_f32(GMM<float>& gmm);

void setup_f32(GMM<float> &gmm);

void free_f32(GMM<float> &gmm);


void init_f64(GMM<double> &gmm,
              double *dmu,  double *dsigma,
              double *dPis,  double *dPis_inv,  double *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full,
              int lddPis, int lddLlhd,
              double *cur_llhd, double reg_covar,
              int nCl, int nDim, int nObs);

void compute_lbow_f64(GMM<double> &gmm);

void update_llhd_f64(double* dX, GMM<double>& gmm);

void update_rhos_f64(GMM<double>& gmm, double* dX);

void update_mus_f64(double* dX, GMM<double>& gmm);

void update_sigmas_f64(double* dX, GMM<double>& gmm);

void update_pis_f64(GMM<double>& gmm);

void setup_f64(GMM<double> &gmm);

void free_f64(GMM<double> &gmm);


}
