#pragma once
#include "hmm/hmm_variables.h"

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
