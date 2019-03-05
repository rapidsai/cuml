#pragma once

template <typename T>
void init_f32(GMM<T> &gmm,
              float *dmu,  float *dsigma,  float *dPis,  float *dPis_inv,  float *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs);

template <typename T>
void update_rhos_f32(float* dX, GMM<float>& gmm);

template <typename T>
void update_mus_f32(float* dX, GMM<float>& gmm);

template <typename T>
void update_sigmas_f32(float* dX, GMM<float>& gmm);

template <typename T>
void update_pis_f32(GMM<float>& gmm);
