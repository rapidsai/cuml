#pragma once

#include "hmm/hmm_variables.h"

namespace hmm {
void init_f64(HMM<double> &hmm,
              std::vector<double*> dmu, std::vector<double*> dsigma, std::vector<double*> dPis, std::vector<double*> dPis_inv, double* dB, double* cur_llhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs,
              double reg_covar,
              int nStates,
              double* dT,
              int lddt);

// void forward_f32(HMM<float>& hmm,
//                  float* dX,
//                  int* len_array,
//                  int nObs);
//
// void backward_f32(HMM<float>& hmm,
//                   float* dX,
//                   int* len_array,
//                   int nObs);
//
// void setup_hmm_f32(HMM<float> &hmm);
//
// void free_hmm_f32(HMM<float> &hmm);

}
