#pragma once

#include "hmm/hmm_variables.h"

namespace hmm {
void init_f64(HMM<double> &hmm,
              std::vector<gmm::GMM<double> > &gmms,
              int nStates,
              double* dT, int lddt, double* dB, int lddb);

void forward_f64(HMM<double>& hmm,
                 double* dX,
                 int* len_array,
                 int nObs);

// void backward_f32(HMM<float>& hmm,
//                   float* dX,
//                   int* len_array,
//                   int nObs);
//
// void setup_hmm_f32(HMM<float> &hmm);
//
// void free_hmm_f32(HMM<float> &hmm);

}
