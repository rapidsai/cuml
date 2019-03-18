#pragma once

#include "hmm/hmm_variables.h"

namespace hmm {
void init_f64(HMM<double> &hmm,
              std::vector<gmm::GMM<double> > &gmms,
              int nStates,
              double* dT, int lddt,
              double* dB, int lddb,
              double* dGamma, int lddgamma
              );

void setup_f64(HMM<double> &hmm);

void forward_backward_f64(HMM<double> &hmm,
                          double* dX, int* dlenghts, int nSeq,
                          bool doForward, bool doBackward);
}

void viterbi_f64(HMM<double>& hmm,
                 int* dStates, int* dlenghts, int nSeq);
