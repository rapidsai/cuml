#pragma once

#include "hmm/hmm_variables.h"
#include "gmm/gmm_variables.h"
#include "hmm/dists/dists_variables.h"

namespace multinomial {

void init_multinomial_f64(multinomial::Multinomial<double> &multinomial,
                          double* dPis, int nCl);
}

namespace hmm {
void init_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                     std::vector<gmm::GMM<double> > &gmms,
                     int nStates,
                     double* dStartProb, int lddsp,
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma,
                     double* logllhd
                     );

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm, double* dLlhd);

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX, unsigned short int* dlenghts, int nSeq,
                                 bool doForward, bool doBackward, bool doGamma);

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &gmms,
                   int nStates,
                   double* dStartProb, int lddsp,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma,
                   double* logllhd
                   );

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                    int nObs, int nSeq, double* dLlhd);

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               unsigned short int* dX, unsigned short int* dlenghts, int nSeq,
                               bool doForward, bool doBackward, bool doGamma);

void viterbi_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                      unsigned short int* dVstates, unsigned short int* dX, unsigned short int* dlenghts, int nSeq);


void m_step_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                     unsigned short int* dX, unsigned short int* dlenghts, int nSeq);
}
