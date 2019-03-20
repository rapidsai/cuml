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
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma
                     );

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm);

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX, int* dlenghts, int nSeq,
                                 bool doForward, bool doBackward);

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &gmms,
                   int nStates,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma
                   );

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm);

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               int* dX, int* dlenghts, int nSeq,
                               bool doForward, bool doBackward);



}
