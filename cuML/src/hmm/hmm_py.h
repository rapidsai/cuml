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

#include "hmm/hmm_variables.h"
#include "gmm/gmm_variables.h"
#include "hmm/dists/dists_variables.h"

namespace multinomial {

void init_multinomial_f32(multinomial::Multinomial<float> &multinomial,
                          float* dPis, int nCl);


void init_multinomial_f64(multinomial::Multinomial<double> &multinomial,
                          double* dPis, int nCl);

}

namespace hmm {
void init_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                   std::vector<multinomial::Multinomial<float> > &multinomials,
                   int nStates,
                   float* dStartProb, int lddsp,
                   float* dT, int lddt,
                   float* dB, int lddb,
                   float* dGamma, int lddgamma,
                   float* logllhd,
                   int nObs, int nSeq,
                   float* dLlhd
                   );

size_t get_workspace_size_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm);

void create_handle_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                            void* workspace);

void setup_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                    int nObs, int nSeq, float* dLlhd);

void forward_backward_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                               unsigned short int* dX, unsigned short int* dlenghts, int nSeq,
                               bool doForward, bool doBackward, bool doGamma);

void viterbi_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                      unsigned short int* dVstates, unsigned short int* dX, unsigned short int* dlenghts, int nSeq);


void m_step_mhmm_f32(HMM<float, multinomial::Multinomial<float> > &hmm,
                     unsigned short int* dX, unsigned short int* dlenghts, int nSeq);

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &multinomials,
                   int nStates,
                   double* dStartProb, int lddsp,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma,
                   double* logllhd,
                   int nObs, int nSeq,
                   double* dLlhd
                   );

size_t get_workspace_size_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm);

void create_handle_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                            void* workspace);

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                    int nObs, int nSeq, double* dLlhd);

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               unsigned short int* dX, unsigned short int* dlenghts, int nSeq,
                               bool doForward, bool doBackward, bool doGamma);

void viterbi_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                      unsigned short int* dVstates, unsigned short int* dX, unsigned short int* dlenghts, int nSeq);


void m_step_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                     unsigned short int* dX, unsigned short int* dlenghts, int nSeq);



void init_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                     std::vector<gmm::GMM<float> > &gmms,
                     int nStates,
                     float* dStartProb, int lddsp,
                     float* dT, int lddt,
                     float* dB, int lddb,
                     float* dGamma, int lddgamma,
                     float* logllhd,
                     int nObs, int nSeq,
                     float* dLlhd
                     );

size_t get_workspace_size_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm);

void create_handle_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                              void* workspace);

void setup_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                      int nObs, int nSeq, float* dLlhd);

void forward_backward_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                                 float* dX,
                                 unsigned short int* dlenghts,
                                 int nSeq,
                                 bool doForward, bool doBackward, bool doGamma);

void viterbi_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                        unsigned short int* dVstates,
                        float* dX,
                        unsigned short int* dlenghts,
                        int nSeq);


void m_step_gmmhmm_f32(HMM<float, gmm::GMM<float> > &hmm,
                       float* dX,
                       unsigned short int* dlenghts,
                       int nSeq);

void init_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                     std::vector<gmm::GMM<double> > &gmms,
                     int nStates,
                     double* dStartProb, int lddsp,
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma,
                     double* logllhd,
                     int nObs, int nSeq,
                     double* dLlhd
                     );

size_t get_workspace_size_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm);

void create_handle_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                              void* workspace);

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                      int nObs, int nSeq, double* dLlhd);

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX,
                                 unsigned short int* dlenghts,
                                 int nSeq,
                                 bool doForward, bool doBackward, bool doGamma);

void viterbi_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                        unsigned short int* dVstates,
                        double* dX,
                        unsigned short int* dlenghts,
                        int nSeq);


void m_step_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                       double* dX,
                       unsigned short int* dlenghts,
                       int nSeq);

}
