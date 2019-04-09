# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.gmm.gaussian_mixture_backend cimport GMM

cdef extern from "hmm/hmm_variables.h" namespace "hmm":
    cdef cppclass HMM[T, D]:
        pass
cdef extern from "hmm/dists/dists_variables.h" namespace "multinomial":
    cdef cppclass Multinomial[T]:
        pass


ctypedef HMM[float, GMM[float]] floatGMMHMM
ctypedef HMM[double, GMM[double]] doubleGMMHMM
ctypedef HMM[float, Multinomial[float]] floatMultinomialHMM
ctypedef HMM[double, Multinomial[double]] doubleMultinomialHMM

ctypedef fused floatTHMM:
    floatGMMHMM
    floatMultinomialHMM

ctypedef fused doubleTHMM:
    doubleGMMHMM
    doubleMultinomialHMM

ctypedef fused floatTDist:
    GMM[float]
    Multinomial[float]

ctypedef fused doubleTDist:
    GMM[double]
    Multinomial[double]


cdef extern from "hmm/hmm_py.h" namespace "hmm" nogil:
    # cdef void init_gmmhmm_f64(doubleGMMHMM &hmm,
    #                    vector[GMM[double]] &gmms,
    #                    int nStates,
    #                    double* dT,
    #                    int lddt,
    #                    double* dB,
    #                    int lddb,
    #                    double* dGamma,
    #                    int lddgamma)
    #
    # cdef void forward_backward_gmmhmm_f64(doubleGMMHMM &hmm,
    #                                double* dX,
    #                                int* dlenghts,
    #                                int nSeq,
    #                                bool doForward,
    #                                bool doBackward)

    cdef void init_mhmm_f32(floatMultinomialHMM &hmm,
                            vector[Multinomial[float]] &multinomials,
                            int nStates,
                            float* dStartProb,
                            int lddsp,
                            float* dT,
                            int lddt,
                            float* dB,
                            int lddb,
                            float* dGamma,
                            int lddgamma,
                            float* logllhd,
                            int nObs,
                            int nSeq,
                            float* dLlhd)

    cdef size_t get_workspace_size_mhmm_f32(floatMultinomialHMM &hmm)
    cdef void create_handle_mhmm_f32(floatMultinomialHMM &hmm,
                                     void* ws)

    cdef void viterbi_mhmm_f32(floatMultinomialHMM &hmm,
                               unsigned short int* dVStates,
                               unsigned short int* dX,
                               unsigned short int* dlenghts,
                               int nSeq)

    cdef void forward_backward_mhmm_f32(floatMultinomialHMM &hmm,
                                        unsigned short int* dX,
                                        unsigned short int* dlenghts,
                                        int nSeq,
                                        bool doForward,
                                        bool doBackward,
                                        bool doGamma)

    cdef void setup_mhmm_f32(floatMultinomialHMM &hmm,
                             int nObs,
                             int nSeq,
                             float* dLlhd)

    cdef void m_step_mhmm_f32(floatMultinomialHMM &hmm,
                              unsigned short int* dX,
                              unsigned short int* dlenghts,
                              int nSeq)

    cdef void init_mhmm_f64(doubleMultinomialHMM &hmm,
                            vector[Multinomial[double]] &multinomials,
                            int nStates,
                            double* dStartProb,
                            int lddsp,
                            double* dT,
                            int lddt,
                            double* dB,
                            int lddb,
                            double* dGamma,
                            int lddgamma,
                            double* logllhd,
                            int nObs,
                            int nSeq,
                            double* dLlhd)

    cdef size_t get_workspace_size_mhmm_f64(doubleMultinomialHMM &hmm)
    cdef void create_handle_mhmm_f64(doubleMultinomialHMM &hmm,
                                     void* ws)

    cdef void viterbi_mhmm_f64(doubleMultinomialHMM &hmm,
                               unsigned short int* dVStates,
                               unsigned short int* dX,
                               unsigned short int* dlenghts,
                               int nSeq)

    cdef void forward_backward_mhmm_f64(doubleMultinomialHMM &hmm,
                                        unsigned short int* dX,
                                        unsigned short int* dlenghts,
                                        int nSeq,
                                        bool doForward,
                                        bool doBackward,
                                        bool doGamma)

    cdef void setup_mhmm_f64(doubleMultinomialHMM &hmm,
                             int nObs,
                             int nSeq,
                             double* dLlhd)

    cdef void m_step_mhmm_f64(doubleMultinomialHMM &hmm,
                              unsigned short int* dX,
                              unsigned short int* dlenghts,
                              int nSeq)


    cdef void init_gmmhmm_f32(floatGMMHMM &hmm,
                              vector[GMM[float]] &gmms,
                              int nStates,
                              float* dStartProb,
                              int lddsp,
                              float* dT,
                              int lddt,
                              float* dB,
                              int lddb,
                              float* dGamma,
                              int lddgamma,
                              float* logllhd,
                              int nObs,
                              int nSeq,
                              float* dLlhd)

    cdef size_t get_workspace_size_gmmhmm_f32(floatGMMHMM &hmm)
    cdef void create_handle_gmmhmm_f32(floatGMMHMM &hmm,
                                       void* ws)

    cdef void viterbi_gmmhmm_f32(floatGMMHMM &hmm,
                                 unsigned short int* dVStates,
                                 unsigned short int* dX,
                                 unsigned short int* dlenghts,
                                 int nSeq)

    cdef void forward_backward_gmmhmm_f32(floatGMMHMM &hmm,
                                          unsigned short int* dX,
                                          unsigned short int* dlenghts,
                                          int nSeq,
                                          bool doForward,
                                          bool doBackward,
                                          bool doGamma)

    cdef void setup_gmmhmm_f32(floatGMMHMM &hmm,
                               int nObs,
                               int nSeq,
                               float* dLlhd)

    cdef void m_step_gmmhmm_f32(floatGMMHMM &hmm,
                                unsigned short int* dX,
                                unsigned short int* dlenghts,
                                int nSeq)

    cdef void init_gmmhmm_f64(doubleGMMHMM &hmm,
                              vector[GMM[double]] &gmms,
                              int nStates,
                              double* dStartProb,
                              int lddsp,
                              double* dT,
                              int lddt,
                              double* dB,
                              int lddb,
                              double* dGamma,
                              int lddgamma,
                              double* logllhd,
                              int nObs,
                              int nSeq,
                              double* dLlhd)

    cdef size_t get_workspace_size_gmmhmm_f64(doubleGMMHMM &hmm)
    cdef void create_handle_gmmhmm_f64(doubleGMMHMM &hmm,
                                       void* ws)

    cdef void viterbi_gmmhmm_f64(doubleGMMHMM &hmm,
                                 unsigned short int* dVStates,
                                 unsigned short int* dX,
                                 unsigned short int* dlenghts,
                                 int nSeq)

    cdef void forward_backward_gmmhmm_f64(doubleGMMHMM &hmm,
                                          unsigned short int* dX,
                                          unsigned short int* dlenghts,
                                          int nSeq,
                                          bool doForward,
                                          bool doBackward,
                                          bool doGamma)

    cdef void setup_gmmhmm_f64(doubleGMMHMM &hmm,
                               int nObs,
                               int nSeq,
                               double* dLlhd)

    cdef void m_step_gmmhmm_f64(doubleGMMHMM &hmm,
                                unsigned short int* dX,
                                unsigned short int* dlenghts,
                                int nSeq)

