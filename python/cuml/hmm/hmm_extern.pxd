from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.gmm.gaussian_mixture cimport GMM

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
                       int lddgamma)

    cdef void forward_backward_mhmm_f64(doubleMultinomialHMM &hmm,
                                   int* dX,
                                   int* dlenghts,
                                   int nSeq,
                                   bool doForward,
                                   bool doBackward)

    cdef void setup_mhmm_f64(doubleMultinomialHMM &hmm,
                    int nObs,
                   int nSeq)