from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.gmm.gaussian_mixture cimport GMM

cdef extern from "hmm/hmm_variables.h" namespace "hmm":
    cdef cppclass HMM[T, D]:
        pass

ctypedef HMM[float, GMM[float]] floatGMMHMM
ctypedef HMM[double, GMM[double]] doubleGMMHMM

cdef extern from "hmm/hmm_py.h" namespace "hmm" nogil:
    cdef void init_f64(doubleGMMHMM &hmm,
                       vector[GMM[double]] &gmms,
                       int nStates,
                       double* dT,
                       int lddt,
                       double* dB,
                       int lddb,
                       double* dGamma,
                       int lddgamma)

    cdef void forward_backward_f64(doubleGMMHMM &hmm,
                                   double* dX,
                                   int* dlenghts,
                                   int nSeq,
                                   bool doForward,
                                   bool doBackward)