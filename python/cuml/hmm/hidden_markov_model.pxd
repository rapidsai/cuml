from libcpp.vector cimport vector
from cuml.gmm.gaussian_mixture cimport GMM

cdef extern from "hmm/hmm_variables.h" namespace "hmm":
    cdef cppclass HMM[T]:
        pass

cdef extern from "hmm/hmm_py.h" namespace "hmm" nogil:
    cdef void init_f64(HMM[double] &hmm,
                       vector[GMM[double]] &gmms,
                       int nStates,
                       double* dT,
                       int lddt,
                       double* dB,
                       int lddb,
                       double* dGamma,
                       int lddgamma)

    cdef void forward_backward_f64(HMM[double] &hmm,
                                   double* dX,
                                   int* dlenghts,
                                   int nSeq,
                                   bool doForward,
                                   bool doBackward)