import numpy as np
from libcpp cimport bool

cdef extern from "hmm/magma/b_likelihood.h" namespace "MLCommon":

  cdef void likelihood_batched(int nCl, int nDim,
                               int nObs,
                               float** &dX_array, int lddx,
                               float** &dmu_array, int lddmu,
                               float** &dsigma_array, int lddsigma_full, int lddsigma,
                               float* dLlhd, int lddLlhd)
