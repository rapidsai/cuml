import cudf

from numba import cuda
from cuml import numba_utils

from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.gmm.gmm_base import _BaseGMM
from cuml.gmm.sample_utils import *
from cuml.hmm.base.hmm_base import _BaseCUML


RUP_SIZE = 32

cdef _setup_multinomial(self,
                        floatMultinomial& multinomial32,
                        doubleMultinomial& multinomial64):
    cdef uintptr_t _dPis_ptr = self.dParams["pis"].device_ctypes_pointer.value
    cdef int lddLlhd =self.lddllhd

    cdef int nCl = self.nCl

    if self.precision == 'double':
        with nogil:
            init_multinomial_f64(multinomial64, _dPis_ptr, nCl)

class Multinomial(_BaseCUML):
    def __init__(self, n_components, precision):
        super().__init__(precision=precision)
        self.nCl = n_components

    def _initialize(self):
        self.dPis = sample_matrix(self.n_components, self.1, random_state=self.random_state, isRowNorm=True)
        self.lddpu = self.nCl
        self.dPis = process_parameter(self.dT, self.lddt, self.dtype)