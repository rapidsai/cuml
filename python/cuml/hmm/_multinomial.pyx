from libc.stdint cimport uintptr_t

from cuml.hmm.base.hmm_base import _BaseCUML
from cuml.gmm.sample_utils import *
from cuml.hmm._multinomial cimport *

RUP_SIZE = 32

cdef _setup_multinomial(self,
                        floatMultinomial& multinomial32,
                        doubleMultinomial& multinomial64):
    cdef uintptr_t _dPis_ptr = self.dParams["pis"].device_ctypes_pointer.value
    cdef int lddLlhd =self.lddllhd

    cdef int nCl = self.nCl

    if self.precision == 'double':
        with nogil:
            init_multinomial_f64(multinomial64,
                                 <double*> _dPis_ptr,
                                 <int> nCl)

class _Multinomial(_BaseCUML):
    def __init__(self, n_features, precision, random_state):
        super().__init__(precision=precision, random_state=random_state)
        self.n_features = n_features

    def _initialize(self):
        self.dPis = sample_matrix(self.n_features, 1, random_state=self.random_state, isColNorm=True)
        self.lddpi = self.n_features
        self.dPis = process_parameter(self.dPis, self.lddpi, self.dtype)

    @property
    def emissionprob_(self):
        pis = self.dPis.copy_to_host()
        pis = pis.flatten()
        return pis