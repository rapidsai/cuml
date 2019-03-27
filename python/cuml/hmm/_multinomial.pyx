from libc.stdint cimport uintptr_t

from cuml.hmm.core.hmm_base import _BaseCUML
from cuml.gmm.sample_utils import *
from cuml.hmm._multinomial cimport *

RUP_SIZE = 32

cdef _setup_multinomial(self,
                        floatMultinomial& multinomial32,
                        doubleMultinomial& multinomial64):
    cdef uintptr_t _dPis_ptr = self.dPis.device_ctypes_pointer.value

    cdef int n_features = self.n_features
    if self.precision == 'double':
        with nogil:
            init_multinomial_f64(multinomial64,
                                 <double*> _dPis_ptr,
                                 <int> n_features)

class _Multinomial(_BaseCUML):
    def __init__(self, init_params, precision, random_state):
        super().__init__(precision=precision, random_state=random_state)
        self.init_params = init_params

    def _initialize(self):
        if "e" in self.init_params :
            dPis = sample_matrix(1, self.n_features, random_state=self.random_state, isRowNorm=True)[0]
            self.set_emissionprob_(dPis)

    def _set_dims(self, X):
        self.n_features = np.max(X)

    def _setup(self, X):
        pass
        # self.n_features = np.max(X)

    def get_emissionprob_(self):
        pis = self.dPis.copy_to_host()
        pis = pis.flatten()
        return pis

    def set_emissionprob_(self, prob):
        self.n_features = prob.shape[0]
        self.dPis = process_parameter(prob[:, None], self.n_features, self.dtype)

    _emissionprob_ = property(get_emissionprob_, set_emissionprob_)