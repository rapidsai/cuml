from libc.stdint cimport uintptr_t


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

