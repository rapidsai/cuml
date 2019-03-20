cdef extern from "hmm/hmm_variables.h" namespace "hmm":
    cdef cppclass Multinomial[T]:
        pass

ctypedef Multinomial[float] floatMultinomial
ctypedef Multinomial[double] doubleMultinomial

cdef extern from "hmm/dists/multinomial.h" namespace "multinomial" nogil:

    cdef void init_multinomial_f32(floatMultinomial&,
                       float*,
                       int)

    cdef void init_multinomial_f64(doubleMultinomial&,
                   double*,
                   int)