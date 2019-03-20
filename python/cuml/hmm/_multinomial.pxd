cdef extern from "hmm/dists/dists_variables.h" namespace "multinomial":
    cdef cppclass Multinomial[T]:
        pass

ctypedef Multinomial[float] floatMultinomial
ctypedef Multinomial[double] doubleMultinomial

cdef extern from "hmm/hmm_py.h" namespace "multinomial" nogil:

    # cdef void init_multinomial_f32(floatMultinomial&,
    #                    float*,
    #                    int)

    cdef void init_multinomial_f64(doubleMultinomial&,
                   double*,
                   int)

cdef _setup_multinomial(self,
                        floatMultinomial& multinomial32,
                        doubleMultinomial& multinomial64)