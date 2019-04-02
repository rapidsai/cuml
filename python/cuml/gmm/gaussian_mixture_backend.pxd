cdef extern from "gmm/gmm_variables.h" namespace "gmm":
    cdef cppclass GMM[T]:
        pass

cdef _setup_gmm(self, GMM[float]& gmm32, GMM[double]& gmm64, toAllocate=*)