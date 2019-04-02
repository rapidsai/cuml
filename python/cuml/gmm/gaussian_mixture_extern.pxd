from cuml.gmm.gaussian_mixture_backend cimport GMM

cdef extern from "gmm/gmm_py.h" namespace "gmm" nogil:

    cdef void init_f32(GMM[float]&,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       float*,
                       float,
                       int,
                       int,
                       int)

    cdef void setup_f32(GMM[float]&)
    cdef void compute_lbow_f32(GMM[float]&)
    cdef void update_llhd_f32(float*, GMM[float]&)
    cdef void update_rhos_f32(GMM[float]&, float*)
    cdef void update_mus_f32(float*, GMM[float]&)
    cdef void update_sigmas_f32(float*, GMM[float]&)
    cdef void update_pis_f32(GMM[float]&)

    cdef void init_f64(GMM[double]&,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       double*,
                       double,
                       int,
                       int,
                       int)

    cdef void setup_f64(GMM[double]&)
    cdef void compute_lbow_f64(GMM[double]&)
    cdef void update_llhd_f64(double*, GMM[double]&)
    cdef void update_rhos_f64(GMM[double]&, double*)
    cdef void update_mus_f64(double*, GMM[double]&)
    cdef void update_sigmas_f64(double*, GMM[double]&)
    cdef void update_pis_f64(GMM[double]&)
