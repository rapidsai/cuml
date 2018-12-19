
from libcpp cimport bool

cdef extern from "glm/glm_c.h" namespace "ML::GLM":

    cdef void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef, float *intercept, bool fit_intercept,
                bool normalize, int algo)
    cdef void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef, double *intercept, bool fit_intercept,
                bool normalize, int algo)
    cdef void olsPredict(const float *input, int n_rows, int n_cols, const float *coef, float intercept, float *preds)
    cdef void olsPredict(const double *input, int n_rows, int n_cols, const double *coef, double intercept, double *preds)

