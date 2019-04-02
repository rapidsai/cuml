import numpy as np

import ctypes
cimport numpy as np

cdef extern from "ts/kalman.h":
  cdef void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                          double* ptr_vs, double* ptr_Fs, double* ptr_loglike, double* ptr_sigma2)


def kfilter(np.ndarray[double, ndim=1, mode="c"] ys,
            np.ndarray[double, ndim=2, mode="c"] Z,
            np.ndarray[double, ndim=2, mode="c"] R,
            np.ndarray[double, ndim=2, mode="c"] T,
            int r):

    cdef int nobs = len(ys)
    cdef np.ndarray[double, ndim=1, mode="c"] vs = np.zeros(nobs)
    cdef np.ndarray[double, ndim=1, mode="c"] Fs = np.zeros(nobs)
    cdef double loglike = 0.0
    cdef double sigma2 = 0.0

    print("Z:",Z)
    print("R:",R)
    print("T:",T)

    kalman_filter(&ys[0], len(ys), &Z[0,0], &R[0,0], &T[0,0], r, &vs[0], &Fs[0], &loglike, &sigma2)

    print("LogLike C++: ", loglike)
    print("Sigma2 C++: ", sigma2)

    loglikelihood = 0
    alpha = np.zeros((r, 1))

    # use a single kalman iteration as covariance (P) initialization
    P = T@T.T - T@Z.T@Z@T.T + R@R.T

    nobs = len(ys)
    Fs = np.ones(nobs)
    vs = np.zeros(nobs)
    it = 0
    F = 0

    Ptm1 = np.copy(P)

    # TODO: Why stop at F==1.0? (and it's basically never exactly 1.0)
    while F != 1.0 and it < nobs:
        v = ys[it] - alpha[0, 0]
        F = P[0, 0]

        if F < 0:
            raise AssertionError("ERROR: F must be > 0. Possible non-positive definite covariance P: {}".format(P))

        Fs[it] = F
        vs[it] = v

        # Recall: '@' is Python3 matrix multiplication operator
        # set_trace()
        K = 1.0/Fs[it] * (T @ P @ Z.T)
        alpha = T*alpha + K*vs[it]
        L = T - K @ Z
        P = T @ P @ L.T + R @ R.T

        loglikelihood += np.log(F)
        it += 1

    for i in range(it, nobs):
        v = ys[i] - alpha[0, 0]
        vs[i] = v
        alpha = T @ alpha + K * v

    sigma2 = np.mean(vs**2 / Fs)
    assert(sigma2 > 0)
    loglike = -.5 * (loglikelihood + nobs * np.log(sigma2))
    loglike -= nobs / 2. * (np.log(2 * np.pi) + 1)
    # print("P vs P0 ||P-P0||", P, P0, np.linalg.norm(P-P0))
    print("LogLike PyX: ", loglike)
    print("Sigma2 PyX: ", sigma2)
    return vs, Fs, loglike, sigma2


def init_kalman_matrices(ar_params, ma_params):
    p = len(ar_params)
    q = len(ma_params)
    r = max(p, q+1)  # see (3.18) in TSA by D&K

    Z = np.zeros((1, r), order="F")
    Z[0, 0] = 1.0

    R = np.zeros((r, 1), order="F")
    # for (i, ma_i) in enumerate(ma_params):
    R[1:q + 1, 0] = ma_params[:]

    R[0] = 1.0

    T = np.zeros((r, r))
    params_padded = np.zeros(r)
    # handle zero coefficients if necessary
    params_padded[:p] = ar_params[:]
    T[:, 0] = params_padded
    T[:-1, 1:] = np.eye(r - 1)

    return Z, R, T, r
