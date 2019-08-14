# cython: language_level = 3

import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "ts/batched_kalman.h":

  void batched_kalman_filter(double* ptr_ys_b,
                             int nobs,
                             # double* h_Zb,
                             # double* h_Rb,
                             # double* h_Tb,
                             const vector[double*]& ptr_Zb,
                             const vector[double*]& ptr_Rb,
                             const vector[double*]& ptr_Tb,
                             int r,
                             int num_batches,
                             vector[double]& vec_loglike_b,
                             vector[vector[double]]& vec_vs_b,
                             bool initP_with_kalman_iterations)

  void nvtx_range_push(string msg)

  void nvtx_range_pop()

  void batched_jones_transform(int p, int q,
                               int batchSize,
                               bool isInv,
                               const vector[double]& ar,
                               const vector[double]& ma,
                               vector[double]& Tar,
                               vector[double]& Tma)


def unpack(p, q, nb, np.ndarray[double, ndim=1] x):
    """Unpack linearized parameters into mu, ar, and ma batched-groupings"""
    pynvtx_range_push("unpack(x) -> (ar,ma,mu)")
    num_parameters = 1 + p + q
    mu = np.zeros(nb)
    ar = []
    ma = []
    for i in range(nb):
        xi = x[i*num_parameters:(i+1)*num_parameters]
        mu[i] = xi[0]
        ar.append(xi[1:p+1])
        ma.append(xi[p+1:])

    pynvtx_range_pop()
    return (mu, ar, ma)


def pack(p, q, nb, mu, ar, ma):
    """Pack mu, ar, and ma batched-groupings into a linearized vector `x`"""
    pynvtx_range_push("pack(ar,ma,mu) -> x")
    num_parameters = 1 + p + q
    x = np.zeros(num_parameters*nb)
    for i in range(nb):
        xi = np.zeros(num_parameters)
        if mu[i]:
            xi[0] = mu[i]
        
        for j in range(p):
            xi[j+1] = ar[i][j]
        for j in range(q):
            xi[j+p+1] = ma[i][j]
        # xi = np.array([mu[i]])
        # xi = np.r_[xi, ar[i]]
        # xi = np.r_[xi, ma[i]]
        x[i*num_parameters:(i+1)*num_parameters] = xi

    pynvtx_range_pop()
    return x

def batched_transform(p, q, nb, ar, ma, isInv):
    cdef vector[double] vec_ar
    cdef vector[double] vec_ma
    cdef vector[double] vec_Tar
    cdef vector[double] vec_Tma

    # pack ar & ma into C++ vectors
    for ib in range(nb):
        for ip in range(p):
            vec_ar.push_back(ar[ib][ip])
        for iq in range(q):
            vec_ma.push_back(ma[ib][ip])

    batched_jones_transform(p, q, nb, isInv, vec_ar, vec_ma, vec_Tar, vec_Tma)

    # unpack Tar & Tma results into [np.ndarray]
    Tar = []
    Tma = []
    for ib in range(nb):
        Tar_i = np.zeros(p)
        for ip in range(p):
            Tar_i[ip] = vec_Tar[ib*p + ip]

        Tar.append(Tar_i)

        Tma_i = np.zeros(q)
        for iq in range(q):
            Tma_i[iq] = vec_Tma[ib*q + iq]

        Tma.append(Tma_i)

    return (Tar, Tma)

def pynvtx_range_push(msg):
    cdef string s = msg.encode("UTF-8")
    nvtx_range_push(s)

def pynvtx_range_pop():
    nvtx_range_pop()

def batched_kfilter(np.ndarray[double, ndim=2] y,
                    Z_b, # list of numpy arrays
                    R_b,
                    T_b,
                    int r,
                    gpu=True,
                    initP_with_kalman_iterations=False):

    cdef vector[double] vec_loglike_b
    
    cdef int nobs = y.shape[0]
    cdef int num_batches = y.shape[1]

    # cuDF wasn't working well, comment out for now
    # # Extract device pointer from DataFrame. Careful: `y_mat` temporary is to
    # # avoid the "gpu_matrix" object from getting garbage collected. `ytmp`
    # # simply satisfies the Cython compiler.
    # y_mat = y.as_gpu_matrix()
    # cdef unsigned long long ytmp = y_mat.gpu_data.device_pointer.value
    # cdef double* y_ptr = <double*>ytmp

    cdef np.ndarray[double, ndim=2, mode="fortran"] Z_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] R_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] T_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] P0_bi
    cdef vector[double*] vec_Zb
    cdef vector[double*] vec_Rb
    cdef vector[double*] vec_Tb
    cdef vector[double*] vec_P0

    cdef vector[double*] vec_ys_b

    cdef vector[vector[double]] vec_vs_b

    for i in range(num_batches):
        Z_bi = Z_b[i]
        R_bi = R_b[i]
        T_bi = T_b[i]
        vec_Zb.push_back(&Z_bi[0,0])
        vec_Rb.push_back(&R_bi[0,0])
        vec_Tb.push_back(&T_bi[0,0])

    ll_b = np.zeros(num_batches)
    vs = np.zeros((nobs, num_batches))

    if gpu:
        batched_kalman_filter(&y[0,0],
                              nobs,
                              # &Z_dense[0], &R_dense[0], &T_dense[0],
                              vec_Zb, vec_Rb, vec_Tb,
                              r,
                              num_batches,
                              vec_loglike_b,
                              vec_vs_b,
                              initP_with_kalman_iterations)
        # convert C++-results to numpy arrays    
        for i in range(num_batches):
            ll_b[i] = vec_loglike_b[i]
            for j in range(nobs):
                vs[j,i] = vec_vs_b[i][j]

    else:
        # CPU reference version
        for i in range(num_batches):
            ysi = np.copy(y[:, i])
            vsi, _, loglike_i, _ = kfilter_reference(ysi, Z_b[i], R_b[i], T_b[i], r)
            ll_b[i] = loglike_i
            vs[:, i] = np.copy(vsi)

    return ll_b, vs


def init_batched_kalman_matrices(b_ar_params, b_ma_params):
    """Builds batched-versions of the kalman matrices given batched AR and MA parameters"""
    pynvtx_range_push("init_batched_kalman_matrices")

    Zb = []
    Rb = []
    Tb = []

    # find maximum 'r' across batches; see (3.18) in TSA by D&K for definition of 'r'
    r_max = np.max([max(len(ar), len(ma)+1) for (ar, ma) in zip(b_ar_params, b_ma_params)])

    for (ari, mai) in zip(b_ar_params, b_ma_params):
        Z, R, T, r = init_kalman_matrices(ari, mai, r_max)
        Zb.append(Z)
        Rb.append(R)
        Tb.append(T)

    pynvtx_range_pop()
    return Zb, Rb, Tb, r_max


def init_kalman_matrices(ar_params, ma_params, r=None):
    p = len(ar_params)
    q = len(ma_params)

    # for batched case, we input the maximum `r` to zero-pad some matrices.
    if r is None:
        r = max(p, q+1)  # see (3.18) in TSA by D&K

    Z = np.zeros((1, r), order="F")
    Z[0, 0] = 1.0

    R = np.zeros((r, 1), order="F")
    # for (i, ma_i) in enumerate(ma_params):
    R[1:q + 1, 0] = ma_params[:]

    R[0] = 1.0

    T = np.zeros((r, r), order="F")
    params_padded = np.zeros(r)
    # handle zero coefficients if necessary
    params_padded[:p] = ar_params[:]
    T[:, 0] = params_padded
    T[:-1, 1:] = np.eye(r - 1)

    return Z, R, T, r

def kfilter_reference(ys, Z, R, T, r):
    """Reference kalman filter implementation"""
    loglikelihood = 0
    alpha = np.zeros((r, 1))

    # see D&K's TSA 5.6.2 for this formula
    # TODO: Why use psuedo-inverse (seems to be just regular inverse in book)

    invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T, T))
    P0 = np.reshape(invImTT @ (R @ R.T).ravel(), (r, r))

    # original:
    #  P0 = np.reshape(np.dot(np.linalg.pinv(np.eye(r**2) - np.kron(T, T)),
    #                         np.dot(R, R.T).ravel()), (r, r))

    # if P0[0, 0] < 0.0:
    #     print("WARNING: Proposed initial covariance P has negative diagonal entry, switching to P0=I")
    #     P = P0
    # else:

    # use a single kalman iteration as covariance (P) initialization
    P = np.copy(P0)

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

        # print("||P-Pm||=", np.linalg.norm(P-Ptm1))()
        # print("P=\n{}\nPm1=\n{}\n--------------".format(P, Ptm1))
        # set_trace()
        Ptm1 = np.copy(P)

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
    return vs, Fs, loglike, sigma2
