#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from cuml.common.handle cimport cumlHandle
import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.string cimport string
cimport cython
from cuml.ts.batched_kalman import pynvtx_range_push, pynvtx_range_pop
from cuml.ts.batched_kalman import pack, unpack
from cuml.ts.batched_kalman import batched_transform as batched_trans_cuda
from cuml.ts.batched_lbfgs import batched_fmin_lbfgs_b

import cuml
from cuml.utils.input_utils import input_to_dev_array

from libc.stdint cimport uintptr_t


from typing import List, Tuple
import cudf

cdef extern from "ts/batched_arima.hpp" namespace "ML":
  void batched_loglike(cumlHandle& handle, double* y, int num_batches, int nobs, int p,
                       int d, int q, double* params, vector[double]& vec_loglike, bool trans)


class ARIMAModel:
    r"""
    The Batched ARIMA model fits the following to each given input:
    if d=1:
      \delta \tilde{y}_{t} = \mu + \sum_{i=1}^{p} \phi_i \delta y_{t-i}
                                    + \sum_{i=1}^{q} \theta_i (y_{t-i} -
                                                                 \tilde{y}_{t-i})

    Note all fitted parameters, \mu, \phi_i, \theta_i.
    """

    def __init__(self, order: List[Tuple[int, int, int]],
                 mu: np.ndarray,
                 ar_params: List[np.ndarray],
                 ma_params: List[np.ndarray],
                 y: cudf.DataFrame):
        self.order = order
        self.mu = mu
        self.ar_params = ar_params
        self.ma_params = ma_params
        self.y = y
        self.num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
        self.num_batches = y.shape[1]
        self.yp = None
        self.niter = None # number of iterations used during fit

    def __repr__(self):
        return "Batched ARIMA Model {}, mu:{}, ar:{}, ma:{}".format(self.order, self.mu,
                                                                    self.ar_params, self.ma_params)

    def __str__(self):
        return self.__repr__()

    @property
    def bic(self):
        (p, d, q) = self.order[0]
        x = pack(p, d, q, self.num_batches, self.mu, self.ar_params, self.ma_params)
        llb = ll_f(self.num_batches, self.num_samples, self.order[0], self.y, x)
        return [-2 * lli + np.log(len(self.y)) * (_model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]

    @property
    def aic(self):
        (p, d, q) = self.order[0]
        x = pack(p, d, q, self.num_batches, self.mu, self.ar_params, self.ma_params)
        llb = ll_f(self.num_batches, self.num_samples, self.order[0], self.y, x)
        return [-2 * lli + 2 * (_model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]


def _model_complexity(order):
    (p, d, q) = order
    # complexity is number of parameters: mu + ar + ma
    return d + p + q


def batch_trans(p, d, q, nb, x, handle):
    """Apply the stationarity/invertibility guaranteeing transform to batched-parameter vector x."""
    pynvtx_range_push("jones trans")

    Tx = batched_trans_cuda(p, d, q, nb, x, False, handle)
    
    pynvtx_range_pop()
    return Tx


def batch_invtrans(p, d, q, nb, x, handle):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """
    pynvtx_range_push("jones inv-trans")

    Tx = batched_trans_cuda(p, d, q, nb, x, True, handle)

    pynvtx_range_pop()
    return Tx


# def batched_loglike_cuda(y, int num_batches, int nobs,
#                          int p, int d, int q, np.ndarray[double] x, bool trans):

#     cdef vector[double] vec_loglike
#     cdef vector[double] vec_y_cm
#     cdef vector[double] vec_x

#     num_params = (p+d+q)

#     vec_loglike.resize(num_batches)

#     # ensure Column major layout
#     cdef np.ndarray[double, ndim=2] y_cm = np.asfortranarray(y)

#     batched_loglike(&y_cm[0,0], num_batches, nobs, p, d, q, &x[0], vec_loglike, trans)

#     # copy results into numpy array
#     loglike = np.zeros(num_batches)
#     for i in range(num_batches):
#         loglike[i] = vec_loglike[i]

#     pynvtx_range_pop()

#     return loglike

def ll_f(num_batches, nobs, order, y, np.ndarray[double] x, trans=True, handle=None):
    """Computes batched loglikelihood given parameters stored in `x`."""

    cdef vector[double] vec_loglike
    cdef vector[double] vec_y_cm
    cdef vector[double] vec_x

    # if cumlHandle is None:
        # cumlHandle = 

    pynvtx_range_push("ll_f")
    p, d, q = order

    num_params = (p+d+q)

    vec_loglike.resize(num_batches)

    cdef uintptr_t d_y_ptr
    # note: make sure you explicitly have d_y_array. Otherwise it gets garbage collected (I think).
    d_y_array, d_y_ptr, _, _, dtype = input_to_dev_array(y, check_dtype=np.float64)
    # print("d_y_ptr1:", d_y_ptr)
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()
 
    if dtype == np.float64:
        batched_loglike(handle_[0], <double*>d_y_ptr, num_batches, nobs, p, d, q, &x[0], vec_loglike, trans)
    else:
        raise ValueError("Only 64-bit floating point inputs currently supported")

    loglike = np.zeros(num_batches)
    for i in range(num_batches):
        loglike[i] = vec_loglike[i]

    pynvtx_range_pop()

    return loglike

def ll_gf(num_batches, nobs, num_parameters, order, y, x, h=1e-8, trans=True, handle=None):
    """Computes fd-gradient of batched loglikelihood given parameters stored in
    `x`. Because batches are independent, it only compute the function for the
    single-batch number of parameters."""
    pynvtx_range_push("ll_gf")
    
    fd = np.zeros(num_parameters)

    grad = np.zeros(len(x))

    # 1st order FD saves 20% runtime.
    # ll_b0 = ll_f(num_batches, num_parameters, order, y, x, trans=trans)
    assert (len(x) / num_parameters) == float(num_batches)
    for i in range(num_parameters):
        fd[i] = h

        # duplicate the perturbation across batches (they are independent)
        fdph = np.tile(fd, num_batches)

        # reset perturbation
        fd[i] = 0.0

        ll_b_ph = ll_f(num_batches, nobs, order, y, x+fdph, trans=trans, handle=handle)
        ll_b_mh = ll_f(num_batches, nobs, order, y, x-fdph, trans=trans, handle=handle)
        
        np.seterr(all='raise')
        # first derivative second order accuracy
        grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)
        # first derivative first order accuracy
        # grad_i_b = (ll_b_ph - ll_b0)/(h)

        if num_batches == 1:
            grad[i] = grad_i_b
        else:
            assert len(grad[i::num_parameters]) == len(grad_i_b)
            # Distribute the result to all batches
            grad[i::num_parameters] = grad_i_b

    pynvtx_range_pop()
    return grad

def fit(y,
        order: Tuple[int, int, int],
        mu0: np.ndarray,
        ar_params0: List[np.ndarray],
        ma_params0: List[np.ndarray],
        opt_disp=-1, h=1e-9, alpha_max=1000):
    """
    Fits the ARIMA model to each time-series (batched together in a dense numpy matrix)
    with the given initial parameters. `y` has shape (num_samples, num_batches)

    """
    p, d, q = order
    num_parameters = d + p + q

    d_y, d_y_ptr, num_samples, num_batches, dtype = input_to_dev_array(y, check_dtype=np.float64)

    handle = cuml.common.handle.Handle()

    def f(x: np.ndarray) -> np.ndarray:
        """The (batched) energy functional returning the negative loglikelihood (for each series)."""

        # Recall: Maximimize LL means minimize negative
        n_llf = -(ll_f(num_batches, num_samples, order, d_y, x, trans=True, handle=handle))
        return n_llf/(num_samples-1)


    # optimized finite differencing gradient for batches
    def gf(x):
        """The gradient of the (batched) energy functional."""
        # Recall: We maximize LL by minimizing -LL
        n_gllf = -ll_gf(num_batches, num_samples, num_parameters, order, d_y, x, h, trans=True, handle=handle)
        return n_gllf/(num_samples-1)

    x0 = pack(p, d, q, num_batches, mu0, ar_params0, ma_params0)
    x0 = batch_invtrans(p, d, q, num_batches, x0, handle)
        
    # check initial parameter sanity
    if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")


    # Optimize parameters by minimizing log likelihood.
    x, niter, flags = batched_fmin_lbfgs_b(f, x0, num_batches, gf,
                                           iprint=opt_disp, factr=1000)

    # TODO: Better Handle non-zero `flag` array values: 0 -> ok, 1,2 -> optimizer had trouble
    if (flags != 0).any():
        print("WARNING(`fit()`): Some batch members had optimizer problems.")

    Tx = batch_trans(p, d, q, num_batches, x, handle)
    mu, ar, ma = unpack(p, d, q, num_batches, Tx)

    fit_model = ARIMAModel(num_batches*[order], mu, ar, ma, y)
    fit_model.niter = niter
    return fit_model

