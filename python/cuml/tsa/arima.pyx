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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import cupy as cp

import ctypes

from cuml.tsa.batched_lbfgs import batched_fmin_lbfgs_b
import rmm

import cuml
from cuml.utils.input_utils import input_to_dev_array, input_to_host_array
from cuml.utils.input_utils import get_dev_array_ptr

from typing import List, Tuple
import cudf
from cuml.utils import get_dev_array_ptr, zeros

from cuml.common.cuda import nvtx_range_push, nvtx_range_pop

from cuml.common.base import Base

from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cuml.common.handle cimport cumlHandle
from libcpp.vector cimport vector

cdef extern from "arima/batched_arima.hpp" namespace "ML":
    void batched_loglike(cumlHandle& handle,
                         double* y,
                         int num_batches,
                         int nobs,
                         int p,
                         int d,
                         int q,
                         double* params,
                         vector[double]& vec_loglike,
                         double* d_vs,
                         bool trans)

    void predict_in_sample(cumlHandle& handle,
                           double* d_y,
                           int num_batches,
                           int nobs,
                           int p,
                           int d,
                           int q,
                           double* d_params,
                           double* d_vs_ptr,
                           double* d_y_p)

    void residual(cumlHandle& handle,
                  double* d_y,
                  int num_batches,
                  int nobs, int p,
                  int d,
                  int q,
                  double* d_params,
                  double* d_vs,
                  bool trans)

    void forecast(cumlHandle& handle,
                  int num_steps,
                  int p,
                  int d,
                  int q,
                  int batch_size,
                  int nobs,
                  double* d_y,
                  double* d_y_diff,
                  double* d_vs,
                  double* d_params,
                  double* d_y_fc)

    void cpp_estimate_x0 "estimate_x0" (
        cumlHandle& handle,
        double* d_mu,
        double* d_ar,
        double* d_ma,
        const double* d_y,
        int num_batches,
        int nobs,
        int p,
        int d,
        int q,
        int start_ar_lags)


cdef extern from "arima/batched_kalman.hpp" namespace "ML":

    void batched_jones_transform(cumlHandle& handle,
                                 int p,
                                 int d,
                                 int q,
                                 int batchSize,
                                 bool isInv,
                                 const double* h_params,
                                 double* h_Tparams)


class ARIMAModel(Base):
    r"""Implements an ARIMA model for in- and out-of-sample
    time-series prediction.

    The ARIMA model consists of three model parameter classes:
    "AutoRegressive", "Integrated", and "Moving Average" to fit to a given
    time-series input. The library provides both in-sample prediction, and out
    of sample forecasting.

    The Batched ARIMA model fits the following to each given time-series input:
    if d=1:
      \delta \tilde{y}_{t} = \mu + \sum_{i=1}^{p} \phi_i \delta y_{t-i}
                                    + \sum_{i=1}^{q} \theta_i (y_{t-i} -
                                                               \tilde{y}_{t-i})

    Note all fitted parameters, \mu, \phi_i, \theta_i and
    the model order (p, d, q).

    **Limitations**: The library assumes collections (i.e., batches) of
      time-series data of the same length with no missing values.

    Examples
    ---------
    .. code-block:: python

        import numpy as np
        from cuml.tsa.arima import fit
        import matplotlib.pyplot as plt

        # create sample data
        num_samples = 200
        xs = np.linspace(0, 1, num_samples)
        np.random.seed(12)
        noise = np.random.normal(scale=0.05, size=num_samples)
        noise2 = np.random.normal(scale=0.05, size=num_samples)
        ys1 = noise + 0.5*xs + 0.1*np.sin(xs/np.pi)
        ys2 = noise2 + 0.25*xs + 0.15*np.sin(0.8*xs/np.pi)
        ys = np.zeros((num_samples, 2))
        ys[:, 0] = ys1
        ys[:, 1] = ys2

        plt.plot(xs, ys1, xs, ys2)

        # get parameter estimates
        mu0, ar0, ma0 = arima.estimate_x0((1,1,1), ys)

        # fine-tune parameter estimates
        model = arima.fit(ys, (1,1,1), mu0, ar0, ma0)

        # predict and forecast using fitted model
        d_yp = model.predict_in_sample()
        yp = cuml.utils.input_to_host_array(d_yp)
        d_yfc = model.forecast(50)
        yfc = cuml.utils.input_to_host_array(d_yfc)
        dx = xs[1] - xs[0]
        xfc = np.linspace(1, 1+50*dx, 50)
        plt.plot(xs, yp, xfc, yfc)

    Parameters
    ----------
    order : Tuple[int, int, int]
            The ARIMA order (p, d, q) of the model
    mu    : array-like (host)
            (if d>0) Array of trend parameters, one for each series
    ar_params : List[array-like] (host) or numpy.ndarray
                List of AR parameters, grouped (`p`) per series.
                If passed as a numpy array, the shape must be (p, num_batches)
                and the order column-major ('F').
    ma_params : List[array-like] (host) or numpy.ndarray
                List of MA parameters, grouped (`q`) per series.
                If passed as a numpy array, the shape must be (q, num_batches)
                and the order column-major ('F').
    y : array-like (device or host)
        The time series series data. If given as `ndarray`, assumed to have
        each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF
        Series, NumPy ndarray, Numba device ndarray, cuda array interface
        compliant array like CuPy.

    References
    ----------
    The library is heavily influenced by the Python library `statsmodels`,
    particularly the `statsmodels.tsa.arima_model.ARIMA` model and
    corresponding code:
    www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA

    Additionally the following book is a useful reference:
    "Time Series Analysis by State Space Methods",
    J. Durbin, S.J. Koopman, 2nd Edition.

    """

    def __init__(self,
                 order: Tuple[int, int, int],
                 mu,
                 ar_params,
                 ma_params,
                 y,
                 handle=None):
        super().__init__(handle)
        self.order = order

        # Convert the parameters to numpy arrays if needed
        if type(mu) is list:
            self.mu = np.array(mu)
        else:
            self.mu = mu
        if type(ar_params) is list:
            self.ar_params = np.transpose(ar_params)
        else:
            self.ar_params = ar_params
        if type(ma_params) is list:
            self.ma_params = np.transpose(ma_params)
        else:
            self.ma_params = ma_params

        # get host and device pointers. Float64 only for now.
        h_y, h_y_ptr, n_samples, n_series, dtype = \
            input_to_host_array(y, check_dtype=np.float64)
        d_y, d_y_ptr, _, _, _ = input_to_dev_array(y, check_dtype=np.float64)

        self.h_y = h_y
        self.d_y = d_y
        self.num_samples = n_samples
        self.num_batches = n_series
        self.yp = None
        self.niter = None  # number of iterations used during fit

    def __str__(self):
        return "Batched ARIMA Model {}, mu:{}, ar:{}, ma:{}".format(
            self.order, self.mu,
            self.ar_params, self.ma_params)

    # TODO: wrap C++ version
    @property
    def bic(self):
        (p, d, q) = self.order
        x = pack(p, d, q, self.num_batches,
                 self.mu, self.ar_params, self.ma_params)
        llb = ll_f(self.num_batches, self.num_samples, self.order, self.d_y, x)
        return [-2 * lli + np.log(len(self.d_y))
                * (_model_complexity(self.order))
                for (i, lli) in enumerate(llb)]

    @property
    def aic(self):
        (p, d, q) = self.order
        x = pack(p, d, q, self.num_batches, self.mu,
                 self.ar_params, self.ma_params)
        llb = ll_f(self.num_batches, self.num_samples, self.order, self.d_y, x)
        return [-2 * lli + 2 * (_model_complexity(self.order))
                for (i, lli) in enumerate(llb)]

    # def _assert_same_d(self, b_order):
    #     """Checks that all values of d in batched order are same"""
    #     b_d = [d for _, d, _ in b_order]
    #     assert (np.array(b_d) == b_d[0]).all()
    # TODO: unused? remove?

    def predict_in_sample(self):
        """Return in-sample prediction on batched series given batched model

        Returns:
        --------
        y_p : array-like, (device), shape = (n_samples, n_series)

        Example:
        --------
        .. code-block:: python
            from cuml.tsa.arima import fit
            ...
            model = fit(ys, (1,1,1), mu0, ar0, ma0)
            y_pred = model.predict_in_sample()
        """

        p, d, q = self.order

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        x = pack(p, d, q, self.num_batches, self.mu, self.ar_params,
                 self.ma_params)
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = \
            input_to_dev_array(x, check_dtype=np.float64)

        # allocate residual (vs) and prediction (y_p) device memory and get
        # pointers
        cdef uintptr_t d_vs_ptr
        cdef uintptr_t d_y_p_ptr
        d_vs = rmm.device_array((self.num_samples - d, self.num_batches),
                                dtype=np.float64, order="F")
        d_y_p = rmm.device_array((self.num_samples, self.num_batches),
                                 dtype=np.float64, order="F")
        d_vs_ptr = get_dev_array_ptr(d_vs)
        d_y_p_ptr = get_dev_array_ptr(d_y_p)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        predict_in_sample(handle_[0],
                          <double*>d_y_ptr,
                          self.num_batches,
                          self.num_samples,
                          p,
                          d,
                          q,
                          <double*>d_params_ptr,
                          <double*>d_vs_ptr,
                          <double*>d_y_p_ptr)

        self.yp = d_y_p
        return d_y_p

    def forecast(self, nsteps: int):
        """Forecast the given model `nsteps` into the future.

        Parameters:
        ----------
        nsteps : int
                 The number of steps to forecast beyond end of `fit()` signal

        Returns:
        --------
        y_fc : array-like, shape = (nsteps, n_series)
               Forecast, one for each series.

        Example:
        --------
        .. code-block:: python
            from cuml.tsa.arima import fit
            import cuml
            ...
            model = fit(ys, (1,1,1), mu0, ar0, ma0)
            d_y_fc = model.forecast(10)
            y_fc = cuml.utils.input_to_host_array(d_y_fc)
            dx = xs[1] - xs[0]
            xfc = np.linspace(1, 1+50*dx, 50)
            plt.plot(xs, ys, xfc, yfc)

        """

        p, d, q = self.order

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        x = pack(p, d, q, self.num_batches, self.mu,
                 self.ar_params, self.ma_params)
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = \
            input_to_dev_array(x, check_dtype=np.float64)

        d_vs = rmm.device_array((self.num_samples - d,
                                 self.num_batches), dtype=np.float64,
                                order="F")
        cdef uintptr_t d_vs_ptr = get_dev_array_ptr(d_vs)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        residual(handle_[0],
                 <double*>d_y_ptr,
                 self.num_batches,
                 self.num_samples,
                 p, d, q,
                 <double*>d_params_ptr,
                 <double*>d_vs_ptr,
                 False)

        # note: `cp.diff()` returns row-major (regardless of input layout),
        # and thus needs conversion with `cp.asfortranarray()`
        y_diff = cp.asfortranarray(cp.diff(self.d_y, axis=0))
        cdef uintptr_t d_y_diff_ptr = y_diff.data

        d_y_fc = rmm.device_array((nsteps, self.num_batches),
                                  dtype=np.float64, order="F")
        cdef uintptr_t d_y_fc_ptr = get_dev_array_ptr(d_y_fc)

        forecast(handle_[0],
                 nsteps,
                 p, d, q,
                 self.num_batches,
                 self.num_samples,
                 <double*> d_y_ptr,
                 <double*>d_y_diff_ptr,
                 <double*>d_vs_ptr,
                 <double*>d_params_ptr,
                 <double*> d_y_fc_ptr)

        return d_y_fc




def estimate_x0(order, y, handle=None, start_ar_lags=None):
    nvtx_range_push("estimate x0")

    p, d, q = order

    cdef uintptr_t d_y_ptr
    d_y, d_y_ptr, num_samples, num_batches, dtype = \
        input_to_dev_array(y, check_dtype=np.float64)

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    # Create mu, ar and ma arrays
    cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
    cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
    cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
    if p > 0:
        d_ar = zeros((p, num_batches), dtype=dtype, order='F')
        d_ar_ptr = get_dev_array_ptr(d_ar)
    if q > 0:
        d_ma = zeros((q, num_batches), dtype=dtype, order='F')
        d_ma_ptr = get_dev_array_ptr(d_ma)
    if d > 0:
        d_mu = zeros(num_batches, dtype=dtype)
        d_mu_ptr = get_dev_array_ptr(d_mu)

    if start_ar_lags is None:
        start_ar_lags = -1

    # Call C++ function
    cpp_estimate_x0(handle_[0],
                    <double*> d_mu_ptr, <double*> d_ar_ptr, <double*> d_ma_ptr,
                    <double*> d_y_ptr,
                    <int> num_batches, <int> num_samples,
                    <int> p, <int> d, <int> q, <int> start_ar_lags)

    nvtx_range_pop()

    h_mu = d_mu.copy_to_host() if d > 0 else np.array([])
    h_ar = d_ar.copy_to_host() if p > 0 else np.zeros(shape=(0, num_batches))
    h_ma = d_ma.copy_to_host() if q > 0 else np.zeros(shape=(0, num_batches))

    # TODO: return device pointers?
    return h_mu, h_ar, h_ma


def ll_f(num_batches, nobs, order, y, x,
         trans=True, handle=None):
    """Computes batched loglikelihood for given parameters and series.

    Parameters:
    ----------
    num_batches : int
                  Number of series
    nobs : int
           Number of samples in each series (identical across series)
    order : Tuple[int, int, int]
            ARIMA Order (p, d, q)
    y     : array like, shape = (n_samples, n_series)
            Time series data
    x     : array
            dense parameter array, grouped by series,
            and again by [(mu, ar, ma), (mu, ar, ma)]
    trans : bool
            Should the `jones_transform` be applied?
            Note: The parameters from a `fit()` model are already transformed.
    handle : cumlHandle (optional)
             The cumlHandle to be used.
    """

    cdef vector[double] vec_loglike

    vec_loglike = _batched_loglike(num_batches,
                                   nobs, order,
                                   y, x,
                                   trans, handle)

    loglike = np.zeros(num_batches)
    for i in range(num_batches):
        loglike[i] = vec_loglike[i]

    return loglike


def ll_gf(num_batches, nobs, num_parameters, order,
          y, x, h=1e-8, trans=True, handle=None):
    """Computes gradient (via finite differencing) of the batched
    loglikelihood.

    Parameters:
    ----------
    num_batches : int
                  Number of series
    nobs : int
           Number of samples in each series (identical across series)
    num_parameters : int
                     The number of parameters per series (p + d + q)
    order : Tuple[int, int, int]
            ARIMA Order (p, d, q)
    y     : array like, shape = (n_samples, n_series)
            Time series data
    x     : array
            dense parameter array, grouped by series, and
            again by [(mu, ar, ma), (mu, ar, ma)]
    h     : float
            The finite-difference stepsize
    trans : bool
            Should the `jones_transform` be applied?
            Note: The parameters from a `fit()` model are already transformed.
    handle : cumlHandle (optional)
             The cumlHandle to be used.

    """
    nvtx_range_push("ll_gf")

    fd = np.zeros(num_parameters)

    grad = np.zeros(len(x))

    assert (len(x) / num_parameters) == float(num_batches)
    for i in range(num_parameters):
        fd[i] = h

        # duplicate the perturbation across batches (they are independent)
        fdph = np.tile(fd, num_batches)

        # reset perturbation
        fd[i] = 0.0

        ll_b_ph = ll_f(num_batches, nobs, order, y, x+fdph,
                       trans=trans, handle=handle)
        ll_b_mh = ll_f(num_batches, nobs, order, y, x-fdph,
                       trans=trans, handle=handle)

        np.seterr(all='raise')
        # first derivative second order accuracy
        grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)

        if num_batches == 1:
            grad[i] = grad_i_b
        else:
            assert len(grad[i::num_parameters]) == len(grad_i_b)
            # Distribute the result to all batches
            grad[i::num_parameters] = grad_i_b

    nvtx_range_pop()
    return grad


def fit(y,
        order: Tuple[int, int, int],
        mu0: np.ndarray,
        ar_params0: np.ndarray,
        ma_params0: np.ndarray,
        opt_disp=-1,
        h=1e-9,
        handle=None):
    """Fits an ARIMA model to each time-series for the given order and initial
    parameter estimates.

    Parameters
    ----------
    y : array-like (device or host) shape = (n_samples, n_series)
        Time series data.
    order : Tuple[int, int, int]
            The ARIMA order (p, d, q)
    mu0 : array-like
          Array of trend-parameter estimates. Only used if `d>0`.
    ar_params0 : List of arrays (TODO: this is outdated; support lists?)
                 List of AR parameter-arrays, one array per series,
                 each series has `p` parameters.
    ma_params0 : List of arrays (TODO: this is outdated; support lists?)
                 List of MA parameter-arrays, one array per series,
                 each series has `q` parameters.
    opt_disp : int
               Fit diagnostic level (for L-BFGS solver):
               * `-1` for no output,
               * `0<n<100` for output every `n` steps
               * `n>100` for more detailed output
    h        : float
               Finite-differencing step size. The gradient is computed
               using second-order differencing:
                   f(x+h) - f(x - h)
               g = ----------------- + O(h^2)
                         2 * h
    handle   : cumlHandle
               The cumlHandle needed for memory allocation
               and stream management.

    Returns:
    --------
    model : ARIMAModel
            The ARIMA model with the best fit parameters.

    """
    p, d, q = order
    num_parameters = d + p + q

    d_y, d_y_ptr, num_samples, num_batches, dtype = \
        input_to_dev_array(y, check_dtype=np.float64)

    if handle is None:
        handle = cuml.common.handle.Handle()

    def f(x: np.ndarray) -> np.ndarray:
        """The (batched) energy functional returning the negative
        loglikelihood (foreach series)."""

        # Recall: Maximimize LL means minimize negative
        n_llf = -(ll_f(num_batches, num_samples,
                       order,
                       d_y, x,
                       trans=True, handle=handle))
        return n_llf/(num_samples-1)

    # optimized finite differencing gradient for batches
    def gf(x):
        """The gradient of the (batched) energy functional."""
        # Recall: We maximize LL by minimizing -LL
        n_gllf = -ll_gf(num_batches, num_samples,
                        num_parameters, order,
                        d_y, x, h,
                        trans=True, handle=handle)
        return n_gllf/(num_samples-1)

    x0 = pack(p, d, q, num_batches, mu0, ar_params0, ma_params0)
    x0 = _batch_invtrans(p, d, q, num_batches, x0, handle)

    # check initial parameter sanity
    if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
        raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")

    # Optimize parameters by minimizing log likelihood.
    x, niter, flags = batched_fmin_lbfgs_b(f, x0, num_batches, gf,
                                           iprint=opt_disp, factr=1000)

    # Handle non-zero flags with Warning
    if (flags != 0).any():
        print("WARNING(`fit()`): Some batch members had optimizer problems.")

    Tx = _batch_trans(p, d, q, num_batches, x, handle)
    mu, ar, ma = unpack(p, d, q, num_batches, Tx)

    fit_model = ARIMAModel(order, mu, ar, ma, y)
    fit_model.niter = niter
    fit_model.d_y = d_y
    return fit_model


def grid_search(y_b, d=1, max_p=3, max_q=3, method="bic"):
    """Grid search to find optimal model order (p, q), weighing
    model complexity against likelihood.
    Optimality is based on minimizing BIC or AIC, which
    both sum negative log-likelihood against model complexity; Higher model
    complexity might yield a lower negative LL, but at higher `bic` due to
    complexity term.

    Parameters:

    y_b : array-like shape = (n_samples, n_series)
          The batched time-series data.
    d : int
        Trend (d>0) or not (d==0)
    max_p : int
            Maximum `p` in search
    max_q : int
            Maximum `q` in search
    method : str
             Complexity method to use ("bic" or "aic")

    Returns:

    Tuple of "best" order, mu, ar, and ma parameters with the
    corresponding IC for each series.

    (best_order: List[Tuple[int, int, int]],
     best_mu: array,
     best_ar_params: List[array],
     best_ma_params: List[array],
     best_ic: array)

    """

    num_batches = y_b.shape[1]
    best_ic = np.full(num_batches, np.finfo(np.float64).max/2)

    best_order = num_batches*[None]
    best_mu = np.zeros(num_batches)
    best_ar_params = num_batches*[None]
    best_ma_params = num_batches*[None]

    for p in range(0, max_p):
        for q in range(0, max_q):
            # skip 0,0 case
            if p == 0 and q == 0:
                continue

            mu0, ar0, ma0 = estimate_x0((p, d, q), y_b)

            b_model = fit(y_b, (p, d, q), mu0, ar0, ma0)

            if method == "aic":
                ic = b_model.aic
            elif method == "bic":
                ic = b_model.bic
            else:
                raise NotImplementedError("Method '{}' not supported".
                                          format(method))

            for (i, ic_i) in enumerate(ic):
                if ic_i < best_ic[i]:
                    best_order[i] = (p, d, q)
                    best_mu[i] = b_model.mu[i]

                    if p > 0:
                        best_ar_params[i] = b_model.ar_params[:, i]
                    else:
                        best_ar_params[i] = []
                    if q > 0:
                        best_ma_params[i] = b_model.ma_params[:, i]
                    else:
                        best_ma_params[i] = []

                    best_ic[i] = ic_i

    return (best_order, best_mu, best_ar_params, best_ma_params, best_ic)


def unpack(p, d, q, nb, x):
    """Unpack linearized parameters into mu, ar, and ma batched-groupings"""
    nvtx_range_push("unpack(x) -> (mu,ar,ma)")
    if type(x) is list or x.shape != (p + d + q, nb):
        x_mat = np.reshape(x, (p + d + q, nb), order='F')
    else:
        x_mat = x
    if d > 0:
        mu = x_mat[0]
    else:
        mu = np.zeros(nb)
    ar = x_mat[d:d+p]
    ma = x_mat[d+p:]

    nvtx_range_pop()
    return (mu, ar, ma)


def pack(p, d, q, nb, mu, ar, ma):
    """Pack mu, ar, and ma batched-groupings into a linearized vector `x`"""
    nvtx_range_push("pack(mu,ar,ma) -> x")
    if d > 0:
        return np.vstack((mu, ar, ma))
    else:
        return np.vstack((ar, ma))


def _batched_transform(p, d, q, nb, x, isInv, handle=None):
    cdef vector[double] vec_ar
    cdef vector[double] vec_ma
    cdef vector[double] vec_Tar
    cdef vector[double] vec_Tma

    nvtx_range_push("batched_transform")

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()
    Tx = np.zeros(nb*(d+p+q))

    cdef uintptr_t x_ptr = x.ctypes.data
    cdef uintptr_t Tx_ptr = Tx.ctypes.data
    batched_jones_transform(handle_[0], p, d, q, nb, isInv,
                            <double*>x_ptr, <double*>Tx_ptr)

    nvtx_range_pop()
    return (Tx)

def _model_complexity(order):
    (p, d, q) = order
    # complexity is number of parameters: mu + ar + ma
    return d + p + q


def _batch_trans(p, d, q, nb, x, handle=None):
    """Apply the stationarity/invertibility guaranteeing transform
    to batched-parameter vector x."""
    nvtx_range_push("jones trans")

    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(p, d, q, nb, x, False, handle)

    nvtx_range_pop()
    return Tx


def _batch_invtrans(p, d, q, nb, x, handle=None):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """
    nvtx_range_push("jones inv-trans")

    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(p, d, q, nb, x, True, handle)

    nvtx_range_pop()
    return Tx


def _batched_loglike(num_batches, nobs, order, y, x,
                     trans=False, handle=None):
    cdef vector[double] vec_loglike
    cdef vector[double] vec_y_cm
    cdef vector[double] vec_x

    nvtx_range_push("batched loglikelihood")
    p, d, q = order

    num_params = (p+d+q)

    vec_loglike.resize(num_batches)

    cdef uintptr_t d_y_ptr
    cdef uintptr_t d_x_ptr

    # note: make sure you explicitly have d_y_array. Otherwise it gets garbage
    # collected (I think).
    d_y_array, d_y_ptr, _, _, dtype_y = \
        input_to_dev_array(y, check_dtype=np.float64)
    d_x_array, d_x_ptr, _, _, _ = \
        input_to_dev_array(x, check_dtype=np.float64)

    d_vs = rmm.device_array((nobs - d, num_batches),
                            dtype=np.float64, order="F")
    cdef uintptr_t d_vs_ptr = get_dev_array_ptr(d_vs)

    if handle is None:
        handle = cuml.common.handle.Handle()

    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    if dtype_y != np.float64:
        raise \
            ValueError("Only 64-bit floating point inputs currently supported")

    batched_loglike(handle_[0],
                    <double*>d_y_ptr,
                    num_batches,
                    nobs,
                    p, d, q,
                    <double*>d_x_ptr,
                    vec_loglike,
                    <double*>d_vs_ptr,
                    trans)

    nvtx_range_pop()
    return vec_loglike


### deprecated ###

def _start_params(order, y_diff, p_best=1):
    """A quick approach to determine reasonable starting mu (trend),
    AR, and MA parameters"""

    # y is mutated so we need a copy
    y = np.copy(y_diff)
    nobs = len(y)

    p, q, d = order
    params_init = np.zeros(p+q+d)
    if d > 0:
        # center y (in `statsmodels`, this is result when exog = [1, 1, 1...])
        mean_y = np.mean(y)
        params_init[0] = mean_y
        y -= mean_y

    if p == 0 and q == 0:
        return params_init

    if p != 0:
        x = np.zeros((len(y) - p_best, p_best))
        # create lagged series set
        for lag in range(1, p_best+1):
            # create lag and trim appropriately from front
            # so they are all the same size
            x[:, lag-1] = y[p_best-lag:-lag].T

        # LS fit a*X - Y
        y_ar = y[p_best:]

        (ar_fit, _, _, _) = np.linalg.lstsq(x, y_ar.T, rcond=None)

        if q == 0:
            params_init[d:] = ar_fit
        else:
            residual = y[p_best:] - np.dot(x, ar_fit)

            assert p >= p_best
            p_diff = p - p_best

            x_resid = np.zeros((len(residual) - q - p_diff, q))
            x_ar2 = np.zeros((len(residual) - q - p_diff, p))

            # create lagged residual and ar term
            for lag in range(1, q+1):
                x_resid[:, lag-1] = (residual[q-lag:-lag].T)[p_diff:]
            for lag in range(1, p+1):
                x_ar2[:, lag-1] = (y[p-lag:-lag].T)[q:]

            X = np.column_stack((x_ar2, x_resid))

            # print(X)
            # print(y_ar[(q+p_diff):])

            (arma_fit, _, _, _) = np.linalg.lstsq(X, y_ar[(q+p_diff):].T,
                                                  rcond=None)

            print(arma_fit)

            params_init[d:] = arma_fit

    else:
        # case when p == 0 and q>0

        # when p==0, MA params are often -1
        params_init[d:] = -1*np.ones(q)

    return params_init

def _estimate_x0(order: Tuple[int, int, int],
                nb: int,
                yb, p_best=1) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Provide initial estimates to ARIMA parameters `mu`, `ar`, and `ma` for
    the batched input `yb`"""
    nvtx_range_push("estimate x0")
    (p, d, q) = order
    N = p + d + q
    x0 = np.zeros(N * nb)

    for ib in range(nb):
        y = yb[:, ib]

        if d == 1:
            yd = np.diff(y)
        else:
            yd = np.copy(y)

        x0ib = _start_params((p, q, d), yd, p_best)

        x0[ib*N:(ib+1)*N] = x0ib

    mu, ar, ma = unpack(p, d, q, nb, x0)

    nvtx_range_pop()

    return mu, ar, ma
