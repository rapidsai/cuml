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

from typing import List, Tuple, Dict, Mapping, Optional, Union
import cudf
from cuml.utils import get_dev_array_ptr, zeros

from cuml.common.cuda import nvtx_range_wrap

from cuml.common.base import Base

from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cuml.common.handle cimport cumlHandle
from libcpp.vector cimport vector

cdef extern from "arima/batched_arima.hpp" namespace "ML":
    void batched_loglike(
        cumlHandle& handle, const double* y, int batch_size, int nobs, int p,
        int d, int q, int P, int D, int Q, int s, int intercept,
        double* params, double* loglike, double* d_vs, bool trans,
        bool host_loglike)

    void predict_in_sample(
        cumlHandle& handle, const double* d_y, int batch_size, int nobs, int p,
        int d, int q, int P, int D, int Q, int s, int intercept,
        double* d_params, double* d_vs_ptr, double* d_y_p)

    void residual(
        cumlHandle& handle, const double* d_y, int batch_size, int nobs, int p,
        int d, int q, int P, int D, int Q, int s, int intercept,
        double* d_params, double* d_vs, bool trans)

    void forecast(
        cumlHandle& handle, int num_steps, int p, int d, int q, int P, int D,
        int Q, int s, int intercept, int batch_size, int nobs,
        const double* d_y, const double* d_y_prep, double* d_vs,
        double* d_params, double* d_y_fc)

    void information_criterion(
        cumlHandle& handle, const double* d_y, int batch_size, int nobs, int p,
        int d, int q, int P, int D, int Q, int s, int intercept,
        double* d_mu, double* d_ar, double* d_ma, double* d_sar,
        double* d_sma, double* ic, int ic_type)

    void cpp_estimate_x0 "estimate_x0" (
        cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
        double* d_sar, double* d_sma, const double* d_y, int batch_size,
        int nobs, int p, int d, int q, int P, int D, int Q, int s,
        int intercept)


cdef extern from "arima/batched_kalman.hpp" namespace "ML":

    void batched_jones_transform(
        cumlHandle& handle, int p, int q, int P, int Q, int intercept,
        int batchSize, bool isInv, const double* h_params, double* h_Tparams)


class ARIMA(Base):
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

        # fit a model
        model = arima.ARIMA(ys, (1,1,1), fit_intercept=True)
        model.fit()

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
    y : array-like (device or host)
        The time series series data. If given as `ndarray`, assumed to have
        each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF
        Series, NumPy ndarray, Numba device ndarray, cuda array interface
        compliant array like CuPy.
    order : Tuple[int, int, int]
        The ARIMA order (p, d, q) of the model
    fit_intercept : bool or int
        Whether to include a constant trend mu in the model
        Leave to None for automatic selection based on d and D
    TODO: update docs with seasonal parameters

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
                 y,
                 order: Tuple[int, int, int],
                 seasonal_order: Tuple[int, int, int, int]
                 = (0, 0, 0, 0),
                 fit_intercept=None,
                 handle=None):
        super().__init__(handle)

        # Check validity of the ARIMA order and seasonal order
        p, d, q = order
        P, D, Q, s = seasonal_order
        if P + D + Q > 0 and s < 2:
            raise ValueError("Invalid period for seasonal ARIMA: {}"
                             .format(s))
        if P + D + Q == 0 and s > 0:
            raise ValueError("Period specified for non-seasonal ARIMA: {}"
                             .format(s))
        if d + D > 2:
            raise ValueError("Invalid order. Required: d+D <= 2")
        if p + P > 8:
            raise ValueError("Invalid order. Required: p+P <= 8")
        if q + Q > 7:
            raise ValueError("Invalid order. Required: q+Q <= 7")

        self.order = order
        self.seasonal_order = seasonal_order
        if fit_intercept is None:
            # by default, use an intercept only with non differenced models
            fit_intercept = (order[1] + seasonal_order[1] == 0)
        self.intercept = int(fit_intercept)

        # Get device array. Float64 only for now.
        self.d_y, _, self.num_samples, self.batch_size, self.dtype \
            = input_to_dev_array(y, check_dtype=np.float64)

        self.yp = None
        self.niter = None  # number of iterations used during fit

    def __str__(self):
        if self.seasonal_order[3]:
            return "Batched ARIMA{}{}_{}".format(self.order,
                                                 self.seasonal_order[:3],
                                                 self.seasonal_order[3])
        else:
            return "Batched ARIMA{}".format(self.order)

    def _ic(self, ic_type: str):
        """Wrapper around C++ information_criterion
        """
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        # Convert host parameters to device parameters (will be removed later)
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        if self.intercept:
            d_mu, d_mu_ptr, _, _, _ = \
                input_to_dev_array(self.mu, check_dtype=np.float64)
        if p:
            d_ar, d_ar_ptr, _, _, _ = \
                input_to_dev_array(self.ar, check_dtype=np.float64)
        if q:
            d_ma, d_ma_ptr, _, _, _ = \
                input_to_dev_array(self.ma, check_dtype=np.float64)
        if P:
            d_sar, d_sar_ptr, _, _, _ = \
                input_to_dev_array(self.sar, check_dtype=np.float64)
        if Q:
            d_sma, d_sma_ptr, _, _, _ = \
                input_to_dev_array(self.sma, check_dtype=np.float64)

        cdef vector[double] ic
        ic.resize(self.batch_size)
        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        ic_name_to_number = {"aic": 0, "aicc": 1, "bic": 2}
        cdef int ic_type_id
        try:
            ic_type_id = ic_name_to_number[ic_type.lower()]
        except KeyError as e:
            raise NotImplementedError("IC type '{}' unknown". format(ic_type))

        information_criterion(handle_[0], <double*> d_y_ptr,
                              <int> self.batch_size, <int> self.num_samples,
                              <int> p, <int> d, <int> q, <int> P, <int> D,
                              <int> Q, <int> s, <int> self.intercept,
                              <double*> d_mu_ptr, <double*> d_ar_ptr,
                              <double*> d_ma_ptr, <double*> d_sar_ptr,
                              <double*> d_sma_ptr, <double*> ic.data(),
                              <int> ic_type_id)

        return ic

    @property
    def aic(self):
        """Akaike Information Criterion"""
        return self._ic("aic")

    @property
    def aicc(self):
        """Corrected Akaike Information Criterion"""
        return self._ic("aicc")

    @property
    def bic(self):
        """Bayesian Information Criterion"""
        return self._ic("bic")

    @property
    def complexity(self):
        """Model complexity (number of parameters)"""
        p, _, q = self.order
        P, _, Q, s = self.seasonal_order
        return p + P + q + Q + self.intercept

    def get_params(self) -> Dict[str, np.ndarray]:
        """TODO: docs
        """
        params = dict()
        names = ["mu", "ar", "ma", "sar", "sma"]
        criteria = [self.intercept, self.order[0], self.order[2],
                    self.seasonal_order[0], self.seasonal_order[2]]
        for i in range(len(names)):
            if criteria[i] > 0:
                params[names[i]] = self.__dict__[names[i]]
        return params

    def set_params(self, params: Mapping[str, object]):
        """TODO: docs
        """
        for param_name in ["mu", "ar", "ma", "sar", "sma"]:
            if param_name in params:
                array, _, _, _, _ = input_to_host_array(params[param_name])
                self.__dict__[param_name] = array

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
            model = ARIMA(ys, (1,1,1))
            model.fit()
            y_pred = model.predict_in_sample()
        """

        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        x = pack(self.order, self.seasonal_order, self.intercept,
                 self.batch_size, self.get_params())
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = \
            input_to_dev_array(x, check_dtype=np.float64)

        # allocate residual (vs) and prediction (y_p) device memory and get
        # pointers
        cdef uintptr_t d_vs_ptr
        cdef uintptr_t d_y_p_ptr
        d_vs = rmm.device_array((self.num_samples - d, self.batch_size),
                                dtype=np.float64, order="F")
        d_y_p = rmm.device_array((self.num_samples, self.batch_size),
                                 dtype=np.float64, order="F")
        d_vs_ptr = get_dev_array_ptr(d_vs)
        d_y_p_ptr = get_dev_array_ptr(d_y_p)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        predict_in_sample(handle_[0], <double*>d_y_ptr, <int> self.batch_size,
                          <int> self.num_samples, <int> p, <int> d, <int> q,
                          <int> P, <int> D, <int> Q, <int> s,
                          <int> self.intercept, <double*>d_params_ptr,
                          <double*>d_vs_ptr, <double*>d_y_p_ptr)

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
            model = ARIMA(ys, (1,1,1))
            model.fit()
            d_y_fc = model.forecast(10)
            y_fc = cuml.utils.input_to_host_array(d_y_fc)
            dx = xs[1] - xs[0]
            xfc = np.linspace(1, 1+50*dx, 50)
            plt.plot(xs, ys, xfc, yfc)

        """

        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        x = pack(self.order, self.seasonal_order, self.intercept,
                 self.batch_size, self.get_params())
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = \
            input_to_dev_array(x, check_dtype=np.float64)

        d_vs = rmm.device_array((self.num_samples - d,
                                 self.batch_size), dtype=np.float64,
                                order="F")
        cdef uintptr_t d_vs_ptr = get_dev_array_ptr(d_vs)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        residual(handle_[0], <double*>d_y_ptr, <int> self.batch_size,
                 <int> self.num_samples, <int> p, <int> d, <int> q, <int> P,
                 <int> D, <int> Q, <int> s, <int> self.intercept,
                 <double*>d_params_ptr, <double*>d_vs_ptr, False)

        cdef uintptr_t d_y_diff_ptr = <uintptr_t> NULL

        d_y_fc = rmm.device_array((nsteps, self.batch_size),
                                  dtype=np.float64, order="F")
        cdef uintptr_t d_y_fc_ptr = get_dev_array_ptr(d_y_fc)

        forecast(handle_[0], <int> nsteps, <int> p, <int> d, <int> q, <int> P,
                 <int> D, <int> Q, <int> s, <int> self.intercept,
                 <int> self.batch_size, <int> self.num_samples,
                 <double*> d_y_ptr, <double*>d_y_diff_ptr, <double*>d_vs_ptr,
                 <double*>d_params_ptr, <double*> d_y_fc_ptr)

        return d_y_fc

    @nvtx_range_wrap("estimate x0")
    def estimate_x0(self):
        """TODO: docs"""
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        # Create mu, ar and ma arrays
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        if self.intercept > 0:
            d_mu = zeros(self.batch_size, dtype=self.dtype)
            d_mu_ptr = get_dev_array_ptr(d_mu)
        if p > 0:
            d_ar = zeros((p, self.batch_size), dtype=self.dtype, order='F')
            d_ar_ptr = get_dev_array_ptr(d_ar)
        if q > 0:
            d_ma = zeros((q, self.batch_size), dtype=self.dtype, order='F')
            d_ma_ptr = get_dev_array_ptr(d_ma)
        if P > 0:
            d_sar = zeros((P, self.batch_size), dtype=self.dtype, order='F')
            d_sar_ptr = get_dev_array_ptr(d_sar)
        if Q > 0:
            d_sma = zeros((Q, self.batch_size), dtype=self.dtype, order='F')
            d_sma_ptr = get_dev_array_ptr(d_sma)

        # Call C++ function
        cpp_estimate_x0(handle_[0], <double*> d_mu_ptr, <double*> d_ar_ptr,
                        <double*> d_ma_ptr, <double*> d_sar_ptr,
                        <double*> d_sma_ptr, <double*> d_y_ptr,
                        <int> self.batch_size, <int> self.num_samples,
                        <int> p, <int> d, <int> q, <int> P, <int> D, <int> Q,
                        <int> s, <int> self.intercept)

        params = dict()
        if d > 0: params["mu"] = d_mu.copy_to_host()
        if p > 0: params["ar"] = d_ar.copy_to_host()
        if q > 0: params["ma"] = d_ma.copy_to_host()
        if P > 0: params["sar"] = d_sar.copy_to_host()
        if Q > 0: params["sma"] = d_sma.copy_to_host()
        self.set_params(params)

    # TODO: missing range wraps?
    def fit(self,
            start_params: Optional[Mapping[str, object]]=None,
            opt_disp: int=-1,
            h: float=1e-9):
        """Fits the ARIMA model to each time-series for the given initial
        parameter estimates.

        Parameters
        ----------
        start_params : TODO: docs
        opt_disp : int
                Fit diagnostic level (for L-BFGS solver):
                * `-1` for no output,
                * `0<n<100` for output every `n` steps
                * `n>100` for more detailed output
        h : float
            Finite-differencing step size. The gradient is computed
            using second-order differencing:
                    f(x+h) - f(x - h)
                g = ----------------- + O(h^2)
                            2 * h
        """
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        if start_params is None:
            self.estimate_x0()
        else:
            self.set_params(start_params)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        def f(x: np.ndarray) -> np.ndarray:
            """The (batched) energy functional returning the negative
            loglikelihood (foreach series)."""

            # Recall: Maximimize LL means minimize negative
            n_llf = -(ll_f(self.batch_size, self.num_samples, self.order,
                           self.seasonal_order, self.intercept, self.d_y, x,
                           trans=True, handle=self.handle))
            return n_llf/(self.num_samples-1)

        # optimized finite differencing gradient for batches
        def gf(x):
            """The gradient of the (batched) energy functional."""
            # Recall: We maximize LL by minimizing -LL
            n_gllf = -ll_gf(self.batch_size, self.num_samples,
                            self.complexity, self.order, self.seasonal_order,
                            self.intercept, self.d_y, x, h, trans=True,
                            handle=self.handle)
            return n_gllf/(self.num_samples-1)

        x0 = pack(self.order, self.seasonal_order, self.intercept,
                  self.batch_size, self.get_params())
        x0 = _batch_invtrans(self.order, self.seasonal_order, self.intercept,
                             self.batch_size, x0, self.handle)

        # check initial parameter sanity
        if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")

        # Optimize parameters by minimizing log likelihood.
        x, niter, flags = batched_fmin_lbfgs_b(f, x0, self.batch_size, gf,
                                               iprint=opt_disp, factr=1000)

        # Handle non-zero flags with Warning
        if (flags != 0).any():
            print("WARNING(`fit()`): Some batch members had optimizer problems.")

        Tx = _batch_trans(self.order, self.seasonal_order, self.intercept,
                          self.batch_size, x, self.handle)

        self.set_params(unpack(self.order, self.seasonal_order, self.intercept,
                               self.batch_size, Tx))
        self.niter = niter


@nvtx_range_wrap("ll_f")
def ll_f(batch_size, nobs, order, seasonal_order, intercept, y, x, trans=True,
         handle=None):
    """Computes batched loglikelihood for given parameters and series.

    Parameters:
    ----------
    batch_size : int
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

    vec_loglike = _batched_loglike(batch_size, nobs, order, seasonal_order,
                                   intercept, y, x, trans, handle)

    loglike = np.zeros(batch_size)
    for i in range(batch_size):
        loglike[i] = vec_loglike[i]

    return loglike


@nvtx_range_wrap("ll_gf")
def ll_gf(batch_size, nobs, num_parameters, order, seasonal_order, intercept,
          y, x, h=1e-8, trans=True, handle=None):
    """Computes gradient (via finite differencing) of the batched
    loglikelihood.

    Parameters:
    ----------
    batch_size : int
                  Number of series
    nobs : int
           Number of samples in each series (identical across series)
    num_parameters : int
                     The number of parameters per series
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
    fd = np.zeros(num_parameters)

    grad = np.zeros(len(x))

    # Get current numpy error level and change all to 'raise'
    err_lvl = np.seterr(all='raise')

    assert (len(x) / num_parameters) == float(batch_size)
    for i in range(num_parameters):
        fd[i] = h

        # duplicate the perturbation across batches (they are independent)
        fdph = np.tile(fd, batch_size)

        # reset perturbation
        fd[i] = 0.0

        ll_b_ph = ll_f(batch_size, nobs, order, seasonal_order, intercept, y,
                       x+fdph, trans=trans, handle=handle)
        ll_b_mh = ll_f(batch_size, nobs, order, seasonal_order, intercept, y,
                       x-fdph, trans=trans, handle=handle)

        # first derivative second order accuracy
        grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)

        if batch_size == 1:
            grad[i] = grad_i_b
        else:
            assert len(grad[i::num_parameters]) == len(grad_i_b)
            # Distribute the result to all batches
            grad[i::num_parameters] = grad_i_b

    # Reset numpy error levels
    np.seterr(**err_lvl)

    return grad


# TODO: later replace with auto-arima?
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
     best_ar: List[array],
     best_ma: List[array],
     best_ic: array)

    """

    batch_size = y_b.shape[1]
    best_ic = np.full(batch_size, np.finfo(np.float64).max/2)

    best_order = batch_size*[None]
    best_mu = np.zeros(batch_size)
    best_ar = batch_size*[None]
    best_ma = batch_size*[None]

    for p in range(0, max_p):
        for q in range(0, max_q):
            # skip 0,0 case
            if p == 0 and q == 0:
                continue

            b_model = ARIMA(y_b, (p, d, q), fit_intercept=True)
            b_model.fit()

            ic = b_model._ic(method)

            for (i, ic_i) in enumerate(ic):
                if ic_i < best_ic[i]:
                    best_order[i] = (p, d, q)
                    best_mu[i] = b_model.mu[i]

                    if p > 0:
                        best_ar[i] = b_model.ar[:, i]
                    else:
                        best_ar[i] = []
                    if q > 0:
                        best_ma[i] = b_model.ma[:, i]
                    else:
                        best_ma[i] = []

                    best_ic[i] = ic_i

    return (best_order, best_mu, best_ar, best_ma, best_ic)


@nvtx_range_wrap("unpack")
def unpack(order: Tuple[int, int, int],
           seasonal_order: Tuple[int, int, int, int],
           k: int, nb: int, x: Union[list, np.ndarray]
           ) -> Dict[str, np.ndarray]:
    """Unpack linearized parameter vector `x` into separarate arrays
       Parameters: order, seasonal order, intercept, batch size, vector
    """
    p, _, q = order
    P, _, Q, _ = seasonal_order
    N = p + q + P + Q + k

    if type(x) is list or x.shape != (N, nb):
        x_mat = np.reshape(x, (N, nb), order='F')
    else:
        x_mat = x

    params = dict()
    if k > 0: params["mu"] = x_mat[0]
    if p > 0: params["ar"] = x_mat[k:k+p]
    if q > 0: params["ma"] = x_mat[k+p:k+p+q]
    if P > 0: params["sar"] = x_mat[k+p+q:k+p+q+P]
    if Q > 0: params["sma"] = x_mat[k+p+q+P:k+p+q+P+Q]

    return params


@nvtx_range_wrap("pack")
def pack(order: Tuple[int, int, int],
         seasonal_order: Tuple[int, int, int, int],
         k: int, nb: int, params: Mapping[str, np.ndarray]
         ) -> np.ndarray:
    """Pack parameters into a linearized vector `x`
       Parameters: order, seasonal order, intercept, batch size, parameters
    """
    p, _, q = order
    P, _, Q, _ = seasonal_order
    N = p + q + P + Q + k

    x = np.zeros((N, nb), order='F')  # 2D array for convenience

    if k > 0: x[0] = params["mu"]
    if p > 0: x[k:k+p] = params["ar"]
    if q > 0: x[k+p:k+p+q] = params["ma"]
    if P > 0: x[k+p+q:k+p+q+P] = params["sar"]
    if Q > 0: x[k+p+q+P:k+p+q+P+Q] = params["sma"]

    return x.reshape(N * nb, order='F')  # return 1D shape


@nvtx_range_wrap("batched_transform")
def _batched_transform(order, seasonal_order, intercept, nb, x, isInv,
                       handle=None):
    p, _, q = order
    P, _, Q, _ = seasonal_order

    cdef vector[double] vec_ar
    cdef vector[double] vec_ma
    cdef vector[double] vec_Tar
    cdef vector[double] vec_Tma

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()
    Tx = np.zeros(nb * (p + q + P + Q + intercept))

    cdef uintptr_t x_ptr = x.ctypes.data
    cdef uintptr_t Tx_ptr = Tx.ctypes.data
    batched_jones_transform(handle_[0], p, q, P, Q, intercept, nb, isInv,
                            <double*>x_ptr, <double*>Tx_ptr)

    return (Tx)


@nvtx_range_wrap("jones trans")
def _batch_trans(order, seasonal_order, intercept, nb, x, handle=None):
    """Apply the stationarity/invertibility guaranteeing transform
    to batched-parameter vector x."""
    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(order, seasonal_order, intercept, nb, x, False,
                            handle)

    return Tx


@nvtx_range_wrap("jones inv-trans")
def _batch_invtrans(order, seasonal_order, intercept, nb, x, handle=None):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """

    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(order, seasonal_order, intercept, nb, x, True,
                            handle)

    return Tx


@nvtx_range_wrap("batched loglikelihood")
def _batched_loglike(batch_size, nobs, order, seasonal_order, intercept, y, x,
                     trans=False, handle=None):
    cdef vector[double] vec_loglike
    cdef vector[double] vec_y_cm
    cdef vector[double] vec_x

    p, d, q = order
    P, D, Q, s = seasonal_order

    num_params = p + q + P + Q + intercept

    vec_loglike.resize(batch_size)

    cdef uintptr_t d_y_ptr
    cdef uintptr_t d_x_ptr

    # note: make sure you explicitly have d_y_array. Otherwise it gets garbage
    # collected (I think).
    d_y_array, d_y_ptr, _, _, dtype_y = \
        input_to_dev_array(y, check_dtype=np.float64)
    d_x_array, d_x_ptr, _, _, _ = \
        input_to_dev_array(x, check_dtype=np.float64)

    d_vs = rmm.device_array((nobs - d, batch_size),
                            dtype=np.float64, order="F")
    cdef uintptr_t d_vs_ptr = get_dev_array_ptr(d_vs)

    if handle is None:
        handle = cuml.common.handle.Handle()

    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    if dtype_y != np.float64:
        raise \
            ValueError("Only 64-bit floating point inputs currently supported")

    batched_loglike(handle_[0], <double*> d_y_ptr, <int> batch_size,
                    <int> nobs, <int> p, <int> d, <int> q, <int> P, <int> D,
                    <int> Q, <int> s, <int> intercept, <double*> d_x_ptr,
                    <double*> vec_loglike.data(), <double*> d_vs_ptr, trans,
                    <bool> True)

    return vec_loglike
