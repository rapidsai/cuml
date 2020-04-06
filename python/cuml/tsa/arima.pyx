#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import sys

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
from cuml.utils import rmm_cupy_ary, has_scipy

from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cuml.common.handle cimport cumlHandle
from libcpp.vector cimport vector


cdef extern from "cuml/tsa/arima_common.h" namespace "ML":
    ctypedef struct ARIMAOrder:
        int p  # Basic order
        int d
        int q
        int P  # Seasonal order
        int D
        int Q
        int s  # Seasonal period
        int k  # Fit intercept?

    cdef cppclass ARIMAParams[DataT]:
        DataT* mu
        DataT* ar
        DataT* ma
        DataT* sar
        DataT* sma
        DataT* sigma2


cdef extern from "cuml/tsa/batched_arima.hpp" namespace "ML":
    void batched_loglike(
        cumlHandle& handle, const double* y, int batch_size, int nobs,
        const ARIMAOrder& order, const double* params, double* loglike,
        double* d_vs, bool trans, bool host_loglike)

    void cpp_predict "predict" (
        cumlHandle& handle, const double* d_y, int batch_size, int nobs,
        int start, int end, const ARIMAOrder& order,
        const ARIMAParams[double]& params, double* d_vs_ptr, double* d_y_p)

    void information_criterion(
        cumlHandle& handle, const double* d_y, int batch_size, int nobs,
        const ARIMAOrder& order, const ARIMAParams[double]& params,
        double* ic, int ic_type)

    void estimate_x0(
        cumlHandle& handle, ARIMAParams[double]& params, const double* d_y,
        int batch_size, int nobs, const ARIMAOrder& order)


cdef extern from "cuml/tsa/batched_kalman.hpp" namespace "ML":

    void batched_jones_transform(
        cumlHandle& handle, const ARIMAOrder& order, int batchSize,
        bool isInv, const double* h_params, double* h_Tparams)


class ARIMA(Base):
    r"""Implements a batched ARIMA model for in- and out-of-sample
    time-series prediction, with support for seasonality (SARIMA)

    ARIMA stands for Auto-Regressive Integrated Moving Average.
    See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

    This class can fit an ARIMA(p,d,q) or ARIMA(p,d,q)(P,D,Q)_s model to a
    batch of time series of the same length with no missing values.
    The implementation is designed to give the best performance when using
    large batches of time series.

    Examples
    ---------
    .. code-block:: python

        import numpy as np
        from cuml.tsa import arima

        # Create seasonal data with a trend, a seasonal pattern and noise
        n_obs = 100
        np.random.seed(12)
        x = np.linspace(0, 1, n_obs)
        pattern = np.array([[0.05, 0.0], [0.07, 0.03],
                            [-0.03, 0.05], [0.02, 0.025]])
        noise = np.random.normal(scale=0.01, size=(n_obs, 2))
        y = (np.column_stack((0.5*x, -0.25*x)) + noise
             + np.tile(pattern, (25, 1)))

        # Fit a seasonal ARIMA model
        model = arima.ARIMA(y, (0,1,1), (0,1,1,4), fit_intercept=False)
        model.fit()

        # Forecast
        fc = model.forecast(10).copy_to_host()
        print(fc)

    Output:

    .. code-block:: python

        [[ 0.55204599 -0.25681163]
         [ 0.57430705 -0.2262438 ]
         [ 0.48120315 -0.20583011]
         [ 0.535594   -0.24060046]
         [ 0.57207541 -0.26695497]
         [ 0.59433647 -0.23638713]
         [ 0.50123257 -0.21597344]
         [ 0.55562342 -0.25074379]
         [ 0.59210483 -0.27709831]
         [ 0.61436589 -0.24653047]]

    Parameters
    ----------
    y : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    order : Tuple[int, int, int]
        The ARIMA order (p, d, q) of the model
    seasonal_order: Tuple[int, int, int, int]
        The seasonal ARIMA order (P, D, Q, s) of the model
    fit_intercept : bool or int
        Whether to include a constant trend mu in the model
        Leave to None for automatic selection based on the model order
    handle: cuml.Handle
        If it is None, a new one is created just for this instance

    Attributes
    ----------
    order : Tuple[int, int, int]
        The ARIMA order (p, d, q) of the model
    seasonal_order: Tuple[int, int, int, int]
        The seasonal ARIMA order (P, D, Q, s) of the model
    intercept : bool or int
        Whether the model includes a constant trend mu
    d_y: device array
        Time series data on device
    num_samples: int
        Number of observations
    batch_size: int
        Number of time series in the batch
    dtype: numpy.dtype
        Floating-point type of the data and parameters
    niter: numpy.ndarray
        After fitting, contains the number of iterations before convergence
        for each time series.

    Performance
    -----------
    Let `r=max(p+s*P, q+s*Q+1)`. The device memory used for most operations
    is `O(batch_size*n_obs + batch_size*r^2)`. The execution time is a linear
    function of `n_obs` and `batch_size` (if `batch_size` is large), but grows
    very fast with `r`.

    The performance is optimized for very large batch sizes (e.g thousands of
    series).

    References
    ----------
    This class is heavily influenced by the Python library `statsmodels`,
    particularly `statsmodels.tsa.statespace.sarimax.SARIMAX`.
    See https://www.statsmodels.org/stable/statespace.html

    Additionally the following book is a useful reference:
    "Time Series Analysis by State Space Methods",
    J. Durbin, S.J. Koopman, 2nd Edition (2012).
    """

    def __init__(self,
                 y,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int]
                 = (0, 0, 0, 0),
                 fit_intercept=None,
                 handle=None):

        if not has_scipy():
            raise RuntimeError("Scipy is needed to run cuML's ARIMA estimator."
                               " Please install it to enable ARIMA "
                               "estimation.")
        super().__init__(handle)

        cdef ARIMAOrder cpp_order
        cpp_order.p, cpp_order.d, cpp_order.q = order
        cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
        if fit_intercept is None:
            # by default, use an intercept only with non differenced models
            fit_intercept = (order[1] + seasonal_order[1] == 0)
        cpp_order.k = int(fit_intercept)

        self.order = cpp_order

        # Check validity of the ARIMA order and seasonal order
        p, d, q = order
        P, D, Q, s = seasonal_order
        if P + D + Q > 0 and s < 2:
            raise ValueError("ERROR: Invalid period for seasonal ARIMA: {}"
                             .format(s))
        if P + D + Q == 0 and s > 0:
            raise ValueError("ERROR: Period specified for non-seasonal ARIMA:"
                             " {}".format(s))
        if d + D > 2:
            raise ValueError("ERROR: Invalid order. Required: d+D <= 2")
        if s != 0 and (p >= s or q >= s):
            raise ValueError("ERROR: Invalid order. Required: s > p, s > q")
        if p + q + P + Q + cpp_order.k == 0:
            raise ValueError("ERROR: Invalid order. At least one parameter"
                             " among p, q, P, Q and fit_intercept must be"
                             " non-zero")
        if p > 4 or P > 4 or q > 4 or Q > 4:
            raise ValueError("ERROR: Invalid order. Required: p,q,P,Q <= 4")

        # Get device array. Float64 only for now.
        self.d_y, _, self.n_obs, self.batch_size, self.dtype \
            = input_to_dev_array(y, check_dtype=np.float64)

        if self.n_obs < d + s * D + 1:
            raise ValueError("ERROR: Number of observations too small for the"
                             " given order")

        self.niter = None  # number of iterations used during fit

    def __str__(self):
        if self.seasonal_order[3]:
            return "Batched ARIMA{}{}_{}".format(self.order,
                                                 self.seasonal_order[:3],
                                                 self.seasonal_order[3])
        else:
            return "Batched ARIMA{}".format(self.order)

    @nvtx_range_wrap
    def _ic(self, ic_type: str):
        """Wrapper around C++ information_criterion
        """
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef ARIMAOrder order = self.order

        # Convert host parameters to device parameters
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order.k:
            d_mu, d_mu_ptr, _, _, _ = \
                input_to_dev_array(self.mu, check_dtype=np.float64)
        if order.p:
            d_ar, d_ar_ptr, _, _, _ = \
                input_to_dev_array(self.ar, check_dtype=np.float64)
        if order.q:
            d_ma, d_ma_ptr, _, _, _ = \
                input_to_dev_array(self.ma, check_dtype=np.float64)
        if order.P:
            d_sar, d_sar_ptr, _, _, _ = \
                input_to_dev_array(self.sar, check_dtype=np.float64)
        if order.Q:
            d_sma, d_sma_ptr, _, _, _ = \
                input_to_dev_array(self.sma, check_dtype=np.float64)
        d_sigma2, d_sigma2_ptr, _, _, _ = \
            input_to_dev_array(self.sigma2, check_dtype=np.float64)

        cdef ARIMAParams[double] cpp_params
        cpp_params.mu = <double*> d_mu_ptr
        cpp_params.ar = <double*> d_ar_ptr
        cpp_params.ma = <double*> d_ma_ptr
        cpp_params.sar = <double*> d_sar_ptr
        cpp_params.sma = <double*> d_sma_ptr
        cpp_params.sigma2 = <double*> d_sigma2_ptr

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
                              <int> self.batch_size, <int> self.n_obs,
                              <ARIMAOrder> order, cpp_params,
                              <double*> ic.data(), <int> ic_type_id)

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
        cdef ARIMAOrder order = self.order
        return order.p + order.P + order.q + order.Q + order.k + 1

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get the parameters of the model

        Returns:
        --------
        params: Dict[str, np.ndarray]
            A dictionary of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu parameters and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
        """
        cdef ARIMAOrder order = self.order
        params = dict()
        names = ["mu", "ar", "ma", "sar", "sma", "sigma2"]
        criteria = [order.k, order.p, order.q, order.P, order.Q, True]
        for i in range(len(names)):
            if criteria[i] > 0:
                params[names[i]] = getattr(self, names[i])
        return params

    def set_params(self, params: Mapping[str, object]):
        """Set the parameters of the model

        Parameters:
        --------
        params: Mapping[str, np.ndarray]
            A mapping (e.g dictionary) of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu parameters and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
        """
        for param_name in ["mu", "ar", "ma", "sar", "sma", "sigma2"]:
            if param_name in params:
                array, _, _, _, _ = input_to_host_array(params[param_name])
                setattr(self, param_name, array)

    @nvtx_range_wrap
    def predict(self, start=0, end=None):
        """Compute in-sample and/or out-of-sample prediction for each series

        Parameters:
        -----------
        start: int
            Index where to start the predictions (0 <= start <= num_samples)
        end:
            Index where to end the predictions, excluded (end > start)

        Returns:
        --------
        y_p : array-like (device)
            Predictions. Shape = (end - start, batch_size)

        Example:
        --------
        .. code-block:: python
            from cuml.tsa.arima import fit
            ...
            model = ARIMA(ys, (1,1,1))
            model.fit()
            y_pred = model.predict().copy_to_host()
        """
        cdef ARIMAOrder order = self.order

        if start < 0:
            raise ValueError("ERROR(`predict`): start < 0")
        elif start > self.n_obs:
            raise ValueError("ERROR(`predict`): There can't be a gap between"
                             " the data and the prediction")
        elif end <= start:
            raise ValueError("ERROR(`predict`): end <= start")
        elif start < order.d + order.D * order.s:
            print("WARNING(`predict`): predictions before {} are undefined,"
                  " will be set to NaN".format(order.d + order.D * order.s),
                  file=sys.stderr)

        if end is None:
            end = self.n_obs

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order.k:
            d_mu, d_mu_ptr, _, _, _ = \
                input_to_dev_array(self.mu, check_dtype=np.float64)
        if order.p:
            d_ar, d_ar_ptr, _, _, _ = \
                input_to_dev_array(self.ar, check_dtype=np.float64)
        if order.q:
            d_ma, d_ma_ptr, _, _, _ = \
                input_to_dev_array(self.ma, check_dtype=np.float64)
        if order.P:
            d_sar, d_sar_ptr, _, _, _ = \
                input_to_dev_array(self.sar, check_dtype=np.float64)
        if order.Q:
            d_sma, d_sma_ptr, _, _, _ = \
                input_to_dev_array(self.sma, check_dtype=np.float64)
        d_sigma2, d_sigma2_ptr, _, _, _ = \
            input_to_dev_array(self.sigma2, check_dtype=np.float64)

        cdef ARIMAParams[double] cpp_params
        cpp_params.mu = <double*> d_mu_ptr
        cpp_params.ar = <double*> d_ar_ptr
        cpp_params.ma = <double*> d_ma_ptr
        cpp_params.sar = <double*> d_sar_ptr
        cpp_params.sma = <double*> d_sma_ptr
        cpp_params.sigma2 = <double*> d_sigma2_ptr

        predict_size = end - start

        # allocate residual (vs) and prediction (y_p) device memory and get
        # pointers
        cdef uintptr_t d_vs_ptr
        cdef uintptr_t d_y_p_ptr
        d_vs = rmm.device_array((self.n_obs - order.d - order.D * order.s,
                                 self.batch_size), dtype=np.float64, order="F")
        d_y_p = rmm.device_array((predict_size, self.batch_size),
                                 dtype=np.float64, order="F")
        d_vs_ptr = get_dev_array_ptr(d_vs)
        d_y_p_ptr = get_dev_array_ptr(d_y_p)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        cpp_predict(handle_[0], <double*>d_y_ptr, <int> self.batch_size,
                    <int> self.n_obs, <int> start, <int> end, order,
                    cpp_params, <double*>d_vs_ptr, <double*>d_y_p_ptr)

        return d_y_p

    @nvtx_range_wrap
    def forecast(self, nsteps: int):
        """Forecast the given model `nsteps` into the future.

        Parameters:
        ----------
        nsteps : int
            The number of steps to forecast beyond end of the given series

        Returns:
        --------
        y_fc : array-like
               Forecasts. Shape = (nsteps, batch_size)

        Example:
        --------
        .. code-block:: python
            from cuml.tsa.arima import fit
            import cuml
            ...
            model = ARIMA(ys, (1,1,1))
            model.fit()
            y_fc = model.forecast(10).copy_to_host()
        """

        return self.predict(self.n_obs, self.n_obs + nsteps)

    @nvtx_range_wrap
    def _estimate_x0(self):
        """Internal method. Estimate initial parameters of the model.
        """
        cdef ARIMAOrder order = self.order

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        # Create mu, ar and ma arrays
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order.k:
            d_mu = zeros(self.batch_size, dtype=self.dtype)
            d_mu_ptr = get_dev_array_ptr(d_mu)
        if order.p:
            d_ar = zeros((order.p, self.batch_size), dtype=self.dtype,
                         order='F')
            d_ar_ptr = get_dev_array_ptr(d_ar)
        if order.q:
            d_ma = zeros((order.q, self.batch_size), dtype=self.dtype,
                         order='F')
            d_ma_ptr = get_dev_array_ptr(d_ma)
        if order.P:
            d_sar = zeros((order.P, self.batch_size), dtype=self.dtype,
                          order='F')
            d_sar_ptr = get_dev_array_ptr(d_sar)
        if order.Q:
            d_sma = zeros((order.Q, self.batch_size), dtype=self.dtype,
                          order='F')
            d_sma_ptr = get_dev_array_ptr(d_sma)
        d_sigma2 = zeros(self.batch_size, dtype=self.dtype)
        d_sigma2_ptr = get_dev_array_ptr(d_sigma2)

        cdef ARIMAParams[double] cpp_params
        cpp_params.mu = <double*> d_mu_ptr
        cpp_params.ar = <double*> d_ar_ptr
        cpp_params.ma = <double*> d_ma_ptr
        cpp_params.sar = <double*> d_sar_ptr
        cpp_params.sma = <double*> d_sma_ptr
        cpp_params.sigma2 = <double*> d_sigma2_ptr

        # Call C++ function
        estimate_x0(handle_[0], cpp_params, <double*> d_y_ptr,
                    <int> self.batch_size, <int> self.n_obs, order)

        params = dict()
        if order.k:
            params["mu"] = d_mu.copy_to_host()
        if order.p:
            params["ar"] = d_ar.copy_to_host()
        if order.q:
            params["ma"] = d_ma.copy_to_host()
        if order.P:
            params["sar"] = d_sar.copy_to_host()
        if order.Q:
            params["sma"] = d_sma.copy_to_host()
        params["sigma2"] = d_sigma2.copy_to_host()
        self.set_params(params)

    @nvtx_range_wrap
    def fit(self,
            start_params: Optional[Mapping[str, object]] = None,
            opt_disp: int = -1,
            h: float = 1e-9,
            maxiter=1000):
        """Fit the ARIMA model to each time series.

        Parameters
        ----------
        start_params : Mapping[str, object] (optional)
            A mapping (e.g dictionary) of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu parameters and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
            Pass None for automatic estimation (recommended)
        opt_disp : int
            Fit diagnostic level (for L-BFGS solver):
             * `-1` for no output (default)
             * `0<n<100` for output every `n` steps
             * `n>100` for more detailed output
        h : float
            Finite-differencing step size. The gradient is computed
            using second-order differencing:
                    f(x+h) - f(x - h)
                g = ----------------- + O(h^2)
                          2 * h
        maxiter : int
            Maximum number of iterations of L-BFGS-B
        """
        if start_params is None:
            self._estimate_x0()
        else:
            self.set_params(start_params)

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)

        def f(x: np.ndarray) -> np.ndarray:
            """The (batched) energy functional returning the negative
            log-likelihood (foreach series)."""
            # Recall: We maximize LL by minimizing -LL
            n_llf = -self._loglike(x, trans=True)
            return n_llf / (self.n_obs - 1)

        # Optimized finite differencing gradient for batches
        def gf(x):
            """The gradient of the (batched) energy functional."""
            # Recall: We maximize LL by minimizing -LL
            n_gllf = -self._loglike_grad(x, h, trans=True)
            return n_gllf / (self.n_obs - 1)

        x0 = self._batched_transform(self.pack(), True)

        # check initial parameter sanity
        if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")

        # Optimize parameters by minimizing log likelihood.
        x, niter, flags = batched_fmin_lbfgs_b(f, x0, self.batch_size, gf,
                                               iprint=opt_disp, factr=1000,
                                               maxiter=maxiter)

        # Handle non-zero flags with Warning
        if (flags != 0).any():
            print("WARNING(`fit`): Some batch members had optimizer problems.",
                  file=sys.stderr)

        self.unpack(self._batched_transform(x))
        self.niter = niter

    @nvtx_range_wrap
    def _loglike(self, x, trans=True):
        """Compute the batched log-likelihood for the given parameters.

        Parameters:
        ----------
        x : array-like
            Packed parameter array, grouped by series
        trans : bool
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.

        Returns:
        --------
        loglike : numpy.ndarray
            Batched log-likelihood. Shape: (batch_size,)
        """
        cdef vector[double] vec_loglike
        vec_loglike.resize(self.batch_size)

        cdef ARIMAOrder order = self.order

        cdef uintptr_t d_x_ptr
        d_x_array, d_x_ptr, _, _, _ = \
            input_to_dev_array(x, check_dtype=np.float64, order='C')

        cdef uintptr_t d_y_ptr = get_dev_array_ptr(self.d_y)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        d_vs = rmm.device_array((self.n_obs - order.d - order.D * order.s,
                                 self.batch_size),
                                dtype=np.float64, order="F")
        cdef uintptr_t d_vs_ptr = get_dev_array_ptr(d_vs)

        batched_loglike(handle_[0], <double*> d_y_ptr, <int> self.batch_size,
                        <int> self.n_obs, order, <double*> d_x_ptr,
                        <double*> vec_loglike.data(), <double*> d_vs_ptr,
                        <bool> trans, <bool> True)

        return np.array(vec_loglike, dtype=np.float64)

    @nvtx_range_wrap
    def _loglike_grad(self, x, h=1e-8, trans=True):
        """Compute the gradient (via finite differencing) of the batched
        log-likelihood.

        Parameters:
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        h : float
            The finite-difference stepsize
        trans : bool
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.

        Returns:
        --------
        grad : numpy.ndarray
            Batched log-likelihood gradient. Shape: (n_params * batch_size,)
            where n_params is the complexity of the model
        """
        N = self.complexity
        assert len(x) == N * self.batch_size

        fd = np.zeros(N)
        grad = np.zeros(len(x))

        # Get current numpy error level and change all to 'raise'
        err_lvl = np.seterr(all='raise')

        for i in range(N):
            fd[i] = h

            # duplicate the perturbation across batches (they are independent)
            fdph = np.tile(fd, self.batch_size)

            # reset perturbation
            fd[i] = 0.0

            ll_b_ph = self._loglike(x + fdph, trans=trans)
            ll_b_mh = self._loglike(x - fdph, trans=trans)

            # first derivative second order accuracy
            grad_i_b = (ll_b_ph - ll_b_mh) / (2 * h)

            if self.batch_size == 1:
                grad[i] = grad_i_b
            else:
                assert len(grad[i::N]) == len(grad_i_b)
                # Distribute the result to all batches
                grad[i::N] = grad_i_b

        # Reset numpy error levels
        np.seterr(**err_lvl)

        return grad

    @property
    def llf(self):
        """Log-likelihood of a fit model. Shape: (batch_size,)
        """
        return self._loglike(self.pack(), trans=False)

    @nvtx_range_wrap
    def unpack(self, x: Union[list, np.ndarray]):
        """Unpack linearized parameter vector `x` into the separate
        parameter arrays of the model

        Parameters:
        -----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        p, q, P, Q, k = (order.p, order.q, order.P, order.Q, order.k)
        N = self.complexity

        if type(x) is list or x.shape != (N, self.batch_size):
            x_mat = np.reshape(x, (N, self.batch_size), order='F')
        else:
            x_mat = x

        params = dict()
        # Note: going through np.array to avoid getting incorrect strides when
        # batch_size is 1
        if k > 0:
            params["mu"] = np.array(x_mat[0], order='F')
        if p > 0:
            params["ar"] = np.array(x_mat[k:k+p], order='F')
        if q > 0:
            params["ma"] = np.array(x_mat[k+p:k+p+q], order='F')
        if P > 0:
            params["sar"] = np.array(x_mat[k+p+q:k+p+q+P], order='F')
        if Q > 0:
            params["sma"] = np.array(x_mat[k+p+q+P:k+p+q+P+Q], order='F')
        params["sigma2"] = np.array(x_mat[k+p+q+P+Q], order='F')

        self.set_params(params)

    @nvtx_range_wrap
    def pack(self) -> np.ndarray:
        """Pack parameters of the model into a linearized vector `x`

        Returns:
        -----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        p, q, P, Q, k = (order.p, order.q, order.P, order.Q, order.k)
        N = self.complexity

        params = self.get_params()

        # 2D array for convenience
        x = np.zeros((N, self.batch_size), order='F')

        if k > 0:
            x[0] = params["mu"]
        if p > 0:
            x[k:k+p] = params["ar"]
        if q > 0:
            x[k+p:k+p+q] = params["ma"]
        if P > 0:
            x[k+p+q:k+p+q+P] = params["sar"]
        if Q > 0:
            x[k+p+q+P:k+p+q+P+Q] = params["sma"]
        x[k+p+q+P+Q] = params["sigma2"]

        return x.reshape(N * self.batch_size, order='F')  # return 1D shape

    @nvtx_range_wrap
    def _batched_transform(self, x, isInv=False):
        """Applies Jones transform or inverse transform to a parameter vector

        Parameters:
        -----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)

        Returns:
        -----------
        Tx : array-like
            Packed transformed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        N = self.complexity

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        Tx = np.zeros(self.batch_size * N)

        cdef uintptr_t x_ptr = x.ctypes.data
        cdef uintptr_t Tx_ptr = Tx.ctypes.data
        batched_jones_transform(handle_[0], order, <int> self.batch_size,
                                <bool> isInv, <double*>x_ptr, <double*>Tx_ptr)

        return (Tx)


# TODO: later replace with an AutoARIMA class
def grid_search(y_b, d=1, max_p=3, max_q=3, method="bic", fit_intercept=None):
    """Grid search to find optimal model order (p, q), weighing
    model complexity against likelihood.
    Optimality is based on minimizing BIC or AIC, which
    both sum negative log-likelihood against model complexity; Higher model
    complexity might yield a lower negative LL, but at higher `bic` due to
    complexity term.

    Deprecation warning:
    --------------------
    This function will be removed and replaced with an auto-ARIMA class with
    more functionality, support for seasonality, and a more efficient search
    algorithm.

    Parameters:
    -----------
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
    fit_intercept : bool or int
        Whether to include a constant trend mu in the model
        Leave to None for automatic selection based on the model order

    Returns:
    --------
    Tuple of "best" order, mu, ar, and ma parameters with the
    corresponding IC for each series.

    (best_order: List[Tuple[int, int, int]],
     best_mu: array,
     best_ar: List[array],
     best_ma: List[array],
     best_ic: array)

    """

    batch_size = y_b.shape[1] if len(y_b.shape) > 1 else 1
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

            b_model = ARIMA(y_b, (p, d, q), fit_intercept=fit_intercept)
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
