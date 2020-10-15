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

# distutils: language = c++

import numpy as np
import sys

import ctypes
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from typing import List, Tuple, Dict, Mapping, Optional, Union

from cuml.common.array import CumlArray as cumlArray
from cuml.common.base import Base
from cuml.common.cuda import nvtx_range_wrap
from cuml.raft.common.handle cimport handle_t
from cuml.tsa.batched_lbfgs import batched_fmin_lbfgs_b
import cuml.common.logger as logger
from cuml.common import has_scipy
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_host_array


cdef extern from "cuml/tsa/arima_common.h" namespace "ML":
    cdef cppclass ARIMAParams[DataT]:
        DataT* mu
        DataT* ar
        DataT* ma
        DataT* sar
        DataT* sma
        DataT* sigma2


cdef extern from "cuml/tsa/batched_arima.hpp" namespace "ML":
    ctypedef enum LoglikeMethod: CSS, MLE

    void batched_diff(handle_t& handle, double* d_y_diff, const double* d_y,
                      int batch_size, int n_obs, const ARIMAOrder& order)

    void batched_loglike(
        handle_t& handle, const double* y, int batch_size, int nobs,
        const ARIMAOrder& order, const double* params, double* loglike,
        double* d_vs, bool trans, bool host_loglike, LoglikeMethod method,
        int truncate)

    void batched_loglike_grad(
        handle_t& handle, const double* d_y, int batch_size, int nobs,
        const ARIMAOrder& order, const double* d_x, double* d_grad, double h,
        bool trans, LoglikeMethod method, int truncate)

    void cpp_predict "predict" (
        handle_t& handle, const double* d_y, int batch_size, int nobs,
        int start, int end, const ARIMAOrder& order,
        const ARIMAParams[double]& params, double* d_y_p, bool pre_diff,
        double level, double* d_lower, double* d_upper)

    void information_criterion(
        handle_t& handle, const double* d_y, int batch_size, int nobs,
        const ARIMAOrder& order, const ARIMAParams[double]& params,
        double* ic, int ic_type)

    void estimate_x0(
        handle_t& handle, ARIMAParams[double]& params, const double* d_y,
        int batch_size, int nobs, const ARIMAOrder& order)


cdef extern from "cuml/tsa/batched_kalman.hpp" namespace "ML":

    void batched_jones_transform(
        handle_t& handle, const ARIMAOrder& order, int batchSize,
        bool isInv, const double* h_params, double* h_Tparams)


class ARIMA(Base):
    r"""
    Implements a batched ARIMA model for in- and out-of-sample
    time-series prediction, with support for seasonality (SARIMA)

    ARIMA stands for Auto-Regressive Integrated Moving Average.
    See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

    This class can fit an ARIMA(p,d,q) or ARIMA(p,d,q)(P,D,Q)_s model to a
    batch of time series of the same length with no missing values.
    The implementation is designed to give the best performance when using
    large batches of time series.

    Parameters
    ----------
    endog : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    order : Tuple[int, int, int]
        The ARIMA order (p, d, q) of the model
    seasonal_order: Tuple[int, int, int, int]
        The seasonal ARIMA order (P, D, Q, s) of the model
    fit_intercept : bool or int (default = True)
        Whether to include a constant trend mu in the model
    simple_differencing: bool or int (default = True)
        If True, the data is differenced before being passed to the Kalman
        filter. If False, differencing is part of the state-space model.
        In some cases this setting can be ignored: computing forecasts with
        confidence intervals will force it to False ; fitting with the CSS
        method will force it to True.
        Note: that forecasts are always for the original series, whereas
        statsmodels computes forecasts for the differenced series when
        simple_differencing is True.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    order : ARIMAOrder
        The ARIMA order of the model (p, d, q, P, D, Q, s, k)
    seasonal_order: Tuple[int, int, int, int]
        The seasonal ARIMA order (P, D, Q, s) of the model
    intercept : bool or int
        Whether the model includes a constant trend mu
    d_y: device array
        Time series data on device
    n_obs: int
        Number of observations
    batch_size: int
        Number of time series in the batch
    dtype: numpy.dtype
        Floating-point type of the data and parameters
    niter: numpy.ndarray
        After fitting, contains the number of iterations before convergence
        for each time series.

    Notes
    -----
    *Performance:* Let :math:`r=max(p+s*P, q+s*Q+1)`. The device memory used
    for most operations is
    :math:`O(\mathtt{batch\_size}*\mathtt{n\_obs} + \mathtt{batch\_size}*r^2)`.
    The execution time is a linear function of `n_obs` and `batch_size`
    (if `batch_size` is large), but grows very fast with `r`.

    The performance is optimized for very large batch sizes (e.g thousands of
    series).

    References
    ----------
    This class is heavily influenced by the Python library `statsmodels`,
    particularly `statsmodels.tsa.statespace.sarimax.SARIMAX`.
    See https://www.statsmodels.org/stable/statespace.html.

    Additionally the following book is a useful reference:
    "Time Series Analysis by State Space Methods",
    J. Durbin, S.J. Koopman, 2nd Edition (2012).

    Examples
    --------
    .. code-block:: python

            import numpy as np
            from cuml.tsa.arima import ARIMA

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
            model = ARIMA(y, (0,1,1), (0,1,1,4), fit_intercept=False)
            model.fit()

            # Forecast
            fc = model.forecast(10)
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

    """

    def __init__(self,
                 endog,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int]
                 = (0, 0, 0, 0),
                 fit_intercept=True,
                 simple_differencing=True,
                 handle=None,
                 verbose=False,
                 output_type=None):

        if not has_scipy():
            raise RuntimeError("Scipy is needed to run cuML's ARIMA estimator."
                               " Please install it to enable ARIMA "
                               "estimation.")

        # Initialize base class
        super().__init__(handle, verbose, output_type)
        self._set_base_attributes(output_type=endog)

        # Set the ARIMA order
        cdef ARIMAOrder cpp_order
        cpp_order.p, cpp_order.d, cpp_order.q = order
        cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
        cpp_order.k = int(fit_intercept)
        self.order = cpp_order

        # Check validity of the ARIMA order and seasonal order
        p, d, q = order
        P, D, Q, s = seasonal_order
        if P + D + Q > 0 and s < 2:
            raise ValueError("ERROR: Invalid period for seasonal ARIMA: {}"
                             .format(s))
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
        if max(p + s * P, q + s * Q) > 1024:
            raise ValueError("ERROR: Invalid order. "
                             "Required: max(p+s*P, q+s*Q) <= 1024")

        # Get device array. Float64 only for now.
        self._d_y, self.n_obs, self.batch_size, self.dtype \
            = input_to_cuml_array(endog, check_dtype=np.float64)

        if self.n_obs < d + s * D + 1:
            raise ValueError("ERROR: Number of observations too small for the"
                             " given order")

        self.simple_differencing = simple_differencing

        # Compute the differenced series
        self._d_y_diff = cumlArray.empty(
            (self.n_obs - d - s * D, self.batch_size), self.dtype)
        cdef uintptr_t d_y_ptr = self._d_y.ptr
        cdef uintptr_t d_y_diff_ptr = self._d_y_diff.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        batched_diff(handle_[0], <double*> d_y_diff_ptr, <double*> d_y_ptr,
                     <int> self.batch_size, <int> self.n_obs, self.order)

        # Create a version of the order for the differenced series
        cdef ARIMAOrder cpp_order_diff = cpp_order
        cpp_order_diff.d = 0
        cpp_order_diff.D = 0
        self.order_diff = cpp_order_diff
        self.n_obs_diff = self.n_obs - d - D * s

    def __str__(self):
        cdef ARIMAOrder order = self.order
        intercept_str = 'c' if order.k else 'n'
        if order.s:
            return "ARIMA({},{},{})({},{},{})_{} ({}) - {} series".format(
                order.p, order.d, order.q, order.P, order.D, order.Q, order.s,
                intercept_str, self.batch_size)
        else:
            return "ARIMA({},{},{}) ({}) - {} series".format(
                order.p, order.d, order.q, intercept_str, self.batch_size)

    @nvtx_range_wrap
    def _ic(self, ic_type: str):
        """Wrapper around C++ information_criterion
        """
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef ARIMAOrder order_kf = \
            self.order_diff if self.simple_differencing else self.order

        # Convert host parameters to device parameters
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order_kf.k:
            d_mu, *_ = input_to_cuml_array(self.mu, check_dtype=np.float64)
            d_mu_ptr = d_mu.ptr
        if order_kf.p:
            d_ar, *_ = input_to_cuml_array(self.ar, check_dtype=np.float64)
            d_ar_ptr = d_ar.ptr
        if order_kf.q:
            d_ma, *_ = input_to_cuml_array(self.ma, check_dtype=np.float64)
            d_ma_ptr = d_ma.ptr
        if order_kf.P:
            d_sar, *_ = input_to_cuml_array(self.sar, check_dtype=np.float64)
            d_sar_ptr = d_sar.ptr
        if order_kf.Q:
            d_sma, *_ = input_to_cuml_array(self.sma, check_dtype=np.float64)
            d_sma_ptr = d_sma.ptr
        d_sigma2, *_ = input_to_cuml_array(self.sigma2, check_dtype=np.float64)
        d_sigma2_ptr = d_sigma2.ptr

        cdef ARIMAParams[double] cpp_params
        cpp_params.mu = <double*> d_mu_ptr
        cpp_params.ar = <double*> d_ar_ptr
        cpp_params.ma = <double*> d_ma_ptr
        cpp_params.sar = <double*> d_sar_ptr
        cpp_params.sma = <double*> d_sma_ptr
        cpp_params.sigma2 = <double*> d_sigma2_ptr

        ic = cumlArray.empty(self.batch_size, self.dtype)
        cdef uintptr_t d_ic_ptr = ic.ptr
        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.ptr if self.simple_differencing else self._d_y.ptr

        n_obs_kf = (self.n_obs_diff if self.simple_differencing
                    else self.n_obs)

        ic_name_to_number = {"aic": 0, "aicc": 1, "bic": 2}
        cdef int ic_type_id
        try:
            ic_type_id = ic_name_to_number[ic_type.lower()]
        except KeyError as e:
            raise NotImplementedError("IC type '{}' unknown".format(ic_type))

        information_criterion(handle_[0], <double*> d_y_kf_ptr,
                              <int> self.batch_size, <int> n_obs_kf,
                              order_kf, cpp_params, <double*> d_ic_ptr,
                              <int> ic_type_id)

        return ic.to_output(self.output_type)

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

    def get_fit_params(self) -> Dict[str, np.ndarray]:
        """Get all the fit parameters. Not to be confused with get_params
        Note: pack() can be used to get a compact vector of the parameters

        Returns
        -------
        params: Dict[str, np.ndarray]
            A dictionary of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2 and
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

    def set_fit_params(self, params: Mapping[str, object]):
        """Set all the fit parameters. Not to be confused with ``set_params``
        Note: `unpack()` can be used to load a compact vector of the
        parameters

        Parameters
        ----------
        params:
            A dictionary of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2 and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
        """
        for param_name in ["mu", "ar", "ma", "sar", "sma", "sigma2"]:
            if param_name in params:
                array, _, _, _, _ = input_to_host_array(params[param_name])
                setattr(self, param_name, array)

    def get_param_names(self):
        """
        .. warning:: ARIMA is unable to be cloned at this time. The methods:
            `get_param_names()`, `get_params` and `set_params` will raise
            ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    def get_params(self, deep=True):
        """
        .. warning:: ARIMA is unable to be cloned at this time. The methods:
            `get_param_names()`, `get_params` and `set_params` will raise
            ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    def set_params(self, **params):
        """
        .. warning:: ARIMA is unable to be cloned at this time. The methods:
            `get_param_names()`, `get_params` and `set_params` will raise
            ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    @nvtx_range_wrap
    def predict(self, start=0, end=None, level=None):
        """Compute in-sample and/or out-of-sample prediction for each series

        Parameters
        ----------
        start: int (default = 0)
            Index where to start the predictions (0 <= start <= num_samples)
        end: int (default = None)
            Index where to end the predictions, excluded (end > start), or
            ``None`` to predict until the last observation
        level: float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. ``0 < level < 1``

        Returns
        -------
        y_p : array-like (device)
            Predictions. Shape = (end - start, batch_size)
        lower: array-like (device) (optional)
            Lower limit of the prediction interval if ``level != None``
            Shape = (end - start, batch_size)
        upper: array-like (device) (optional)
            Upper limit of the prediction interval if ``level != None``
            Shape = (end - start, batch_size)

        Examples
        --------
        .. code-block:: python

            from cuml.tsa.arima import ARIMA

            model = ARIMA(ys, (1,1,1))
            model.fit()
            y_pred = model.predict()
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
            logger.warn("WARNING(`predict`): predictions before {} are"
                        " undefined, will be set to NaN"
                        .format(order.d + order.D * order.s))

        if level is not None:
            if level <= 0 or level >= 1:
                raise ValueError("ERROR: Invalid confidence level: {}"
                                 .format(level))
            elif level > 0 and start < self.n_obs:
                raise ValueError("ERROR: Prediction intervals can only be"
                                 " computed for out-of-sample predictions")

        if end is None:
            end = self.n_obs

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order.k:
            d_mu, *_ = input_to_cuml_array(self.mu, check_dtype=np.float64)
            d_mu_ptr = d_mu.ptr
        if order.p:
            d_ar, *_ = input_to_cuml_array(self.ar, check_dtype=np.float64)
            d_ar_ptr = d_ar.ptr
        if order.q:
            d_ma, *_ = input_to_cuml_array(self.ma, check_dtype=np.float64)
            d_ma_ptr = d_ma.ptr
        if order.P:
            d_sar, *_ = input_to_cuml_array(self.sar, check_dtype=np.float64)
            d_sar_ptr = d_sar.ptr
        if order.Q:
            d_sma, *_ = input_to_cuml_array(self.sma, check_dtype=np.float64)
            d_sma_ptr = d_sma.ptr
        d_sigma2, *_ = input_to_cuml_array(self.sigma2, check_dtype=np.float64)
        d_sigma2_ptr = d_sigma2.ptr

        cdef ARIMAParams[double] cpp_params
        cpp_params.mu = <double*> d_mu_ptr
        cpp_params.ar = <double*> d_ar_ptr
        cpp_params.ma = <double*> d_ma_ptr
        cpp_params.sar = <double*> d_sar_ptr
        cpp_params.sma = <double*> d_sma_ptr
        cpp_params.sigma2 = <double*> d_sigma2_ptr

        predict_size = end - start

        # allocate predictions and intervals device memory
        cdef uintptr_t d_y_p_ptr = <uintptr_t> NULL
        cdef uintptr_t d_lower_ptr = <uintptr_t> NULL
        cdef uintptr_t d_upper_ptr = <uintptr_t> NULL
        d_y_p = cumlArray.empty((predict_size, self.batch_size),
                                dtype=np.float64, order="F")
        d_y_p_ptr = d_y_p.ptr
        if level is not None:
            d_lower = cumlArray.empty((predict_size, self.batch_size),
                                      dtype=np.float64, order="F")
            d_upper = cumlArray.empty((predict_size, self.batch_size),
                                      dtype=np.float64, order="F")
            d_lower_ptr = d_lower.ptr
            d_upper_ptr = d_upper.ptr

        cdef uintptr_t d_y_ptr = self._d_y.ptr

        cpp_predict(handle_[0], <double*>d_y_ptr, <int> self.batch_size,
                    <int> self.n_obs, <int> start, <int> end, order,
                    cpp_params, <double*>d_y_p_ptr,
                    <bool> self.simple_differencing,
                    <double> (0 if level is None else level),
                    <double*> d_lower_ptr, <double*> d_upper_ptr)

        if level is None:
            return d_y_p.to_output(self.output_type)
        else:
            return (d_y_p.to_output(self.output_type),
                    d_lower.to_output(self.output_type),
                    d_upper.to_output(self.output_type))

    @nvtx_range_wrap
    def forecast(self, nsteps: int, level=None):
        """Forecast the given model `nsteps` into the future.

        Parameters
        ----------
        nsteps : int
            The number of steps to forecast beyond end of the given series
        level: float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. 0 < level < 1

        Returns
        -------
        y_fc : array-like
            Forecasts. Shape = (nsteps, batch_size)
        lower: array-like (device) (optional)
            Lower limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        upper: array-like (device) (optional)
            Upper limit of the prediction interval if level != None
            Shape = (end - start, batch_size)

        Examples
        --------
        .. code-block:: python

            from cuml.tsa.arima import ARIMA
            ...
            model = ARIMA(ys, (1,1,1))
            model.fit()
            y_fc = model.forecast(10)
        """

        return self.predict(self.n_obs, self.n_obs + nsteps, level)

    @nvtx_range_wrap
    def _estimate_x0(self):
        """Internal method. Estimate initial parameters of the model.
        """
        cdef ARIMAOrder order = self.order

        cdef uintptr_t d_y_ptr = self._d_y.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Create mu, ar and ma arrays
        cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> NULL
        if order.k:
            d_mu = cumlArray.zeros(self.batch_size, dtype=np.float64)
            d_mu_ptr = d_mu.ptr
        if order.p:
            d_ar = cumlArray.zeros((order.p, self.batch_size),
                                   dtype=np.float64, order='F')
            d_ar_ptr = d_ar.ptr
        if order.q:
            d_ma = cumlArray.zeros((order.q, self.batch_size),
                                   dtype=np.float64, order='F')
            d_ma_ptr = d_ma.ptr
        if order.P:
            d_sar = cumlArray.zeros((order.P, self.batch_size),
                                    dtype=np.float64, order='F')
            d_sar_ptr = d_sar.ptr
        if order.Q:
            d_sma = cumlArray.zeros((order.Q, self.batch_size),
                                    dtype=np.float64, order='F')
            d_sma_ptr = d_sma.ptr
        d_sigma2 = cumlArray.zeros(self.batch_size, dtype=np.float64)
        d_sigma2_ptr = d_sigma2.ptr

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
            params["mu"] = d_mu.to_output('numpy')
        if order.p:
            params["ar"] = d_ar.to_output('numpy')
        if order.q:
            params["ma"] = d_ma.to_output('numpy')
        if order.P:
            params["sar"] = d_sar.to_output('numpy')
        if order.Q:
            params["sma"] = d_sma.to_output('numpy')
        params["sigma2"] = d_sigma2.to_output('numpy')
        self.set_fit_params(params)

    @nvtx_range_wrap
    def fit(self,
            start_params: Optional[Mapping[str, object]] = None,
            opt_disp: int = -1,
            h: float = 1e-8,
            maxiter: int = 1000,
            method="ml",
            truncate: int = 0):
        r"""Fit the ARIMA model to each time series.

        Parameters
        ----------
        start_params : Mapping[str, array-like] (optional)
            A mapping (e.g dictionary) of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2
            parameters and (n, batch_size) for any other type, where n is the
            corresponding number of parameters of this type.
            Pass None for automatic estimation (recommended)

        opt_disp : int
            Fit diagnostic level (for L-BFGS solver):

            * `-1` for no output (default)
            * `0<n<100` for output every `n` steps
            * `n>100` for more detailed output

        h : float
            Finite-differencing step size. The gradient is computed using
            forward finite differencing:
            :math:`g = \frac{f(x + \mathtt{h}) - f(x)}{\mathtt{h}} + O(\mathtt{h})` # noqa

        maxiter : int
            Maximum number of iterations of L-BFGS-B
        method : str
            Estimation method - "css", "css-ml" or "ml".
            CSS uses a sum-of-squares approximation.
            ML estimates the log-likelihood with statespace methods.
            CSS-ML starts with CSS and refines with ML.
        truncate : int
            When using CSS, start the sum of squares after a given number of
            observations
        """
        def fit_helper(x_in, fit_method):
            cdef uintptr_t d_y_ptr = self._d_y.ptr

            def f(x: np.ndarray) -> np.ndarray:
                """The (batched) energy functional returning the negative
                log-likelihood (foreach series)."""
                # Recall: We maximize LL by minimizing -LL
                n_llf = -self._loglike(x, True, fit_method, truncate)
                return n_llf / (self.n_obs - 1)

            # Optimized finite differencing gradient for batches
            def gf(x) -> np.ndarray:
                """The gradient of the (batched) energy functional."""
                # Recall: We maximize LL by minimizing -LL
                n_gllf = -self._loglike_grad(x, h, True, fit_method, truncate)
                return n_gllf / (self.n_obs - 1)

            # Check initial parameter sanity
            if ((np.isnan(x_in).any()) or (np.isinf(x_in).any())):
                raise FloatingPointError(
                    "Initial parameter vector x has NaN or Inf.")

            # Optimize parameters by minimizing log likelihood.
            x_out, niter, flags = batched_fmin_lbfgs_b(
                f, x_in, self.batch_size, gf, iprint=opt_disp, factr=1000,
                maxiter=maxiter)

            # Handle non-zero flags with Warning
            if (flags != 0).any():
                logger.warn("fit: Some batch members had optimizer problems")

            return x_out, niter

        if start_params is None:
            self._estimate_x0()
        else:
            self.set_fit_params(start_params)

        x0 = self._batched_transform(self.pack(), True)

        method = method.lower()
        if method not in {"css", "css-ml", "ml"}:
            raise ValueError("Unknown method: {}".format(method))
        if method == "css" or method == "css-ml":
            x, self.niter = fit_helper(x0, "css")
        if method == "css-ml" or method == "ml":
            x, niter = fit_helper(x if method == "css-ml" else x0, "ml")
            self.niter = (self.niter + niter) if method == "css-ml" else niter

        self.unpack(self._batched_transform(x))
        return self

    @nvtx_range_wrap
    def _loglike(self, x, trans=True, method="ml", truncate=0):
        """Compute the batched log-likelihood for the given parameters.

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series
        trans : bool
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.
        method : str
            Estimation method: "css" for sum-of-squares, "ml" for
            an estimation with statespace methods
        truncate : int
            When using CSS, start the sum of squares after a given number of
            observations

        Returns
        -------
        loglike : numpy.ndarray
            Batched log-likelihood. Shape: (batch_size,)
        """
        cdef vector[double] vec_loglike
        vec_loglike.resize(self.batch_size)

        cdef LoglikeMethod ll_method = CSS if method == "css" else MLE
        diff = ll_method != MLE or self.simple_differencing

        cdef ARIMAOrder order_kf = self.order_diff if diff else self.order

        d_x_array, *_ = \
            input_to_cuml_array(x, check_dtype=np.float64, order='C')
        cdef uintptr_t d_x_ptr = d_x_array.ptr

        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.ptr if diff else self._d_y.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        n_obs_kf = (self.n_obs_diff if diff else self.n_obs)
        d_vs = cumlArray.empty((n_obs_kf, self.batch_size), dtype=np.float64,
                               order="F")
        cdef uintptr_t d_vs_ptr = d_vs.ptr

        batched_loglike(handle_[0], <double*> d_y_kf_ptr,
                        <int> self.batch_size, <int> n_obs_kf, order_kf,
                        <double*> d_x_ptr, <double*> vec_loglike.data(),
                        <double*> d_vs_ptr, <bool> trans, <bool> True,
                        ll_method, <int> truncate)

        return np.array(vec_loglike, dtype=np.float64)

    @nvtx_range_wrap
    def _loglike_grad(self, x, h=1e-8, trans=True, method="ml", truncate=0):
        """Compute the gradient (via finite differencing) of the batched
        log-likelihood.

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        h : float
            The finite-difference stepsize
        trans : bool
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.
        method : str
            Estimation method: "css" for sum-of-squares, "ml" for
            an estimation with statespace methods
        truncate : int
            When using CSS, start the sum of squares after a given number of
            observations

        Returns
        -------
        grad : numpy.ndarray
            Batched log-likelihood gradient. Shape: (n_params * batch_size,)
            where n_params is the complexity of the model
        """
        N = self.complexity
        assert len(x) == N * self.batch_size

        cdef LoglikeMethod ll_method = CSS if method == "css" else MLE
        diff = ll_method != MLE or self.simple_differencing

        grad = cumlArray.empty(N * self.batch_size, np.float64)
        cdef uintptr_t d_grad = <uintptr_t> grad.ptr

        cdef ARIMAOrder order_kf = self.order_diff if diff else self.order

        d_x_array, *_ = \
            input_to_cuml_array(x, check_dtype=np.float64, order='C')
        cdef uintptr_t d_x_ptr = d_x_array.ptr

        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.ptr if diff else self._d_y.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        batched_loglike_grad(handle_[0], <double*> d_y_kf_ptr,
                             <int> self.batch_size,
                             <int> (self.n_obs_diff if diff else self.n_obs),
                             order_kf, <double*> d_x_ptr, <double*> d_grad,
                             <double> h, <bool> trans, ll_method,
                             <int> truncate)

        return grad.to_output("numpy")

    @property
    def llf(self):
        """Log-likelihood of a fit model. Shape: (batch_size,)
        """
        return self._loglike(self.pack(), trans=False)

    @nvtx_range_wrap
    def unpack(self, x: Union[list, np.ndarray]):
        """Unpack linearized parameter vector `x` into the separate
        parameter arrays of the model

        Parameters
        ----------
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

        self.set_fit_params(params)

    @nvtx_range_wrap
    def pack(self) -> np.ndarray:
        """Pack parameters of the model into a linearized vector `x`

        Returns
        -------
        x : numpy ndarray
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        p, q, P, Q, k = (order.p, order.q, order.P, order.Q, order.k)
        N = self.complexity

        params = self.get_fit_params()

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

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)

        Returns
        -------
        Tx : array-like
            Packed transformed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        N = self.complexity

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        Tx = np.zeros(self.batch_size * N)

        cdef uintptr_t x_ptr = x.ctypes.data
        cdef uintptr_t Tx_ptr = Tx.ctypes.data
        batched_jones_transform(handle_[0], order, <int> self.batch_size,
                                <bool> isInv, <double*>x_ptr, <double*>Tx_ptr)

        return (Tx)
