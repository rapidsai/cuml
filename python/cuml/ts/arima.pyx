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
from cuml.ts.batched_lbfgs import batched_fmin_lbfgs_b

import cuml
from cuml.utils.input_utils import input_to_dev_array
from cuml.common.handle cimport cumlHandle
from libc.stdint cimport uintptr_t

from typing import List, Tuple
import cudf

from cuml.ts.nvtx import pynvtx_range_push, pynvtx_range_pop

cdef extern from "ts/batched_arima.hpp" namespace "ML":
  void batched_loglike(cumlHandle& handle, double* y, int num_batches, int nobs, int p,
                       int d, int q, double* params,
                       vector[double]& vec_loglike, double* d_vs, bool trans)

  void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                         int nobs, int p, int d, int q, double* d_params, double* d_vs_ptr,
                         double* d_y_p)

  void residual(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
                int d, int q, double* d_params, double* d_vs, bool trans)

  void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
                int batch_size, int nobs, double* d_y, double* d_y_diff, double* d_vs,
                double* d_params, double* d_y_fc)

cdef extern from "utils.h" namespace "MLCommon":
  void updateHost[Type](Type* hPtr, const Type* dPtr, size_t len, int stream)

cdef extern from "ts/batched_kalman.hpp" namespace "ML":

  void batched_jones_transform(cumlHandle& handle, int p, int d, int q,
                               int batchSize, bool isInv, const double* h_params,
                               double* h_Tparams);

class ARIMAModel:
    r"""Implements an ARIMA model for in- and out-of-sample time-series prediction.
    The ARIMA model consists of three model parameter classes:
    "AutoRegressive", "Integrated", and "Moving Average" to fit to a given
    time-series input. The library provides both in-sample prediction, and out
    of sample forecasting.

    The Batched ARIMA model fits the following to each given time-series input:
    if d=1:
      \delta \tilde{y}_{t} = \mu + \sum_{i=1}^{p} \phi_i \delta y_{t-i}
                                    + \sum_{i=1}^{q} \theta_i (y_{t-i} -
                                                                 \tilde{y}_{t-i})

    Note all fitted parameters, \mu, \phi_i, \theta_i and the model order (p, d, q).
    
    **Limitations**: The library assumes collections (i.e., batches) of time-series data of the same length with no missing values.

    Examples
    ---------
    .. code-block:: python

        import numpy as np
        from cuml.ts.arima import fit
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
        mu0, ar0, ma0 = arima.estimate_x0((1,1,1), 2, ys)

        # fine-tune parameter estimates
        model = arima.fit(ys, (1,1,1), mu0, ar0, ma0)

        # predict and forecast using fitted model
        yp = model.predict_in_sample()
        yfc = model.forecast(50)
        dx = xs[1] - xs[0]
        xfc = np.linspace(1, 1+50*dx, 50)
        plt.plot(xs, yp, xfc, yfc)

    Parameters
    ----------
    order : Tuple[int, int, int]
            The ARIMA order (p, d, q) of the model
    mu    : ndarray
            (d>0) Array of trend parameters, one for each series
    ar_params : ndarray
                Array of AR parameters, grouped (`p`) per series
    ma_params : ndarray
                Array of MA parameters, grouped (`q`) per series
    y : Array
        The series data. If given as `ndarray`, assumed to be in
        column-major order with series as columns.

    References
    ----------
    The library is heavily influenced by the Python library `statsmodels`, particularly the `statsmodels.tsa.arima_model.ARIMA` model and corresponding code:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html

    Additionally the following book is a useful reference:
    "Time Series Analysis by State Space Methods", J. Durbin, S.J. Koopman, 2nd Edition.
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
        self.d_y = None

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

    def _assert_same_d(self, b_order):
        """Checks that all values of d in batched order are same"""
        b_d = [d for _, d, _ in b_order]
        assert (np.array(b_d) == b_d[0]).all()

    def predict_in_sample(self):
        """Return in-sample prediction on batched series given batched model

        Returns:
        --------
        y_p : array-like, shape = (n_samples, n_series)

        Example:
        --------
        .. code-block:: python
            from cuml.ts.arima import fit
            ...
            model = fit(ys, (1,1,1), mu0, ar0, ma0)
            y_pred = model.predict_in_sample()
        """

        p, d, q = self.order[0]
        
        handle = cuml.common.handle.Handle()
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

        x = pack(p, d, q, self.num_batches, self.mu, self.ar_params, self.ma_params)
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = input_to_dev_array(x, check_dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode="fortran"] y_p = np.zeros(((self.num_samples),
                                                                        self.num_batches), order="F")
        cdef uintptr_t d_y_p_ptr
        d_y_p, d_y_p_ptr, _, _, _ = input_to_dev_array(y_p, check_dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode="fortran"] vs = np.zeros(((self.num_samples - d),
                                                                       self.num_batches), order="F")

        cdef uintptr_t d_vs_ptr
        d_vs, d_vs_ptr, _, _, _ = input_to_dev_array(vs, check_dtype=np.float64)

        cdef uintptr_t d_y_ptr    
        if self.d_y is None:
            d_y, d_y_ptr, _, _, _ = input_to_dev_array(self.y, check_dtype=np.float64)
            self.d_y = (d_y, d_y_ptr)
        
        (_, d_y_ptr) = self.d_y

        predict_in_sample(handle_[0], <double*>d_y_ptr,
                          self.num_batches, self.num_samples, p, d, q,
                          <double*>d_params_ptr, <double*>d_vs_ptr, <double*>d_y_p_ptr)

        
        
        updateHost(&vs[0,0], <double*>d_vs_ptr, (self.num_samples-1) * self.num_batches, 0)
        updateHost(&y_p[0,0], <double*>d_y_p_ptr, (self.num_samples) * self.num_batches, 0)

        self.yp = y_p
        return y_p

    def forecast(self, nsteps: int) -> np.ndarray:
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
            from cuml.ts.arima import fit
            ...
            model = fit(ys, (1,1,1), mu0, ar0, ma0)
            y_fc = model.forecast(10)
            dx = xs[1] - xs[0]
            xfc = np.linspace(1, 1+50*dx, 50)
            plt.plot(xs, ys, xfc, yfc)

        """

        p, d, q = self.order[0]
        
        handle = cuml.common.handle.Handle()
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

        x = pack(p, d, q, self.num_batches, self.mu, self.ar_params, self.ma_params)
        cdef uintptr_t d_params_ptr
        d_params, d_params_ptr, _, _, _ = input_to_dev_array(x, check_dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode="fortran"] y_p = np.zeros(((self.num_samples),
                                                                        self.num_batches), order="F")
        cdef uintptr_t d_y_p_ptr
        d_y_p, d_y_p_ptr, _, _, _ = input_to_dev_array(y_p, check_dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode="fortran"] vs = np.zeros(((self.num_samples - d),
                                                                       self.num_batches), order="F")
        cdef uintptr_t d_vs_ptr
        d_vs, d_vs_ptr, _, _, _ = input_to_dev_array(vs, check_dtype=np.float64)

        cdef uintptr_t d_y_ptr    
        if self.d_y is None:
            d_y, d_y_ptr, _, _, _ = input_to_dev_array(self.y, check_dtype=np.float64)
            self.d_y = (d_y, d_y_ptr)
            
        (_, d_y_ptr) = self.d_y

        residual(handle_[0], <double*>d_y_ptr, self.num_batches, self.num_samples, p, d, q,
                 <double*>d_params_ptr, <double*>d_vs_ptr,
                 False)

        y_diff = np.diff(self.y, axis=0)
        cdef uintptr_t d_y_diff_ptr
        d_y_diff, d_y_diff_ptr, _, _, _ = input_to_dev_array(y_diff, check_dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode="fortran"] y_fc = np.zeros((nsteps, self.num_batches), order="F")
        cdef uintptr_t d_y_fc_ptr
        d_y_fc, d_y_fc_ptr, _, _, _ = input_to_dev_array(y_fc, check_dtype=np.float64)

        forecast(handle_[0], nsteps, p, d, q, self.num_batches, self.num_samples, <double*> d_y_ptr,
                 <double*>d_y_diff_ptr, <double*>d_vs_ptr, <double*>d_params_ptr, <double*> d_y_fc_ptr)

        updateHost(&y_fc[0,0], <double*>d_y_fc_ptr, nsteps * self.num_batches, 0)

        return y_fc


def estimate_x0(order: Tuple[int, int, int],
                nb: int,
                yb) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provide initial estimates to ARIMA parameters `mu`, `ar`, and `ma` for the batched input `yb`"""
    pynvtx_range_push("init x0")
    (p, d, q) = order
    N = p + d + q
    x0 = np.zeros(N * nb)

    for ib in range(nb):
        y = yb[:, ib]

        if d == 1:
            yd = np.diff(y)
        else:
            yd = np.copy(y)

        x0ib = _start_params((p, q, d), yd)

        x0[ib*N:(ib+1)*N] = x0ib

    mu, ar, ma = unpack(p, d, q, nb, x0)

    pynvtx_range_pop()

    return mu, ar, ma



def ll_f(num_batches, nobs, order, y, np.ndarray[double] x, trans=True, handle=None):
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
            dense parameter array, grouped by series, and again by [(mu, ar, ma), (mu, ar, ma)]
    trans : bool
            Should the `jones_transform` be applied?
            Note: The parameters from a `fit()` model are already transformed.
    handle : cumlHandle (optional)
             The cumlHandle to be used.
    """

    cdef vector[double] vec_loglike

    vec_loglike = _batched_loglike(num_batches, nobs, order, y, x, trans, handle)

    loglike = np.zeros(num_batches)
    for i in range(num_batches):
        loglike[i] = vec_loglike[i]

    return loglike

def ll_gf(num_batches, nobs, num_parameters, order, y, x, h=1e-8, trans=True, handle=None):
    """Computes gradient (via finite differencing) of the batched loglikelihood.

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
            dense parameter array, grouped by series, and again by [(mu, ar, ma), (mu, ar, ma)]
    h     : float
            The finite-difference stepsize
    trans : bool
            Should the `jones_transform` be applied?
            Note: The parameters from a `fit()` model are already transformed.
    handle : cumlHandle (optional)
             The cumlHandle to be used.
    """
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
        opt_disp=-1, h=1e-9):
    """
    Fits an ARIMA model to each time-series for the given order and initial parameter estimates.

    Parameters
    ----------
    y : array-like (device or host) shape = (n_samples, n_series)
        Time series data.
    order : Tuple[int, int, int]
            The ARIMA order (p, d, q)
    mu0 : array-like
          Array of trend-parameter estimates. Only used if `d>0`.
    ar_params0 : List of arrays
                 List of AR parameter-arrays, one array per series, each series has `p` parameters.
    ma_params0 : List of arrays
                 List of MA parameter-arrays, one array per series, each series has `q` parameters.
    opt_disp : int
               Fit diagnostic level (for L-BFGS solver):
               * `-1` for no output,
               * `0<n<100` for output every `n` steps
               * `n>100` for more detailed output
    h        : float
               Finite-differencing step size. The gradient is computed using second-order differencing:
                   f(x+h) - f(x - h)
               g = ----------------- + O(h^2)
                         2 * h

    Returns:
    --------
    model : ARIMAModel
            The ARIMA model with the best fit parameters.

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
    x0 = _batch_invtrans(p, d, q, num_batches, x0, handle)

    # check initial parameter sanity
    if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")


    # Optimize parameters by minimizing log likelihood.
    x, niter, flags = batched_fmin_lbfgs_b(f, x0, num_batches, gf,
                                           iprint=opt_disp, factr=1000)

    # TODO: Better Handle non-zero `flag` array values: 0 -> ok, 1,2 -> optimizer had trouble
    if (flags != 0).any():
        print("WARNING(`fit()`): Some batch members had optimizer problems.")

    Tx = _batch_trans(p, d, q, num_batches, x, handle)
    mu, ar, ma = unpack(p, d, q, num_batches, Tx)

    fit_model = ARIMAModel(num_batches*[order], mu, ar, ma, y)
    fit_model.niter = niter
    fit_model.d_y = (d_y, d_y_ptr)
    return fit_model



def grid_search(y_b, d=1, max_p=3, max_q=3, method="bic"):
    """Grid search to find optimal model order (p, q), weighing model complexity
    against likelihood. Optimality is based on minimizing BIC or AIC, which
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
    """

    num_batches = y_b.shape[1]
    best_ic = np.full(num_batches, np.finfo(np.float64).max/2)
    best_model = ARIMAModel([[]]*num_batches, np.zeros(num_batches), [[]]*num_batches, [[]]*num_batches, y_b)
    # best_model =

    for p in range(0, max_p):
        arparams = np.zeros(p)
        for q in range(0, max_q):
            maparams = np.zeros(q)

            # skip 0,0 case
            if p == 0 and q == 0:
                continue

            mu0, ar0, ma0 = estimate_x0((p, d, q), 2, y_b)
            
            b_model = fit(y_b, (p, d, q), mu0, ar0, ma0)

            if method == "aic":
                ic = b_model.aic
            elif method == "bic":
                ic = b_model.bic
            else:
                raise NotImplementedError("Method '{}' not supported".format(method))

            for (i, ic_i) in enumerate(ic):
                if ic_i < best_ic[i]:
                    best_model.order[i] = (p, d, q)
                    best_model.mu[i] = b_model.mu[i]

                    if p > 0:
                        best_model.ar_params[i] = b_model.ar_params[i]
                    else:
                        best_model.ar_params[i] = []
                    if q > 0:
                        best_model.ma_params[i] = b_model.ma_params[i]
                    else:
                        best_model.ma_params[i] = []

                    best_ic[i] = ic_i

    return (best_model, best_ic)


def unpack(p, d, q, nb, np.ndarray[double, ndim=1] x):
    """Unpack linearized parameters into mu, ar, and ma batched-groupings"""
    pynvtx_range_push("unpack(x) -> (ar,ma,mu)")
    num_parameters = d + p + q
    mu = np.zeros(nb)
    ar = []
    ma = []
    for i in range(nb):
        xi = x[i*num_parameters:(i+1)*num_parameters]
        if d > 0:
            mu[i] = xi[0]
        if p > 0:
            ar.append(xi[d:(d+p)])
        ma.append(xi[d+p:])

    pynvtx_range_pop()
    return (mu, ar, ma)


def pack(p, d, q, nb, mu, ar, ma):
    """Pack mu, ar, and ma batched-groupings into a linearized vector `x`"""
    pynvtx_range_push("pack(ar,ma,mu) -> x")
    num_parameters = d + p + q
    x = np.zeros(num_parameters*nb)
    for i in range(nb):
        xi = np.zeros(num_parameters)
        if d > 0:
            xi[0] = mu[i]

        for j in range(p):
            xi[j+d] = ar[i][j]
        for j in range(q):
            xi[j+p+d] = ma[i][j]
        # xi = np.array([mu[i]])
        # xi = np.r_[xi, ar[i]]
        # xi = np.r_[xi, ma[i]]
        x[i*num_parameters:(i+1)*num_parameters] = xi

    pynvtx_range_pop()
    return x


def _batched_transform(p, d, q, nb, np.ndarray[double] x, isInv, handle=None):
    cdef vector[double] vec_ar
    cdef vector[double] vec_ma
    cdef vector[double] vec_Tar
    cdef vector[double] vec_Tma

    pynvtx_range_push("batched_transform")
    
    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()
    cdef np.ndarray[double] Tx = np.zeros(nb*(d+p+q))

    batched_jones_transform(handle_[0], p, d, q, nb, isInv, &x[0], &Tx[0])

    pynvtx_range_pop()
    return (Tx)



def _start_params(order, y_diff):
    """A quick approach to determine reasonable starting mu (trend), AR, and MA parameters"""

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

        # TODO: `statsmodels` uses BIC to pick the "best" `p` for this initial
        # fit. The "best" model is probably a p=1, so we will assume that for now.
        p_best = 1
        x = np.zeros((len(y) - p_best, p_best))
        # create lagged series set
        for lag in range(1, p_best+1):
            # create lag and trim appropriately from front so they are all the same size
            x[:, lag-1] = y[p_best-lag:-lag].T

        # LS fit a*X - Y
        y_ar = y[p_best:]
        
        (ar_fit, _, _, _) = np.linalg.lstsq(x, y_ar.T, rcond=None)
        # print("initial_ar_fit:", ar_fit)
        # set_trace()
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
            (arma_fit, _, _, _) = np.linalg.lstsq(X, y_ar[(q+p_diff):].T, rcond=None)

            params_init[d:] = arma_fit

    else:
        # case when p == 0 and q>0

        # when p==0, MA params are often -1
        # TODO: See how `statsmodels` handles this case
        params_init[d:] = -1*np.ones(q)

    return params_init



def _model_complexity(order):
    (p, d, q) = order
    # complexity is number of parameters: mu + ar + ma
    return d + p + q


def _batch_trans(p, d, q, nb, x, handle=None):
    """Apply the stationarity/invertibility guaranteeing transform to batched-parameter vector x."""
    pynvtx_range_push("jones trans")

    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(p, d, q, nb, x, False, handle)
    
    pynvtx_range_pop()
    return Tx


def _batch_invtrans(p, d, q, nb, x, handle=None):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """
    pynvtx_range_push("jones inv-trans")

    if handle is None:
        handle = cuml.common.handle.Handle()

    Tx = _batched_transform(p, d, q, nb, x, True, handle)

    pynvtx_range_pop()
    return Tx


def _batched_loglike(num_batches, nobs, order, y, np.ndarray[double] x, trans=False, handle=None):
    cdef vector[double] vec_loglike
    cdef vector[double] vec_y_cm
    cdef vector[double] vec_x

    # if cumlHandle is None:
    # cumlHandle = 

    pynvtx_range_push("batched loglikelihood")
    p, d, q = order

    num_params = (p+d+q)

    vec_loglike.resize(num_batches)

    cdef uintptr_t d_y_ptr
    cdef uintptr_t d_x_ptr
    cdef uintptr_t d_vs_ptr
    cdef np.ndarray[double, ndim=2, mode="fortran"] vs = np.zeros(((nobs-d), num_batches), order="F")
    # note: make sure you explicitly have d_y_array. Otherwise it gets garbage collected (I think).
    d_y_array, d_y_ptr, _, _, dtype_y = input_to_dev_array(y, check_dtype=np.float64)
    d_x_array, d_x_ptr, _, _, _ = input_to_dev_array(x, check_dtype=np.float64)
    d_vs, d_vs_ptr, _, _, _ = input_to_dev_array(vs, check_dtype=np.float64)
    

    if handle is None:
        handle = cuml.common.handle.Handle()

    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    if dtype_y != np.float64:
        raise ValueError("Only 64-bit floating point inputs currently supported")

    batched_loglike(handle_[0], <double*>d_y_ptr, num_batches, nobs, p, d, q, <double*>d_x_ptr,
                    vec_loglike, <double*>d_vs_ptr, trans)

    pynvtx_range_pop()
    return vec_loglike

def _residual(num_batches, nobs, order, y, np.ndarray[double] x, trans=False, handle=None):
    """ Computes and returns the kalman residual """
    
    cdef vector[double] vec_loglike
    cdef uintptr_t d_vs_ptr
    cdef uintptr_t d_params_ptr
    cdef uintptr_t d_y_ptr

    p, d, q = order

    cdef np.ndarray[double, ndim=2, mode="fortran"] vs = np.zeros(((nobs-d), num_batches), order="F")
    
    d_params, d_params_ptr, _, _, _ = input_to_dev_array(x, check_dtype=np.float64)
    d_y, d_y_ptr, _, _, _ = input_to_dev_array(y, check_dtype=np.float64)
    d_vs, d_vs_ptr, _, _, _ = input_to_dev_array(vs, check_dtype=np.float64)

    if handle is None:
        handle = cuml.common.handle.Handle()

    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    residual(handle_[0], <double*>d_y_ptr, num_batches, nobs, p, d, q,
             <double*>d_params_ptr, <double*>d_vs_ptr,
             trans)

    updateHost(&vs[0,0], <double*>d_vs_ptr, (nobs-d) * num_batches, 0)
    
    return vs

