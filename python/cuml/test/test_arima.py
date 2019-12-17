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

###############################################################################
#                             How these tests work                            #
###############################################################################
#
# This test file contains some unit tests and an integration test.
#
# The integration test has wider tolerance marging because the optimization
# algorithm may fit the model with different parameters than the reference
# implementation. These margins have been found empirically when creating the
# datasets. They will help to identify regressions but might have to be
# changed, e.g if there are changes in the optimization algorithm.
#
# The units tests use some ground truth (e.g the parameters found by the
# reference implementation) to test a unique piece of code. The error margin
# is then very small.

import pytest

from collections import namedtuple
import numpy as np
import os
from timeit import default_timer as timer
import warnings

import pandas as pd
from scipy.optimize.optimize import _approx_fprime_helper
import statsmodels.api as sm

import cudf
import cuml.tsa.arima as arima


###############################################################################
#                                  Test data                                  #
###############################################################################

# Common structure to hold the data, the reference and the testing parameters
ARIMAData = namedtuple('ARIMAData', ['batch_size', 'n_obs', 'dataset', 'start',
                                     'end', 'tolerance_integration_in',
                                     'tolerance_integration_out',
                                     'tolerance_sameparam_inout'])

# ARIMA(2,0,0)
test_200 = ARIMAData(
    batch_size=8,
    n_obs=15,
    dataset="long_term_arrivals_by_citizenship",
    start=10,
    end=25,
    tolerance_integration_in=0.05,
    tolerance_integration_out=0.5,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(0,0,2) with intercept
test_002c = ARIMAData(
    batch_size=10,
    n_obs=20,
    dataset="net_migrations_auckland_by_age",
    start=15,
    end=30,
    tolerance_integration_in=210.0,
    tolerance_integration_out=180.0,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(0,1,0) with intercept
test_010c = ARIMAData(
    batch_size=4,
    n_obs=17,
    dataset="cattle",
    start=10,
    end=25,
    tolerance_integration_in=0.0001,
    tolerance_integration_out=0.0001,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(1,1,0)
test_110 = ARIMAData(
    batch_size=1,
    n_obs=137,
    dataset="police_recorded_crime",
    start=100,
    end=150,
    tolerance_integration_in=45.0,
    tolerance_integration_out=45.0,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(0,1,1)
test_011 = ARIMAData(
    batch_size=16,
    n_obs=28,
    dataset="deaths_by_region",
    start=20,
    end=40,
    tolerance_integration_in=30.0,
    tolerance_integration_out=20.0,
    tolerance_sameparam_inout=0.05
)

# ARIMA(1,2,1)
test_121 = ARIMAData(
    batch_size=2,
    n_obs=137,
    dataset="population_estimate",
    start=100,
    end=150,
    tolerance_integration_in=2.5,
    tolerance_integration_out=55.0,
    tolerance_sameparam_inout=0.05
)

# ARIMA(1,0,1)(1,1,1)_4
test_101_111_4 = ARIMAData(
    batch_size=3,
    n_obs=101,
    dataset="alcohol",
    start=80,
    end=110,
    tolerance_integration_in=0.01,
    tolerance_integration_out=0.01,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(1,1,1)(2,0,0)_4
test_111_200_4 = ARIMAData(
    batch_size=14,
    n_obs=123,
    dataset="hourly_earnings_by_industry",
    start=115,
    end=130,
    tolerance_integration_in=0.35,
    tolerance_integration_out=0.7,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(1,1,2)(0,1,2)_4
test_112_012_4 = ARIMAData(
    batch_size=2,
    n_obs=179,
    dataset="passenger_movements",
    start=160,
    end=200,
    tolerance_integration_in=0.01,
    tolerance_integration_out=0.05,
    tolerance_sameparam_inout=0.0001
)

# ARIMA(1,1,1)(1,1,1)_12
test_111_111_12 = ARIMAData(
    batch_size=12,
    n_obs=279,
    dataset="guest_nights_by_region",
    start=260,
    end=290,
    tolerance_integration_in=0.05,
    tolerance_integration_out=0.05,
    tolerance_sameparam_inout=0.005
)

# Dictionary matching a test case to a tuple of model parameters
# (a test case could be used with different models)
# (p, d, q, P, D, Q, s, k) -> ARIMAData
test_data = {
    (2, 0, 0, 0, 0, 0, 0, 0): test_200,
    (0, 0, 2, 0, 0, 0, 0, 1): test_002c,
    (0, 1, 0, 0, 0, 0, 0, 1): test_010c,
    (1, 1, 0, 0, 0, 0, 0, 0): test_110,
    (0, 1, 1, 0, 0, 0, 0, 0): test_011,
    (1, 2, 1, 0, 0, 0, 0, 0): test_121,
    (1, 0, 1, 1, 1, 1, 4, 0): test_101_111_4,
    (1, 1, 1, 2, 0, 0, 4, 0): test_111_200_4,
    (1, 1, 2, 0, 1, 2, 4, 0): test_112_012_4,
    (1, 1, 1, 1, 1, 1, 12, 0): test_111_111_12,
}

# Dictionary for lazy-loading of datasets
# (name, dtype) -> dataframe
lazy_data = {}

# Dictionary for lazy-evaluation of reference fits
# (p, d, q, P, D, Q, s, k, name, dtype) -> SARIMAXResults
lazy_ref_fit = {}


def extract_order(tup):
    """Extract the order from a tuple of parameters"""
    p, d, q, P, D, Q, s, k = tup
    return (p, d, q), (P, D, Q, s), k


data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'ts_datasets')


# TODO: store both pandas and cuDF dataframes?
def get_dataset(data, dtype):
    """Load a dataset with a given dtype or return a previously loaded dataset
    """
    key = (data.dataset, np.dtype(dtype).name)
    if key not in lazy_data:
        lazy_data[key] = pd.read_csv(
            os.path.join(data_path, "{}.csv".format(data.dataset)),
            usecols=range(1, data.batch_size + 1), dtype=dtype)
    return lazy_data[key]


def get_ref_fit(data, order, seasonal_order, intercept, dtype):
    """Compute a reference fit of a dataset with the given parameters and dtype
    or return a previously computed fit
    """
    y = get_dataset(data, dtype)
    key = order + seasonal_order + \
        (intercept, data.dataset, np.dtype(dtype).name)
    if key not in lazy_ref_fit:
        ref_model = [sm.tsa.SARIMAX(y[col], order=order,
                                    seasonal_order=seasonal_order,
                                    trend='c' if intercept else 'n')
                     for col in y.columns]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lazy_ref_fit[key] = [model.fit(disp=0) for model in ref_model]
    return lazy_ref_fit[key]


###############################################################################
#                                    Tests                                    #
###############################################################################

@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_integration(test_case, dtype):
    """Full integration test: estimate, fit, predict (in- and out-of-sample)
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)

    y = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create and fit cuML model
    cuml_model = arima.ARIMA(cudf.from_pandas(
        y), order, seasonal_order, fit_intercept=intercept)
    cuml_model.fit()

    # Predict
    cuml_pred = cuml_model.predict(data.start, data.end).copy_to_host()
    ref_preds = np.zeros((data.end - data.start, data.batch_size))
    for i in range(data.batch_size):
        ref_preds[:, i] = ref_fits[i].get_prediction(
            data.start, data.end - 1).predicted_mean

    # Compare results
    boundary = data.n_obs - data.start
    max_err_in = (np.absolute(cuml_pred[:boundary, :]
                              - ref_preds[:boundary, :]).max()
                  if boundary > 0 else 0)
    max_err_out = (np.absolute(cuml_pred[boundary:, :]
                               - ref_preds[boundary:, :]).max()
                   if data.end > data.n_obs else 0)
    assert max_err_in < data.tolerance_integration_in, \
        "In-sample prediction error {} > tolerance {}".format(
            max_err_in, data.tolerance_integration_in)
    assert max_err_out < data.tolerance_integration_out, \
        "Out-of-sample prediction error {} > tolerance {}".format(
            max_err_out, data.tolerance_integration_out)


def _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype):
    """Utility function to transfer the parameters from a statsmodels'
    SARIMAXResults object to a cuML ARIMA object.
    Note: be cautious with the intercept, it is not always equivalent
    in statsmodels and cuML models (it depends on the order).
    """
    p, _, q = order
    P, _, Q, _ = seasonal_order
    nb = cuml_model.batch_size
    params = dict()
    if p:
        params["ar"] = np.zeros((p, nb), dtype=dtype)
    if q:
        params["ma"] = np.zeros((q, nb), dtype=dtype)
    if P:
        params["sar"] = np.zeros((P, nb), dtype=dtype)
    if Q:
        params["sma"] = np.zeros((Q, nb), dtype=dtype)
    if intercept:
        params["mu"] = np.zeros(nb, dtype=dtype)
    for i in range(nb):
        if p:
            params["ar"][:, i] = ref_fits[i].arparams[:]
        if q:
            params["ma"][:, i] = ref_fits[i].maparams[:]
        if P:
            params["sar"][:, i] = ref_fits[i].seasonalarparams[:]
        if Q:
            params["sma"][:, i] = ref_fits[i].seasonalmaparams[:]
        if intercept:
            params["mu"][i] = ref_fits[i].params[0]
    cuml_model.set_params(params)


def _predict_common(test_case, dtype, start, end, num_steps=None):
    """Utility function used by test_predict and test_forecast to avoid
    code duplication.
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)
    p, d, q = order
    P, D, Q, s = seasonal_order

    y = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(cudf.from_pandas(
        y), order, seasonal_order, fit_intercept=intercept)

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype)

    # Predict or forecast
    ref_preds = np.zeros((end - start, data.batch_size))
    for i in range(data.batch_size):
        ref_preds[:, i] = ref_fits[i].get_prediction(
            start, end - 1).predicted_mean
    if num_steps is None:
        cuml_pred = cuml_model.predict(start, end).copy_to_host()
    else:
        cuml_pred = cuml_model.forecast(num_steps).copy_to_host()

    # Compare results
    max_err = np.absolute(cuml_pred[:, :] - ref_preds[:, :]).max()
    assert max_err < data.tolerance_sameparam_inout, \
        "Prediction error {} > tolerance {}".format(
            max_err, data.tolerance_sameparam_inout)


@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_predict(test_case, dtype):
    """Test in-sample prediction against statsmodels (with the same values
    for the model parameters)
    """
    n_obs = test_case[1].n_obs
    _predict_common(test_case, dtype, n_obs // 2, n_obs)


@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_forecast(test_case, dtype):
    """Test out-of-sample forecasting against statsmodels (with the same
    values for the model parameters)
    """
    n_obs = test_case[1].n_obs
    _predict_common(test_case, dtype, n_obs, n_obs + 10, 10)


# TODO: aic / bic tests against a statsmodels ARMA?

# # TODO: test with seasonality
# def test_transform():
#     """Test the parameter transformation code."""
#     x0 = np.array([-36.24493319, -0.76159416, -0.76159516, -167.65533746,
#                    -0.76159416, -0.76159616])

#     # Without corrections to the MA parameters, this inverse transform will
#     # return NaN
#     Tx0 = arima._batch_invtrans((0, 1, 2), (0, 0, 0, 0), 1, 2, x0)

#     assert(not np.isnan(Tx0).any())

#     Tx0 = arima._batch_invtrans((2, 1, 0), (0, 0, 0, 0), 1, 2, x0)

#     assert(not np.isnan(Tx0).any())

#     Tx0 = arima._batch_invtrans((1, 1, 1), (0, 0, 0, 0), 1, 2,
#                                 np.array([-1.27047619e+02,
#                                           1.90024682e-02,
#                                           -5.88867176e-01,
#                                           -1.20404762e+02,
#                                           5.12333137e-05,
#                                           -6.14485076e-01]))
#     np.testing.assert_allclose(Tx0, np.array([-1.27047619e+02,
#                                               3.80095119e-02,
#                                               -1.35186024e+00,
#                                               -1.20404762e+02,
#                                               1.02466627e-04,
#                                               -1.43219144e+00]))


# def test_log_likelihood():
#     """
#     Test loglikelihood against reference results using reference parameters
#     """
#     x0 = [[-220.35376518754148,
#            -0.2617000627224417,
#            -2.1893003751753457],
#           [-2.3921544864718811e+02, -1.3525124433776395e-01,
#            -7.5978156540072991e-02,
#            -2.4055488944465053e+00]]
#     ref_ll = [-415.7117855771454, -415.32341960785186]

#     _, y = get_data()

#     for p in range(1, 3):
#         order = (p, 1, 1)
#         y0 = np.zeros((len(t), 1), order='F')
#         y0[:, 0] = y[:, 0]
#         ll = arima.ll_f(1, len(t), order, (0, 0, 0, 0), 1,
#                         y0, np.copy(x0[p-1]), trans=True)
#         np.testing.assert_almost_equal(ll, ref_ll[p-1])

#     x = [-1.2704761899e+02, 3.8009501900e-02, -1.3518602400e+00,
#          -1.2040476199e+02, 1.0245662700e-04, -1.4321914400e+00]
#     ll = arima.ll_f(2, len(t), (1, 1, 1), (0, 0, 0, 0), 1, y, np.array(x))
#     np.set_printoptions(precision=14)
#     ll_ref = np.array([-418.2732740315433, -413.7692130741877])
#     np.testing.assert_allclose(ll, ll_ref)


# def test_gradient_ref():
#     """Tests the gradient based on a reference output"""
#     x = np.array([-1.2704761899e+02, 3.8009511900e-02, -1.3518602400e+00,
#                   -1.2040476199e+02, 1.0246662700e-04, -1.4321914400e+00])

#     _, y = get_data()
#     np.set_printoptions(precision=14)
#     g = arima.ll_gf(2, len(t), 3, (1, 1, 1), (0, 0, 0, 0), 1, y, x)
#     g_ref = np.array([-7.16227077646181e-04, -4.09565927839139e+00,
#                       -4.10715017551411e+00, -1.02602371043758e-03,
#                       -4.46265460141149e+00,
#                       -4.18378931499319e+00])
#     np.testing.assert_allclose(g, g_ref, rtol=1e-6)


# def test_gradient():
#     """test gradient implementation using FD"""
#     num_samples = 100
#     xs = np.linspace(0, 1, num_samples)
#     np.random.seed(12)
#     noise = np.random.normal(scale=0.1, size=num_samples)
#     ys = noise + 0.5*xs
#     for num_batches in range(1, 5):
#         ys_df = np.reshape(np.tile(np.reshape(ys,
#                                               (num_samples, 1)),
#                                    num_batches),
#                            (num_batches, num_samples), order="C").T
#         order = (1, 1, 1)
#         mu = 0.0
#         arparams = np.array([-0.01])
#         maparams = np.array([-1.0])
#         x = np.r_[mu, arparams, maparams]
#         x = np.tile(x, num_batches)
#         num_samples = ys_df.shape[0]
#         num_batches = ys_df.shape[1]

#         p, d, q = order
#         num_parameters = d + p + q
#         g = arima.ll_gf(num_batches, num_samples, num_parameters, order,
#                         (0, 0, 0, 0), 1, ys_df, x)
#         grad_fd = np.zeros(len(x))
#         h = 1e-8
#         for i in range(len(x)):
#             def fx(xp):
#                 return arima.ll_f(num_batches, num_samples, order,
#                                   (0, 0, 0, 0), 1, ys_df, xp).sum()

#             xph = np.copy(x)
#             xmh = np.copy(x)
#             xph[i] += h
#             xmh[i] -= h
#             f_ph = fx(xph)
#             f_mh = fx(xmh)
#             grad_fd[i] = (f_ph-f_mh)/(2*h)

#         np.testing.assert_allclose(g, grad_fd, rtol=1e-4)

#         def f(xk):
#             return arima.ll_f(num_batches, num_samples, order, (0, 0, 0, 0),
#                               1, ys_df, xk).sum()

#         # from scipy
#         g_sp = _approx_fprime_helper(x, f, h)
#         np.testing.assert_allclose(g, g_sp, rtol=1e-4)


# def test_bic():
#     """Test "Bayesian Information Criterion" metric. BIC penalizes the
#     log-likelihood with the number of parameters.

#     """
#     np.set_printoptions(precision=16)

#     bic_reference = [[851.0904458614862, 842.6620993460326],
#                      [854.747970752074, 846.2220267762417]]

#     _, y = get_data()

#     for p in range(1, 3):
#         order = (p, 1, 1)

#         batched_model = arima.ARIMA(y, order, fit_intercept=True)
#         batched_model.fit()

#         np.testing.assert_allclose(batched_model.bic,
#                                    bic_reference[p-1], rtol=1e-4)


# def test_grid_search(num_batches=2):
#     """Tests grid search using random data over the default range of p,q
#     parameters"""
#     ns = len(t)
#     y_b = np.zeros((ns, num_batches))

#     for i in range(num_batches):
#         y_b[:, i] = np.random.normal(size=ns, scale=2000) + data_smooth

#     best_order, best_mu, best_ar, best_ma, best_ic = arima.grid_search(y_b,
#                                                                        d=1)

#     if num_batches == 2:
#         np.testing.assert_array_equal(best_order, [(0, 1, 1), (0, 1, 1)])


# def demo():
#     """Demo example from the documentation"""
#     import matplotlib.pyplot as plt
#     num_samples = 200
#     xs = np.linspace(0, 1, num_samples)
#     np.random.seed(12)
#     noise = np.random.normal(scale=0.05, size=num_samples)
#     noise2 = np.random.normal(scale=0.05, size=num_samples)
#     ys1 = noise + 0.5*xs + 0.1*np.sin(xs/np.pi)
#     ys2 = noise2 + 0.25*xs + 0.15*np.sin(0.8*xs/np.pi)
#     ys = np.zeros((num_samples, 2))
#     ys[:, 0] = ys1
#     ys[:, 1] = ys2

#     plt.plot(xs, ys1, xs, ys2)

#     model = arima.ARIMA(ys, (1, 1, 1), fit_intercept=True)
#     model.fit()

#     d_yp = model.predict_in_sample()
#     yp = input_to_host_array(d_yp).array
#     d_yfc = model.forecast(50)
#     yfc = input_to_host_array(d_yfc).array
#     dx = xs[1] - xs[0]
#     xfc = np.linspace(1, 1+50*dx, 50)
#     plt.plot(xs, yp)
#     plt.plot(xfc, yfc)
