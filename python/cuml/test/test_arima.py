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

###############################################################################
#                             How these tests work                            #
###############################################################################
#
# This test file contains some unit tests and an integration test.
#
# The integration test has a wider tolerance margin, set separately for each
# dataset. These margins have been found empirically when creating the
# datasets. They will help to identify regressions.
#
# The units tests use some ground truth (e.g the parameters found by the
# reference implementation) to test a unique piece of code. The error margin
# is then very small.
#
# Note: when using an intercept, in certain cases our model and the reference
# will converge to slightly different parameters. It is not an issue, but these
# cases need to be removed for the tests

import pytest

from collections import namedtuple
import numpy as np
import os
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
                                     'end', 'tolerance_integration'])

# ARIMA(2,0,0)
test_200 = ARIMAData(
    batch_size=8,
    n_obs=15,
    dataset="long_term_arrivals_by_citizenship",
    start=10,
    end=25,
    tolerance_integration=0.001
)

# ARIMA(0,0,2) with intercept
test_002c = ARIMAData(
    batch_size=7,
    n_obs=20,
    dataset="net_migrations_auckland_by_age",
    start=15,
    end=30,
    tolerance_integration=0.001
)

# ARIMA(0,1,0) with intercept
test_010c = ARIMAData(
    batch_size=4,
    n_obs=17,
    dataset="cattle",
    start=10,
    end=25,
    tolerance_integration=0.001
)

# ARIMA(1,1,0)
test_110 = ARIMAData(
    batch_size=1,
    n_obs=137,
    dataset="police_recorded_crime",
    start=100,
    end=150,
    tolerance_integration=0.001
)

# ARIMA(0,1,1)
test_011 = ARIMAData(
    batch_size=16,
    n_obs=28,
    dataset="deaths_by_region",
    start=20,
    end=40,
    tolerance_integration=0.005
)

# ARIMA(1,2,1)
test_121 = ARIMAData(
    batch_size=2,
    n_obs=137,
    dataset="population_estimate",
    start=100,
    end=150,
    tolerance_integration=0.05
)

# ARIMA(1,0,1)(1,1,1)_4
test_101_111_4 = ARIMAData(
    batch_size=3,
    n_obs=101,
    dataset="alcohol",
    start=80,
    end=110,
    tolerance_integration=0.02
)

# ARIMA(1,1,1)(2,0,0)_4
test_111_200_4 = ARIMAData(
    batch_size=14,
    n_obs=123,
    dataset="hourly_earnings_by_industry",
    start=115,
    end=130,
    tolerance_integration=0.05
)

# ARIMA(1,1,2)(0,1,2)_4
test_112_012_4 = ARIMAData(
    batch_size=2,
    n_obs=179,
    dataset="passenger_movements",
    start=160,
    end=200,
    tolerance_integration=0.001
)

# ARIMA(1,1,1)(1,1,1)_12
test_111_111_12 = ARIMAData(
    batch_size=12,
    n_obs=279,
    dataset="guest_nights_by_region",
    start=260,
    end=290,
    tolerance_integration=0.001
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
# (name, dtype) -> (pandas dataframe, cuDF dataframe)
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


def get_dataset(data, dtype):
    """Load a dataset with a given dtype or return a previously loaded dataset
    """
    key = (data.dataset, np.dtype(dtype).name)
    if key not in lazy_data:
        y = pd.read_csv(
            os.path.join(data_path, "{}.csv".format(data.dataset)),
            usecols=range(1, data.batch_size + 1), dtype=dtype)
        y_cudf = cudf.from_pandas(y)
        lazy_data[key] = (y, y_cudf)
    return lazy_data[key]


def get_ref_fit(data, order, seasonal_order, intercept, dtype):
    """Compute a reference fit of a dataset with the given parameters and dtype
    or return a previously computed fit
    """
    y, _ = get_dataset(data, dtype)
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

    y, y_cudf = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create and fit cuML model
    cuml_model = arima.ARIMA(
        y_cudf, order, seasonal_order, fit_intercept=intercept)
    cuml_model.fit()

    # Predict
    cuml_pred = cuml_model.predict(data.start, data.end).copy_to_host()
    ref_preds = np.zeros((data.end - data.start, data.batch_size))
    for i in range(data.batch_size):
        ref_preds[:, i] = ref_fits[i].get_prediction(
            data.start, data.end - 1).predicted_mean

    # Compare results
    np.testing.assert_allclose(cuml_pred, ref_preds,
                               rtol=data.tolerance_integration,
                               atol=data.tolerance_integration)


def _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype):
    """Utility function to transfer the parameters from a statsmodels'
    SARIMAXResults object to a cuML ARIMA object.
    Note: be cautious with the intercept, it is not always equivalent
    in statsmodels and cuML models (it depends on the order).
    """
    nb = cuml_model.batch_size
    N = cuml_model.complexity
    x = np.zeros(nb * N, dtype=np.float64)

    for ib in range(nb):
        x[ib*N:(ib+1)*N] = ref_fits[ib].params[:N]

    cuml_model.unpack(x)


def _predict_common(test_case, dtype, start, end, num_steps=None):
    """Utility function used by test_predict and test_forecast to avoid
    code duplication.
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)

    y, y_cudf = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(
        y_cudf, order, seasonal_order, fit_intercept=intercept)

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
    np.testing.assert_allclose(cuml_pred, ref_preds, rtol=0.001, atol=0.01)


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


@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_loglikelihood(test_case, dtype):
    """Test loglikelihood against statsmodels (with the same values for the
    model parameters)
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)

    y, y_cudf = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(
        y_cudf, order, seasonal_order, fit_intercept=intercept)

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype)

    # Compute loglikelihood
    cuml_llf = cuml_model.llf
    ref_llf = np.array([ref_fit.llf for ref_fit in ref_fits])

    # Compare results
    np.testing.assert_allclose(cuml_llf, ref_llf, rtol=0.01, atol=0.01)


@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_gradient(test_case, dtype):
    """Test batched gradient implementation against scipy non-batched
    gradient. Note: it doesn't test that the loglikelihood is correct!
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)
    p, _, q = order
    P, _, Q, _ = seasonal_order
    N = p + P + q + Q + intercept + 1
    h = 1e-8

    y, y_cudf = get_dataset(data, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(y_cudf, order, seasonal_order,
                             fit_intercept=intercept)

    # Get an estimate of the parameters and pack them into a vector
    cuml_model._estimate_x0()
    x = cuml_model.pack()

    # Compute the batched loglikelihood gradient
    batched_grad = cuml_model._loglike_grad(x, h)

    # Iterate over the batch to compute a reference gradient
    scipy_grad = np.zeros(N * data.batch_size)
    for i in range(data.batch_size):
        # Create a model with only the current series
        model_i = arima.ARIMA(y_cudf[y_cudf.columns[i]], order, seasonal_order,
                              fit_intercept=intercept)

        def f(x):
            return model_i._loglike(x)

        scipy_grad[N * i: N * (i + 1)] = \
            _approx_fprime_helper(x[N * i: N * (i + 1)], f, h)

    # Compare
    np.testing.assert_allclose(batched_grad, scipy_grad, rtol=0.001, atol=0.01)


@pytest.mark.parametrize('test_case', test_data.items())
@pytest.mark.parametrize('dtype', [np.float64])
def test_start_params(test_case, dtype):
    """Test starting parameters against statsmodels
    """
    key, data = test_case
    order, seasonal_order, intercept = extract_order(key)

    y, y_cudf = get_dataset(data, dtype)

    # Create models
    cuml_model = arima.ARIMA(
        y_cudf, order, seasonal_order, fit_intercept=intercept)
    ref_model = [sm.tsa.SARIMAX(y[col], order=order,
                                seasonal_order=seasonal_order,
                                trend='c' if intercept else 'n')
                 for col in y.columns]

    # Estimate reference starting parameters
    N = cuml_model.complexity
    nb = data.batch_size
    x_ref = np.zeros(N * nb, dtype=dtype)
    for ib in range(nb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_ref[ib*N:(ib+1)*N] = ref_model[ib].start_params[:N]

    # Estimate cuML starting parameters
    cuml_model._estimate_x0()
    x_cuml = cuml_model.pack()

    # Compare results
    np.testing.assert_allclose(x_cuml, x_ref, rtol=0.001, atol=0.01)
