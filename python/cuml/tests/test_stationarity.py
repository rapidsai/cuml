# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

# TODO: update!

from cuml.tsa import stationarity
from statsmodels.tsa import stattools
import warnings
import pytest

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


###############################################################################
#                       Helpers and reference functions                       #
###############################################################################


def prepare_data(y, d, D, s):
    """Applies differencing and seasonal differencing to the data"""
    n_obs, batch_size = y.shape
    s1 = s if D else (1 if d else 0)
    s2 = 1 if d + D == 2 else 0
    y_diff = np.zeros((n_obs - d - s * D, batch_size), dtype=y.dtype)
    for i in range(batch_size):
        temp = y[s1:, i] - y[:-s1, i] if s1 else y[:, i]
        y_diff[:, i] = temp[s2:] - temp[:-s2] if s2 else temp[:]
    return y_diff


def kpss_ref(y):
    """Wrapper around statsmodels' KPSS test"""
    batch_size = y.shape[1]
    test_results = np.zeros(batch_size, dtype=bool)
    for i in range(batch_size):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, pval, *_ = stattools.kpss(
                y[:, i], regression="c", nlags="legacy"
            )
        test_results[i] = pval > 0.05
    return test_results


cuml_tests = {
    "kpss": stationarity.kpss_test,
}

ref_tests = {
    "kpss": kpss_ref,
}


###############################################################################
#                                    Tests                                    #
###############################################################################


@pytest.mark.parametrize("batch_size", [25, 100])
@pytest.mark.parametrize("n_obs", [50, 130])
@pytest.mark.parametrize("dD", [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("s", [4, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_type", ["kpss"])
def test_stationarity(batch_size, n_obs, dD, s, dtype, test_type):
    """Test stationarity tests against a reference implementation"""
    d, D = dD

    # Fix seed for stability
    np.random.seed(42)

    # Generate seasonal patterns with random walks
    pattern = np.zeros((s, batch_size))
    pattern[0, :] = np.random.uniform(-1.0, 1.0, batch_size)
    for i in range(1, s):
        pattern[i, :] = pattern[i - 1, :] + np.random.uniform(
            -1.0, 1.0, batch_size
        )
    pattern /= s

    # Decide for each series whether to include a linear and/or quadratic
    # trend and/or a seasonal pattern
    linear_mask = np.random.choice([False, True], batch_size, p=[0.50, 0.50])
    quadra_mask = np.random.choice([False, True], batch_size, p=[0.75, 0.25])
    season_mask = np.random.choice([False, True], batch_size, p=[0.75, 0.25])

    # Generate coefficients for the linear, quadratic and seasonal terms,
    # taking into account the masks computed above and avoiding coefficients
    # close to zero
    linear_coef = (
        linear_mask
        * np.random.choice([-1.0, 1.0], batch_size)
        * np.random.uniform(0.2, 2.0, batch_size)
    )
    quadra_coef = (
        quadra_mask
        * np.random.choice([-1.0, 1.0], batch_size)
        * np.random.uniform(0.2, 2.0, batch_size)
    )
    season_coef = season_mask * np.random.uniform(0.4, 0.8, batch_size)

    # Generate the data
    x = np.linspace(0.0, 2.0, n_obs)
    offset = np.random.uniform(-2.0, 2.0, batch_size)
    y = np.zeros((n_obs, batch_size), order="F", dtype=dtype)
    for i in range(n_obs):
        y[i, :] = (
            offset[:]
            + linear_coef[:] * x[i]
            + quadra_coef[:] * x[i] * x[i]
            + season_coef[:] * pattern[i % s, :]
            + np.random.normal(0.0, 0.2, batch_size)
        )

    # Call the cuML function
    test_cuml = cuml_tests[test_type](y, d, D, s)

    # Compute differenced data and call the reference function
    y_diff = prepare_data(y, d, D, s)
    test_ref = ref_tests[test_type](y_diff)

    np.testing.assert_array_equal(test_cuml, test_ref)
