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


def _is_stationary(yi: np.ndarray, pval_threshold=0.05) -> bool:
    """Single series stationarity test."""
    ns = len(yi)

    # Null hypothesis: data is stationary around a constant
    e = yi - yi.mean()

    # Table 1, Kwiatkowski 1992
    crit_vals = [0.347, 0.463, 0.574, 0.739]
    pvals = [0.10, 0.05, 0.025, 0.01]

    s = np.cumsum(e)
    # eq. 11
    eta = s.dot(s) / ns**2

    #########################
    # compute eq. 10

    # from Kwiatkowski et al. referencing Schwert (1989)
    lags = int(np.ceil(12. * np.power(ns / 100., 1 / 4.)))

    s2_A = e.dot(e)/ns

    def w(_s, _l):
        return 1-_s/(_l+1)

    s2_B = 0.0
    for j in range(1, lags+1):
        eprod = np.dot(e[j:], e[:ns - j])
        s2_B += 2/ns * w(j, lags) * eprod

    s2 = s2_A + s2_B
    #########################

    kpss_stat = eta / s2
    p_value = np.interp(kpss_stat, crit_vals, pvals)

    # print("kpss_stat: {}, pvalue:{}".format(kpss_stat, p_value))

    # higher p_value means higher chance data is stationary around constant.
    return p_value > pval_threshold


def stationarity(y: np.ndarray, pval_threshold=0.05) -> np.ndarray:
    """Return recommended trend parameter `d=0 or 1` for a batched series.

    Parameters:
    -----------
    y : array-like shape = (n_samples, n_series)
         Series to test (not-batched!)
    pval_threshold : float
                     The stationarity threshold.

    Returns:
    --------
    stationarity : array[int]
                   The recommended `d` for each series


    Example:
    --------
    .. code-block:: python

         num_samples = 200
         xs = np.linspace(0, 1, num_samples)
         np.random.seed(12)
         noise = np.random.normal(scale=0.1, size=num_samples)
         ys1 = noise + 0.5*xs # d = 1
         ys2 = noise # d = 0

         num_batches = 2
         ys_df = np.zeros((num_samples, num_batches), order="F")
         ys_df[:, 0] = ys1
         ys_df[:, 1] = ys2

         d_b = stationarity(ys_df)
         # d_b = [1, 0]

    References
    ----------
    Based on paper 'Testing the
    null hypothesis of stationary against the alternative of a unit root' by
    Kwiatkowski et al. 1992. See
    https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html#kpss
    for additional details."""

    nb = y.shape[1]
    d = np.zeros(nb, dtype=np.int32)
    for i in range(nb):

        yi = y[:, i]

        if _is_stationary(yi, pval_threshold):
            d[i] = 0
        elif _is_stationary(np.diff(yi), pval_threshold):
            d[i] = 1
        else:
            raise ValueError("Stationarity failed for d=0 or 1.")

    return d
