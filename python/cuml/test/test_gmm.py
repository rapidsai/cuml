# Copyright (c) 2018, NVIDIA CORPORATION.
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

import pytest
import cudf
import numpy as np

from numba import cuda
from math import sqrt
import cuml
from cuml.hmm.sample_utils import *
from sklearn.mixture import GaussianMixture


def np_to_dataframe(df):
    pdf = cudf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:, c]
    return pdf

def mse(x, y):
    return np.sum((x - y) ** 2)

def compute_error(params_pred, params_true):
    mse_dict = dict(
        (key, mse(params_pred[key], params_true[key]) )
         for key in params_pred.keys())
    error = sum([mse_dict[key] for key in mse_dict.keys()])
    return mse_dict, error

def run_sklearn(X, n_iter):
    gmm = GaussianMixture(n_components=nCl,
                             covariance_type="full",
                             tol=100,
                             reg_covar=0,
                             max_iter=n_iter,
                             n_init=1,
                             init_params="random",
                             weights_init=None,
                             means_init=None,
                             precisions_init=None,
                             random_state=None,
                             warm_start=False,
                             verbose=0,
                             verbose_interval=10)
    gmm.fit(X)
    params = {"mus" : gmm.means_,
              "sigmas" : gmm.covariances_,
              "pis" : gmm.weights_}

    return params

def run_cuml(X, n_iter):
    gmm = cuml.GaussianMixture(precision=precision)

    gmm.fit(X, nCl, n_iter)

    params = {"mus" : gmm.dmu.copy_to_host(),
              "sigmas" : gmm.dsigma.copy_to_host(),
              "pis" : gmm.dPis.copy_to_host()}
    return params

def sample():
    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    params = sample_parameters()
    data = sample_data()
    return data, params


if __name__ == '__main__':
    n_iter = 3

    precision = 'single'
    nCl = 2
    nDim = 1
    nObs = 1000

    data, true_params = sample()
    sk_params = run_sklearn(data, n_iter, nCl)
    cuml_params = run_cuml(data, n_iter, nCl, precision)

    sk_error = compute_error(sk_params, true_params)

    assert sk_error < 0.1
