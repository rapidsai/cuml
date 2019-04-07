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

import cuml
from cuml.gmm.utils.sample_utils import *
from sklearn.mixture import GaussianMixture
from cuml.gmm.utils.utils import timer, info


def np_to_dataframe(df):
    pdf = cudf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:, c]
    return pdf


def mae(x, y):
    return np.mean(np.abs(x - y))


def compute_error(params_pred, params_true):
    mae_dict = dict(
        (key, mae(params_pred[key], params_true[key]))
        for key in params_pred.keys())
    error = sum([mae_dict[key] for key in mae_dict.keys()]) / 3
    return mae_dict, error


@info
def sample(nDim, nCl, nObs, precision):
    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    params = sample_parameters(nDim=nDim, nCl=nCl)
    X = sample_data(nObs, params)
    params = cast_parameters(params, dtype=dt)
    return X, params


@timer("sklearn")
@info
def run_sklearn(X, n_iter, nCl, tol, reg_covar, random_state):
    gmm = GaussianMixture(n_components=nCl,
                          covariance_type="full",
                          tol=tol,
                          reg_covar=reg_covar,
                          max_iter=n_iter,
                          n_init=1,
                          init_params="random",
                          weights_init=None,
                          means_init=None,
                          precisions_init=None,
                          random_state=random_state,
                          warm_start=False)
    gmm.fit(X)

    params = {"mus": gmm.means_,
              "sigmas": gmm.covariances_,
              "pis": gmm.weights_}

    return params


@timer("cuml")
@info
def run_cuml(X, n_iter, precision, nCl, tol, reg_covar, random_state):
    gmm = cuml.GaussianMixture(n_components=nCl,
                               max_iter=n_iter,
                               precision=precision,
                               reg_covar=reg_covar,
                               random_state=random_state,
                               warm_start=False,
                               tol=tol)

    gmm.fit(X)

    params = {"mus": gmm.means_,
              "sigmas": gmm.covars_,
              "pis": gmm.weights_}
    return params


def print_info(true_params, sk_params, cuml_params):
    print("\n true params")
    print(true_params)

    print('\nsklearn')
    mse_dict_sk, error_sk = compute_error(sk_params, true_params)
    print('error')
    print(mse_dict_sk)
    print("params")
    print(sk_params)

    print('\ncuml')
    mse_dict_cuml, error_cuml = compute_error(cuml_params, true_params)
    print('error')
    print(mse_dict_cuml)
    print("params")
    print(cuml_params)

    print('\ncuml-sk')
    mse_dict_cuml, error_cuml = compute_error(cuml_params, sk_params)
    print('errors')
    print(mse_dict_cuml)


# @pytest.mark.parametrize('n_iter', [100])
# @pytest.mark.parametrize('nCl', [5, 10])
# @pytest.mark.parametrize('nDim', [5, 10])
# @pytest.mark.parametrize('nObs', [1000])
@pytest.mark.parametrize('n_iter', [5])
@pytest.mark.parametrize('nCl', [10, 40])
@pytest.mark.parametrize('nDim', [10])
@pytest.mark.parametrize('nObs', [300])
@pytest.mark.parametrize('precision', ['double'])
@pytest.mark.parametrize('tol', [1e-03])
@pytest.mark.parametrize('reg_covar', [0])
@pytest.mark.parametrize('random_state', [10, 45])
def test_gmm(n_iter, nCl, nDim, nObs, precision, tol, reg_covar, random_state):

    X, true_params = sample(nDim=nDim, nCl=nCl, nObs=nObs, precision=precision)

    cuml_params = run_cuml(X, n_iter, precision, nCl,
                           tol, reg_covar, random_state)
    sk_params = run_sklearn(X, n_iter, nCl, tol, reg_covar, random_state)

    # print_info(true_params, sk_params, cuml_params)
    error_dict, error = compute_error(cuml_params, sk_params)
    if precision is "single":
        # I susspect that sklearn is implemented in double precision therefore the computational differences propagate and lead to different results at single precision
        assert error < 1e-01
    else:
        # Tests have always passed on double precision
        assert error < 1e-11
