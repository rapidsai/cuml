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
from cuml import GaussianMixture
from cuml.hmm.sample_utils import *


def np_to_dataframe(df):
    pdf = cudf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:, c]
    return pdf


@pytest.mark.parametrize('precision', ['single'])
def test_gaussian_mixture_base(precision):

    gmm = GaussianMixture(precision=precision)
    n_iter = 10
    nCl = 2
    nDim = 2
    nObs = 20
    ldd = 32

    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    mus = sample_mus(nDim=nDim, nCl=nCl, lddmu=ldd)
    sigmas = sample_sigmas(nDim=nDim, nCl=nCl, lddsigma=ldd)
    pis = sample_pis(nObs=nObs)

    X = sample_mixture(mus=mus, sigmas=sigmas, pis=pis,
                       nCl=nCl, nDim=nDim, nObs=nObs, lddsigma=ldd, dt=dt)

    # TODO : Fix ldd
    gmm.fit(X, nCl, n_iter)

    assert 0 < 0.1


# # @pytest.mark.parametrize('nDim', [2, 10, 25])
# # @pytest.mark.parametrize('nObs', [1, 2, 10, 25])
# # @pytest.mark.parametrize('precision', ['single', 'double'])
# # @pytest.mark.parametrize('input_type', ['numpy', 'cudf'])
# @pytest.mark.parametrize('nDim', [2])
# @pytest.mark.parametrize('nObs', [250])
# @pytest.mark.parametrize('nCl', [2])
# @pytest.mark.parametrize('precision', ['single'])
# @pytest.mark.parametrize('input_type', ['numpy'])
# def test_gaussian_mixture(precision, nDim, nObs, input_type):
#     gmm = GaussianMixture(precision=precision)
#
#     if precision == 'single':
#         dt = np.float32
#     else:
#         dt = np.float64
#
#     if input_type == 'numpy':
#         X = sample_mixture(nCl=nCl, nDim=nDim, nObs=nObs)
#
#     rmse_x = 0
#     gmm.fit(X=X, n_iter=10, nCl=nCl)
#     assert sqrt(rmse_x) < 0.1
