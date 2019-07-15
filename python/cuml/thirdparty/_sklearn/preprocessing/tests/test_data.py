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

import pytest
import cupy as cp
import cudf
import numpy as np
from cupyx.scipy import sparse

from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import clean_warning_registry
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import skip_if_32bit

# from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.preprocessing.data import _handle_zeros_in_scale
from sklearn.preprocessing.data import Binarizer
from sklearn.preprocessing.data import KernelCenterer
from sklearn.preprocessing.data import Normalizer
from sklearn.preprocessing.data import normalize
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing.data import scale
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.preprocessing.data import minmax_scale
# from sklearn.preprocessing.data import QuantileTransformer
# from sklearn.preprocessing.data import quantile_transform
from sklearn.preprocessing.data import MaxAbsScaler
from sklearn.preprocessing.data import maxabs_scale
from sklearn.preprocessing.data import RobustScaler
from sklearn.preprocessing.data import robust_scale
from sklearn.preprocessing.data import add_dummy_feature
# from sklearn.preprocessing.data import PolynomialFeatures
# from sklearn.preprocessing.data import PowerTransformer
# from sklearn.preprocessing.data import power_transform

from sklearn import datasets

iris = datasets.load_iris()

# Make some data to be used many times
rng = cp.random.RandomState(0)
n_features = 30
n_samples = 1000
offsets = rng.uniform(-1, 1, size=n_features)
scales = rng.uniform(1, 10, size=n_features)
X_2d = rng.randn(n_samples, n_features) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_features)
X_1col = X_2d[:, 0].reshape(n_samples, 1)
X_list_1row = X_1row.tolist()
X_list_1col = X_1col.tolist()


# def test_standard_scaler_1d():
#     # Test scaling of dataset along single axis
#     for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

#         scaler = StandardScaler()
#         X_scaled = scaler.fit(X).transform(X, copy=True)

#         if isinstance(X, list):
#             X = cp.array(X)  # cast only after scaling done

#         if _check_dim_1axis(X) == 1:
#             assert_almost_equal(scaler.mean_, X.ravel())
#             assert_almost_equal(scaler.scale_, np.ones(n_features))
#             assert_array_almost_equal(X_scaled.mean(axis=0),
#                                       np.zeros_like(n_features))
#             assert_array_almost_equal(X_scaled.std(axis=0),
#                                       np.zeros_like(n_features))
#         else:
#             assert_almost_equal(scaler.mean_, X.mean())
#             assert_almost_equal(scaler.scale_, X.std())
#             assert_array_almost_equal(X_scaled.mean(axis=0),
#                                       cp.zeros_like(n_features))
#             assert_array_almost_equal(X_scaled.mean(axis=0), .0)
#             assert_array_almost_equal(X_scaled.std(axis=0), 1.)
#         assert_equal(scaler.n_samples_seen_, X.shape[0])

#         # check inverse transform
#         X_scaled_back = scaler.inverse_transform(X_scaled)
#         assert_array_almost_equal(X_scaled_back, X)

#     # Constant feature
#     X = cp.ones((5, 1))
#     scaler = StandardScaler()
#     X_scaled = scaler.fit(X).transform(X, copy=True)
#     assert_almost_equal(scaler.mean_, 1.)
#     assert_almost_equal(scaler.scale_, 1.)
#     assert_array_almost_equal(X_scaled.mean(axis=0), .0)
#     assert_array_almost_equal(X_scaled.std(axis=0), .0)
#     assert_equal(scaler.n_samples_seen_, X.shape[0])
