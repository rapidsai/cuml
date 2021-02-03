# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
import pytest

from cuml.test.utils import get_pattern, unit_param, \
    quality_param, stress_param, array_equal, assert_dbscan_equal

from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


@pytest.mark.mg
@pytest.mark.parametrize('max_mbytes_per_batch', [1e3, None])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(20), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('out_dtype', [unit_param("int32"),
                                       unit_param(np.int32),
                                       unit_param("int64"),
                                       unit_param(np.int64),
                                       quality_param("int32"),
                                       stress_param("int32")])
def test_dbscan(datatype, nrows, ncols,
                max_mbytes_per_batch, out_dtype, client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    n_samples = nrows
    n_feats = ncols
    X, y = make_blobs(n_samples=n_samples, cluster_std=0.01,
                      n_features=n_feats, random_state=0)

    eps = 1
    cuml_dbscan = cuDBSCAN(eps=eps, min_samples=2,
                           max_mbytes_per_batch=max_mbytes_per_batch,
                           output_type='numpy')

    cu_labels = cuml_dbscan.fit_predict(X, out_dtype=out_dtype)

    if nrows < 500000:
        sk_dbscan = skDBSCAN(eps=1, min_samples=2, algorithm="brute")
        sk_labels = sk_dbscan.fit_predict(X)

        # Check the core points are equal
        assert array_equal(cuml_dbscan.core_sample_indices_,
                           sk_dbscan.core_sample_indices_)

        # Check the labels are correct
        assert_dbscan_equal(sk_labels, cu_labels, X,
                            cuml_dbscan.core_sample_indices_, eps)

    if out_dtype == "int32" or out_dtype == np.int32:
        assert cu_labels.dtype == np.int32
    elif out_dtype == "int64" or out_dtype == np.int64:
        assert cu_labels.dtype == np.int64


@pytest.mark.mg
@pytest.mark.parametrize("name", [
                                 'noisy_moons',
                                 'blobs',
                                 'no_structure'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
# Vary the eps to get a range of core point counts
@pytest.mark.parametrize('eps', [0.05, 0.1, 0.5])
def test_dbscan_sklearn_comparison(name, nrows, eps, client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    default_base = {'quantile': .2,
                    'eps': eps,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 2}

    n_samples = nrows
    pat = get_pattern(name, n_samples)
    params = default_base.copy()
    params.update(pat[1])
    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cuml_dbscan = cuDBSCAN(eps=params['eps'], min_samples=5,
                           output_type='numpy')
    cu_labels = cuml_dbscan.fit_predict(X)

    if nrows < 500000:
        sk_dbscan = skDBSCAN(eps=params['eps'], min_samples=5)
        sk_labels = sk_dbscan.fit_predict(X)

        assert_dbscan_equal(sk_labels, cu_labels, X,
                            cuml_dbscan.core_sample_indices_, eps)

        # Check the core points are equal
        assert array_equal(cuml_dbscan.core_sample_indices_,
                           sk_dbscan.core_sample_indices_)

        # Check the labels are correct
        assert_dbscan_equal(sk_labels, cu_labels, X,
                            cuml_dbscan.core_sample_indices_, eps)


@pytest.mark.mg
@pytest.mark.parametrize("name", [
                                 'noisy_moons',
                                 'blobs',
                                 'no_structure'])
def test_dbscan_default(name, client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    eps = 0.5
    default_base = {'quantile': .3,
                    'eps': eps,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 2}
    n_samples = 500
    pat = get_pattern(name, n_samples)
    params = default_base.copy()
    params.update(pat[1])
    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cuml_dbscan = cuDBSCAN(output_type='numpy')
    cu_labels = cuml_dbscan.fit_predict(X)

    sk_dbscan = skDBSCAN(eps=params['eps'], min_samples=5)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(cuml_dbscan.core_sample_indices_,
                       sk_dbscan.core_sample_indices_)

    # Check the labels are correct
    assert_dbscan_equal(sk_labels, cu_labels, X,
                        cuml_dbscan.core_sample_indices_, eps)


@pytest.mark.mg
@pytest.mark.xfail(strict=True, raises=ValueError)
def test_dbscan_out_dtype_fails_invalid_input(client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    X, _ = make_blobs(n_samples=500)

    cuml_dbscan = cuDBSCAN(output_type='numpy')
    cuml_dbscan.fit_predict(X, out_dtype="bad_input")


@pytest.mark.mg
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('out_dtype', ["int32", np.int32, "int64", np.int64])
def test_dbscan_propagation(datatype, out_dtype, client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    X, y = make_blobs(5000, centers=1, cluster_std=8.0,
                      center_box=(-100.0, 100.0), random_state=8)
    X = X.astype(datatype)

    eps = 0.5
    cuml_dbscan = cuDBSCAN(eps=eps, min_samples=5,
                           output_type='numpy')
    cu_labels = cuml_dbscan.fit_predict(X, out_dtype=out_dtype)

    sk_dbscan = skDBSCAN(eps=eps, min_samples=5)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(cuml_dbscan.core_sample_indices_,
                       sk_dbscan.core_sample_indices_)

    # Check the labels are correct
    assert_dbscan_equal(sk_labels, cu_labels, X,
                        cuml_dbscan.core_sample_indices_, eps)


@pytest.mark.mg
def test_dbscan_no_calc_core_point_indices(client):
    from cuml.dask.cluster.dbscan import DBSCAN as cuDBSCAN

    params = {'eps': 1.1, 'min_samples': 4}
    n_samples = 1000
    pat = get_pattern("noisy_moons", n_samples)

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    # Set calc_core_sample_indices=False
    cuml_dbscan = cuDBSCAN(eps=params['eps'], min_samples=5,
                           output_type='numpy', calc_core_sample_indices=False)
    cuml_dbscan.fit_predict(X)

    # Make sure we are None
    assert(cuml_dbscan.core_sample_indices_ is None)
