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

import cuml
import numpy as np
import pytest

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.metrics import adjusted_rand_score as cu_ars
from cuml.metrics.cluster import accuracy_score as cu_acc_score
from cuml.test.utils import get_handle, \
    fit_predict, get_pattern

from numba import cuda

from sklearn import cluster
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score as sk_acc_score
from sklearn.metrics.cluster import adjusted_rand_score as sk_ars
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_r2_score(datatype, use_handle):
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=datatype)
    b = np.array([0.12, 0.22, 0.32, 0.42, 0.52], dtype=datatype)

    a_dev = cuda.to_device(a)
    b_dev = cuda.to_device(b)

    handle, stream = get_handle(use_handle)

    score = cuml.metrics.r2_score(a_dev, b_dev, handle=handle)

    np.testing.assert_almost_equal(score, 0.98, decimal=7)


def test_sklearn_search():
    """Test ensures scoring function works with sklearn machinery
    """
    import numpy as np
    from cuml import Ridge as cumlRidge
    import cudf
    from sklearn import datasets
    from sklearn.model_selection import train_test_split, GridSearchCV
    diabetes = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data,
                                                        diabetes.target,
                                                        test_size=0.2,
                                                        shuffle=False,
                                                        random_state=1)

    alpha = np.array([1.0])
    fit_intercept = True
    normalize = False

    params = {'alpha': np.logspace(-3, -1, 10)}
    cu_clf = cumlRidge(alpha=alpha, fit_intercept=fit_intercept,
                       normalize=normalize, solver="eig")

    assert getattr(cu_clf, 'score', False)
    sk_cu_grid = GridSearchCV(cu_clf, params, cv=5, iid=False)

    record_data = (('fea%d' % i, X_train[:, i]) for i in
                   range(X_train.shape[1]))
    gdf_data = cudf.DataFrame(record_data)
    gdf_train = cudf.DataFrame(dict(train=y_train))

    sk_cu_grid.fit(gdf_data, gdf_train.train)
    assert sk_cu_grid.best_params_ == {'alpha': 0.1}


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('nrows', [unit_param(30), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(10), quality_param(100),
                         stress_param(200)])
@pytest.mark.parametrize('n_info', [unit_param(7), quality_param(50),
                         stress_param(100)])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_accuracy(nrows, ncols, n_info, datatype):

    use_handle = True
    train_rows = np.int32(nrows*0.8)
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=123, n_classes=5)
    X_test = np.asarray(X[train_rows:, 0:]).astype(datatype)
    y_test = np.asarray(y[train_rows:, ]).astype(np.int32)
    X_train = np.asarray(X[0:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[0:train_rows, ]).astype(np.int32)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=1.0,
                       n_bins=8, split_algo=0, split_criterion=0,
                       min_rows_per_node=2,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=-1)
    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_acc = cu_acc_score(y_test, cu_predict)

    # sklearn random forest classification model
    # initialization, fit and predict
    sk_model = skrfc(n_estimators=40, max_depth=None,
                     min_samples_split=2, max_features=1.0,
                     random_state=10)
    sk_model.fit(X_train, y_train)
    sk_predict = sk_model.predict(X_test)
    sk_acc = sk_acc_score(y_test, sk_predict)
    # compare the accuracy of the two models
    assert cu_acc >= (sk_acc - 0.07)

dataset_names = ['noisy_circles', 'noisy_moons', 'aniso'] + \
                [pytest.param(ds, marks=pytest.mark.xfail)
                 for ds in ['blobs', 'varied']]


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
def test_rand_index_score(name, nrows):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred, _ = fit_predict(cuml_kmeans,
                               'cuml_Kmeans', X)

    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    sk_y_pred, _ = fit_predict(kmeans,
                               'sk_Kmeans', X)
    cu_score = cu_ars(y, cu_y_pred)
    sk_score = sk_ars(y, sk_y_pred)

    assert cu_score >= sk_score
