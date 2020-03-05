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

import cuml
import cupy as cp
import numpy as np
import pytest

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.metrics.cluster import adjusted_rand_score as cu_ars
from cuml.metrics import accuracy_score as cu_acc_score
from cuml.test.utils import get_handle, get_pattern, array_equal, \
    unit_param, quality_param, stress_param, generate_random_labels, \
    score_labeling_with_handle

from numba import cuda
from numpy.testing import assert_almost_equal

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score as sk_acc_score
from sklearn.metrics.cluster import adjusted_rand_score as sk_ars
from sklearn.metrics.cluster import homogeneity_score as sk_homogeneity_score
from sklearn.metrics.cluster import mutual_info_score as sk_mutual_info_score
from sklearn.metrics.cluster import completeness_score as sk_completeness_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error as sklearn_mse

from cuml.metrics.cluster import entropy
from cuml.metrics.regression import mean_squared_error

from scipy.stats import entropy as sp_entropy


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

    gdf_data = cudf.DataFrame.from_gpu_matrix(cuda.to_device(X_train))
    gdf_train = cudf.DataFrame(dict(train=y_train))

    sk_cu_grid.fit(gdf_data, gdf_train.train)
    assert sk_cu_grid.best_params_ == {'alpha': 0.1}


@pytest.mark.parametrize('nrows', [unit_param(30), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(10), quality_param(100),
                         stress_param(200)])
@pytest.mark.parametrize('n_info', [unit_param(7), quality_param(50),
                         stress_param(100)])
@pytest.mark.parametrize('datatype', [np.float32])
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
    handle, stream = get_handle(use_handle, n_streams=8)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=1.0,
                       n_bins=8, split_algo=0, split_criterion=0,
                       min_rows_per_node=2,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16)

    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_acc = cu_acc_score(y_test, cu_predict)
    cu_acc_using_sk = sk_acc_score(y_test, cu_predict)
    # compare the accuracy of the two models
    assert array_equal(cu_acc, cu_acc_using_sk)


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

    cu_y_pred = cuml_kmeans.fit_predict(X)

    cu_score = cu_ars(y, cu_y_pred)
    cu_score_using_sk = sk_ars(y, cp.asnumpy(cu_y_pred))

    assert array_equal(cu_score, cu_score_using_sk)


def score_homogeneity(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(cuml.metrics.homogeneity_score,
                                      ground_truth,
                                      predictions,
                                      use_handle,
                                      dtype=np.int32)


def score_mutual_info(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(cuml.metrics.mutual_info_score,
                                      ground_truth,
                                      predictions,
                                      use_handle,
                                      dtype=np.int32)


def score_completeness(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(cuml.metrics.completeness_score,
                                      ground_truth,
                                      predictions,
                                      use_handle,
                                      dtype=np.int32)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [1, 1, 0, 0]),
                                  ([0, 0, 1, 1], [0, 0, 1, 1])])
def test_homogeneity_perfect_labeling(use_handle, data):
    # Perfect labelings are homogeneous
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 1.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [0, 0, 1, 2]),
                                  ([0, 0, 1, 1], [0, 1, 2, 3])])
def test_homogeneity_non_perfect_labeling(use_handle, data):
    # Non-perfect labelings that further split classes into more clusters can
    # be perfectly homogeneous
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 1.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [0, 1, 0, 1]),
                                  ([0, 0, 1, 1], [0, 0, 0, 0])])
def test_homogeneity_non_homogeneous_labeling(use_handle, data):
    # Clusters that include samples from different classes do not make for an
    # homogeneous labeling
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 0.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_range', [[0, 1000],
                                         [-1000, 1000]])
def test_homogeneity_score_big_array(use_handle, input_range):
    a, b = generate_random_labels(lambda rng: rng.randint(*input_range,
                                                          int(10e4),
                                                          dtype=np.int32))
    score = score_homogeneity(a, b, use_handle)
    ref = sk_homogeneity_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_labels', [([0, 0, 1, 1], [1, 1, 0, 0]),
                                          ([0, 0, 1, 1], [0, 0, 1, 1]),
                                          ([0, 0, 1, 1], [0, 0, 1, 2]),
                                          ([0, 0, 1, 1], [0, 1, 2, 3]),
                                          ([0, 0, 1, 1], [0, 1, 0, 1]),
                                          ([0, 0, 1, 1], [0, 0, 0, 0])])
def test_mutual_info_score(use_handle, input_labels):
    score = score_mutual_info(*input_labels, use_handle)
    ref = sk_mutual_info_score(*input_labels)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_range', [[0, 1000],
                                         [-1000, 1000]])
def test_mutual_info_score_big_array(use_handle, input_range):
    a, b = generate_random_labels(lambda rng: rng.randint(*input_range,
                                                          int(10e4),
                                                          dtype=np.int32))
    score = score_mutual_info(a, b, use_handle)
    ref = sk_mutual_info_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_range', [[0, 2],
                                         [-5, 20],
                                         [int(-10e3), int(10e3)]])
def test_homogeneity_completeness_symmetry(use_handle, input_range):
    a, b = generate_random_labels(lambda rng: rng.randint(*input_range,
                                                          int(10e3),
                                                          dtype=np.int32))
    hom = score_homogeneity(a, b, use_handle)
    com = score_completeness(b, a, use_handle)
    np.testing.assert_almost_equal(hom, com, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [1, 1, 0, 0]),
                                  ([0, 0, 1, 1], [0, 0, 1, 1])])
def test_completeness_perfect_labeling(use_handle, data):
    # Perfect labelings are complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 1.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [0, 0, 0, 0]),
                                  ([0, 1, 2, 3], [0, 0, 1, 1])])
def test_completeness_non_perfect_labeling(use_handle, data):
    # Non-perfect labelings that assign all classes members to the same
    # clusters are still complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 1.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('data', [([0, 0, 1, 1], [0, 1, 0, 1]),
                                  ([0, 0, 0, 0], [0, 1, 2, 3])])
def test_completeness_non_complete_labeling(use_handle, data):
    # If classes members are split across different clusters, the assignment
    # cannot be complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 0.0, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_range', [[0, 1000],
                                         [-1000, 1000]])
def test_completeness_score_big_array(use_handle, input_range):
    a, b = generate_random_labels(lambda rng: rng.randint(*input_range,
                                                          int(10e4),
                                                          dtype=np.int32))
    score = score_completeness(a, b, use_handle)
    ref = sk_completeness_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('input_range', [[0, 19],
                                         [0, 2],
                                         [-5, 20]])
@pytest.mark.parametrize('n_samples', [129, 258])
def test_mutual_info_score_many_blocks(use_handle, input_range, n_samples):
    a, b = generate_random_labels(lambda rng: rng.randint(*input_range,
                                                          n_samples,
                                                          dtype=np.int32))
    score = score_mutual_info(a, b, use_handle)
    ref = sk_mutual_info_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


def test_mean_squared_error():
    y_true = np.arange(50, dtype=np.int)
    y_pred = y_true + 1
    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.)


@pytest.mark.parametrize('n_samples', [50, stress_param(500000)])
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_mean_squared_error_random(n_samples, dtype):
    if dtype == np.float32 and n_samples == 500000:
        # stress test for float32 fails because of floating point precision
        pytest.xfail()

    y_true, y_pred = generate_random_labels(
        lambda rng: rng.randint(0, 1000, n_samples).astype(dtype))
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    skl_mse = sklearn_mse(y_true, y_pred, multioutput='raw_values')
    cp.testing.assert_array_almost_equal(mse, skl_mse, decimal=2)


def test_mean_squared_error_at_limits():
    y_true = np.array([0.], dtype=np.float)
    y_pred = np.array([0.], dtype=np.float)
    assert_almost_equal(mean_squared_error(y_true, y_pred), 0.00, decimal=2)
    assert_almost_equal(mean_squared_error(y_true, y_pred, squared=False),
                        0.00, decimal=2)


def test_mean_squared_error_multioutput_array():
    y_true = np.array([[1, 2], [2.5, -1], [4.5, 3], [5, 7]], dtype=np.float)
    y_pred = np.array([[1, 1], [2, -1], [5, 4], [5, 6.5]], dtype=np.float)

    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    cp.testing.assert_array_almost_equal(mse, [0.125, 0.5625], decimal=2)

    weights = np.array([0.4, 0.6], dtype=np.float)
    msew = mean_squared_error(y_true, y_pred, multioutput=weights)
    rmsew = mean_squared_error(y_true, y_pred, multioutput=weights,
                               squared=False)
    assert_almost_equal(msew, 0.39, decimal=2)
    assert_almost_equal(rmsew, 0.62, decimal=2)

    y_true = np.array([[0, 0]] * 4, dtype=np.int)
    y_pred = np.array([[1, 1]] * 4, dtype=np.int)
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    cp.testing.assert_array_almost_equal(mse, [1., 1.], decimal=2)


def test_mean_squared_error_custom_weights():
    y_true = np.array([1, 2, 2.5, -1], dtype=np.float)
    y_pred = np.array([1, 1, 2, -1], dtype=np.float)
    weights = np.array([0.2, 0.25, 0.4, 0.15], dtype=np.float)

    mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
    skl_mse = sklearn_mse(y_true, y_pred, sample_weight=weights)

    assert_almost_equal(mse, skl_mse, decimal=2)


@pytest.mark.parametrize('use_handle', [True, False])
def test_entropy(use_handle):
    handle, stream = get_handle(use_handle)

    # The outcome of a fair coin is the most uncertain:
    # in base 2 the result is 1 (One bit of entropy).
    cluster = np.array([0, 1], dtype=np.int32)
    assert_almost_equal(entropy(cluster, base=2., handle=handle), 1.)

    # The outcome of a biased coin is less uncertain:
    cluster = np.array(([0] * 9) + [1], dtype=np.int32)
    assert_almost_equal(entropy(cluster, base=2., handle=handle), 0.468995593)
    # base e
    assert_almost_equal(entropy(cluster, handle=handle), 0.32508297339144826)


@pytest.mark.parametrize('n_samples', [50, stress_param(500000)])
@pytest.mark.parametrize('base', [None, 2, 10, 50])
@pytest.mark.parametrize('use_handle', [True, False])
def test_entropy_random(n_samples, base, use_handle):
    handle, stream = get_handle(use_handle)

    clustering, _ = \
        generate_random_labels(lambda rng: rng.randint(0, 1000, n_samples))

    # generate unormalized probabilities from clustering
    pk = np.bincount(clustering)

    # scipy's entropy uses probabilities
    sp_S = sp_entropy(pk, base=base)
    # we use a clustering
    S = entropy(np.array(clustering, dtype=np.int32), base, handle=handle)

    assert_almost_equal(S, sp_S, decimal=2)
