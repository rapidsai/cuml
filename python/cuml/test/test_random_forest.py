# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import cudf
import numpy as np
import pytest
import random
import json
import io
from contextlib import redirect_stdout

from numba import cuda

import cuml
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.metrics import r2_score
from cuml.test.utils import get_handle, unit_param, \
    quality_param, stress_param
import cuml.common.logger as logger

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import fetch_california_housing, \
    make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture(
    scope="session",
    params=[
        unit_param({'n_samples': 350, 'n_features': 20, 'n_informative': 10}),
        quality_param({'n_samples': 5000, 'n_features': 200,
                      'n_informative': 80}),
        stress_param({'n_samples': 500000, 'n_features': 400,
                     'n_informative': 180})
    ])
def small_clf(request):
    X, y = make_classification(n_samples=request.param['n_samples'],
                               n_features=request.param['n_features'],
                               n_clusters_per_class=1,
                               n_informative=request.param['n_informative'],
                               random_state=123, n_classes=2)
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({'n_samples': 350, 'n_features': 30, 'n_informative': 15}),
        quality_param({'n_samples': 5000, 'n_features': 200,
                      'n_informative': 80}),
        stress_param({'n_samples': 500000, 'n_features': 400,
                     'n_informative': 180})
    ])
def mclass_clf(request):
    X, y = make_classification(n_samples=request.param['n_samples'],
                               n_features=request.param['n_features'],
                               n_clusters_per_class=1,
                               n_informative=request.param['n_informative'],
                               random_state=123, n_classes=10)
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({'n_samples': 500, 'n_features': 20, 'n_informative': 10}),
        quality_param({'n_samples': 5000, 'n_features': 200,
                      'n_informative': 50}),
        stress_param({'n_samples': 500000, 'n_features': 400,
                     'n_informative': 100})
    ])
def large_clf(request):
    X, y = make_classification(n_samples=request.param['n_samples'],
                               n_features=request.param['n_features'],
                               n_clusters_per_class=1,
                               n_informative=request.param['n_informative'],
                               random_state=123, n_classes=2)
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({'n_samples': 1500, 'n_features': 20, 'n_informative': 10}),
        quality_param({'n_samples': 12000, 'n_features': 200,
                      'n_informative': 100}),
        stress_param({'n_samples': 500000, 'n_features': 500,
                     'n_informative': 350})
    ])
def large_reg(request):
    X, y = make_regression(n_samples=request.param['n_samples'],
                           n_features=request.param['n_features'],
                           n_informative=request.param['n_informative'],
                           random_state=123)
    return X, y


special_reg_params = [
        unit_param({'mode': 'unit', 'n_samples': 500,
                   'n_features': 20, 'n_informative': 10}),
        quality_param({'mode': 'quality', 'n_samples': 500,
                      'n_features': 20, 'n_informative': 10}),
        quality_param({'mode': 'quality', 'n_features': 200,
                      'n_informative': 50}),
        stress_param({'mode': 'stress', 'n_samples': 500,
                     'n_features': 20, 'n_informative': 10}),
        stress_param({'mode': 'stress', 'n_features': 200,
                     'n_informative': 50}),
        stress_param({'mode': 'stress', 'n_samples': 1000,
                     'n_features': 400, 'n_informative': 100})
    ]


@pytest.fixture(
    scope="session",
    params=special_reg_params)
def special_reg(request):
    if request.param['mode'] == 'quality':
        X, y = fetch_california_housing(return_X_y=True)
    else:
        X, y = make_regression(n_samples=request.param['n_samples'],
                               n_features=request.param['n_features'],
                               n_informative=request.param['n_informative'],
                               random_state=123)
    return X, y


@pytest.mark.parametrize('max_samples', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
@pytest.mark.parametrize('use_experimental_backend', [True, False])
def test_rf_classification(small_clf, datatype, split_algo,
                           max_samples, max_features,
                           use_experimental_backend):
    use_handle = True

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=max_features, max_samples=max_samples,
                       n_bins=16, split_algo=split_algo, split_criterion=0,
                       min_samples_leaf=2, random_state=123, n_streams=1,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16,
                       use_experimental_backend=use_experimental_backend)
    f = io.StringIO()
    with redirect_stdout(f):
        cuml_model.fit(X_train, y_train)
    captured_stdout = f.getvalue()
    if use_experimental_backend:
        is_fallback_used = False
        if split_algo != 1:
            assert ('Experimental backend does not yet support histogram ' +
                    'split algorithm' in captured_stdout)
            is_fallback_used = True
        if is_fallback_used:
            assert ('Not using the experimental backend due to above ' +
                    'mentioned reason(s)' in captured_stdout)
        else:
            assert ('Using experimental backend for growing trees'
                    in captured_stdout)
    else:
        assert captured_stdout == ''
    fil_preds = cuml_model.predict(X_test,
                                   predict_model="GPU",
                                   threshold=0.5,
                                   algo='auto')
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))
    cuml_acc = accuracy_score(y_test, cu_preds)
    fil_acc = accuracy_score(y_test, fil_preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(n_estimators=40,
                         max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert fil_acc >= (sk_acc - 0.07)
    assert fil_acc >= (cuml_acc - 0.02)


@pytest.mark.parametrize('max_samples', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize(
    'split_algo,max_features,use_experimental_backend,n_bins',
    [(0, 1.0, False, 16),
     (1, 1.0, False, 11),
     (0, 'auto', False, 128),
     (1, 'log2', False, 100),
     (1, 'sqrt', False, 100),
     (1, 1.0, True, 17),
     (1, 1.0, True, 32),
     ])
def test_rf_regression(special_reg, datatype, split_algo, max_features,
                       max_samples, use_experimental_backend, n_bins):

    use_handle = True

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(max_features=max_features, max_samples=max_samples,
                       n_bins=n_bins, split_algo=split_algo, split_criterion=2,
                       min_samples_leaf=2, random_state=123, n_streams=1,
                       n_estimators=50, handle=handle, max_leaves=-1,
                       max_depth=16, accuracy_metric='mse',
                       use_experimental_backend=use_experimental_backend)
    cuml_model.fit(X_train, y_train)
    # predict using FIL
    fil_preds = cuml_model.predict(X_test, predict_model="GPU")
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))

    cu_r2 = r2_score(y_test, cu_preds, convert_dtype=datatype)
    fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype)
    # Initialize, fit and predict using
    # sklearn's random forest regression model
    if X.shape[0] < 1000:  # mode != "stress"
        sk_model = skrfr(n_estimators=50, max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds, convert_dtype=datatype)
        assert fil_r2 >= (sk_r2 - 0.07)
    assert fil_r2 >= (cu_r2 - 0.02)


@pytest.mark.parametrize('datatype', [np.float32])
def test_rf_classification_seed(small_clf, datatype):

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    for i in range(8):
        seed = random.randint(100, 1e5)
        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class = curfc(random_state=seed, n_streams=1)
        cu_class.fit(X_train, y_train)

        # predict using FIL
        fil_preds_orig = cu_class.predict(X_test,
                                          predict_model="GPU")
        cu_preds_orig = cu_class.predict(X_test,
                                         predict_model="CPU")
        cu_acc_orig = accuracy_score(y_test, cu_preds_orig)
        fil_preds_orig = np.reshape(fil_preds_orig, np.shape(cu_preds_orig))

        fil_acc_orig = accuracy_score(y_test, fil_preds_orig)

        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class2 = curfc(random_state=seed, n_streams=1)
        cu_class2.fit(X_train, y_train)

        # predict using FIL
        fil_preds_rerun = cu_class2.predict(X_test,
                                            predict_model="GPU")
        cu_preds_rerun = cu_class2.predict(X_test, predict_model="CPU")
        cu_acc_rerun = accuracy_score(y_test, cu_preds_rerun)
        fil_preds_rerun = np.reshape(fil_preds_rerun, np.shape(cu_preds_rerun))

        fil_acc_rerun = accuracy_score(y_test, fil_preds_rerun)

        assert fil_acc_orig == fil_acc_rerun
        assert cu_acc_orig == cu_acc_rerun
        assert (fil_preds_orig == fil_preds_rerun).all()
        assert (cu_preds_orig == cu_preds_rerun).all()


@pytest.mark.parametrize('datatype', [(np.float64, np.float32),
                                      (np.float32, np.float64)])
@pytest.mark.parametrize('convert_dtype', [True, False])
def test_rf_classification_float64(small_clf, datatype, convert_dtype):

    X, y = small_clf
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_test = X_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_acc = accuracy_score(y_test, cu_preds)

    # sklearn random forest classification model
    # initialization, fit and predict
    if X.shape[0] < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert cu_acc >= (sk_acc - 0.07)

    # predict using cuML's GPU based prediction
    if datatype[0] == np.float32 and convert_dtype:
        fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                       convert_dtype=convert_dtype)
        fil_preds = np.reshape(fil_preds, np.shape(cu_preds))

        fil_acc = accuracy_score(y_test, fil_preds)
        assert fil_acc >= (cu_acc - 0.02)
    else:
        with pytest.raises(TypeError):
            fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                           convert_dtype=convert_dtype)


@pytest.mark.parametrize('datatype', [(np.float64, np.float32),
                                      (np.float32, np.float64)])
def test_rf_regression_float64(large_reg, datatype):

    X, y = large_reg
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_train = X_train.astype(datatype[0])
    y_train = y_train.astype(datatype[0])
    X_test = X_test.astype(datatype[1])
    y_test = y_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_r2 = r2_score(y_test, cu_preds, convert_dtype=datatype[0])

    # sklearn random forest classification model
    # initialization, fit and predict
    if X.shape[0] < 500000:
        sk_model = skrfr(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds, convert_dtype=datatype[0])
        assert cu_r2 >= (sk_r2 - 0.09)

    # predict using cuML's GPU based prediction
    if datatype[0] == np.float32:
        fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                       convert_dtype=True)
        fil_preds = np.reshape(fil_preds, np.shape(cu_preds))
        fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype[0])
        assert fil_r2 >= (cu_r2 - 0.02)

    #  because datatype[0] != np.float32 or datatype[0] != datatype[1]
    with pytest.raises(TypeError):
        fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                       convert_dtype=False)


def check_predict_proba(test_proba, baseline_proba, y_test, rel_err):
    y_proba = np.zeros(np.shape(baseline_proba))
    for count, _class in enumerate(y_test):
        y_proba[count, _class] = 1
    baseline_mse = mean_squared_error(y_proba, baseline_proba)
    test_mse = mean_squared_error(y_proba, test_proba)
    # using relative error is more stable when changing decision tree
    # parameters, column or class count
    assert test_mse <= baseline_mse * (1.0 + rel_err)


def rf_classification(datatype, array_type, max_features, max_samples,
                      fixture):
    X, y = fixture
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_test = X_test.astype(datatype[1])

    handle, stream = get_handle(True, n_streams=1)
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=max_features, max_samples=max_samples,
                       n_bins=16, split_criterion=0,
                       min_samples_leaf=2, random_state=123,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16)
    if array_type == 'dataframe':
        X_train_df = cudf.DataFrame(X_train)
        y_train_df = cudf.Series(y_train)
        X_test_df = cudf.DataFrame(X_test)
        cuml_model.fit(X_train_df, y_train_df)
        cu_proba_gpu = np.array(cuml_model.predict_proba(X_test_df)
                                .as_gpu_matrix())
        cu_preds_cpu = cuml_model.predict(X_test_df,
                                          predict_model="CPU").to_array()
        cu_preds_gpu = cuml_model.predict(X_test_df,
                                          predict_model="GPU").to_array()
    else:
        cuml_model.fit(X_train, y_train)
        cu_proba_gpu = cuml_model.predict_proba(X_test)
        cu_preds_cpu = cuml_model.predict(X_test, predict_model="CPU")
        cu_preds_gpu = cuml_model.predict(X_test, predict_model="GPU")
    np.testing.assert_array_equal(cu_preds_gpu,
                                  np.argmax(cu_proba_gpu, axis=1))

    cu_acc_cpu = accuracy_score(y_test, cu_preds_cpu)
    cu_acc_gpu = accuracy_score(y_test, cu_preds_gpu)
    assert cu_acc_cpu == pytest.approx(cu_acc_gpu, abs=0.01, rel=0.1)

    # sklearn random forest classification model
    # initialization, fit and predict
    if y.size < 500000:
        sk_model = skrfc(n_estimators=40,
                         max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        sk_proba = sk_model.predict_proba(X_test)
        assert cu_acc_cpu >= sk_acc - 0.07
        assert cu_acc_gpu >= sk_acc - 0.07
        # 0.06 is the highest relative error observed on CI, within
        # 0.0061 absolute error boundaries seen previously
        check_predict_proba(cu_proba_gpu, sk_proba, y_test, 0.1)


@pytest.mark.parametrize('datatype', [(np.float32, np.float32)])
@pytest.mark.parametrize('array_type', ['dataframe', 'numpy'])
def test_rf_classification_multi_class(mclass_clf, datatype, array_type):
    rf_classification(datatype, array_type, 1.0, 1.0, mclass_clf)


@pytest.mark.parametrize('datatype', [(np.float32, np.float32)])
@pytest.mark.parametrize('max_samples', [unit_param(1.0),
                         stress_param(0.95)])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_classification_proba(small_clf, datatype,
                                 max_samples, max_features):
    rf_classification(datatype, 'numpy', max_features, max_samples,
                      small_clf)


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('fil_sparse_format', ['not_supported', True,
                                               'auto', False])
@pytest.mark.parametrize('algo', ['auto', 'naive', 'tree_reorg',
                                  'batch_tree_reorg'])
def test_rf_classification_sparse(small_clf, datatype,
                                  fil_sparse_format, algo):
    use_handle = True
    num_treees = 50

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(n_bins=16, split_criterion=0,
                       min_samples_leaf=2, random_state=123, n_streams=1,
                       n_estimators=num_treees, handle=handle, max_leaves=-1,
                       max_depth=40)
    cuml_model.fit(X_train, y_train)

    if ((not fil_sparse_format or algo == 'tree_reorg' or
            algo == 'batch_tree_reorg') or
            fil_sparse_format == 'not_supported'):
        with pytest.raises(ValueError):
            fil_preds = cuml_model.predict(X_test,
                                           predict_model="GPU",
                                           threshold=0.5,
                                           fil_sparse_format=fil_sparse_format,
                                           algo=algo)
    else:
        fil_preds = cuml_model.predict(X_test,
                                       predict_model="GPU",
                                       threshold=0.5,
                                       fil_sparse_format=fil_sparse_format,
                                       algo=algo)
        fil_preds = np.reshape(fil_preds, np.shape(y_test))
        fil_acc = accuracy_score(y_test, fil_preds)
        np.testing.assert_almost_equal(fil_acc,
                                       cuml_model.score(X_test, y_test))

        fil_model = cuml_model.convert_to_fil_model()

        with cuml.using_output_type("numpy"):
            fil_model_preds = fil_model.predict(X_test)
            fil_model_acc = accuracy_score(y_test, fil_model_preds)
            assert fil_acc == fil_model_acc

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features

        if X.shape[0] < 500000:
            sk_model = skrfc(n_estimators=50,
                             max_depth=40,
                             min_samples_split=2,
                             random_state=10)
            sk_model.fit(X_train, y_train)
            sk_preds = sk_model.predict(X_test)
            sk_acc = accuracy_score(y_test, sk_preds)
            assert fil_acc >= (sk_acc - 0.07)


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('fil_sparse_format', ['not_supported', True,
                                               'auto', False])
@pytest.mark.parametrize('algo', ['auto', 'naive', 'tree_reorg',
                                  'batch_tree_reorg'])
def test_rf_regression_sparse(special_reg, datatype, fil_sparse_format, algo):
    use_handle = True
    num_treees = 50

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(n_bins=16, split_criterion=2,
                       min_samples_leaf=2, random_state=123, n_streams=1,
                       n_estimators=num_treees, handle=handle, max_leaves=-1,
                       max_depth=40, accuracy_metric='mse')
    cuml_model.fit(X_train, y_train)

    # predict using FIL
    if ((not fil_sparse_format or algo == 'tree_reorg' or
            algo == 'batch_tree_reorg') or
            fil_sparse_format == 'not_supported'):
        with pytest.raises(ValueError):
            fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                           fil_sparse_format=fil_sparse_format,
                                           algo=algo)
    else:
        fil_preds = cuml_model.predict(X_test, predict_model="GPU",
                                       fil_sparse_format=fil_sparse_format,
                                       algo=algo)
        fil_preds = np.reshape(fil_preds, np.shape(y_test))
        fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype)

        fil_model = cuml_model.convert_to_fil_model()

        with cuml.using_output_type("numpy"):
            fil_model_preds = fil_model.predict(X_test)
            fil_model_preds = np.reshape(fil_model_preds, np.shape(y_test))
            fil_model_r2 = r2_score(y_test, fil_model_preds,
                                    convert_dtype=datatype)
            assert fil_r2 == fil_model_r2

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features

        # Initialize, fit and predict using
        # sklearn's random forest regression model
        if X.shape[0] < 1000:  # mode != "stress":
            sk_model = skrfr(n_estimators=50, max_depth=40,
                             min_samples_split=2,
                             random_state=10)
            sk_model.fit(X_train, y_train)
            sk_preds = sk_model.predict(X_test)
            sk_r2 = r2_score(y_test, sk_preds, convert_dtype=datatype)
            assert fil_r2 >= (sk_r2 - 0.07)


@pytest.mark.xfail(reason='Need rapidsai/rmm#415 to detect memleak robustly')
@pytest.mark.memleak
@pytest.mark.parametrize('fil_sparse_format', [True, False, 'auto'])
@pytest.mark.parametrize('n_iter', [unit_param(5), quality_param(30),
                         stress_param(80)])
def test_rf_memory_leakage(small_clf, fil_sparse_format, n_iter):
    datatype = np.float32
    use_handle = True

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Warmup. Some modules that are used in RF allocate space on the device
    # and consume memory. This is to make sure that the allocation is done
    # before the first call to get_memory_info.
    base_model = curfc(handle=handle)
    base_model.fit(X_train, y_train)
    handle.sync()  # just to be sure
    free_mem = cuda.current_context().get_memory_info()[0]

    def test_for_memory_leak():
        cuml_mods = curfc(handle=handle)
        cuml_mods.fit(X_train, y_train)
        handle.sync()  # just to be sure
        # Calculate the memory free after fitting the cuML model
        delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
        assert delta_mem == 0

        for i in range(2):
            cuml_mods.predict(X_test, predict_model="GPU",
                              fil_sparse_format=fil_sparse_format)
            handle.sync()  # just to be sure
            # Calculate the memory free after predicting the cuML model
            delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
            assert delta_mem == 0

    for i in range(n_iter):
        test_for_memory_leak()


@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
@pytest.mark.parametrize('max_depth', [10, 13, 16])
@pytest.mark.parametrize('n_estimators', [10, 20, 100])
@pytest.mark.parametrize('n_bins', [8, 9, 10])
def test_create_classification_model(max_features,
                                     max_depth, n_estimators, n_bins):

    # random forest classification model
    cuml_model = curfc(max_features=max_features,
                       n_bins=n_bins,
                       n_estimators=n_estimators,
                       max_depth=max_depth)
    params = cuml_model.get_params()
    cuml_model2 = curfc()
    cuml_model2.set_params(**params)
    verfiy_params = cuml_model2.get_params()
    assert params['max_features'] == verfiy_params['max_features']
    assert params['max_depth'] == verfiy_params['max_depth']
    assert params['n_estimators'] == verfiy_params['n_estimators']
    assert params['n_bins'] == verfiy_params['n_bins']


@pytest.mark.parametrize('n_estimators', [10, 20, 100])
@pytest.mark.parametrize('n_bins', [8, 9, 10])
def test_multiple_fits_classification(large_clf, n_estimators, n_bins):

    datatype = np.float32
    X, y = large_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfc(n_bins=n_bins,
                       n_estimators=n_estimators,
                       max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params['n_estimators'] == n_estimators
    assert params['n_bins'] == n_bins


@pytest.mark.parametrize('column_info', [unit_param([100, 50]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_estimators', [10, 20, 100])
@pytest.mark.parametrize('n_bins', [8, 9, 10])
def test_multiple_fits_regression(column_info, nrows, n_estimators, n_bins):
    datatype = np.float32
    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info,
                           random_state=123)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfr(n_bins=n_bins,
                       n_estimators=n_estimators,
                       max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params['n_estimators'] == n_estimators
    assert params['n_bins'] == n_bins


@pytest.mark.parametrize('n_estimators', [5, 10, 20])
@pytest.mark.parametrize('detailed_text', [True, False])
def test_rf_get_text(n_estimators, detailed_text):

    X, y = make_classification(n_samples=500, n_features=10,
                               n_clusters_per_class=1, n_informative=5,
                               random_state=94929, n_classes=2)

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Create a handle for the cuml model
    handle, stream = get_handle(True, n_streams=1)

    # Initialize cuML Random Forest classification model
    cuml_model = curfc(handle=handle, max_features=1.0, max_samples=1.0,
                       n_bins=16, split_algo=0, split_criterion=0,
                       min_samples_leaf=2, random_state=23707, n_streams=1,
                       n_estimators=n_estimators, max_leaves=-1,
                       max_depth=16)

    # Train model on the data
    cuml_model.fit(X, y)

    if detailed_text:
        text_output = cuml_model.get_detailed_text()
    else:
        text_output = cuml_model.get_summary_text()

    # Test 1: Output is non-zero
    assert '' != text_output

    # Count the number of trees printed
    tree_count = 0
    for line in text_output.split('\n'):
        if line.strip().startswith('Tree #'):
            tree_count += 1

    # Test 2: Correct number of trees are printed
    assert n_estimators == tree_count


@pytest.mark.parametrize('max_depth', [1, 2, 3, 5, 10, 15, 20])
@pytest.mark.parametrize('n_estimators', [5, 10, 20])
@pytest.mark.parametrize('estimator_type', ['regression', 'classification'])
def test_rf_get_json(estimator_type, max_depth, n_estimators):
    X, y = make_classification(n_samples=350, n_features=20,
                               n_clusters_per_class=1, n_informative=10,
                               random_state=123, n_classes=2)
    X = X.astype(np.float32)
    if estimator_type == 'classification':
        cuml_model = curfc(max_features=1.0, max_samples=1.0,
                           n_bins=16, split_algo=0, split_criterion=0,
                           min_samples_leaf=2, random_state=23707, n_streams=1,
                           n_estimators=n_estimators, max_leaves=-1,
                           max_depth=max_depth)
        y = y.astype(np.int32)
    elif estimator_type == 'regression':
        cuml_model = curfr(max_features=1.0, max_samples=1.0,
                           n_bins=16, split_algo=0,
                           min_samples_leaf=2, random_state=23707, n_streams=1,
                           n_estimators=n_estimators, max_leaves=-1,
                           max_depth=max_depth)
        y = y.astype(np.float32)
    else:
        assert False

    # Train model on the data
    cuml_model.fit(X, y)

    json_out = cuml_model.get_json()
    json_obj = json.loads(json_out)

    # Test 1: Output is non-zero
    assert '' != json_out

    # Test 2: JSON object contains correct number of trees
    assert isinstance(json_obj, list)
    assert len(json_obj) == n_estimators

    # Test 3: Traverse JSON trees and get the same predictions as cuML RF
    def predict_with_json_tree(tree, x):
        if 'children' not in tree:
            assert 'leaf_value' in tree
            return tree['leaf_value']
        assert 'split_feature' in tree
        assert 'split_threshold' in tree
        assert 'yes' in tree
        assert 'no' in tree
        if x[tree['split_feature']] <= tree['split_threshold']:
            return predict_with_json_tree(tree['children'][0], x)
        return predict_with_json_tree(tree['children'][1], x)

    def predict_with_json_rf_classifier(rf, x):
        # Returns the class with the highest vote. If there is a tie, return
        # the list of all classes with the highest vote.
        vote = []
        for tree in rf:
            vote.append(predict_with_json_tree(tree, x))
        vote = np.bincount(vote)
        max_vote = np.max(vote)
        majority_vote = np.nonzero(np.equal(vote, max_vote))[0]
        return majority_vote

    def predict_with_json_rf_regressor(rf, x):
        pred = 0.
        for tree in rf:
            pred += predict_with_json_tree(tree, x)
        return pred / len(rf)

    if estimator_type == 'classification':
        expected_pred = cuml_model.predict(X).astype(np.int32)
        for idx, row in enumerate(X):
            majority_vote = predict_with_json_rf_classifier(json_obj, row)
            assert expected_pred[idx] in majority_vote
    elif estimator_type == 'regression':
        expected_pred = cuml_model.predict(X).astype(np.float32)
        pred = []
        for idx, row in enumerate(X):
            pred.append(predict_with_json_rf_regressor(json_obj, row))
        pred = np.array(pred, dtype=np.float32)
        np.testing.assert_almost_equal(pred, expected_pred, decimal=6)


@pytest.mark.parametrize('max_depth', [1, 2, 3, 5, 10, 15, 20])
@pytest.mark.parametrize('n_estimators', [5, 10, 20])
@pytest.mark.parametrize('use_experimental_backend', [True, False])
def test_rf_instance_count(max_depth, n_estimators, use_experimental_backend):
    X, y = make_classification(n_samples=350, n_features=20,
                               n_clusters_per_class=1, n_informative=10,
                               random_state=123, n_classes=2)
    X = X.astype(np.float32)
    cuml_model = curfc(max_features=1.0, max_samples=1.0,
                       n_bins=16, split_algo=1, split_criterion=0,
                       min_samples_leaf=2, random_state=23707, n_streams=1,
                       n_estimators=n_estimators, max_leaves=-1,
                       max_depth=max_depth,
                       use_experimental_backend=use_experimental_backend)
    y = y.astype(np.int32)

    # Train model on the data
    cuml_model.fit(X, y)

    json_out = cuml_model.get_json()
    json_obj = json.loads(json_out)

    # The instance count of each node must be equal to the sum of
    # the instance counts of its children. Note that the instance count
    # is only available with the new backend.
    if use_experimental_backend:
        def check_instance_count_for_non_leaf(tree):
            assert 'instance_count' in tree
            if 'children' not in tree:
                return
            assert 'instance_count' in tree['children'][0]
            assert 'instance_count' in tree['children'][1]
            assert (tree['instance_count']
                    == tree['children'][0]['instance_count']
                    + tree['children'][1]['instance_count'])
            check_instance_count_for_non_leaf(tree['children'][0])
            check_instance_count_for_non_leaf(tree['children'][1])
        for tree in json_obj:
            check_instance_count_for_non_leaf(tree)
            # The root's count must be equal to the number of rows in the data
            assert tree['instance_count'] == X.shape[0]
    else:
        def assert_instance_count_absent(tree):
            assert 'instance_count' not in tree
            if 'children' not in tree:
                return
            assert_instance_count_absent(tree['children'][0])
            assert_instance_count_absent(tree['children'][1])
        for tree in json_obj:
            assert_instance_count_absent(tree)


@pytest.mark.memleak
@pytest.mark.parametrize('estimator_type', ['classification'])
def test_rf_host_memory_leak(large_clf, estimator_type):
    import gc
    import os

    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    process = psutil.Process(os.getpid())

    X, y = large_clf
    X = X.astype(np.float32)
    params = {'max_depth': 50}
    if estimator_type == 'classification':
        base_model = curfc(max_depth=10,
                           n_estimators=100,
                           random_state=123)
        y = y.astype(np.int32)
    else:
        base_model = curfr(max_depth=10,
                           n_estimators=100,
                           random_state=123)
        y = y.astype(np.float32)

    # Pre-fit once - this is our baseline and memory usage
    # should not significantly exceed it after later fits
    base_model.fit(X, y)
    gc.collect()
    initial_baseline_mem = process.memory_info().rss

    for i in range(5):
        base_model.fit(X, y)
        base_model.set_params(**params)
        gc.collect()
        final_mem = process.memory_info().rss

    # Some tiny allocations may occur, but we shuld not leak
    # without bounds, which previously happened
    assert (final_mem - initial_baseline_mem) < 2e6


@pytest.mark.memleak
@pytest.mark.parametrize('estimator_type', ['regression', 'classification'])
def test_concat_memory_leak(large_clf, estimator_type):
    import gc
    import os

    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    process = psutil.Process(os.getpid())

    X, y = large_clf
    X = X.astype(np.float32)

    # Build a series of RF models
    n_models = 10
    if estimator_type == 'classification':
        base_models = [curfc(max_depth=10,
                             n_estimators=100,
                             random_state=123) for i in range(n_models)]
        y = y.astype(np.int32)
    elif estimator_type == 'regression':
        base_models = [curfr(max_depth=10,
                             n_estimators=100,
                             random_state=123) for i in range(n_models)]
        y = y.astype(np.float32)
    else:
        assert False

    # Pre-fit once - this is our baseline and memory usage
    # should not significantly exceed it after later fits
    for model in base_models:
        model.fit(X, y)

    # Just concatenate over and over in a loop
    concat_models = base_models[1:]
    init_model = base_models[0]
    other_handles = [
        model._obtain_treelite_handle() for model in concat_models
    ]
    init_model._concatenate_treelite_handle(other_handles)

    gc.collect()
    initial_baseline_mem = process.memory_info().rss
    for i in range(10):
        init_model._concatenate_treelite_handle(other_handles)
        gc.collect()
        used_mem = process.memory_info().rss
        logger.debug("memory at rep %2d: %d m" % (
                    i, (used_mem - initial_baseline_mem)/1e6))

    gc.collect()
    used_mem = process.memory_info().rss
    logger.info("Final memory delta: %d" % (
        (used_mem - initial_baseline_mem)/1e6))
    assert (used_mem - initial_baseline_mem) < 1e6


@pytest.mark.xfail(strict=True, raises=ValueError)
def test_rf_nbins_small(small_clf):

    X, y = small_clf
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    cuml_model.fit(X_train[0:3, :], y_train[0:3])


@pytest.mark.parametrize('split_criterion', [2, 3], ids=['mse', 'mae'])
@pytest.mark.parametrize('use_experimental_backend', [True, False])
def test_rf_regression_with_identical_labels(split_criterion,
                                             use_experimental_backend):
    X = np.array([[-1, 0], [0, 1], [2, 0], [0, 3], [-2, 0]], dtype=np.float32)
    y = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Degenerate case: all labels are identical.
    # RF Regressor must not create any split. It must yield an empty tree
    # with only the root node.
    clf = curfr(max_features=1.0, max_samples=1.0, n_bins=5, split_algo=1,
                bootstrap=False, split_criterion=split_criterion,
                min_samples_leaf=1, min_samples_split=2, random_state=0,
                n_streams=1, n_estimators=1, max_depth=1,
                use_experimental_backend=use_experimental_backend)
    clf.fit(X, y)
    model_dump = json.loads(clf.get_json())
    assert len(model_dump) == 1
    expected_dump = {'nodeid': 0, 'leaf_value': 1.0}
    if use_experimental_backend:
        expected_dump['instance_count'] = 5
    assert model_dump[0] == expected_dump
