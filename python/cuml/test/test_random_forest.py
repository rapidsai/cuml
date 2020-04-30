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

import cudf
import numpy as np
import pytest
import random
import rmm

from numba import cuda

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.metrics import r2_score
from cuml.test.utils import get_handle, unit_param, \
    quality_param, stress_param

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


@pytest.mark.parametrize('rows_sample', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_classification(small_clf, datatype, split_algo,
                           rows_sample, max_features):
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
    cuml_model = curfc(max_features=max_features, rows_sample=rows_sample,
                       n_bins=16, split_algo=split_algo, split_criterion=0,
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16)
    cuml_model.fit(X_train, y_train)
    fil_preds = cuml_model.predict(X_test,
                                   predict_model="GPU",
                                   output_class=True,
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


@pytest.mark.parametrize('rows_sample', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_regression(special_reg, datatype, split_algo, max_features,
                       rows_sample):

    use_handle = True

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(max_features=max_features, rows_sample=rows_sample,
                       n_bins=16, split_algo=split_algo, split_criterion=2,
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=50, handle=handle, max_leaves=-1,
                       max_depth=16, accuracy_metric='mse')
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
        cu_class = curfc(seed=seed, n_streams=1)
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
        cu_class2 = curfc(seed=seed, n_streams=1)
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


@pytest.mark.parametrize('datatype', [(np.float32, np.float32)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_classes', [10])
@pytest.mark.parametrize('type', ['dataframe', 'numpy'])
def test_rf_classification_multi_class(datatype, column_info, nrows,
                                       n_classes, type):

    ncols, n_info = column_info
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=0, n_classes=n_classes)
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_test = X_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    if type == 'dataframe':
        X_train_df = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X_train))
        y_train_df = cudf.Series(y_train)
        X_test_df = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X_test))
        cuml_model.fit(X_train_df, y_train_df)
        cu_preds = cuml_model.predict(X_test_df,
                                      predict_model="CPU").to_array()
    else:
        cuml_model.fit(X_train, y_train)
        cu_preds = cuml_model.predict(X_test, predict_model="CPU")

    cu_acc = accuracy_score(y_test, cu_preds)

    # sklearn random forest classification model
    # initialization, fit and predict
    if nrows < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert cu_acc >= (sk_acc - 0.07)


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
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=num_treees, handle=handle, max_leaves=-1,
                       max_depth=40)
    cuml_model.fit(X_train, y_train)

    if ((not fil_sparse_format or algo == 'tree_reorg' or
            algo == 'batch_tree_reorg') or
            fil_sparse_format == 'not_supported'):
        with pytest.raises(ValueError):
            fil_preds = cuml_model.predict(X_test,
                                           predict_model="GPU",
                                           output_class=True,
                                           threshold=0.5,
                                           fil_sparse_format=fil_sparse_format,
                                           algo=algo)
    else:
        fil_preds = cuml_model.predict(X_test,
                                       predict_model="GPU",
                                       output_class=True,
                                       threshold=0.5,
                                       fil_sparse_format=fil_sparse_format,
                                       algo=algo)
        fil_preds = np.reshape(fil_preds, np.shape(y_test))
        fil_acc = accuracy_score(y_test, fil_preds)

        fil_model = cuml_model.convert_to_fil_model()
        input_type = 'numpy'
        fil_model_preds = fil_model.predict(X_test,
                                            output_type=input_type)
        fil_model_acc = accuracy_score(y_test, fil_model_preds)
        assert fil_acc == fil_model_acc

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features
        del tl_model

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
                       min_rows_per_node=2, seed=123, n_streams=1,
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

        input_type = 'numpy'
        fil_model_preds = fil_model.predict(X_test,
                                            output_type=input_type)
        fil_model_preds = np.reshape(fil_model_preds, np.shape(y_test))
        fil_model_r2 = r2_score(y_test, fil_model_preds,
                                convert_dtype=datatype)
        assert fil_r2 == fil_model_r2

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features
        del tl_model

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


@pytest.mark.parametrize('rows_sample', [unit_param(1.0),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_classification_proba(small_clf, datatype,
                                 rows_sample, max_features):
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
    cuml_model = curfc(max_features=max_features, rows_sample=rows_sample,
                       n_bins=16, split_criterion=0,
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16)
    cuml_model.fit(X_train, y_train)
    fil_preds_proba = cuml_model.predict_proba(X_test,
                                               output_class=True,
                                               threshold=0.5,
                                               algo='auto')
    y_proba = np.zeros(np.shape(fil_preds_proba))
    y_proba[:, 1] = y_test
    y_proba[:, 0] = 1.0 - y_test
    fil_mse = mean_squared_error(y_proba, fil_preds_proba)
    if X.shape[0] < 500000:
        sk_model = skrfc(n_estimators=40,
                         max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds_proba = sk_model.predict_proba(X_test)
        sk_mse = mean_squared_error(y_proba, sk_preds_proba)
        # Max difference of 0.0061 is seen between the mse values of
        # predict proba function of fil and sklearn
        assert fil_mse <= (sk_mse + 0.0061)
