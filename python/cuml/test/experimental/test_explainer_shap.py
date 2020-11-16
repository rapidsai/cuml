#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cuml.experimental.explainer
import numpy as np
import pytest

from cuml.common.import_utils import has_shap
from cuml.test.utils import array_equal
from cuml.test.utils import ClassEnumerator
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()

golden_results = {
    (4, cuml.LinearRegression): [58.13167305, 139.33765425, 28.08136872,
                                 13.12541971],
    (10, cuml.LinearRegression): [-3.47197726, -12.13657959, -43.05540892,
                                  -41.44955195, -4.1909009, -30.91657623,
                                  -14.73675613, 23.92447365, 15.73265123,
                                  -45.94585396],
    (4, cuml.KNeighborsRegressor): [58.13167305, 139.33765425, 28.08136872,
                                    13.12541971],
    (10, cuml.KNeighborsRegressor): [-3.47197726, -12.13657959, -43.05540892,
                                     -41.44955195, -4.1909009, -30.91657623,
                                     -14.73675613, 23.92447365, 15.73265123,
                                     -45.94585396]
}


# todo: use tags to generate the correct dataset
@pytest.fixture(scope="session")
def single_dataset():
    X, y = make_classification(100, 5, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X, y


def func_positional_arg(func):
    if hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nfeatures", [4, 10])
@pytest.mark.parametrize("nbackground", [80])
@pytest.mark.parametrize("model", [cuml.LinearRegression,
                                   cuml.KNeighborsRegressor])
def test_kernel_shap_standalone(dtype, nfeatures, nbackground, model):
    X, y = cuml.datasets.make_regression(n_samples=nbackground + 1,
                                         n_features=nfeatures,
                                         noise=0.1,
                                         random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    mod = model().fit(X_train, y_train)

    cu_explainer = cuml.experimental.explainer.KernelSHAP(model=mod.predict,
                                                          data=X_train,
                                                          gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test[0])

    assert array_equal(cu_shap_values, golden_results[nfeatures, model],
                       1e-1, with_sign=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nfeatures", [4, 100])
@pytest.mark.parametrize("nbackground", [10, 80])
@pytest.mark.parametrize("model", [cuml.LinearRegression,
                                   cuml.KNeighborsRegressor])
def test_kernel_gpu_cpu_shap(dtype, nfeatures, nbackground, model):
    if not has_shap():
        pytest.skip("Need SHAP installed for these tests")

    import shap

    X, y = cuml.datasets.make_regression(n_samples=nbackground + 1,
                                         n_features=nfeatures,
                                         noise=0.1,
                                         random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    mod = model().fit(X_train, y_train)

    explainer = shap.KernelExplainer(mod.predict, X_train)
    shap_values = explainer.shap_values(X_test[0])

    cu_explainer = cuml.experimental.explainer.KernelSHAP(model=mod.predict,
                                                          data=X_train,
                                                          gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test[0])

    assert array_equal(cu_shap_values, shap_values,
                       1e-1, with_sign=True)


@pytest.mark.parametrize("model_name", list(models.keys()))
def test_cuml_models(single_dataset, model_name):
    n_pos_args_constr = func_positional_arg(models[model_name].__init__)

    if model_name in ["SparseRandomProjection", "GaussianRandomProjection"]:
        model = models[model_name](n_components=2)
    elif model_name in ["ARIMA", "AutoARIMA", "ExponentialSmoothing"]:
        model = models[model_name](np.random.normal(0.0, 1.0, (10,)))
    else:
        if n_pos_args_constr == 1:
            model = models[model_name]()
        elif n_pos_args_constr == 2:
            model = models[model_name](5)
        else:
            model = models[model_name](5, 5)

    X, y = single_dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    mod = model().fit(X_train, y_train)

    cu_explainer = cuml.experimental.explainer.KernelSHAP(model=mod.predict,
                                                          data=X_train,
                                                          gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test[0])

    if has_shap():
        import shap
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test[0])
        assert array_equal(cu_shap_values, shap_values,
                           1e-1, with_sign=True)
