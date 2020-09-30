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

import pytest
import cuml
import numpy as np

from cuml.test.utils import get_classes_from_package
from sklearn import datasets

all_models = get_classes_from_package(cuml)


@pytest.mark.parametrize('model', all_models)
def test_exact_shap_interop(model):
    shap = pytest.importorskip('shap')

    if model in ['ExponentialSmoothing',
                 'ARIMA',
                 'SparseRandomProjection',
                 'GaussianRandomProjection',
                 'OneHotEncoder',
                 'LabelEncoder',
                 'LabelBinarizer',
                 'ForestInference',
                 'Base',
                 'MultinomialNB',
                 'RandomForestClassifier',
                 'RandomForestRegressor']:
        pytest.skip("Models not yet compatible.")

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    y[y == 2] = 0  # making it 2 classes so any estimator can be trained on it
    if model == 'SVC':
        model = all_models[model](probability=True)
    else:
        model = all_models[model]()
    try:
        model.fit(X)
    except (TypeError, AttributeError):
        model.fit(X, y)

    if hasattr(model, 'predict'):
        explainer = shap.explainers.Exact(model.predict, X)
        shap_values = explainer(X[:2])
        assert(isinstance(shap_values, shap._explanation.Explanation))

    if hasattr(model, 'predict_proba'):
        explainer = shap.explainers.Exact(model.predict_proba, X)
        shap_values = explainer(X[:2])
        assert(isinstance(shap_values, shap._explanation.Explanation))


@pytest.mark.parametrize('model', all_models)
def test_permutation_shap_interop(model):
    shap = pytest.importorskip('shap')

    if model in ['ExponentialSmoothing',
                 'ARIMA',
                 'SparseRandomProjection',
                 'GaussianRandomProjection',
                 'OneHotEncoder',
                 'LabelEncoder',
                 'LabelBinarizer',
                 'ForestInference',
                 'Base',
                 'MultinomialNB',
                 'RandomForestClassifier',
                 'RandomForestRegressor']:
        pytest.skip("Models not yet compatible.")

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    y[y == 2] = 0  # making it 2 classes so any estimator can be trained on it
    if model == 'SVC':
        model = all_models[model](probability=True)
    else:
        model = all_models[model]()
    try:
        model.fit(X)
    except (TypeError, AttributeError):
        model.fit(X, y)

    if hasattr(model, 'predict'):
        explainer = shap.explainers.Permutation(model.predict, X)
        shap_values = explainer(X[:2])
        assert(isinstance(shap_values, shap._explanation.Explanation))

    if hasattr(model, 'predict_proba'):
        explainer = shap.explainers.Permutation(model.predict_proba, X)
        shap_values = explainer(X[:2])
        assert(isinstance(shap_values, shap._explanation.Explanation))


@pytest.mark.parametrize('model', all_models)
def test_kernel_shap_interop(model):
    shap = pytest.importorskip('shap')
    if model in ['ExponentialSmoothing',
                 'ARIMA',
                 'SparseRandomProjection',
                 'GaussianRandomProjection',
                 'OneHotEncoder',
                 'LabelEncoder',
                 'LabelBinarizer',
                 'ForestInference',
                 'Base',
                 'MultinomialNB',
                 'RandomForestClassifier',
                 'RandomForestRegressor',
                 'SGD',
                 'CD',
                 'Ridge',
                 'ElasticNet',
                 'Lasso',
                 'MBSGDClassifier',
                 'MBSGDRegressor']:
        pytest.skip("Models not yet compatible.")

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    y[y == 2] = 0  # making it 2 classes so any estimator can be trained on it
    if model == 'SVC':
        model = all_models[model](probability=True)
    else:
        model = all_models[model]()
    try:
        model.fit(X)
    except (TypeError, AttributeError):
        model.fit(X, y)

    if hasattr(model, 'predict'):
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X[:2])
        assert(isinstance(shap_values,
                          np.ndarray) or isinstance(shap_values, list))

    if hasattr(model, 'predict_proba'):
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X[:2])
        assert(isinstance(shap_values,
                          np.ndarray) or isinstance(shap_values, list))
