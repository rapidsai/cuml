# Copyright (c) 2024, NVIDIA CORPORATION.
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
import tempfile
import os
from cuml import (
    KMeans,
    DBSCAN,
    PCA,
    TruncatedSVD,
    KernelRidge,
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    Lasso,
    TSNE,
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
)


# List of estimators with their types, parameters, and data-specific parameters
estimators = [
    (KMeans, 'clusterer', {'n_clusters': 3, 'random_state': 42}, {}),
    (DBSCAN, 'clusterer', {}, {}),
    (PCA, 'transformer', {'n_components': 5}, {}),
    (TruncatedSVD, 'transformer', {'n_components': 5}, {}),
    (LinearRegression, 'regressor', {}, {}),
    (ElasticNet, 'regressor', {'max_iter': 1000}, {}),
    (Ridge, 'regressor', {}, {}),
    (Lasso, 'regressor', {'max_iter': 1000}, {}),
    (
        LogisticRegression,
        'classifier',
        {'random_state': 42, 'solver': 'liblinear', 'max_iter': 1000},
        {'n_classes': 2},
    ),
    (TSNE, 'transformer', {'n_components': 2, 'random_state': 42}, {}),
    (NearestNeighbors, 'neighbors', {'n_neighbors': 5}, {}),
    (KNeighborsClassifier, 'classifier', {'n_neighbors': 5}, {'n_classes': 3}),
    (KNeighborsRegressor, 'regressor', {'n_neighbors': 5}, {}),
]


def get_y(estimator_type: str, n_samples: int, data_params: dict):
    if estimator_type in ['classifier', 'regressor']:
        if estimator_type == 'classifier':
            n_classes = data_params.get('n_classes', 2)
            y = np.random.randint(0, n_classes, size=n_samples)
        else:
            y = np.random.rand(n_samples)
    else:
        y = None  # Unsupervised methods don't use y

    return y


def predict_transform(estimator, estimator_type, X):
    if estimator_type in ['regressor', 'classifier']:
        output = estimator.predict(X)
    elif estimator_type == 'clusterer':
        if hasattr(estimator, 'predict'):
            output = estimator.predict(X)
        else:
            output = estimator.labels_
    elif estimator_type == 'transformer':
        if hasattr(estimator, 'transform'):
            output = estimator.transform(X)
        elif hasattr(estimator, 'embedding_'):
            output = estimator.embedding_
    elif estimator_type == 'neighbors':
        output = estimator.kneighbors(X)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")
    
    return output


@pytest.mark.parametrize("Estimator, estimator_type, est_params, data_params", estimators)
def test_estimator_to_from_sklearn(Estimator, estimator_type, est_params, data_params):
    # Generate data based on estimator type
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    y = get_y(estimator_type, n_samples, data_params)

    # Instantiate estimator
    est = Estimator(**est_params)

    # Fit estimator
    if y is not None:
        est.fit(X, y)
    else:
        if estimator_type == 'transformer' and hasattr(est, 'fit_transform') and not hasattr(est, 'transform'):
            # For TSNE
            output1 = est.fit_transform(X)
        else:
            est.fit(X)

    # Make predictions or transformations

    output1 = predict_transform(est, estimator_type, X)
    
    # Save and load the estimator using temporary file
    with tempfile.NamedTemporaryFile(suffix='.pickle', delete=False) as tmp_file:
        filename = tmp_file.name
    try:
        est.to_sklearn(filename=filename)
        est2 = Estimator.from_sklearn(filename=filename)
    finally:
        # Clean up the temporary file
        os.remove(filename)

    output2 = predict_transform(est2, estimator_type, X)
    # Make predictions or transformations with the loaded estimator
   
    # Compare outputs
    if estimator_type in ['regressor', 'transformer']:
        assert np.allclose(output1, output2)
    elif estimator_type in ['classifier', 'clusterer']:
        assert np.array_equal(output1, output2)
    elif estimator_type == 'neighbors':
        distances1, indices1 = output1
        distances2, indices2 = output2
        assert np.allclose(distances1, distances2)
        assert np.array_equal(indices1, indices2)