#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import importlib
import inspect
import numpy as np

from cuml.internals.global_settings import GlobalSettings
from sklearn.datasets import make_classification, make_regression

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Starting from version 22.04, the default method of TSNE is 'fft'."
    ),
    pytest.mark.filterwarnings(
        "ignore:The default value of `n_init` will change from 1 to 'auto' in 25.04"
    ),
]


estimators = [
    "KMeans",
    "DBSCAN",
    "PCA",
    "TruncatedSVD",
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Ridge",
    "Lasso",
    "TSNE",
    "NearestNeighbors",
    "UMAP",
    "HDBSCAN",
]


estimator_module_mapping = {
    "KMeans": "cluster",
    "DBSCAN": "cluster",
    "PCA": "decomposition",
    "TruncatedSVD": "decomposition",
    "LinearRegression": "linear_model",
    "LogisticRegression": "linear_model",
    "ElasticNet": "linear_model",
    "Ridge": "linear_model",
    "Lasso": "linear_model",
    "TSNE": "manifold",
    "NearestNeighbors": "neighbors",
}


# Categorize estimators based on their type
supervised_estimators = [
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Ridge",
    "Lasso",
]


regression_estimators = [
    "LinearRegression",
    "ElasticNet",
    "Ridge",
    "Lasso",
]


classification_estimators = [
    "LogisticRegression",
]


unsupervised_estimators = [
    "KMeans",
    "DBSCAN",
    "PCA",
    "TruncatedSVD",
    "TSNE",
    "NearestNeighbors",
    "UMAP",
    "HDBSCAN",
]


@pytest.mark.parametrize("estimator_name", estimators)
def test_UniversalBase_estimators(estimator_name):
    # importing dynamically will also implicitly test that cuML
    # estimators can be imported identically to host ones
    if estimator_name == "UMAP":
        host_module_name = "umap"
        cuml_module_name = "cuml"

    elif estimator_name == "HDBSCAN":
        host_module_name = "hdbscan"
        cuml_module_name = "cuml"
    else:
        host_module_name = (
            "sklearn." + estimator_module_mapping[estimator_name]
        )
        cuml_module_name = "cuml." + estimator_module_mapping[estimator_name]

    # Import the estimator from scikit-learn
    host_module = importlib.import_module(host_module_name)
    host_estimator_class = getattr(host_module, estimator_name)

    # Import the estimator from cuml
    cuml_module = importlib.import_module(cuml_module_name)
    cuml_estimator_class = getattr(cuml_module, estimator_name)

    # Get the attributes and methods of both estimators
    host_attrs = dir(host_estimator_class)
    cuml_attrs = dir(cuml_estimator_class)

    # Filter out private attributes (those starting with '_')
    host_public_attrs = [
        attr for attr in host_attrs if not attr.startswith("_")
    ]
    cuml_public_attrs = [
        attr for attr in cuml_attrs if not attr.startswith("_")
    ]

    # Compare the sets of public attributes and methods
    missing_in_host = set(cuml_public_attrs) - set(host_public_attrs)
    missing_in_cuml = set(host_public_attrs) - set(cuml_public_attrs)

    if GlobalSettings().accelerator_active:
        assert len(missing_in_cuml) == 0, (
            f"Mismatch in attributes/methods for {estimator_name}:\n"
            f"Missing in host: {missing_in_host}\n"
            f"Missing in cuML: {missing_in_cuml}"
        )

    # Prepare a small dataset
    if estimator_name in regression_estimators:
        X, y = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
    elif estimator_name in classification_estimators:
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
    else:
        X = np.random.rand(100, 5)
        y = None
        # Unsupervised estimators don't use 'y'

    # Instantiate the estimators
    host_estimator = host_estimator_class()
    cuml_estimator = cuml_estimator_class()

    # Fit the estimators
    if y is not None:
        host_estimator.fit(X, y)
        cuml_estimator.fit(X, y)
    else:
        host_estimator.fit(X)
        cuml_estimator.fit(X)

        if estimator_name == "HDBSCAN":
            cuml_estimator.generate_prediction_data()

    errors = []

    # Call public methods and ensure they can be executed without errors
    for method_name in host_public_attrs:
        # Skip special methods and attributes
        if method_name.startswith("__"):
            continue

        # Get the method from both estimators
        host_method = getattr(host_estimator, method_name, None)
        cuml_method = getattr(cuml_estimator, method_name, None)

        # Ensure both methods exist and are callable
        if callable(host_method):
            if callable(cuml_method):
                # Prepare arguments based on method name
                if method_name in ["fit", "partial_fit"]:
                    continue

                # Already fitted
                elif method_name in [
                    "predict",
                    "transform",
                    "predict_proba",
                    "decision_function",
                    "score_samples",
                    "kneighbors",
                ]:
                    args = (X,)
                elif method_name == "fit_predict":
                    args = (X, y) if y is not None else (X,)
                else:
                    continue  # Skip other methods for simplicity

                # Call the methods
                try:
                    host_method(*args)
                    cuml_method(*args)
                except Exception as e:
                    errors.append(
                        f"Method {method_name} failed for {estimator_name}: {e}"
                    )

            elif cuml_method is None:
                cuml_estimator._experimental_dispatching = True
                dispatched_method = getattr(cuml_estimator, method_name, None)

                if dispatched_method is None:
                    errors.append(
                        f"Method {method_name} was not dispatched correctly"
                    )
                else:
                    if not (
                        inspect.ismethod(dispatched_method)
                        or inspect.isfunction(dispatched_method)
                    ):
                        errors.append(
                            f"Dispatched method {method_name} is not a "
                            "method or function"
                        )

                cuml_estimator._experimental_dispatching = False

    # Report all errors at the end
    if errors:
        pytest.fail("\n".join(errors))
