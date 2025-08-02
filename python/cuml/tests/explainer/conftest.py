# Copyright (c) 2025, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from cuml.testing.datasets import with_dtype


@pytest.fixture(scope="session")
def exact_shap_regression_dataset():
    """Generate a synthetic regression dataset for SHAP testing.

    This fixture creates a synthetic regression dataset with known
    properties for testing SHAP (SHapley Additive exPlanations) values.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) split of the synthetic dataset
    """
    X, y = with_dtype(
        make_regression(
            n_samples=101, n_features=11, noise=0.1, random_state=42
        ),
        np.float32,
    )
    return train_test_split(X, y, test_size=3, random_state=42)


@pytest.fixture(scope="session")
def exact_shap_classification_dataset():
    """Generate a synthetic classification dataset for SHAP testing.

    This fixture creates a synthetic classification dataset with known
    properties for testing SHAP (SHapley Additive exPlanations) values.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) split of the synthetic dataset
    """
    X, y = with_dtype(
        make_classification(
            n_samples=101, n_features=11, n_classes=2, random_state=42
        ),
        np.float32,
    )
    return train_test_split(X, y, test_size=3, random_state=42)
