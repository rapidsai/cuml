#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
from cuml.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    HalvingGridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)


@pytest.mark.parametrize(
    "method",
    [
        train_test_split,
        GridSearchCV,
        StratifiedKFold,
        HalvingGridSearchCV,
        ParameterGrid,
        ParameterSampler,
        RandomizedSearchCV,
        HalvingRandomSearchCV,
    ],
)
def test_sklearn_docs(method):
    assert method.__doc__ is not None
    # Ensure the customized documentation has been added
    assert "cuML includes tests to ensure full compatibility" in method.__doc__


def test_add_sklearn_documentation_not_sklearn_installed(monkeypatch):
    # Simulate the case where sklearn is not installed
    monkeypatch.setattr(
        "cuml.internals.import_utils.has_sklearn", lambda: False
    )

    from cuml.model_selection import add_sklearn_documentation

    # Attempt to add documentation when sklearn is not installed
    add_sklearn_documentation("GridSearchCV", "Test description")

    # Assert that the method's doc remains unchanged
    assert GridSearchCV.__doc__ is None
