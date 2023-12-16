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

from cuml.model_selection._split import train_test_split
from cuml.model_selection._split import StratifiedKFold
from cuml.internals.import_utils import has_sklearn
from sklearn import model_selection
import sklearn


def add_sklearn_documentation(sklearn_method, description):
    if has_sklearn():
        imported_method = getattr(sklearn.model_selection, sklearn_method)
        imported_method.__doc__ = (
            f"This code is developed and maintained by scikit-learn and imported by cuML to maintain the familiar sklearn namespace structure. cuML includes tests to ensure full compatibility of these wrappers with CUDA-based data and cuML estimators, but all of the underlying code is due to the scikit-learn developers.\n\n{description}\n\n"
            + imported_method.__doc__
        )


hpo_methods = [
    {
        "method_name": "GridSearchCV",
        "description": "Exhaustive search over specified parameter values for an estimator.",
    },
    {
        "method_name": "HalvingGridSearchCV",
        "description": "Search over specified parameter values with successive halving.",
    },
    {
        "method_name": "ParameterGrid",
        "description": "Grid of parameters with a discrete number of values for each.",
    },
    {
        "method_name": "ParameterSampler",
        "description": "Generator on parameters sampled from given distributions.",
    },
    {
        "method_name": "RandomizedSearchCV",
        "description": "Randomized search on hyperparameters.",
    },
    {
        "method_name": "HalvingRandomSearchCV",
        "description": "Randomized search on hyperparameters with successive halving.",
    },
]

for method_info in hpo_methods:
    add_sklearn_documentation(
        method_info["method_name"], method_info["description"]
    )

__all__ = [
    "train_test_split",
    "GridSearchCV",
    "StratifiedKFold",
    "HalvingGridSearchCV",
    "ParameterGrid",
    "ParameterSampler",
    "RandomizedSearchCV",
    "HalvingRandomSearchCV",
]
