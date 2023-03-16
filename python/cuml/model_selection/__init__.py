#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

if has_sklearn():
    from sklearn.model_selection import GridSearchCV

    GridSearchCV.__doc__ = (
        """
    This code is developed and maintained by scikit-learn and imported
    by cuML to maintain the familiar sklearn namespace structure.
    cuML includes tests to ensure full compatibility of these wrappers
    with CUDA-based data and cuML estimators, but all of the underlying code
    is due to the scikit-learn developers.\n\n"""
        + GridSearchCV.__doc__
    )

__all__ = ["train_test_split", "GridSearchCV", "StratifiedKFold"]
