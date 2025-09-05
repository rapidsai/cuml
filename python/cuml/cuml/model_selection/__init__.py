#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
"""This code is developed and maintained by scikit-learn and imported
by cuML to maintain the familiar sklearn namespace structure.
cuML includes tests to ensure full compatibility of these wrappers
with CUDA-based data and cuML estimators, but all of the underlying code
is due to the scikit-learn developers."""

from cuml.model_selection._split import StratifiedKFold, train_test_split

__all__ = ["train_test_split", "GridSearchCV", "StratifiedKFold"]


def __getattr__(name):
    if name == "GridSearchCV":
        from sklearn.model_selection import GridSearchCV

        return GridSearchCV
    raise AttributeError(f"module {__name__} has no attribute {name}")
