# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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


from sklearn.utils import estimator_checks

from cuml import LogisticRegression


@estimator_checks.parametrize_with_checks([LogisticRegression()])
def test_sklearn_compatible_estimator(estimator, check):
    # Check that all estimators pass the "common estimator" checks
    # provided by scikit-learn
    check(estimator)
