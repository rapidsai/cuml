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

from cuml.internals.utils import all_estimators
from cuml import LogisticRegression


DEFAULT_PARAMETERS = {
    "MulticlassClassifier": dict(estimator=LogisticRegression()),
    "OneVsOneClassifier": dict(estimator=LogisticRegression()),
    "OneVsRestClassifier": dict(estimator=LogisticRegression()),
}


def constructed_estimators():
    """Build list of instances of all estimators in cuml"""
    for name, Estimator in all_estimators(
        type_filter=["classifier", "regressor", "cluster"]
    ):
        if name in DEFAULT_PARAMETERS:
            yield Estimator(**DEFAULT_PARAMETERS[name])
        else:
            yield Estimator()


@estimator_checks.parametrize_with_checks(list(constructed_estimators()))
def test_sklearn_compatible_estimator(estimator, check):
    # Check that all estimators pass the "common estimator" checks
    # provided by scikit-learn
    check(estimator)
