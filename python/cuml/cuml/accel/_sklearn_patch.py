#
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
#
from collections import defaultdict
from operator import itemgetter

from sklearn.utils.discovery import all_estimators as sklearn_all_estimators

from cuml.accel.estimator_proxy import is_proxy


def _patched_all_estimators(*args, **kwargs):
    """Monkeypatch sklearn's all_estimators to prioritize proxy estimators.

    This function replaces sklearn's all_estimators function with a version
    that filters out duplicate estimator names, keeping only proxy estimators
    when both proxy and non-proxy versions exist.
    """
    # Obtain the list of all estimators from sklearn
    ret = sklearn_all_estimators(*args, **kwargs)

    # Group estimators by name
    estimator_groups = defaultdict(list)
    for name, cls in ret:
        estimator_groups[name].append(cls)

    # Drop non-proxies wherever there are multiple classes with the same name
    estimators = []
    for name, cls_list in estimator_groups.items():
        if len(cls_list) == 1:
            estimators.append((name, cls_list[0]))
        else:
            proxied_cls = [cls for cls in cls_list if is_proxy(cls)]
            assert len(proxied_cls) == 1
            estimators.append((name, proxied_cls[0]))

    # Return the sorted list of estimators like the original
    return sorted(set(estimators), key=itemgetter(0))


def apply_sklearn_patches():
    """Apply all sklearn patches necessary for the accelerator testing."""

    # Monkeypatch sklearn's all_estimators to prioritize proxy estimators
    #
    # When the accelerator is installed, sklearn's all_estimators() returns
    # duplicate entries for the same estimator name (e.g., LinearSVC appears
    # both with sklearn.svm.LinearSVC and sklearn._classes.LinearSVC). This causes
    # a TypeError during test collection when sklearn tries to sort the
    # estimators as it attempts to sort based on the type.
    #
    # The patch filters out duplicates by keeping only proxy estimators when
    # multiple classes with the same name exist, ensuring test collection
    # succeeds.
    import sklearn.utils

    sklearn.utils.all_estimators = _patched_all_estimators
