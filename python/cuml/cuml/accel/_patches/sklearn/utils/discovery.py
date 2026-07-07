#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import functools
from collections import defaultdict
from operator import itemgetter

from sklearn.utils.discovery import all_estimators as _all_estimators

from cuml.accel.estimator_proxy import is_proxy

__all__ = ("all_estimators",)


@functools.wraps(_all_estimators)
def all_estimators(*args, **kwargs):
    # This function replaces sklearn's all_estimators function with a version
    # that filters out duplicate estimator names, keeping only proxy estimators
    # when both proxy and non-proxy versions exist.
    #
    # When the accelerator is installed, sklearn's all_estimators() returns
    # duplicate entries for the same estimator name (e.g., LinearSVC appears
    # both with sklearn.svm.LinearSVC and sklearn._classes.LinearSVC). This causes
    # a TypeError during test collection when sklearn tries to sort the
    # estimators as it attempts to sort based on the type.

    # Obtain the list of all estimators from sklearn
    ret = _all_estimators(*args, **kwargs)

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
