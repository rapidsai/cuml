#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings
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


def _construct_instances_for_proxy(init_params, skipped):
    """Return a _construct_instances implementation that looks up INIT_PARAMS by proxy or _cpu_class."""

    def _construct_instances(Estimator):
        if Estimator in skipped:
            msg = f"Can't instantiate estimator {Estimator.__name__}"
            from sklearn.exceptions import SkipTestWarning
            from sklearn.utils._testing import SkipTest

            warnings.warn(msg, SkipTestWarning)
            raise SkipTest(msg)
        key = (
            Estimator
            if Estimator in init_params
            else getattr(Estimator, "_cpu_class", None)
        )
        if key is not None and key in init_params:
            param_sets = init_params[key]
            if not isinstance(param_sets, list):
                param_sets = [param_sets]
            for params in param_sets:
                yield Estimator(**params)
        else:
            yield Estimator()

    return _construct_instances


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

    # Patch _construct_instances so INIT_PARAMS lookup works for proxy classes.
    # INIT_PARAMS is keyed by the class at import time; all_estimators() yields
    # proxy classes, so "Estimator in INIT_PARAMS" can be False. Look up by
    # _cpu_class when the estimator is a proxy so we use the same param sets
    # (e.g. Pipeline(steps=...) instead of Pipeline()).
    try:
        from sklearn.utils._test_common import instance_generator
    except ImportError:
        return
    instance_generator._construct_instances = _construct_instances_for_proxy(
        instance_generator.INIT_PARAMS,
        instance_generator.SKIPPED_ESTIMATORS,
    )
