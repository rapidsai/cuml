#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import platform

from packaging.version import Version

from cuml.internals.safe_imports import gpu_only_import, UnavailableError


numba = gpu_only_import("numba")


def has_dask():
    try:
        import dask  # NOQA
        import dask.distributed  # NOQA
        import dask.dataframe  # NOQA

        return True
    except ImportError:
        return False


def has_dask_cudf():
    try:
        import dask_cudf  # NOQA

        return True
    except ImportError:
        return False


def has_dask_sql():
    try:
        import dask_sql  # NOQA

        return True
    except ImportError:
        return False


def has_cupy():
    try:
        import cupy  # NOQA

        return True
    except ImportError:
        return False


def has_ucp():
    try:
        import ucp  # NOQA

        return True
    except ImportError:
        return False


def has_umap():
    if platform.processor() == "aarch64":
        return False
    try:
        import umap  # NOQA

        return True
    except ImportError:
        return False


def has_lightgbm():
    try:
        import lightgbm  # NOQA

        return True
    except ImportError:
        return False


def has_xgboost():
    try:
        import xgboost  # NOQA

        return True
    except ImportError:
        return False
    except Exception as ex:
        import warnings

        warnings.warn(
            (
                "The XGBoost library was found but raised an exception during "
                "import. Importing xgboost will be skipped. "
                "Error message:\n{}"
            ).format(str(ex))
        )
        return False


def has_pytest_benchmark():
    try:
        import pytest_benchmark  # NOQA

        return True
    except ImportError:
        return False


def check_min_dask_version(version):
    try:
        import dask

        return Version(dask.__version__) >= Version(version)
    except ImportError:
        return False


def check_min_numba_version(version):
    try:
        return Version(str(numba.__version__)) >= Version(version)
    except UnavailableError:
        return False


def check_min_cupy_version(version):
    if has_cupy():
        import cupy

        return Version(str(cupy.__version__)) >= Version(version)
    else:
        return False


def has_scipy(raise_if_unavailable=False, min_version=None):
    try:
        import scipy  # NOQA

        if min_version is None:
            return True
        else:
            return Version(str(scipy.__version__)) >= Version(min_version)

    except ImportError:
        if not raise_if_unavailable:
            return False
        else:
            raise ImportError("Scipy is not available.")


def has_sklearn():
    try:
        import sklearn  # NOQA

        return True
    except ImportError:
        return False


def has_hdbscan(raise_if_unavailable=False):
    try:
        import hdbscan  # NOQA

        return True
    except ImportError:
        if not raise_if_unavailable:
            return False
        else:
            raise ImportError(
                "hdbscan is not available. Please install hdbscan."
            )


def has_shap(min_version="0.37"):
    try:
        import shap  # noqa

        if min_version is None:
            return True
        else:
            return Version(str(shap.__version__)) >= Version(min_version)
    except ImportError:
        return False


def has_daskglm(min_version=None):
    try:
        import dask_glm  # noqa

        if min_version is None:
            return True
        else:
            return Version(str(dask_glm.__version__)) >= Version(min_version)
    except ImportError:
        return False


def dummy_function_always_false(*args, **kwargs):
    return False


class DummyClass(object):
    pass
