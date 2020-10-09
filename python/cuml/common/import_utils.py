#
# Copyright (c) 2019, NVIDIA CORPORATION.
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


import inspect
import numba

from distutils.version import LooseVersion
from functools import wraps


def has_dask():
    try:
        import dask   # NOQA
        import dask.distributed   # NOQA
        import dask.dataframe   # NOQA
        return True
    except ImportError:
        return False


def has_cupy():
    try:
        import cupy   # NOQA
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


def has_pytest_benchmark():
    try:
        import pytest_benchmark  # NOQA
        return True
    except ImportError:
        return False


def check_min_numba_version(version):
    return LooseVersion(str(numba.__version__)) >= LooseVersion(version)


def check_min_cupy_version(version):
    if has_cupy():
        import cupy
        return LooseVersion(str(cupy.__version__)) >= LooseVersion(version)
    else:
        return False


def has_scipy(raise_if_unavailable=False):
    try:
        import scipy   # NOQA
        return True
    except ImportError:
        if not raise_if_unavailable:
            return False
        else:
            raise ImportError("Scipy is not available.")


def has_sklearn():
    try:
        import sklearn   # NOQA
        return True
    except ImportError:
        return False


def dummy_function_always_false(*args, **kwargs):
    return False


class DummyClass(object):
    pass


def check_cupy8(conf=None):
    """Decorator checking availability of CuPy 8.0+

    Parameters:
    conf: string (optional, default None): If set to 'pytest' will skip tests.
    Will otherwise raise an error in case CuPy 8.0+ is unavailable.

    """
    def check_cupy8_dec(func):

        assert not inspect.isclass(func), \
            ("Do not use this decorator on classes. Instead decorate "
             "__init__  and any static or class methods.")

        @wraps(func)
        def inner(*args, **kwargs):
            import cupy as cp
            if LooseVersion(str(cp.__version__)) >= LooseVersion('8.0'):
                return func(*args, **kwargs)
            else:
                err_msg = 'Could not import required module CuPy 8.0+'
                if conf == 'pytest':
                    import pytest
                    pytest.skip(err_msg)
                else:
                    raise ImportError(err_msg)
        return inner
    return check_cupy8_dec
