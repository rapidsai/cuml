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
def has_dask():
    try:
        import dask  # NOQA
        import dask.dataframe  # NOQA
        import dask.distributed  # NOQA

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


def has_sklearn():
    try:
        import sklearn  # NOQA

        return True
    except ImportError:
        return False
