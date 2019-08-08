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


def has_treelite():
    try:
        import treelite  # NOQA
        return True
    except ImportError:
        return False


def has_xgboost():
    try:
        import xgboost  # NOQA
        return True
    except ImportError:
        return False
