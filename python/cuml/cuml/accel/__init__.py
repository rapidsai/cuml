#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from cuml.accel.core import enabled, install
from cuml.accel.estimator_proxy_mixin import ProxyMixin as _ProxyMixin
from cuml.accel.magics import load_ipython_extension
from cuml.accel.pytest_plugin import (
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_load_initial_conftests,
)

try:
    from cuml.accel.estimator_proxy import ProxyBase as _ProxyBase
except ImportError:

    class _ProxyBase:
        pass


# TODO: move back to estimator_proxy.py once sklearn is a required dependency
def is_proxy(instance_or_class) -> bool:
    """Check if an instance or class is a proxy object created by the accelerator."""

    if isinstance(instance_or_class, type):
        cls = instance_or_class
    else:
        cls = type(instance_or_class)
    return issubclass(cls, (_ProxyMixin, _ProxyBase))


__all__ = (
    "enabled",
    "install",
    "is_proxy",
    "load_ipython_extension",
    "pytest_load_initial_conftests",
    "pytest_collection_modifyitems",
    "pytest_addoption",
)
