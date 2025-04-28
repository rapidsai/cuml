#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from cuml.internals.api_decorators import (
    api_base_return_any,
    api_base_return_array,
    api_base_return_generic,
    api_base_return_sparse_array,
    api_return_any,
)
from cuml.internals.base_return_types import _get_base_return_type
from cuml.internals.constants import CUML_WRAPPED_FLAG


def _wrap_attribute(class_name: str, attribute_name: str, attribute, **kwargs):

    # Skip items marked with autowrap_ignore
    if attribute.__dict__.get(CUML_WRAPPED_FLAG, False):
        return attribute

    return_type = _get_base_return_type(class_name, attribute)

    if return_type == "generic":
        attribute = api_base_return_generic(**kwargs)(attribute)
    elif return_type == "array":
        attribute = api_base_return_array(**kwargs)(attribute)
    elif return_type == "sparsearray":
        attribute = api_base_return_sparse_array(**kwargs)(attribute)
    elif return_type == "base":
        attribute = api_base_return_any(**kwargs)(attribute)
    elif not attribute_name.startswith("_"):
        # Only replace public functions with return any
        attribute = api_return_any()(attribute)

    return attribute


class BaseMetaClass(type):
    """
    Metaclass for all estimators in cuML.

    This metaclass will get called for estimators deriving from `cuml.common.Base`
    as well as `cuml.dask.common.BaseEstimator`. It automatically wraps methods and
    properties in the API decorators (`cuml.common.Base` only).
    """

    def __new__(cls, classname, bases, namespace):
        # Skip wrapping methods in dask estimators
        if not namespace["__module__"].startswith("cuml.dask"):
            for name, attribute in namespace.items():
                if callable(attribute):
                    # Wrap method
                    namespace[name] = _wrap_attribute(
                        classname, name, attribute
                    )

                elif (
                    isinstance(attribute, property)
                    and attribute.fget is not None
                ):
                    # Wrap property getters
                    namespace[name] = attribute.getter(
                        _wrap_attribute(
                            classname,
                            name,
                            attribute.fget,
                            input_arg=None,
                        )
                    )

        return type.__new__(cls, classname, bases, namespace)


class _tags_class_and_instance:
    """
    Decorator for Base class to allow for dynamic and static _get_tags.
    In general, most methods are either dynamic or static, so this decorator
    is only meant to be used in the Base estimator _get_tags.
    """

    def __init__(self, _class, _instance=None):
        self._class = _class
        self._instance = _instance

    def instance_method(self, _instance):
        """
        Factory to create a _tags_class_and_instance instance method with
        the existing class associated.
        """
        return _tags_class_and_instance(self._class, _instance)

    def __get__(self, _instance, _class):
        # if the caller had no instance (i.e. it was a class) or there is no
        # instance associated we the method we return the class call
        if _instance is None or self._instance is None:
            return self._class.__get__(_class, None)

        # otherwise return instance call
        return self._instance.__get__(_instance, _class)
