#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from inspect import Parameter, signature
import typing

from cuml.internals.api_decorators import (
    api_base_return_generic,
    api_base_return_array,
    api_base_return_sparse_array,
    api_base_return_any,
    api_return_any,
    _deprecate_pos_args
)
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base_return_types import _get_base_return_type
from cuml.internals.constants import CUML_WRAPPED_FLAG


def _process_generic(gen_type):

    # Check if the type is not a generic. If not, must return "generic" if
    # subtype is CumlArray otherwise None
    if (not isinstance(gen_type, typing._GenericAlias)):
        if (issubclass(gen_type, CumlArray)):
            return "generic"

        # We don't handle SparseCumlArray at this time
        if (issubclass(gen_type, SparseCumlArray)):
            raise NotImplementedError(
                "Generic return types with SparseCumlArray are not supported "
                "at this time")

        # Otherwise None (keep processing)
        return None

    # Its a generic type by this point. Support Union, Tuple, Dict and List
    supported_gen_types = [
        tuple,
        dict,
        list,
        typing.Union,
    ]

    if (gen_type.__origin__ in supported_gen_types):
        # Check for a CumlArray type in the args
        for arg in gen_type.__args__:
            inner_type = _process_generic(arg)

            if (inner_type is not None):
                return inner_type
    else:
        raise NotImplementedError("Unknow generic type: {}".format(gen_type))

    return None


def _wrap_attribute(class_name: str,
                    attribute_name: str,
                    attribute,
                    **kwargs):

    # Skip items marked with autowrap_ignore
    if (attribute.__dict__.get(CUML_WRAPPED_FLAG, False)):
        return attribute

    return_type = _get_base_return_type(class_name, attribute)

    if (return_type == "generic"):
        attribute = api_base_return_generic(**kwargs)(attribute)
    elif (return_type == "array"):
        attribute = api_base_return_array(**kwargs)(attribute)
    elif (return_type == "sparsearray"):
        attribute = api_base_return_sparse_array(
            **kwargs)(attribute)
    elif (return_type == "base"):
        attribute = api_base_return_any(**kwargs)(attribute)
    elif (not attribute_name.startswith("_")):
        # Only replace public functions with return any
        attribute = api_return_any()(attribute)

    return attribute


def _check_and_wrap_init(attribute, **kwargs):

    # Check if the decorator has already been added
    if (attribute.__dict__.get(_deprecate_pos_args.FLAG_NAME)):
        return attribute

    # Get the signature to test if all args are keyword only
    sig = signature(attribute)

    incorrect_params = [
        n for n,
        p in sig.parameters.items()
        if n != "self" and (p.kind == Parameter.POSITIONAL_ONLY
                            or p.kind == Parameter.POSITIONAL_OR_KEYWORD)
    ]

    assert len(incorrect_params) == 0, \
        (
            "Error in `{}`!. Positional arguments for estimators (that derive "
            "from `Base`) have been deprecated but parameters '{}' can still "
            "be used as positional arguments. Please specify all parameters "
            "after `self` as keyword only by using the `*` argument"
        ).format(attribute.__qualname__, ", ".join(incorrect_params))

    return _deprecate_pos_args(**kwargs)(attribute)


class BaseMetaClass(type):
    """
    Metaclass for all estimators in cuML. This metaclass will get called for
    estimators deriving from `cuml.common.Base` as well as
    `cuml.dask.common.BaseEstimator`. It serves 2 primary functions:

     1. Set the `@_deprecate_pos_args()` decorator on all `__init__` functions
     2. Wrap any functions and properties in the API decorators
        [`cuml.common.Base` only]

    """
    def __new__(cls, classname, bases, classDict):

        is_dask_module = classDict["__module__"].startswith("cuml.dask")

        for attributeName, attribute in classDict.items():

            # If attributeName is `__init__`, wrap in the decorator to
            # deprecate positional args
            if (attributeName == "__init__"):
                attribute = _check_and_wrap_init(attribute, version="21.06")
                classDict[attributeName] = attribute

            # For now, skip all additional processing if we are a dask
            # estimator
            if is_dask_module:
                continue

            # Must be a function
            if callable(attribute):

                classDict[attributeName] = _wrap_attribute(
                    classname, attributeName, attribute)

            elif isinstance(attribute, property):
                # Need to wrap the getter if it exists
                if (hasattr(attribute, "fget") and attribute.fget is not None):
                    classDict[attributeName] = attribute.getter(
                        _wrap_attribute(classname,
                                        attributeName,
                                        attribute.fget,
                                        input_arg=None))

        return type.__new__(cls, classname, bases, classDict)


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
