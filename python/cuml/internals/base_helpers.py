#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import typing

import cuml
import cuml.internals
import cuml.common


def _process_generic(gen_type):

    # Check if the type is not a generic. If not, must return "generic" if
    # subtype is CumlArray otherwise None
    if (not isinstance(gen_type, typing._GenericAlias)):
        if (issubclass(gen_type, cuml.common.CumlArray)):
            return "generic"

        # We don't handle SparseCumlArray at this time
        if (issubclass(gen_type, cuml.common.SparseCumlArray)):
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


def _get_base_return_type(class_name, attr):

    if (not hasattr(attr, "__annotations__")
            or "return" not in attr.__annotations__):
        return None

    try:
        type_hints = typing.get_type_hints(attr)

        if ("return" in type_hints):

            ret_type = type_hints["return"]

            is_generic = isinstance(ret_type, typing._GenericAlias)

            if (is_generic):
                return _process_generic(ret_type)
            elif (issubclass(ret_type, cuml.common.CumlArray)):
                return "array"
            elif (issubclass(ret_type, cuml.common.SparseCumlArray)):
                return "sparsearray"
            elif (issubclass(ret_type, cuml.Base)):
                return "base"
            else:
                return None
    except NameError:
        # A NameError is raised if the return type is the same as the
        # type being defined (which is incomplete). Check that here and
        # return base if the name matches
        if (attr.__annotations__["return"] == class_name):
            return "base"
    except Exception:
        assert False, "Shouldnt get here"
        return None

    return None


def _wrap_attribute(class_name: str,
                    attribute_name: str,
                    attribute,
                    **kwargs):

    # Skip items marked with autowrap_ignore
    if ("__cuml_is_wrapped" in attribute.__dict__
            and attribute.__dict__["__cuml_is_wrapped"]):
        return attribute

    return_type = _get_base_return_type(class_name, attribute)

    if (return_type == "generic"):
        attribute = cuml.internals.api_base_return_generic(**kwargs)(attribute)
    elif (return_type == "array"):
        attribute = cuml.internals.api_base_return_array(**kwargs)(attribute)
    elif (return_type == "sparsearray"):
        attribute = cuml.internals.api_base_return_sparse_array(
            **kwargs)(attribute)
    elif (return_type == "base"):
        attribute = cuml.internals.api_base_return_any(**kwargs)(attribute)
    elif (not attribute_name.startswith("_")):
        # Only replace public functions with return any
        attribute = cuml.internals.api_return_any()(attribute)

    return attribute


class BaseMetaClass(type):
    def __new__(cls, classname, bases, classDict):

        for attributeName, attribute in classDict.items():
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
