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
from dataclasses import dataclass

import cuml
import cuml.internals
import cuml.common


@dataclass
class BaseFunctionMetadata:
    ignore: bool = False
    returns_self: bool = False
    returns_cumlarray: bool = False

    func_dict_str: typing.ClassVar[str] = "__cuml_base_wrapper"


# def _process_function(func: typing.Callable, args, kwargs, func_meta: BaseFunctionMetadata):

#     typing.cast(dict, func.__dict__).

#     if (BaseFunctionMetadata.func_dict_str in func.__dict__):

#     else:
#         pass

#     return func


def _process_generic(gen_type):

    if (gen_type.__origin__ is tuple):
        # Check for a CumlArray type in the args
        for arg in gen_type.__args__:
            if (issubclass(arg, cuml.common.CumlArray)):
                return "generic"

    return None

class BaseMetaClass(type):
    def __new__(meta, classname, bases, classDict):

        newClassDict = {}

        def get_base_return_type(attr):

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
                    elif (issubclass(ret_type, cuml.Base)):
                        return "base"
                    else:
                        print("Other")
            except NameError:
                if (attr.__annotations__["return"] == classname):
                    return "base"
            except Exception as ex:
                return None

            return None

        for attributeName, attribute in classDict.items():
            if callable(attribute):

                # Skip items marked with autowrap_ignore
                if ("__cuml_do_not_wrap" in attribute.__dict__
                        and attribute.__dict__["__cuml_do_not_wrap"]):
                    pass
                else:
                    return_type = get_base_return_type(attribute)

                    if (return_type == "generic"):
                        attribute = cuml.internals.api_base_return_genericarray()(
                            attribute)
                    elif (return_type == "array"):
                        attribute = cuml.internals.api_base_return_array()(
                            attribute)
                    elif (return_type == "base"):
                        attribute = cuml.internals.wrap_api_base_return_any()(
                            attribute)
                    elif (not attributeName.startswith("_")):
                        # Only replace public functions with return any
                        attribute = cuml.internals.wrap_api_return_any()(
                            attribute)

                # # Check if we have an __cuml_internals object
                # if (BaseFunctionMetadata.func_dict_str in attribute.__dict__):

            newClassDict[attributeName] = attribute

        return type.__new__(meta, classname, bases, newClassDict)
