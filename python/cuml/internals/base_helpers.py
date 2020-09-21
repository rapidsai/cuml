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

class BaseMetaClass(type):
    def __new__(meta, classname, bases, classDict):

        newClassDict = {}

        for attributeName, attribute in classDict.items():
            if callable(attribute) and not attributeName.startswith("_"):

                # Skip items marked with autowrap_ignore
                if ("__cuml_do_not_wrap" in attribute.__dict__
                      and attribute.__dict__["__cuml_do_not_wrap"]):
                    pass
                else:
                    type_hints = typing.get_type_hints(attribute)

                    if ("return" in type_hints
                            and type_hints["return"] == cuml.common.CumlArray):
                        attribute = cuml.internals.api_base_return_array()(attribute)
                    else:
                        # replace it with a wrapped version
                        attribute = cuml.internals.cuml_internal_func(attribute)

                # # Check if we have an __cuml_internals object
                # if (BaseFunctionMetadata.func_dict_str in attribute.__dict__):
                    
            newClassDict[attributeName] = attribute

        return type.__new__(meta, classname, bases, newClassDict)
