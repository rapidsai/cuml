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
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.array import CumlArray
from enum import Enum, auto


class CumlReturnType(Enum):
    """An enum used to control what happens to CumlArrays at the cuML API
    boundary

    CumlArray offers an internal representation of data which may be provided
    to cuML via numpy, cupy, Pandas, cuDF or some other framework, but it is
    not intended to be returned as output from any cuML API function. This enum
    specifies what kind of output is returned from a cuML function/method in
    order to automatically convert any CumlArray in the output to the expected
    output type.
    """

    # Return an array, which can directly be converted to the
    # globally-specified output type
    array = auto()
    # Return a generator whose elements can be converted to the
    # globally-specified output type
    generator_of_arrays = auto()
    # Return an iterable whose elements can be converted to the
    # globally-specified output type
    iterable_of_arrays = auto()
    # Return a dictionary, some of whose values can be converted to the
    # globally-specified output type
    dict_of_arrays = auto()
    # Return a value of unspecified type which will not undergo
    # conversion
    raw = auto()


class UnexpectedReturnType(Exception):
    """Exception raised when the specified CumlReturnType is unexpected"""


def _cuml_array_to_output(arr):
    if isinstance(arr, CumlArray):
        return arr.to_output(
            output_type=GlobalSettings().output_type,
            output_dtype=GlobalSettings().output_dtype,
            output_mem_type=GlobalSettings().output_mem_type,
        )
    else:
        return arr


def _generator_of_arrays_to_output(gen):
    def generator():
        for val in gen:
            yield _cuml_array_to_output(val)

    return generator()


def _iterable_of_arrays_to_output(iterable):
    return type(iterable)(_cuml_array_to_output(val) for val in iterable)


def _dict_of_arrays_to_output(dictionary):
    return {key: _cuml_array_to_output(val) for key, val in dictionary.items()}


def cuml_return_value_to_external_output(result, return_type):
    if return_type == CumlReturnType.array:
        return _cuml_array_to_output(result)
    if return_type == CumlReturnType.generator_of_arrays:
        return _generator_of_arrays_to_output(result)
    if return_type == CumlReturnType.iterable_of_arrays:
        return _iterable_of_arrays_to_output(result)
    if return_type == CumlReturnType.dict_of_arrays:
        return _dict_of_arrays_to_output(result)
    if return_type == CumlReturnType.raw:
        return result
    raise UnexpectedReturnType(f"Unrecognized return type {return_type}")
