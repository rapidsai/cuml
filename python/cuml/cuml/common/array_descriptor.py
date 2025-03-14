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

from dataclasses import dataclass, field
from cuml.internals.array import CumlArray
import cuml
from cuml.internals.input_utils import (
    input_to_cuml_array,
    determine_array_type,
)


@dataclass
class CumlArrayDescriptorMeta:

    # The type for the input value. One of: _input_type_to_str
    input_type: str

    # Dict containing values in different formats. One entry per type. Both the
    # input type and any cached converted types will be stored. Erased on set
    values: dict = field(default_factory=dict)

    def get_input_value(self):

        assert (
            self.input_type in self.values
        ), "Missing value for input_type {}".format(self.input_type)

        return self.values[self.input_type]

    def __getstate__(self):
        # Need to only return the input_value from
        return {
            "input_type": self.input_type,
            "input_value": self.get_input_value(),
        }

    def __setstate__(self, d):
        self.input_type = d["input_type"]
        self.values = {self.input_type: d["input_value"]}


class CumlArrayDescriptor:
    """
    Python descriptor object to control getting/setting `CumlArray` attributes
    on `Base` objects. See the Estimator Guide for an in depth guide.
    """

    def __init__(self, order="K"):
        # order corresponds to the order that the CumlArray attribute
        # should be in to work with the C++ algorithms.
        self.order = order

    def __set_name__(self, owner, name):
        self.name = name

    def _get_meta(
        self, instance, throw_on_missing=False
    ) -> CumlArrayDescriptorMeta:

        if throw_on_missing:
            if self.name not in instance.__dict__:
                raise AttributeError()

        return instance.__dict__.setdefault(
            self.name, CumlArrayDescriptorMeta(input_type=None, values={})
        )

    def _to_output(self, instance, to_output_type, to_output_dtype=None):
        existing = self._get_meta(instance, throw_on_missing=True)

        # Handle input_type==None which means we have a non-array object stored
        if existing.input_type is None:
            # Dont save in the cache. Just return the value
            return existing.values[existing.input_type]

        # Return a cached value if it exists
        if to_output_type in existing.values:
            return existing.values[to_output_type]

        # If the input type was anything but CumlArray, need to create one now
        if "cuml" not in existing.values:
            existing.values["cuml"] = input_to_cuml_array(
                existing.get_input_value(), order="K"
            ).array

        cuml_arr: CumlArray = existing.values["cuml"]

        # Do the conversion
        output = cuml_arr.to_output(
            output_type=to_output_type, output_dtype=to_output_dtype
        )

        # Cache the value
        existing.values[to_output_type] = output

        return output

    def __get__(self, instance, owner):

        if instance is None:
            return self

        existing = self._get_meta(instance, throw_on_missing=True)

        assert len(existing.values) > 0

        # Get the global output type
        output_type = cuml.global_settings.output_type

        # First, determine if we need to call to_output at all
        if output_type == "mirror":
            # We must be internal, just return the input type
            return existing.get_input_value()

        else:
            # We are external, determine the target output type
            if output_type is None:
                # Default to the owning base object output_type
                output_type = instance.output_type

            if output_type == "input":
                # Default to the owning base object, _input_type
                output_type = instance._input_type

            return self._to_output(instance, output_type)

    def __set__(self, instance, value):

        existing = self._get_meta(instance)

        # Determine the type
        existing.input_type = determine_array_type(value)

        # Clear any existing values
        existing.values.clear()

        # Set the existing value
        existing.values[existing.input_type] = value

    def __delete__(self, instance):

        if instance is not None:
            del instance.__dict__[self.name]
