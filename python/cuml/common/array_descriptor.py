# Internal, non class owned helper functions
from dataclasses import dataclass, field
from cudf.core import Series as cuSeries
from cudf.core import DataFrame as cuDataFrame
from cuml.common.array import CumlArray
from cupy import ndarray as cupyArray
from numba.cuda import devicearray as numbaArray
from numpy import ndarray as numpyArray
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
import cupy as cp
import cuml

_input_type_to_str = {
    numpyArray: 'numpy',
    cupyArray: 'cupy',
    cuSeries: 'cudf',
    cuDataFrame: 'cudf',
    pdSeries: 'numpy',
    pdDataFrame: 'numpy',
    CumlArray: "cuml",
}


def _input_to_type(input):
    # function to access _input_to_str, while still using the correct
    # numba check for a numba device_array
    if type(input) in _input_type_to_str.keys():
        return _input_type_to_str[type(input)]
    elif numbaArray.is_cuda_ndarray(input):
        return 'numba'
    else:
        return 'cupy'


@dataclass
class CumlArrayDescriptorMeta:
    input_type: str
    values: dict = field(default_factory=dict)


class CumlArrayDescriptor():
    '''Descriptor for a meter.'''
    def __set_name__(self, owner, name):
        self.name = name
        self.internal_name = self.name

    # def __init__(self, value=None):
    #     self.value = float(value)

    def _get_value(self, instance) -> CumlArrayDescriptorMeta:
        return instance.__dict__.setdefault(
            self.internal_name,
            CumlArrayDescriptorMeta(input_type=None, values={}))

    def _to_output(self, instance, to_output_type):

        existing = self._get_value(instance)

        # Return a cached value if it exists
        if (to_output_type in existing.values):
            return existing.values[to_output_type]

        # If the input type was anything but CumlArray, need to create one now
        if ("cuml" not in existing.values):
            existing.values["cuml"] = CumlArray(
                existing.values[existing.input_type])

        cumlArr = existing.values["cuml"]

        # Do the conversion
        output = cumlArr.to_output(to_output_type)

        # Cache the value
        existing.values[to_output_type] = output

        return output

    def __get__(self, instance, owner):

        if (instance is None):
            return None

        existing = self._get_value(instance)

        assert len(existing.values) > 0

        # Get the global output type
        output_type = cuml.global_output_type

        # First, determine if we need to call to_output at all
        if (output_type == "mirror"):
            # We must be internal, just return the input type
            return existing.values[existing.input_type]

        else:
            # We are external, determine the target output type
            if (output_type is None or output_type == "input"):
                # Default to the owning base object
                output_type = instance.output_type

            return self._to_output(instance, output_type)

    def __set__(self, instance, value):

        existing = self._get_value(instance)

        # Determine the type
        existing.input_type = _input_to_type(value)

        # Clear any existing values
        existing.values.clear()

        # Set the existing value
        existing.values[existing.input_type] = value
