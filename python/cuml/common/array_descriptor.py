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


def _input_to_type(input_array):
    # function to access _input_to_str, while still using the correct
    # numba check for a numba device_array
    if type(input_array) in _input_type_to_str.keys():
        return _input_type_to_str[type(input_array)]
    elif numbaArray.is_cuda_ndarray(input_array):
        return 'numba'
    elif input_array is None:
        return "none"
    else:
        return 'cupy'


@dataclass
class CumlArrayDescriptorMeta:

    # The type for the input value. One of: _input_type_to_str
    input_type: str

    # Specifies the `output_dtype` argument when calling to_output. Use `None`
    # to use the same dtype as the input
    output_dtype: str = None

    # Dict containing values in different formats. One entry per type. Both the
    # input type and any cached converted types will be stored. Erased on set
    values: dict = field(default_factory=dict)

    def get_input_value(self):
        if (self.input_type is None):
            # TODO: Need to determine if this raises an error or not
            return None

        assert self.input_type in self.values, \
            "Missing value for input_type {}".format(self.input_type)

        return self.values[self.input_type]


class CumlArrayDescriptor():
    '''Descriptor for a meter.'''
    def __set_name__(self, owner, name):
        self.name = name

    # def __init__(self, value=None):
    #     self.value = float(value)

    def _get_value(self, instance, throw_on_missing = False) -> CumlArrayDescriptorMeta:

        if (throw_on_missing):
            if (self.name not in instance.__dict__):
                raise AttributeError()

        return instance.__dict__.setdefault(
            self.name,
            CumlArrayDescriptorMeta(input_type=None, values={}))

    def _to_output(self, instance, to_output_type, to_output_dtype=None):

        existing = self._get_value(instance, throw_on_missing=True)

        # Handle setting npone

        if (existing.input_type == "none"):
            return None

        # Return a cached value if it exists
        if (to_output_type in existing.values):
            return existing.values[to_output_type]

        # If the input type was anything but CumlArray, need to create one now
        if ("cuml" not in existing.values):
            existing.values["cuml"] = CumlArray(
                existing.get_input_value())

        cuml_arr: CumlArray = existing.values["cuml"]

        # Do the conversion
        output = cuml_arr.to_output(output_type=to_output_type, output_dtype=to_output_dtype)

        # Cache the value
        existing.values[to_output_type] = output

        return output

    def __get__(self, instance, owner):

        if (instance is None):
            return self

        existing = self._get_value(instance, throw_on_missing=True)

        assert len(existing.values) > 0

        # Get the global output type
        output_type = cuml.global_output_type

        # First, determine if we need to call to_output at all
        if (output_type == "mirror"):
            # We must be internal, just return the input type
            return existing.get_input_value()

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
