#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import functools
import inspect

from cuml.internals.api_context_managers import (
    CumlAPIContext,
    GlobalSettingsContext
)
from cuml.internals.array import (
    cuml_array_to_output,
    generator_of_arrays_to_output,
    iterable_of_arrays_to_output,
    dict_of_arrays_to_output
)
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import (
    determine_array_type,
    determine_array_dtype
)
from enum import Enum, auto
from functools import wraps

global_settings = GlobalSettings()


class CumlReturnType(Enum):
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



class cuml_function:
    """A decorator for cuML API functions

    This decorator's primary purpose is to track the type of data which is
    provided as inputs to cuML and ensure that data of the corresponding type
    are returned upon exiting the cuML API boundary. For instance, if a user
    provides numpy input, they should receive numpy output. If the user
    provides cuDF input, they should receive cuDF output.
    """
    def __init__(
        self,
        input_param='X',
        target_param='y',
        return_type=CumlReturnType.raw
    ):
        # Dictionary mapping parameter names to their purpose (e.g. as input,
        # target, etc.) for use in parsing input to a wrapped function
        self.params_to_handle = {
            "self": "self",
        }
        if input_param is not None:
            self.params_to_handle[input_param] = "input"
        if target_param is not None:
            self.params_to_handle[target_param] = "target"
        self.return_type = return_type

    def __call__(self, func):

        signature = inspect.signature(func)
        # Mapping from parameter purpose (input, target, ...) to a two-element
        # tuple. The first element is the index where the param may appear for
        # positional inputs (if any) and the second is the name of that param
        # as it may appear in a dictionary of kwargs
        param_parsing_dict = {}

        for param_index, (name, spec) in enumerate(signature.parameters.items()):
            if name in self.params_to_handle:
                if spec.kind == inspect.Parameter.POSITIONAL_ONLY:
                    param_parsing_dict[
                        self.params_to_handle[name]
                    ] = (param_index, None)
                elif spec.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    param_parsing_dict[
                        self.params_to_handle[name]
                    ] = (param_index, name)
                else:
                    param_parsing_dict[
                        self.params_to_handle[name]
                    ] = (None, name)

        def get_value_from_params(parse_spec, *args, **kwargs):
            try:
                return kwargs[parse_spec[1]]
            except KeyError:
                if parse_spec[0] is not None:
                    try:
                        return args[parse_spec[0]]
                    except IndexError:
                        pass
            return None

        @functools.wraps(func)
        def cuml_func(*args, **kwargs):
            special_param_values = {
                param_purpose: get_value_from_params(param_spec, *args, **kwargs)
                for param_purpose, param_spec in param_parsing_dict
            }
            special_param_values = {
                key: value for key, value in special_param_values.items()
                if value is not None
            }

            api_output_type = None
            api_output_dtype = None
            if "input" in special_param_values:
                if "self" in special_param_values:
                    estimator = special_param_values["self"]
                    estimator._set_output_type(
                        special_param_values["input"]
                    )
                    if len(special_param_values["input"].shape) > 1:
                        estimator._set_n_features_in(
                            special_param_values["input"]
                        )
                else:
                    api_output_type = determine_array_type(
                        special_param_values["input"]
                    )
            elif "self" in special_param_values:
                estimator = special_param_values["self"]
                api_output_type = estimator.output_type
                if api_output_type == "input":
                    api_output_type = estimator._input_type

            if "target" in special_param_values:
                if "self" in special_param_values:
                    estimator = special_param_values["self"]
                    estimator._set_target_dtype(
                        determine_array_dtype(
                            special_param_values["target"]
                        )
                    )
                    api_output_dtype = estimator._get_target_dtype()
                else:
                    api_output_dtype = determine_array_dtype(
                        special_param_values["target"]
                    )
            elif "self" in special_param_values:
                estimator = special_param_values["self"]
                api_output_dtype = estimator._get_target_dtype()

            with GlobalSettingsContext(
                output_type = api_output_type,
                output_dtype = api_output_dtype
            ):
                with CumlAPIContext():
                    result = func(*args, **kwargs)

                if (
                    global_settings.in_internal_api() or
                    self.return_type == CumlReturnType.raw
                ):
                    return result

                if self.return_type == CumlReturnType.array:
                    return cuml_array_to_output(result)
                elif self.return_type == CumlReturnType.generator_of_arrays:
                    return generator_of_arrays_to_output(result)
                elif self.return_type == CumlReturnType.iterable_of_arrays:
                    return iterable_of_arrays_to_output(result)
                elif self.return_type == CumlReturnType.dict_of_arrays_to_output:
                    return dict_of_arrays_to_output(result)
                else:
                    return result

        return cuml_func


class _deprecate_pos_args:
    """
    Decorator that issues a warning when using positional args that should be
    keyword args. Mimics sklearn's `_deprecate_positional_args` with added
    functionality.

    For any class that derives from `cuml.Base`, this decorator will be
    automatically added to `__init__`. In this scenario, its assumed that all
    arguments are keyword arguments. To override the functionality this
    decorator can be manually added, allowing positional arguments if
    necessary.

    Parameters
    ----------
    version : str
        This version will be specified in the warning message as the
        version when positional arguments will be removed

    """

    FLAG_NAME: typing.ClassVar[str] = "__cuml_deprecated_pos"

    def __init__(self, version: str):

        self._version = version

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        sig = inspect.signature(func)
        kwonly_args = []
        all_args = []

        # Store all the positional and keyword only args
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps_typed(func)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args > 0:
                # ignore first 'self' argument for instance methods
                args_msg = [
                    "{}={}".format(name, arg)
                    for name, arg in zip(
                        kwonly_args[:extra_args], args[-extra_args:]
                    )
                ]
                warnings.warn(
                    "Pass {} as keyword args. From version {}, "
                    "passing these as positional arguments will "
                    "result in an error".format(
                        ", ".join(args_msg), self._version
                    ),
                    FutureWarning,
                    stacklevel=2,
                )

            # Convert all positional args to keyword
            kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})

            return func(**kwargs)

        # Set this flag to prevent auto adding this decorator twice
        inner_f.__dict__[_deprecate_pos_args.FLAG_NAME] = True

        return inner_f


def device_interop_preparation(init_func):
    """
    This function serves as a decorator for cuML estimators that implement
    the CPU/GPU interoperability feature. It processes the estimator's
    hyperparameters by saving them and filtering them for GPU execution.
    """

    @functools.wraps(init_func)
    def processor(self, *args, **kwargs):
        # if child class is already prepared for interop, skip
        if hasattr(self, "_full_kwargs"):
            return init_func(self, *args, **kwargs)

        # Save all kwargs
        self._full_kwargs = kwargs
        # Generate list of available cuML hyperparameters
        gpu_hyperparams = list(inspect.signature(init_func).parameters.keys())

        # Filter provided parameters for cuML estimator initialization
        filtered_kwargs = {}
        for keyword, arg in self._full_kwargs.items():
            if keyword in gpu_hyperparams:
                filtered_kwargs[keyword] = arg
            else:
                logger.info(
                    "Unused keyword parameter: {} "
                    "during cuML estimator "
                    "initialization".format(keyword)
                )

        return init_func(self, *args, **filtered_kwargs)

    return processor


def enable_device_interop(gpu_func):
    @functools.wraps(gpu_func)
    def dispatch(self, *args, **kwargs):
        # check that the estimator implements CPU/GPU interoperability
        if hasattr(self, "dispatch_func"):
            func_name = gpu_func.__name__
            return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
        else:
            return gpu_func(self, *args, **kwargs)

    return dispatch
