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

import functools
import inspect

from cuml.internals.api_context_managers import (
    CumlAPIContext,
    GlobalSettingsContext,
)
from cuml.internals.array import (
    CumlReturnType,
    cuml_array_to_output,
    generator_of_arrays_to_output,
    iterable_of_arrays_to_output,
    dict_of_arrays_to_output,
)
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import (
    determine_array_type,
    determine_array_dtype,
)

global_settings = GlobalSettings()


class cuml_function:
    """A decorator for cuML API functions

    This decorator's primary purpose is to track the type of data which is
    provided as inputs to cuML and ensure that data of the corresponding type
    are returned upon exiting the cuML API boundary. For instance, if a user
    provides numpy input, they should receive numpy output. If the user
    provides cuDF input, they should receive cuDF output.
    """

    def __init__(
        self, input_param="X", target_param="y", return_type=CumlReturnType.raw
    ):
        # Dictionary mapping parameter names to their purpose (e.g. as input,
        # target, etc.) for use in parsing input to a wrapped function. IF YOU
        # ARE DEBUGGING ISSUES WITH HOW GLOBAL OUTPUT_TYPE OR OUTPUT_DTYPE ARE
        # BEING SET FOR AN ESTIMATOR, ANY PARAMS IN THIS DICTIONARY MAY BE
        # RELEVANT TO YOU.
        self.params_to_process = {
            "self": "self",
        }
        if input_param is not None:
            self.params_to_process[input_param] = "input"
        if target_param is not None:
            self.params_to_process[target_param] = "target"
        self.return_type = return_type

    def __call__(self, func):

        signature = inspect.signature(func)
        # Mapping from parameter purpose (input, target, ...) to a two-element
        # tuple. The first element is the index where the param may appear for
        # positional inputs (if any) and the second is the name of that param
        # as it may appear in a dictionary of kwargs
        param_parsing_dict = {}

        for param_index, (name, spec) in enumerate(
            signature.parameters.items()
        ):
            if name in self.params_to_process:
                if spec.kind == inspect.Parameter.POSITIONAL_ONLY:
                    param_parsing_dict[self.params_to_process[name]] = (
                        param_index,
                        None,
                    )
                elif spec.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    param_parsing_dict[self.params_to_process[name]] = (
                        param_index,
                        name,
                    )
                else:
                    param_parsing_dict[self.params_to_process[name]] = (
                        None,
                        name,
                    )

        # This helper function lets us retrieve values for different parameters
        # that may be passed either positionally or as keywords. The parse_spec
        # argument is a tuple whose first element is the positional index (if
        # any) and the second is the keyword name (if any) for the required
        # param.
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

        # When the underlying func is called, first update global settings with
        # the correct output_type based on the arguments to the function.
        #
        # Then enter the cuML API context, incrementing the global API depth
        # counter so that we can track whether or not values returned from this
        # function will be returned external to the cuML API.
        #
        # Call the function.
        #
        # Finally, if we are leaving the cuML API, perform special handling of
        # its output to convert CumlArray objects to the global output type
        # based on the specified return type for this decorator.
        @functools.wraps(func)
        def cuml_func(*args, **kwargs):
            special_param_values = {
                param_purpose: get_value_from_params(
                    param_spec, *args, **kwargs
                )
                for param_purpose, param_spec in param_parsing_dict
            }
            special_param_values = {
                key: value
                for key, value in special_param_values.items()
                if value is not None
            }

            api_output_type = None
            api_output_dtype = None
            if "input" in special_param_values:
                if "self" in special_param_values:
                    estimator = special_param_values["self"]
                    estimator._set_output_type(special_param_values["input"])
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
                        determine_array_dtype(special_param_values["target"])
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
                output_type=api_output_type, output_dtype=api_output_dtype
            ):
                with CumlAPIContext():
                    result = func(*args, **kwargs)

                if (
                    global_settings.in_internal_api()
                    or self.return_type == CumlReturnType.raw
                ):
                    return result

                if self.return_type == CumlReturnType.array:
                    return cuml_array_to_output(result)
                elif self.return_type == CumlReturnType.generator_of_arrays:
                    return generator_of_arrays_to_output(result)
                elif self.return_type == CumlReturnType.iterable_of_arrays:
                    return iterable_of_arrays_to_output(result)
                elif (
                    self.return_type == CumlReturnType.dict_of_arrays_to_output
                ):
                    return dict_of_arrays_to_output(result)
                else:
                    return result

        return cuml_func
