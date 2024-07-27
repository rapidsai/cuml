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

import contextlib
import functools
import inspect
import typing
import warnings

# TODO: Try to resolve circular import that makes this necessary:
from cuml.internals import input_utils as iu
from cuml.internals.api_context_managers import BaseReturnAnyCM
from cuml.internals.api_context_managers import BaseReturnArrayCM
from cuml.internals.api_context_managers import BaseReturnGenericCM
from cuml.internals.api_context_managers import BaseReturnSparseArrayCM
from cuml.internals.api_context_managers import InternalAPIContextBase
from cuml.internals.api_context_managers import ReturnAnyCM
from cuml.internals.api_context_managers import ReturnArrayCM
from cuml.internals.api_context_managers import ReturnGenericCM
from cuml.internals.api_context_managers import ReturnSparseArrayCM
from cuml.internals.api_context_managers import set_api_output_dtype
from cuml.internals.api_context_managers import set_api_output_type
from cuml.internals.constants import CUML_WRAPPED_FLAG
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.memory_utils import using_output_type
from cuml.internals.type_utils import _DecoratorType, wraps_typed
from cuml.internals import logger


def _wrap_once(wrapped, *args, **kwargs):
    """Prevent wrapping functions multiple times."""
    setattr(wrapped, CUML_WRAPPED_FLAG, True)
    return functools.wraps(wrapped, *args, **kwargs)


def _has_self(sig):
    return "self" in sig.parameters and list(sig.parameters)[0] == "self"


def _find_arg(sig, arg_name, default_position):
    params = list(sig.parameters)

    # Check for default name in input args
    if arg_name in sig.parameters:
        return arg_name, params.index(arg_name)
    # Otherwise use argument in list by position
    elif arg_name is ...:
        index = int(_has_self(sig)) + default_position
        return params[index], index
    else:
        raise ValueError(f"Unable to find parameter '{arg_name}'.")


def _get_value(args, kwargs, name, index):
    """Determine value for a given set of args, kwargs, name and index."""
    try:
        return kwargs[name]
    except KeyError:
        try:
            return args[index]
        except IndexError:
            raise IndexError(
                f"Specified arg idx: {index}, and argument name: {name}, "
                "were not found in args or kwargs."
            )


def _make_decorator_function(
    context_manager_cls: InternalAPIContextBase,
    process_return=True,
    needs_self: bool = False,
    **defaults,
) -> typing.Callable[..., _DecoratorType]:
    # This function generates a function to be applied as decorator to a
    # wrapped function. For example:
    #
    #       a_decorator = _make_decorator_function(...)
    #
    #       ...
    #
    #       @a_decorator(...)  # apply decorator where appropriate
    #       def fit(X, y):
    #           ...
    #
    # Note: The decorator function can be partially closed by directly
    # providing keyword arguments to this function to be used as defaults.

    def decorator_function(
        input_arg: str = ...,
        target_arg: str = ...,
        get_output_type: bool = False,
        set_output_type: bool = False,
        get_output_dtype: bool = False,
        set_output_dtype: bool = False,
        set_n_features_in: bool = False,
    ) -> _DecoratorType:
        def decorator_closure(func):
            # This function constitutes the closed decorator that will return
            # the wrapped function. It performs function introspection at
            # function definition time. The code within the wrapper function is
            # executed at function execution time.

            # Prepare arguments
            sig = inspect.signature(func, follow_wrapped=True)

            has_self = _has_self(sig)
            if needs_self and not has_self:
                raise Exception("No self found on function!")

            if input_arg is not None and (
                set_output_type
                or set_output_dtype
                or set_n_features_in
                or get_output_type
            ):
                input_arg_ = _find_arg(sig, input_arg or "X", 0)
            else:
                input_arg_ = None

            if set_output_dtype or (get_output_dtype and not has_self):
                target_arg_ = _find_arg(sig, target_arg or "y", 1)
            else:
                target_arg_ = None

            @_wrap_once(func)
            def wrapper(*args, **kwargs):
                # Wraps the decorated function, executed at runtime.

                with context_manager_cls(func, args) as cm:

                    self_val = args[0] if has_self else None

                    if input_arg_:
                        input_val = _get_value(args, kwargs, *input_arg_)
                    else:
                        input_val = None
                    if target_arg_:
                        target_val = _get_value(args, kwargs, *target_arg_)
                    else:
                        target_val = None

                    if set_output_type:
                        assert self_val is not None
                        self_val._set_output_type(input_val)
                    if set_output_dtype:
                        assert self_val is not None
                        self_val._set_target_dtype(target_val)
                    if set_n_features_in and len(input_val.shape) >= 2:
                        assert self_val is not None
                        self_val._set_n_features_in(input_val)

                    if get_output_type:
                        if self_val is None:
                            assert input_val is not None
                            out_type = iu.determine_array_type(input_val)
                        elif input_val is None:
                            out_type = self_val.output_type
                            if out_type == "input":
                                out_type = self_val._input_type
                        else:
                            out_type = self_val._get_output_type(input_val)

                        set_api_output_type(out_type)

                    if get_output_dtype:
                        if self_val is None:
                            assert target_val is not None
                            output_dtype = iu.determine_array_dtype(target_val)
                        else:
                            output_dtype = self_val._get_target_dtype()

                        set_api_output_dtype(output_dtype)

                    if process_return:
                        ret = func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                return cm.process_return(ret)

            return wrapper

        return decorator_closure

    return functools.partial(decorator_function, **defaults)


api_return_any = _make_decorator_function(ReturnAnyCM, process_return=False)
api_base_return_any = _make_decorator_function(
    BaseReturnAnyCM,
    needs_self=True,
    set_output_type=True,
    set_n_features_in=True,
)
api_return_array = _make_decorator_function(ReturnArrayCM, process_return=True)
api_base_return_array = _make_decorator_function(
    BaseReturnArrayCM,
    needs_self=True,
    process_return=True,
    get_output_type=True,
)
api_return_generic = _make_decorator_function(
    ReturnGenericCM, process_return=True
)
api_base_return_generic = _make_decorator_function(
    BaseReturnGenericCM,
    needs_self=True,
    process_return=True,
    get_output_type=True,
)
api_base_fit_transform = _make_decorator_function(
    # TODO: add tests for this decorator(
    BaseReturnArrayCM,
    needs_self=True,
    process_return=True,
    get_output_type=True,
    set_output_type=True,
    set_n_features_in=True,
)

api_return_sparse_array = _make_decorator_function(
    ReturnSparseArrayCM, process_return=True
)
api_base_return_sparse_array = _make_decorator_function(
    BaseReturnSparseArrayCM,
    needs_self=True,
    process_return=True,
    get_output_type=True,
)

api_base_return_any_skipall = api_base_return_any(
    set_output_type=False, set_n_features_in=False
)
api_base_return_array_skipall = api_base_return_array(get_output_type=False)
api_base_return_generic_skipall = api_base_return_generic(
    get_output_type=False
)


@contextlib.contextmanager
def exit_internal_api():

    assert GlobalSettings().root_cm is not None

    try:
        old_root_cm = GlobalSettings().root_cm

        GlobalSettings().root_cm = None

        # Set the global output type to the previous value to pretend we never
        # entered the API
        with using_output_type(old_root_cm.prev_output_type):

            yield

    finally:
        GlobalSettings().root_cm = old_root_cm


def mirror_args(
    wrapped: _DecoratorType,
    assigned=("__doc__", "__annotations__"),
    updated=functools.WRAPPER_UPDATES,
) -> typing.Callable[[_DecoratorType], _DecoratorType]:
    return _wrap_once(wrapped=wrapped, assigned=assigned, updated=updated)


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
