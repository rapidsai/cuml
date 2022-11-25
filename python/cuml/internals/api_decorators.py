#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import cuml.internals.array
import cuml.internals.array_sparse
import cuml.internals.input_utils
from cuml.internals.type_utils import _DecoratorType, wraps_typed
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
from cuml.internals import logger


def _wrap_once(wrapped, *args, **kwargs):
    """Prevent wrapping functions multiple times."""
    setattr(wrapped, CUML_WRAPPED_FLAG, True)
    return functools.wraps(wrapped, *args, **kwargs)


def _has_self(sig):
    return "self" in sig.parameters and list(sig.parameters)[0] == "self"


def _find_arg(sig, default_name, default_position):
    params = list(sig.parameters)

    # Check for default name in input args
    if default_name in sig.parameters:
        return default_name, params.index(default_name)
    # Otherwise use argument in list by position
    else:
        index = int(_has_self(sig)) + default_position
        return params[index], index


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
                "were not found in args or kwargs.")


class WithArgsDecoratorMixin(object):
    """
    This decorator mixin handles processing the input arguments for all api
    decorators. It supplies the input_arg, target_arg properties
    """
    def __init__(self,
                 *,
                 input_arg: typing.Optional[str] = None,
                 target_arg: typing.Optional[str] = None,
                 needs_self=True,
                 needs_input=False,
                 needs_target=False):

        super().__init__()

        self.input_arg = None if input_arg is ... else input_arg
        self.target_arg = None if target_arg is ... else target_arg

        self.needs_self = needs_self
        self.needs_input = needs_input
        self.needs_target = needs_target

    def prep_arg_to_use(self, func) -> tuple:
        """"
        Determine from function signature what processing needs to be done.

        This function is executed once per function definition.

        Return tuple of:
            - has_self
            - input_arg (name, index)
            - target_arg (name, index)
        """

        sig = inspect.signature(func, follow_wrapped=True)

        has_self = _has_self(sig)

        if self.needs_self and not has_self:
            raise Exception("No self found on function!")

        if self.needs_input:
            input_arg = _find_arg(sig, self.input_arg or "X", 0)
        else:
            input_arg = (None, None)

        if self.needs_target:
            target_arg = _find_arg(sig, self.target_arg or "y", 1)
        else:
            target_arg = (None, None)

        return has_self, input_arg, target_arg


class HasSettersDecoratorMixin(object):
    """
    This mixin is responsible for handling any "set_XXX" methods used by api
    decorators. Mostly used by `fit()` functions
    """
    def __init__(self,
                 *,
                 set_output_type=True,
                 set_output_dtype=False,
                 set_n_features_in=True) -> None:

        super().__init__()

        self.set_output_type = set_output_type
        self.set_output_dtype = set_output_dtype
        self.set_n_features_in = set_n_features_in

        self.has_setters = (
            set_output_type or set_output_dtype or set_n_features_in)

    def do_setters(self, *, self_val, input_val, target_val):
        if (self.set_output_type):
            assert input_val is not None, \
                "`set_output_type` is False but no input_arg detected"
            self_val._set_output_type(input_val)

        if (self.set_output_dtype):
            assert target_val is not None, \
                "`set_output_dtype` is True but no target_arg detected"
            self_val._set_target_dtype(target_val)

        if (self.set_n_features_in):
            assert input_val is not None, \
                "`set_n_features_in` is False but no input_arg detected"
            if (len(input_val.shape) >= 2):
                self_val._set_n_features_in(input_val)


class HasGettersDecoratorMixin(object):
    """
    This mixin is responsible for handling any "get_XXX" methods used by api
    decorators. Used for many functions like `predict()`, `transform()`, etc.
    """
    def __init__(self,
                 *,
                 get_output_type=False,
                 get_output_dtype=False) -> None:

        super().__init__()

        self.get_output_type = get_output_type
        self.get_output_dtype = get_output_dtype

        self.has_getters = get_output_type or get_output_dtype

    class _NOT_GIVEN:
        """Sentinel that indicates the argument was not provided."""

    def do_getters(self, *,
                   self_val=_NOT_GIVEN,
                   input_val=_NOT_GIVEN,
                   target_val=_NOT_GIVEN):

        if self.get_output_type:
            if self_val is self._NOT_GIVEN:
                assert input_val is not self._NOT_GIVEN
                out_type = \
                    cuml.common.input_utils.determine_array_type(input_val)
            elif input_val is self._NOT_GIVEN:
                out_type = self_val.output_type

                if out_type == "input":
                    out_type = self_val._input_type
            else:
                out_type = self_val._get_output_type(input_val)

            assert out_type is not None
            set_api_output_type(out_type)

        if self.get_output_dtype:
            if self_val is self._NOT_GIVEN:
                assert target_val is not self._NOT_GIVEN
                output_dtype = \
                    cuml.internals.input_utils.determine_array_dtype(target_val)
            else:
                output_dtype = self_val._get_target_dtype()
            set_api_output_dtype(output_dtype)

    def has_getters_target(self, needs_self):
        return False if needs_self else self.get_output_dtype


class ReturnDirectDecorator():

    cm_class = None

    def __init__(self):
        super().__init__()

        self.do_autowrap = False

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        @_wrap_once(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        return inner

    @classmethod
    def _recreate_cm(self, func, args) -> InternalAPIContextBase:
        if self.cm_class is None:
            raise NotImplementedError()
        return self.cm_class(func, args)


class ReturnUnwindDecorator(ReturnDirectDecorator):

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        @_wrap_once(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        return inner


class ReturnAnyDecorator(ReturnDirectDecorator):
    cm_class = ReturnAnyCM


class BaseReturnAnyDecorator(ReturnDirectDecorator,
                             HasSettersDecoratorMixin,
                             WithArgsDecoratorMixin):

    cm_class = BaseReturnAnyCM

    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 set_output_type=True,
                 set_output_dtype=False,
                 set_n_features_in=True) -> None:

        ReturnDirectDecorator.__init__(self)
        HasSettersDecoratorMixin.__init__(self,
                                          set_output_type=set_output_type,
                                          set_output_dtype=set_output_dtype,
                                          set_n_features_in=set_n_features_in)
        WithArgsDecoratorMixin.__init__(self,
                                        input_arg=input_arg,
                                        target_arg=target_arg,
                                        needs_self=True,
                                        needs_input=set_output_type
                                        or set_n_features_in,
                                        needs_target=set_output_dtype)

        self.do_autowrap = self.has_setters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        has_self, input_arg, target_arg = self.prep_arg_to_use(func)

        if self.has_setters:

            @_wrap_once(func)
            def inner_with_setters(*args, **kwargs):

                with self._recreate_cm(func, args):

                    self_val = args[0] if has_self else None
                    input_val = _get_value(args, kwargs, * input_arg) \
                        if self.needs_input else None
                    target_val = _get_value(args, kwargs, * target_arg) \
                        if self.needs_target else None

                    self.do_setters(self_val=self_val,
                                    input_val=input_val,
                                    target_val=target_val)

                    return func(*args, **kwargs)

            return inner_with_setters

        else:
            return super().__call__(func)


class ReturnArrayDecorator(ReturnUnwindDecorator,
                           HasGettersDecoratorMixin,
                           WithArgsDecoratorMixin):

    cm_class = ReturnArrayCM

    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 get_output_type=False,
                 get_output_dtype=False) -> None:

        ReturnDirectDecorator.__init__(self)
        HasGettersDecoratorMixin.__init__(self,
                                          get_output_type=get_output_type,
                                          get_output_dtype=get_output_dtype)
        WithArgsDecoratorMixin.__init__(
            self,
            input_arg=input_arg,
            target_arg=target_arg,
            needs_self=False,
            needs_input=get_output_type,
            needs_target=self.has_getters_target(False))

        self.do_autowrap = self.has_getters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        _, input_arg, target_arg = self.prep_arg_to_use(func)

        if self.has_getters:

            @_wrap_once(func)
            def inner_with_getters(*args, **kwargs):
                with self._recreate_cm(func, args) as cm:

                    # Get input/target values
                    input_val = _get_value(args, kwargs, * input_arg) \
                        if self.needs_input else None
                    target_val = _get_value(args, kwargs, * target_arg) \
                        if self.needs_target else None

                    # Now execute the getters
                    self.do_getters(input_val=input_val, target_val=target_val)

                    # Call the function
                    ret_val = func(*args, **kwargs)

                return cm.process_return(ret_val)

            return inner_with_getters

        else:

            return super().__call__(func)


class ReturnSparseArrayDecorator(ReturnArrayDecorator):

    cm_class = ReturnSparseArrayCM


class BaseReturnArrayDecorator(ReturnUnwindDecorator,
                               HasSettersDecoratorMixin,
                               HasGettersDecoratorMixin,
                               WithArgsDecoratorMixin):

    cm_class = BaseReturnArrayCM

    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 get_output_type=True,
                 get_output_dtype=False,
                 set_output_type=False,
                 set_output_dtype=False,
                 set_n_features_in=False) -> None:

        ReturnDirectDecorator.__init__(self)
        HasSettersDecoratorMixin.__init__(self,
                                          set_output_type=set_output_type,
                                          set_output_dtype=set_output_dtype,
                                          set_n_features_in=set_n_features_in)
        HasGettersDecoratorMixin.__init__(self,
                                          get_output_type=get_output_type,
                                          get_output_dtype=get_output_dtype)
        WithArgsDecoratorMixin.__init__(
            self,
            input_arg=input_arg,
            target_arg=target_arg,
            needs_self=True,
            needs_input=input_arg is not None
            and (set_output_type or set_n_features_in or get_output_type),
            needs_target=set_output_dtype
            or (False if True else get_output_dtype)
        )

        self.do_autowrap = self.has_setters or self.has_getters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        has_self, input_arg, target_arg = self.prep_arg_to_use(func)

        @_wrap_once(func)
        def inner_set_get(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val = args[0] if has_self else None
                input_val = _get_value(args, kwargs, * input_arg) \
                    if input_arg else None
                target_val = _get_value(args, kwargs, * target_arg) \
                    if target_arg else None

                # Must do the setters first
                self.do_setters(self_val=self_val,
                                input_val=input_val,
                                target_val=target_val)

                # Now execute the getters
                if (self.needs_input):
                    self.do_getters(self_val=self_val, input_val=input_val)
                else:
                    self.do_getters(self_val=self_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @_wrap_once(func)
        def inner_set(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val = args[0] if has_self else None
                input_val = _get_value(args, kwargs, * input_arg)
                target_val = _get_value(args, kwargs, * target_arg)

                # Must do the setters first
                self.do_setters(self_val=self_val,
                                input_val=input_val,
                                target_val=target_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @_wrap_once(func)
        def inner_get(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val = args[0] if has_self else None

                # Do the getters
                if self.needs_input:
                    input_val = _get_value(args, kwargs, * input_arg)
                    self.do_getters(self_val=self_val, input_val=input_val)
                else:
                    self.do_getters(self_val=self_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        # Return the function depending on whether or not we do any automatic
        # wrapping
        if (self.has_getters and self.has_setters):
            return inner_set_get
        elif (self.has_getters):
            return inner_get
        elif (self.has_setters):
            return inner_set
        else:
            return super().__call__(func)


class BaseReturnSparseArrayDecorator(BaseReturnArrayDecorator):
    cm_class = BaseReturnSparseArrayCM


class ReturnGenericDecorator(ReturnArrayDecorator):
    cm_class = ReturnGenericCM


class BaseReturnGenericDecorator(BaseReturnArrayDecorator):
    cm_class = BaseReturnGenericCM


class BaseReturnArrayFitTransformDecorator(BaseReturnArrayDecorator):
    """
    Identical to `BaseReturnArrayDecorator`, however the defaults have been
    changed to better suit `fit_transform` methods
    """
    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 get_output_type=True,
                 get_output_dtype=False,
                 set_output_type=True,
                 set_output_dtype=False,
                 set_n_features_in=True) -> None:

        super().__init__(input_arg=input_arg,
                         target_arg=target_arg,
                         get_output_type=get_output_type,
                         get_output_dtype=get_output_dtype,
                         set_output_type=set_output_type,
                         set_output_dtype=set_output_dtype,
                         set_n_features_in=set_n_features_in)


api_return_any = ReturnAnyDecorator
api_base_return_any = BaseReturnAnyDecorator
api_return_array = ReturnArrayDecorator
api_base_return_array = BaseReturnArrayDecorator
api_return_generic = ReturnGenericDecorator
api_base_return_generic = BaseReturnGenericDecorator
api_base_fit_transform = BaseReturnArrayFitTransformDecorator

api_return_sparse_array = ReturnSparseArrayDecorator
api_base_return_sparse_array = BaseReturnSparseArrayDecorator

api_return_array_skipall = ReturnArrayDecorator(get_output_dtype=False,
                                                get_output_type=False)

api_base_return_any_skipall = BaseReturnAnyDecorator(set_output_type=False,
                                                     set_n_features_in=False)
api_base_return_array_skipall = BaseReturnArrayDecorator(get_output_type=False)
api_base_return_generic_skipall = BaseReturnGenericDecorator(
    get_output_type=False)


@contextlib.contextmanager
def exit_internal_api():

    assert (GlobalSettings().root_cm is not None)

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
    assigned=('__doc__', '__annotations__'),
    updated=functools.WRAPPER_UPDATES
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
                    '{}={}'.format(name, arg) for name,
                    arg in zip(kwonly_args[:extra_args], args[-extra_args:])
                ]
                warnings.warn(
                    "Pass {} as keyword args. From version {}, "
                    "passing these as positional arguments will "
                    "result in an error".format(", ".join(args_msg),
                                                self._version),
                    FutureWarning,
                    stacklevel=2)

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
        if hasattr(self, '_full_kwargs'):
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
                logger.info("Unused keyword parameter: {} "
                            "during cuML estimator "
                            "initialization".format(keyword))

        return init_func(self, *args, **filtered_kwargs)
    return processor


def enable_device_interop(gpu_func):
    @functools.wraps(gpu_func)
    def dispatch(self, *args, **kwargs):
        # check that the estimator implements CPU/GPU interoperability
        if hasattr(self, 'dispatch_func'):
            func_name = gpu_func.__name__
            return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
        else:
            return gpu_func(self, *args, **kwargs)
    return dispatch
