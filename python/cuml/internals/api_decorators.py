#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from functools import wraps
import warnings

import cuml
import cuml.common
import cuml.common.array
import cuml.common.array_sparse
import cuml.common.input_utils
from cuml.common.type_utils import _DecoratorType, wraps_typed
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
from cuml.internals.base_helpers import _get_base_return_type

CUML_WRAPPED_FLAG = "__cuml_is_wrapped"


class DecoratorMetaClass(type):
    """
    This metaclass is used to prevent wrapping functions multiple times by
    adding `__cuml_is_wrapped = True` to the function __dict__
    """
    def __new__(cls, classname, bases, classDict):

        if ("__call__" in classDict):

            func = classDict["__call__"]

            @wraps(func)
            def wrap_call(*args, **kwargs):
                ret_val = func(*args, **kwargs)

                ret_val.__dict__[CUML_WRAPPED_FLAG] = True

                return ret_val

            classDict["__call__"] = wrap_call

        return type.__new__(cls, classname, bases, classDict)


class WithArgsDecoratorMixin(object):
    """
    This decorator mixin handles processing the input arguments for all api
    decorators. It supplies the input_arg, target_arg properties
    """
    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 needs_self=True,
                 needs_input=False,
                 needs_target=False):
        super().__init__()

        # For input_arg and target_arg, use Ellipsis to auto detect, None to
        # skip (this has different functionality on Base where it can determine
        # the output type like CumlArrayDescriptor)
        self.input_arg = input_arg
        self.target_arg = target_arg

        self.needs_self = needs_self
        self.needs_input = needs_input
        self.needs_target = needs_target

    def prep_arg_to_use(self, func) -> bool:

        # Determine from the signature what processing needs to be done. This
        # is executed once per function on import
        sig = inspect.signature(func, follow_wrapped=True)
        sig_args = list(sig.parameters.keys())

        self.has_self = "self" in sig.parameters and sig_args.index(
            "self") == 0

        if (not self.has_self and self.needs_self):
            raise Exception("No self found on function!")

        # Return early if we dont need args
        if (not self.needs_input and not self.needs_target):
            return

        self_offset = (1 if self.has_self else 0)

        if (self.needs_input):
            input_arg_to_use = self.input_arg
            input_arg_to_use_name = None

            # if input_arg is None, then set to first non self argument
            if (input_arg_to_use is ...):

                # Check for "X" in input args
                if ("X" in sig_args):
                    input_arg_to_use = "X"
                else:
                    if (len(sig.parameters) <= self_offset):
                        raise Exception("No input_arg could be determined!")

                    input_arg_to_use = sig_args[self_offset]

            # Now convert that to an index
            if (isinstance(input_arg_to_use, str)):
                input_arg_to_use_name = input_arg_to_use
                input_arg_to_use = sig_args.index(input_arg_to_use)

            assert input_arg_to_use != -1 and input_arg_to_use is not None, \
                "Could not determine input_arg"

            # Save the name and argument to use later
            self.input_arg_to_use = input_arg_to_use
            self.input_arg_to_use_name = input_arg_to_use_name

        if (self.needs_target):

            target_arg_to_use = self.target_arg
            target_arg_to_use_name = None

            # if input_arg is None, then set to first non self argument
            if (target_arg_to_use is ...):

                # Check for "y" in args
                if ("y" in sig_args):
                    target_arg_to_use = "y"
                else:
                    if (len(sig.parameters) <= self_offset + 1):
                        raise Exception("No target_arg could be determined!")

                    target_arg_to_use = sig_args[self_offset + 1]

            # Now convert that to an index
            if (isinstance(target_arg_to_use, str)):
                target_arg_to_use_name = target_arg_to_use
                target_arg_to_use = sig_args.index(target_arg_to_use)

            assert target_arg_to_use != -1 and target_arg_to_use is not None, \
                "Could not determine target_arg"

            # Save the name and argument to use later
            self.target_arg_to_use = target_arg_to_use
            self.target_arg_to_use_name = target_arg_to_use_name

        return True

    def get_arg_values(self, *args, **kwargs):
        """
        This function is called once per function invocation to get the values
        of self, input and target.

        Returns
        -------
        tuple
            Returns a tuple of self, input, target values

        Raises
        ------
        IndexError
            Raises an exception if the specified input argument is not
            available or called with the wrong number of arguments
        """
        self_val = None
        input_val = None
        target_val = None

        if (self.has_self):
            self_val = args[0]

        if (self.needs_input):
            # Check if its set to a string
            if (isinstance(self.input_arg_to_use, str)):
                input_val = kwargs[self.input_arg_to_use]

            # If all arguments are set by name, then this can happen
            elif (self.input_arg_to_use >= len(args)):
                # Check for the name in kwargs
                if (self.input_arg_to_use_name in kwargs):
                    input_val = kwargs[self.input_arg_to_use_name]
                else:
                    raise IndexError(
                        ("Specified input_arg idx: {}, and argument name: {}, "
                         "were not found in args or kwargs").format(
                             self.input_arg_to_use,
                             self.input_arg_to_use_name))
            else:
                # Otherwise return the index
                input_val = args[self.input_arg_to_use]

        if (self.needs_target):
            # Check if its set to a string
            if (isinstance(self.target_arg_to_use, str)):
                target_val = kwargs[self.target_arg_to_use]

            # If all arguments are set by name, then this can happen
            elif (self.target_arg_to_use >= len(args)):
                # Check for the name in kwargs
                if (self.target_arg_to_use_name in kwargs):
                    target_val = kwargs[self.target_arg_to_use_name]
                else:
                    raise IndexError((
                        "Specified target_arg idx: {}, and argument name: {}, "
                        "were not found in args or kwargs").format(
                            self.target_arg_to_use,
                            self.target_arg_to_use_name))
            else:
                # Otherwise return the index
                target_val = args[self.target_arg_to_use]

        return self_val, input_val, target_val


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

        self.has_setters = (self.set_output_type or self.set_output_dtype
                            or self.set_n_features_in)

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

    def has_setters_input(self):
        return self.set_output_type or self.set_n_features_in

    def has_setters_target(self):
        return self.set_output_dtype


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

        self.has_getters = (self.get_output_type or self.get_output_dtype)

    def do_getters_with_self_no_input(self, *, self_val):
        if (self.get_output_type):
            out_type = self_val.output_type

            if (out_type == "input"):
                out_type = self_val._input_type

            set_api_output_type(out_type)

        if (self.get_output_dtype):
            set_api_output_dtype(self_val._get_target_dtype())

    def do_getters_with_self(self, *, self_val, input_val):
        if (self.get_output_type):
            out_type = self_val._get_output_type(input_val)
            assert out_type is not None, \
                ("`get_output_type` is False but output_type could not "
                 "be determined from input_arg")
            set_api_output_type(out_type)

        if (self.get_output_dtype):
            set_api_output_dtype(self_val._get_target_dtype())

    def do_getters_no_self(self, *, input_val, target_val):
        if (self.get_output_type):
            assert input_val is not None, \
                "`get_output_type` is False but no input_arg detected"
            set_api_output_type(
                cuml.common.input_utils.determine_array_type(input_val))

        if (self.get_output_dtype):
            assert target_val is not None, \
                "`get_output_dtype` is False but no target_arg detected"
            set_api_output_dtype(
                cuml.common.input_utils.determine_array_dtype(target_val))

    def has_getters_input(self):
        return self.get_output_type

    def has_getters_target(self, needs_self):
        return False if needs_self else self.get_output_dtype


class ReturnDecorator(metaclass=DecoratorMetaClass):
    def __init__(self):
        super().__init__()

        self.do_autowrap = False

    def __call__(self, func: _DecoratorType) -> _DecoratorType:
        raise NotImplementedError()

    def _recreate_cm(self, func, args) -> InternalAPIContextBase:
        raise NotImplementedError()


class ReturnAnyDecorator(ReturnDecorator):
    def __call__(self, func: _DecoratorType) -> _DecoratorType:
        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        return inner

    def _recreate_cm(self, func, args):
        return ReturnAnyCM(func, args)


class BaseReturnAnyDecorator(ReturnDecorator,
                             HasSettersDecoratorMixin,
                             WithArgsDecoratorMixin):
    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 set_output_type=True,
                 set_output_dtype=False,
                 set_n_features_in=True) -> None:

        ReturnDecorator.__init__(self)
        HasSettersDecoratorMixin.__init__(self,
                                          set_output_type=set_output_type,
                                          set_output_dtype=set_output_dtype,
                                          set_n_features_in=set_n_features_in)
        WithArgsDecoratorMixin.__init__(self,
                                        input_arg=input_arg,
                                        target_arg=target_arg,
                                        needs_self=True,
                                        needs_input=self.has_setters_input(),
                                        needs_target=self.has_setters_target())

        self.do_autowrap = self.has_setters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        self.prep_arg_to_use(func)

        @wraps(func)
        def inner_with_setters(*args, **kwargs):

            with self._recreate_cm(func, args):

                self_val, input_val, target_val = \
                    self.get_arg_values(*args, **kwargs)

                self.do_setters(self_val=self_val,
                                input_val=input_val,
                                target_val=target_val)

                return func(*args, **kwargs)

        @wraps(func)
        def inner(*args, **kwargs):

            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        # Return the function depending on whether or not we do any automatic
        # wrapping
        return inner_with_setters if self.has_setters else inner

    def _recreate_cm(self, func, args):
        return BaseReturnAnyCM(func, args)


class ReturnArrayDecorator(ReturnDecorator,
                           HasGettersDecoratorMixin,
                           WithArgsDecoratorMixin):
    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 get_output_type=False,
                 get_output_dtype=False) -> None:

        ReturnDecorator.__init__(self)
        HasGettersDecoratorMixin.__init__(self,
                                          get_output_type=get_output_type,
                                          get_output_dtype=get_output_dtype)
        WithArgsDecoratorMixin.__init__(
            self,
            input_arg=input_arg,
            target_arg=target_arg,
            needs_self=False,
            needs_input=self.has_getters_input(),
            needs_target=self.has_getters_target(False))

        self.do_autowrap = self.has_getters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        self.prep_arg_to_use(func)

        @wraps(func)
        def inner_with_getters(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                _, input_val, target_val = self.get_arg_values(*args, **kwargs)

                # Now execute the getters
                self.do_getters_no_self(input_val=input_val,
                                        target_val=target_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        return inner_with_getters if self.has_getters else inner

    def _recreate_cm(self, func, args):

        return ReturnArrayCM(func, args)


class ReturnSparseArrayDecorator(ReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return ReturnSparseArrayCM(func, args)


class BaseReturnArrayDecorator(ReturnDecorator,
                               HasSettersDecoratorMixin,
                               HasGettersDecoratorMixin,
                               WithArgsDecoratorMixin):
    def __init__(self,
                 *,
                 input_arg: str = ...,
                 target_arg: str = ...,
                 get_output_type=True,
                 get_output_dtype=False,
                 set_output_type=False,
                 set_output_dtype=False,
                 set_n_features_in=False) -> None:

        ReturnDecorator.__init__(self)
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
            needs_input=(self.has_setters_input() or self.has_getters_input())
            and input_arg is not None,
            needs_target=self.has_setters_target()
            or self.has_getters_target(True))

        self.do_autowrap = self.has_setters or self.has_getters

    def __call__(self, func: _DecoratorType) -> _DecoratorType:

        self.prep_arg_to_use(func)

        @wraps(func)
        def inner_set_get(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val, input_val, target_val = \
                    self.get_arg_values(*args, **kwargs)

                # Must do the setters first
                self.do_setters(self_val=self_val,
                                input_val=input_val,
                                target_val=target_val)

                # Now execute the getters
                if (self.needs_input):
                    self.do_getters_with_self(self_val=self_val,
                                              input_val=input_val)
                else:
                    self.do_getters_with_self_no_input(self_val=self_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner_set(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val, input_val, target_val = \
                    self.get_arg_values(*args, **kwargs)

                # Must do the setters first
                self.do_setters(self_val=self_val,
                                input_val=input_val,
                                target_val=target_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner_get(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                # Get input/target values
                self_val, input_val, _ = self.get_arg_values(*args, **kwargs)

                # Do the getters
                if (self.needs_input):
                    self.do_getters_with_self(self_val=self_val,
                                              input_val=input_val)
                else:
                    self.do_getters_with_self_no_input(self_val=self_val)

                # Call the function
                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

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
            return inner

    def _recreate_cm(self, func, args):

        return BaseReturnArrayCM(func, args)


class BaseReturnSparseArrayDecorator(BaseReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return BaseReturnSparseArrayCM(func, args)


class ReturnGenericDecorator(ReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return ReturnGenericCM(func, args)


class BaseReturnGenericDecorator(BaseReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return BaseReturnGenericCM(func, args)


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


def api_ignore(func: _DecoratorType) -> _DecoratorType:

    func.__dict__[CUML_WRAPPED_FLAG] = True

    return func


@contextlib.contextmanager
def exit_internal_api():

    assert (cuml.global_settings.root_cm is not None)

    try:
        old_root_cm = cuml.global_settings.root_cm

        cuml.global_settings.root_cm = None

        # Set the global output type to the previous value to pretend we never
        # entered the API
        with cuml.using_output_type(old_root_cm.prev_output_type):

            yield

    finally:
        cuml.global_settings.root_cm = old_root_cm


def mirror_args(
    wrapped: _DecoratorType,
    assigned=('__doc__', '__annotations__'),
    updated=functools.WRAPPER_UPDATES
) -> typing.Callable[[_DecoratorType], _DecoratorType]:
    return wraps(wrapped=wrapped, assigned=assigned, updated=updated)


@mirror_args(BaseReturnArrayDecorator)
def api_base_return_autoarray(*args, **kwargs):
    def inner(func: _DecoratorType) -> _DecoratorType:
        # Determine the array return type and choose
        return_type = _get_base_return_type(None, func)

        if (return_type == "generic"):
            func = api_base_return_generic(*args, **kwargs)(func)
        elif (return_type == "array"):
            func = api_base_return_array(*args, **kwargs)(func)
        elif (return_type == "sparsearray"):
            func = api_base_return_sparse_array(*args, **kwargs)(func)
        elif (return_type == "base"):
            assert False, \
                ("Must use api_base_return_autoarray decorator on function "
                 "that returns some array")

        return func

    return inner


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
