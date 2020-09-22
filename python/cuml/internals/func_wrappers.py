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

import contextlib
import threading
import typing
from dataclasses import dataclass
from functools import wraps

import cuml
import cuml.common
from cuml.internals.base_helpers import BaseFunctionMetadata
import rmm

try:
    from cupy.cuda import using_allocator as cupy_using_allocator
except ImportError:
    try:
        from cupy.cuda.memory import using_allocator as cupy_using_allocator
    except ImportError:
        pass


@dataclass
class TempOutputState:
    output_type: str = None
    target_dtype: str = None


global_output_type_data = threading.local()

global_output_type_data.internal_func_count = 0
global_output_type_data.target_type = None
global_output_type_data.target_dtype = None
global_output_type_data.target_stack = []
global_output_type_data.root_cm = None


def set_api_output_type(output_type: str):
    assert (global_output_type_data.root_cm is not None)

    global_output_type_data.root_cm.target_type = output_type


def set_api_target_dtype(target_dtype):
    assert (global_output_type_data.root_cm is not None)

    global_output_type_data.root_cm.target_dtype = target_dtype


def cuml_internal_func(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            # Increment the internal_func_counter
            global_output_type_data.internal_func_count += 1

            old_target_type = global_output_type_data.target_type
            old_target_dtype = global_output_type_data.target_dtype

            with cupy_using_allocator(rmm.rmm_cupy_allocator):
                with cuml.using_output_type("mirror") as prev_type:
                    ret_val = func(*args, **kwargs)

                    # Determine what the target type and dtype should be. Use
                    # the non-None value from the lowest value in the stack
                    global_output_type_data.target_type = (
                        old_target_type if old_target_type is not None else
                        global_output_type_data.target_type)
                    global_output_type_data.target_dtype = (
                        old_target_dtype if old_target_dtype is not None else
                        global_output_type_data.target_dtype)

                    if isinstance(
                            ret_val, cuml.common.CumlArray
                    ) and global_output_type_data.internal_func_count == 1:

                        target_type = (global_output_type_data.target_type
                                       if global_output_type_data.target_type
                                       is not None else prev_type)

                        if (target_type == "input"):

                            # If we are on the Base object, get output_type
                            if (len(args) > 0
                                    and isinstance(args[0], cuml.Base)):
                                target_type = args[0].output_type

                        return ret_val.to_output(
                            output_type=target_type,
                            output_dtype=global_output_type_data.target_dtype)
                    else:
                        return ret_val
        finally:
            global_output_type_data.internal_func_count -= 1

            # On exiting the API, reset the target types
            if (global_output_type_data.internal_func_count == 0):
                global_output_type_data.target_type = None
                global_output_type_data.target_dtype = None

    return wrapped


@contextlib.contextmanager
def func_with_cumlarray_return():
    try:
        old_target_type = global_output_type_data.target_type
        old_target_dtype = global_output_type_data.target_dtype

        yield

    finally:
        global_output_type_data.target_type = (
            old_target_type if old_target_type is not None else
            global_output_type_data.target_type)
        global_output_type_data.target_dtype = (
            old_target_dtype if old_target_dtype is not None else
            global_output_type_data.target_dtype)


class InternalAPIContext(contextlib.ExitStack):
    def __init__(self):
        super().__init__()

        def cleanup():
            global_output_type_data.root_cm = None

        self.callback(cleanup)

        self.enter_context(cupy_using_allocator(rmm.rmm_cupy_allocator))
        self.prev_output_type = self.enter_context(
            cuml.using_output_type("mirror"))

        self.target_type = self.prev_output_type if self.prev_output_type == "input" else None
        self.target_dtype = None

        self._count = 0

        global_output_type_data.root_cm = self

    def __enter__(self) -> int:

        self._count += 1

        return self._count

    def __exit__(self, *exc_details):

        self._count -= 1

        return

    @contextlib.contextmanager
    def push_target_types(self):
        try:
            old_target_type = self.target_type
            old_target_dtype = self.target_dtype

            self.target_type = None
            self.old_target_dtype = None

            yield

        finally:
            self.target_type = (old_target_type if old_target_type is not None
                                else self.target_type)
            self.target_dtype = (old_target_dtype if old_target_dtype
                                 is not None else self.target_dtype)


class InternalAPIContextManager(contextlib.ExitStack):
    def __init__(self, func, args):
        super().__init__()

        self._func = func
        self._args = args

        self.root_cm = get_internal_context()

    def __enter__(self):

        # Enter the root context to know if we are the root cm
        self.is_root = self.enter_context(self.root_cm) == 1

        # If we are the first, push any callbacks from the root into this CM
        # If we are not the first, this will have no effect
        self.push(self.root_cm.pop_all())

        # if (len(self._args) > 0 and isinstance(self._args[0], cuml.Base)
        #         and self._args[0]._mirror_input):
        #     self.enter_context(
        #         global_output_type_data.root_cm.internal_func_ret_base())
        # elif (global_output_type_data.root_cm.prev_output_type == "input"):
        #     self.enter_context(
        #         global_output_type_data.root_cm.internal_func_ret_base())

        return super().__enter__()

    # def __exit__(self,
    #              __exc_type: Optional[Type[BaseException]],
    #              __exc_value: Optional[BaseException],
    #              __traceback: Optional[TracebackType]) -> Optional[bool]:

    #     return False

    def process_return(self, ret_val):

        return ret_val


class InternalAPIWithReturnContextManager(InternalAPIContextManager):
    def __init__(self, func, args):

        # Check this before calling super().__init__(). We can detect if we are
        # root here
        super().__init__(func, args)

        # self.target_type = None
        # self.target_dtype = None

        self.old_target_type = None
        self.old_target_dtype = None

    def __enter__(self):

        # Call super to ensure we get any root callbacks
        super().__enter__()

        # self.old_target_type = None
        # self.old_target_dtype = None

        self.enter_context(self.root_cm.push_target_types())

        # Now return an object based on if we are root or not
        if (self.is_root):
            return RootCumlArrayReturnConverter()
        else:
            return CumlArrayReturnConverter()

    # def __exit__(self, *exc_details) -> Optional[bool]:

    #     return


class InternalAPIBaseWithReturnContextManager(
        InternalAPIWithReturnContextManager):
    def __init__(self, func, args, base_obj):

        # Check this before calling super().__init__(). We can detect if we are
        # root here
        super().__init__(func, args)

        self.base_obj = base_obj

    def __exit__(self, *exc_details):

        root_cm = get_internal_context()

        target_type = (root_cm.target_type if root_cm.target_type is not None
                       else root_cm.prev_output_type)

        if (target_type == "input"):
            target_type = self.base_obj.output_type

            set_api_output_type(target_type)


class CumlArrayReturnConverter(object):
    def process_return(self, ret_val):

        if (not isinstance(ret_val, cuml.common.CumlArray)):
            ret_val, _, _, _ = cuml.common.input_to_cuml_array(ret_val, order="K")

        return ret_val


class RootCumlArrayReturnConverter(CumlArrayReturnConverter):
    def process_return(self, ret_val):

        # This ensures we are a CumlArray
        ret_val = super().process_return(ret_val)

        return ret_val.to_output(
            output_type=global_output_type_data.root_cm.target_type,
            output_dtype=global_output_type_data.root_cm.target_dtype)



# class RootOutputTypeContextManager(OutputTypeContextManager):
#     def __init__(self, func, args):
#         super().__init__(func, args, self)

#         def cleanup():
#             global_output_type_data.root_cm = None

#         self.callback(cleanup)

#         self.enter_context(cupy_using_allocator(rmm.rmm_cupy_allocator))
#         self.prev_output_type = self.enter_context(
#             cuml.using_output_type("mirror"))

#         self.target_type = None
#         self.target_dtype = None

#         global_output_type_data.root_cm = self

#     @contextlib.contextmanager
#     def internal_func_ret_base(self):
#         try:
#             old_target_type = self.target_type
#             old_target_dtype = self.target_dtype

#             yield

#         finally:
#             self.target_type = (old_target_type if old_target_type is not None
#                                 else self.target_type)
#             self.target_dtype = (old_target_dtype if old_target_dtype
#                                  is not None else self.target_dtype)

#     def process_return(self, ret_val):

#         if isinstance(ret_val, cuml.common.CumlArray):

#             target_type = (self.target_type if self.target_type is not None
#                            else self.prev_output_type)

#             if (target_type == "input"):

#                 # If we are on the Base object, get output_type
#                 if (len(self._args) > 0
#                         and isinstance(self._args[0], cuml.Base)):
#                     target_type = self._args[0].output_type

#             return ret_val.to_output(output_type=target_type,
#                                      output_dtype=self.target_dtype)
#         else:
#             return ret_val


def get_internal_context() -> InternalAPIContext:
    if (global_output_type_data.root_cm is None):
        return InternalAPIContext()

    return global_output_type_data.root_cm


# def get_internal_cm(func: typing.Callable, args) -> OutputTypeContextManager:

#     internal_context = get_internal_context()

#     return internal_context.create_cm(func, args)

#     # if (global_output_type_data.root_cm is None):
#     #     return RootOutputTypeContextManager(func, args)
#     # elif (func.__dict__.get("__cuml_return_array", False)):
#     #     return OutputTypeContextManager(func,
#     #                                     args,
#     #                                     global_output_type_data.root_cm)
#     # else:
#     #     return InternalAPIContextManager(func,
#     #                                      args,
#     #                                      global_output_type_data.root_cm)

# def cuml_internal_func(func):
#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         with get_internal_cm(func, args) as cm:

#             ret_val = func(*args, **kwargs)

#         return cm.process_return(ret_val)

#     return wrapped

# class cuml_internal_func_decorator(object):
#     "A base class or mixin that enables context managers to work as decorators."

#     def __init__(self,
#                  input_arg: str = None,
#                  skip_output_type=False,
#                  skip_target_dtype=True) -> None:

#         self.input_arg = input_arg
#         self.skip_output_type = skip_output_type
#         self.skip_target_dtype = skip_target_dtype

#     def _recreate_cm(self):
#         """Return a recreated instance of self.

#         Allows an otherwise one-shot context manager like
#         _GeneratorContextManager to support use as
#         a decorator via implicit recreation.

#         This is a private interface just for _GeneratorContextManager.
#         See issue #11647 for details.
#         """
#         return self

#     def __call__(self, func):
#         @wraps(func)
#         def inner(*args, **kwds):
#             with self._recreate_cm():
#                 return func(*args, **kwds)

#         return inner


def cuml_internal_func_check_type(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with cupy_using_allocator(rmm.rmm_cupy_allocator):
            with cuml.using_output_type("mirror") as prev_type:
                ret_val = func(*args, **kwargs)

                if isinstance(ret_val, cuml.common.CumlArray):
                    if (prev_type == "input"):

                        if (len(args) > 0 and isinstance(args[0], cuml.Base)):
                            prev_type = args[0].output_type

                    return ret_val.to_output(prev_type)
                else:
                    return ret_val

    return wrapped


def autowrap_ignore(func: typing.Callable):

    func_dict: BaseFunctionMetadata = typing.cast(
        dict, func.__dict__).setdefault(BaseFunctionMetadata.func_dict_str,
                                        BaseFunctionMetadata())

    func_dict.ignore = True

    return func


def autowrap_return_self(func):

    func_dict: BaseFunctionMetadata = typing.cast(
        dict, func.__dict__).setdefault(BaseFunctionMetadata.func_dict_str,
                                        BaseFunctionMetadata())

    func_dict.returns_self = True

    return func


# def autowrap_return_cumlarray(input_arg: str = None,
#                               skip_output_type=False,
#                               skip_target_dtype=True):
#     def inner(func):
#         @wraps(func)
#         def wrapped(*args, **kwargs):

#             with get_internal_cm(func, args) as cm:

#                 input_arg_val = None

#                 if (input_arg is not None):
#                     input_arg_val = kwargs[input_arg]
#                 else:
#                     # By default, use the first parameter after self
#                     input_arg_val = args[1]

#                 func_self: cuml.Base = args[0]

#                 # Defaults
#                 output_type = None
#                 target_dtype = None

#                 if (not skip_output_type):
#                     output_type = func_self._get_output_type(input_arg_val)

#                 if (not skip_target_dtype):
#                     target_dtype = func_self._get_target_dtype()

#                 ret_val = func(*args, **kwargs)

#                 if (not isinstance(ret_val, cuml.common.CumlArray)):
#                     ret_val, _, _, _ = cuml.common.input_to_cuml_array(ret_val, order="K")

#             return typing.cast(OutputTypeContextManager,
#                                cm).process_return(ret_val)

#         return wrapped

#     return inner

# def autowrap_return_cumlarray(func):

#     func_dict: BaseFunctionMetadata = typing.cast(
#         dict, func.__dict__).setdefault(BaseFunctionMetadata.func_dict_str,
#                                         BaseFunctionMetadata())

#     func_dict.returns_cumlarray = True

#     return func


class ReturnAnyDecorator(object):
    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)

        return inner

    def _recreate_cm(self):
        return InternalAPIContextManager(None, None)


class BaseReturnArrayDecorator(ReturnAnyDecorator):
    def __init__(self,
                 input_arg: str = None,
                 skip_output_type=False,
                 skip_target_dtype=True) -> None:

        super().__init__()

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):

            with self._recreate_cm():
                return func(*args, **kwds)

        return inner


class ReturnArrayDecorator(object):
    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm() as cm:

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        return inner

    def _recreate_cm(self):

        return InternalAPIWithReturnContextManager(None, None)


class BaseReturnArrayDecorator(ReturnArrayDecorator):
    def __init__(self,
                 input_arg: str = None,
                 skip_output_type=False,
                 skip_target_dtype=True) -> None:

        super().__init__()

        self.input_arg = input_arg
        self.skip_output_type = skip_output_type
        self.skip_target_dtype = skip_target_dtype

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(args[0]) as cm:

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner_mirror_input(*args, **kwargs):
            with self._recreate_cm(args[0]) as cm:

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        return inner

    def _recreate_cm(self, base_obj=None):

        root_cm = get_internal_context()

        if (root_cm.prev_output_type == "input"):
            if (base_obj is not None and base_obj._mirror_input):
                return InternalAPIBaseWithReturnContextManager(
                    None, None, base_obj)

        return super()._recreate_cm()


wrap_api_return_any = ReturnAnyDecorator
wrap_api_base_return_any = ReturnAnyDecorator
api_return_array = ReturnArrayDecorator
api_base_return_array = BaseReturnArrayDecorator
