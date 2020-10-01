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

from abc import ABC, abstractmethod
import contextlib
import inspect
from inspect import FullArgSpec
import threading
import typing
from dataclasses import dataclass
from functools import wraps
from collections import deque

import cuml
import cuml.common
import cuml.common.array
import cuml.common.base
import cuml.common.input_utils
from cuml.internals.base_helpers import BaseFunctionMetadata
from cuml.common.input_utils import determine_array_type, is_array_like
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

    # Quick exit
    if (isinstance(output_type, str)):
        global_output_type_data.root_cm.output_type = output_type
        return

    # Try to convert any array objects to their type
    array_type = cuml.common.input_utils.determine_array_type(output_type)

    # Ensure that this is an array-like object
    assert output_type is None or array_type is not None

    global_output_type_data.root_cm.output_type = array_type


def set_api_output_dtype(output_dtype):
    assert (global_output_type_data.root_cm is not None)

    # Try to convert any array objects to their type
    if (output_dtype is not None and cuml.common.input_utils.is_array_like(output_dtype)):
        output_dtype = cuml.common.input_utils.determine_array_dtype(output_dtype)

        assert(output_dtype is not None)

    global_output_type_data.root_cm.output_dtype = output_dtype


# def cuml_internal_func(func):
#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         try:
#             # Increment the internal_func_counter
#             global_output_type_data.internal_func_count += 1

#             old_target_type = global_output_type_data.target_type
#             old_target_dtype = global_output_type_data.target_dtype

#             with cupy_using_allocator(rmm.rmm_cupy_allocator):
#                 with cuml.using_output_type("mirror") as prev_type:
#                     ret_val = func(*args, **kwargs)

#                     # Determine what the target type and dtype should be. Use
#                     # the non-None value from the lowest value in the stack
#                     global_output_type_data.target_type = (
#                         old_target_type if old_target_type is not None else
#                         global_output_type_data.target_type)
#                     global_output_type_data.target_dtype = (
#                         old_target_dtype if old_target_dtype is not None else
#                         global_output_type_data.target_dtype)

#                     if isinstance(
#                             ret_val, cuml.common.CumlArray
#                     ) and global_output_type_data.internal_func_count == 1:

#                         target_type = (global_output_type_data.target_type
#                                        if global_output_type_data.target_type
#                                        is not None else prev_type)

#                         if (target_type == "input"):

#                             # If we are on the Base object, get output_type
#                             if (len(args) > 0
#                                     and isinstance(args[0], cuml.Base)):
#                                 target_type = args[0].output_type

#                         return ret_val.to_output(
#                             output_type=target_type,
#                             output_dtype=global_output_type_data.target_dtype)
#                     else:
#                         return ret_val
#         finally:
#             global_output_type_data.internal_func_count -= 1

#             # On exiting the API, reset the target types
#             if (global_output_type_data.internal_func_count == 0):
#                 global_output_type_data.target_type = None
#                 global_output_type_data.target_dtype = None

#     return wrapped


# @contextlib.contextmanager
# def func_with_cumlarray_return():
#     try:
#         old_target_type = global_output_type_data.target_type
#         old_target_dtype = global_output_type_data.target_dtype

#         yield

#     finally:
#         global_output_type_data.target_type = (
#             old_target_type if old_target_type is not None else
#             global_output_type_data.target_type)
#         global_output_type_data.target_dtype = (
#             old_target_dtype if old_target_dtype is not None else
#             global_output_type_data.target_dtype)

class InternalAPIContext(contextlib.ExitStack):
    def __init__(self):
        super().__init__()

        def cleanup():
            global_output_type_data.root_cm = None

        self.callback(cleanup)

        self.enter_context(cupy_using_allocator(rmm.rmm_cupy_allocator))
        self.prev_output_type = self.enter_context(
            cuml.using_output_type("mirror"))

        self.output_type = None if self.prev_output_type == "input" else self.prev_output_type
        self.output_dtype = None

        self._count = 0

        self.call_stack = {}

        global_output_type_data.root_cm = self

    def pop_all(self):
        """Preserve the context stack by transferring it to a new instance."""
        new_stack = contextlib.ExitStack()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        return new_stack

    def __enter__(self) -> int:

        self._count += 1

        return self._count

    def __exit__(self, *exc_details):

        del self.call_stack[self._count]

        self._count -= 1

        return

    def push_func(self, func):

        self.call_stack[self._count] = func

    def get_current_func(self):

        if (self._count in self.call_stack):
            return self.call_stack[self._count]

        return None

    @contextlib.contextmanager
    def push_output_types(self):
        try:
            old_output_type = self.output_type
            old_output_dtype = self.output_dtype

            self.output_type = None
            self.output_dtype = None

            yield

        finally:
            self.output_type = (old_output_type if old_output_type is not None
                                else self.output_type)
            self.output_dtype = (old_output_dtype if old_output_dtype
                                 is not None else self.output_dtype)


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

        # self.output_type = None
        # self.output_dtype = None

        self.old_output_type = None
        self.old_output_dtype = None

    def __enter__(self):

        # Call super to ensure we get any root callbacks
        super().__enter__()

        # self.old_output_type = None
        # self.old_output_dtype = None

        self.enter_context(self.root_cm.push_output_types())

        # Now return an object based on if we are root or not
        if (self.is_root):
            return RootCumlArrayReturnConverter()
        else:
            return CumlArrayReturnConverter()


class InternalAPIBaseWithReturnContextManager(
        InternalAPIWithReturnContextManager):
    def __init__(self, func, args, base_obj):

        # Check this before calling super().__init__(). We can detect if we are
        # root here
        super().__init__(func, args)

        self.base_obj = base_obj

    def __exit__(self, *exc_details):

        # Get a copy of the root_cm before calling exit
        root_cm = get_internal_context()

        super().__exit__(*exc_details)

        output_type = (root_cm.output_type if root_cm.output_type is not None
                       else root_cm.prev_output_type)

        if (output_type == "input"):
            output_type = self.base_obj.output_type

            set_api_output_type(output_type)


class CumlArrayReturnConverter(object):
    def process_return(self, ret_val):

        if (not isinstance(ret_val, cuml.common.CumlArray)):
            ret_val, _, _, _ = cuml.common.input_to_cuml_array(ret_val, order="K")

        return ret_val


class RootCumlArrayReturnConverter(CumlArrayReturnConverter):
    def __init__(self) -> None:
        # Save the context because this will need to function after the global root_cm is set to None
        self._root_cm = get_internal_context()

    def process_return(self, ret_val):

        # This ensures we are a CumlArray
        ret_val = super().process_return(ret_val)

        return ret_val.to_output(output_type=self._root_cm.output_type,
                                 output_dtype=self._root_cm.output_dtype)


class ProcessEnter(object):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__()

        self._context = context

        self._process_enter_cbs: typing.Deque[typing.Callable] = deque()

    def process_enter(self):

        for cb in self._process_enter_cbs:
            cb()


class ProcessReturn(object):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__()

        self._context = context

        self._process_return_cbs: typing.Deque[typing.Callable[
            [typing.Any], typing.Any]] = deque()

    def process_return(self, ret_val):

        for cb in self._process_return_cbs:
            ret_val = cb(ret_val)

        return ret_val


EnterT = typing.TypeVar("EnterT", bound=ProcessEnter)
ProcessT = typing.TypeVar("ProcessT", bound=ProcessReturn)


class InternalAPIContextBase(contextlib.ExitStack,
                             typing.Generic[EnterT, ProcessT]):

    ProcessEnter_Type: typing.Type[EnterT] = None
    ProcessReturn_Type: typing.Type[ProcessT] = None

    def __init__(self, func=None, args=None):
        super().__init__()

        self._func = func
        self._args = args

        self.root_cm = get_internal_context()

        self._enter_obj: ProcessEnter = self.ProcessEnter_Type(self)
        self._process_obj: ProcessReturn = None

    def __enter__(self):

        # Enter the root context to know if we are the root cm
        self.is_root = self.enter_context(self.root_cm) == 1

        self.root_cm.push_func(self._func)

        # If we are the first, push any callbacks from the root into this CM
        # If we are not the first, this will have no effect
        self.push(self.root_cm.pop_all())

        self._enter_obj.process_enter()

        # Now create the process functions since we know if we are root or not
        self._process_obj = self.ProcessReturn_Type(self)

        return super().__enter__()

    def process_return(self, ret_val):

        return self._process_obj.process_return(ret_val)

    def __class_getitem__(cls: typing.Type["InternalAPIContextBase"], params):

        param_names = [
            param.__name__ if hasattr(param, '__name__') else str(param)
            for param in params
        ]

        type_name = f'{cls.__name__}[{", ".join(param_names)}]'

        ns = {
            "ProcessEnter_Type": params[0],
            "ProcessReturn_Type": params[1],
        }

        return type(type_name, (cls, ), ns)


class ProcessEnterBaseMixin(ProcessEnter):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self.base_obj: cuml.Base = self._context._args[0]


class ProcessEnterReturnAny(ProcessEnterBaseMixin):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)


class ProcessEnterReturnArray(ProcessEnter):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self._process_enter_cbs.append(self.push_output_types)

    def push_output_types(self):

        self._context.enter_context(self._context.root_cm.push_output_types())


class ProcessEnterBaseReturnArray(ProcessEnterReturnArray,
                                  ProcessEnterBaseMixin):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        if (self._context.root_cm.prev_output_type == "input"):
            self._process_enter_cbs.append(self.base_output_type_callback)

    def base_output_type_callback(self):

        root_cm = self._context.root_cm

        def set_output_type():
            output_type = (root_cm.output_type if root_cm.output_type
                           is not None else root_cm.prev_output_type)

            if (output_type == "input"):
                output_type = self.base_obj.output_type

                set_api_output_type(output_type)

            assert (output_type != "mirror")

        self._context.callback(set_output_type)


class ProcessEnterBaseSetOutputTypes(ProcessEnterBaseMixin):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

    def set_output_types(self):
        pass


class ProcessReturnAny(ProcessReturn):
    pass


class ProcessReturnArray(ProcessReturn):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self._process_return_cbs.append(self.convert_to_cumlarray)

        if (self._context.is_root):
            self._process_return_cbs.append(self.convert_to_outputtype)

    def convert_to_cumlarray(self, ret_val):

        # Get the output type
        ret_val_type_str = determine_array_type(ret_val)

        # If we are a supported array and not already cuml, convert to cuml
        if (ret_val_type_str is not None and ret_val_type_str != "cuml"):
            ret_val, _, _, _ = cuml.common.input_to_cuml_array(ret_val, order="K")

        return ret_val

    def convert_to_outputtype(self, ret_val):
        return ret_val.to_output(
            output_type=self._context.root_cm.output_type,
            output_dtype=self._context.root_cm.output_dtype)


class ProcessReturnGeneric(ProcessReturnArray):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        # Clear the existing callbacks to allow processing one at a time
        self._single_array_cbs = self._process_return_cbs

        # Make a new queue
        self._process_return_cbs = deque()

        type_hints = typing.get_type_hints(self._context._func)
        gen_type = type_hints["return"]

        assert (isinstance(gen_type, typing._GenericAlias))

        # Add the right processing function based on the generic type
        if (gen_type.__origin__ is typing.Union):

            found_gen_type = None

            # If we are a Union, the supported types must only be either CumlArray, Tuple, Dict, or List
            for gen_arg in gen_type.__args__:

                if (isinstance(gen_arg, typing._GenericAlias)):
                    assert found_gen_type is None or found_gen_type == gen_arg

                    found_gen_type = gen_arg
                else:
                    assert issubclass(gen_arg, cuml.common.CumlArray)

            assert found_gen_type is not None

            self._process_return_cbs.append(self.process_generic)

        elif (gen_type.__origin__ is tuple):
            self._process_return_cbs.append(self.process_tuple)
        elif (gen_type.__origin__ is dict):
            self._process_return_cbs.append(self.process_dict)
        elif (gen_type.__origin__ is list):
            self._process_return_cbs.append(self.process_list)
        else:
            raise NotImplementedError("Unsupported origin type: {}".format(
                gen_type.__origin__))

    def process_single(self, ret_val):
        for cb in self._single_array_cbs:
            ret_val = cb(ret_val)

        return ret_val

    def process_tuple(self, ret_val: tuple):

        # Convert to a list
        out_val = list(ret_val)

        for idx, item in enumerate(out_val):

            out_val[idx] = self.process_generic(item)

        return tuple(out_val)

    def process_dict(self, ret_val):

        return ret_val

    def process_list(self, ret_val):

        for idx, item in enumerate(ret_val):

            ret_val[idx] = self.process_generic(item)

        return ret_val

    def process_generic(self, ret_val):

        if (is_array_like(ret_val)):
            return self.process_single(ret_val)

        if (isinstance(ret_val, tuple)):
            return self.process_tuple(ret_val)

        if (isinstance(ret_val, dict)):
            return self.process_dict(ret_val)

        if (isinstance(ret_val, list)):
            return self.process_list(ret_val)

        return ret_val


class ReturnAnyCM(InternalAPIContextBase[ProcessEnterReturnAny,
                                         ProcessReturnAny]):
    pass


class ReturnArrayCM(InternalAPIContextBase[ProcessEnterReturnArray,
                                           ProcessReturnArray]):
    pass


class BaseReturnAnyCM(InternalAPIContextBase[ProcessEnterReturnAny,
                                             ProcessReturnAny]):
    pass


class BaseReturnArrayCM(InternalAPIContextBase[ProcessEnterBaseReturnArray,
                                               ProcessReturnArray]):
    pass


class BaseReturnGenericCM(InternalAPIContextBase[ProcessEnterBaseReturnArray,
                                                 ProcessReturnGeneric]):
    pass


def get_internal_context() -> InternalAPIContext:
    if (global_output_type_data.root_cm is None):
        return InternalAPIContext()

    return global_output_type_data.root_cm


# def cuml_internal_func_check_type(func):
#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         with cupy_using_allocator(rmm.rmm_cupy_allocator):
#             with cuml.using_output_type("mirror") as prev_type:
#                 ret_val = func(*args, **kwargs)

#                 if isinstance(ret_val, cuml.common.CumlArray):
#                     if (prev_type == "input"):

#                         if (len(args) > 0 and isinstance(args[0], cuml.Base)):
#                             prev_type = args[0].output_type

#                     return ret_val.to_output(prev_type)
#                 else:
#                     return ret_val

#     return wrapped


def api_ignore(func: typing.Callable):

    func.__dict__["__cuml_is_wrapped"] = True

    return func


# def autowrap_return_self(func):

#     func_dict: BaseFunctionMetadata = typing.cast(
#         dict, func.__dict__).setdefault(BaseFunctionMetadata.func_dict_str,
#                                         BaseFunctionMetadata())

#     func_dict.returns_self = True

#     return func


class DecoratorMetaClass(type):
    def __new__(meta, classname, bases, classDict):

        if ("__call__" in classDict):

            func = classDict["__call__"]

            @wraps(func)
            def wrap_call(*args, **kwargs):
                ret_val = func(*args, **kwargs)

                ret_val.__dict__["__cuml_is_wrapped"] = True

                return ret_val

            classDict["__call__"] = wrap_call

        return type.__new__(meta, classname, bases, classDict)


class InputArgDecoratorMixin(object):
    def __init__(self, input_arg: str = None, should_have_self=True):
        super().__init__()

        self.input_arg = input_arg
        self.should_have_self = should_have_self

    def prep_arg_to_use(self, func) -> bool:

        sig = inspect.signature(func, follow_wrapped=True)
        sig_args = list(sig.parameters.keys())

        self.has_self = "self" in sig.parameters and sig_args.index("self") == 0

        if (not self.has_self and self.should_have_self):
            raise Exception("No self found!")

        arg_to_use = self.input_arg
        arg_to_use_name = None

        self_offset = (1 if self.has_self else 0)

        # if input_arg is None, then set to first non self argument
        if (arg_to_use is None):

            if (len(sig.parameters) <= self_offset):
                return False

            arg_to_use = sig_args[self_offset]

        # Now convert that to an index
        if (isinstance(arg_to_use, str)):
            arg_to_use_name = arg_to_use
            arg_to_use = sig_args.index(arg_to_use)

        assert arg_to_use != -1

        self.arg_to_use = arg_to_use
        self.arg_to_use_name = arg_to_use_name

        return True

    def get_arg_value(self, *args, **kwargs):

        self_val = None

        if (self.has_self):
            self_val = args[0]

        # Check if its set to a string
        if (isinstance(self.arg_to_use, str)):
            return self_val, kwargs[self.arg_to_use]

        # If all arguments are set by name, then this can happen
        if (self.arg_to_use >= len(args)):
            # Check for the name in kwargs
            if (self.arg_to_use_name in kwargs):
                return self_val, kwargs[self.arg_to_use_name]

            raise IndexError(
                "Specified argument idx: {}, and argument name: {}, were not found in args or kwargs"
                .format(self.arg_to_use, self.arg_to_use_name))

        # Otherwise return the index
        return self_val, args[self.arg_to_use]


class ReturnDecorator(metaclass=DecoratorMetaClass):
    def __init__(self):
        super().__init__()

        self.do_autowrap = False

    def __call__(self, func) -> typing.Callable:
        raise NotImplementedError()

    def _recreate_cm(self, func, args) -> InternalAPIContextBase:
        raise NotImplementedError()


class ReturnAnyDecorator(ReturnDecorator):
    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        return inner

    def _recreate_cm(self, func, args):
        return ReturnAnyCM(func, args)


class BaseReturnAnyDecorator(ReturnAnyDecorator, InputArgDecoratorMixin):
    def __init__(self,
                 input_arg: str = None,
                 skip_set_output_type=False,
                 skip_set_output_dtype=True,
                 skip_set_n_features_in=False) -> None:

        ReturnAnyDecorator.__init__(self)
        InputArgDecoratorMixin.__init__(self, input_arg=input_arg)

        self.skip_set_output_type = skip_set_output_type
        self.skip_set_output_dtype = skip_set_output_dtype
        self.skip_set_n_features_in = skip_set_n_features_in
        self.has_setters = not (self.skip_set_output_type
                                and self.skip_set_output_dtype
                                and self.skip_set_n_features_in)
        self.do_autowrap = self.has_setters

    def __call__(self, func):

        if (self.do_autowrap):
            self.do_autowrap = self.prep_arg_to_use(func)

        @wraps(func)
        def inner(*args, **kwargs):

            self_val, arg_val = self.get_arg_value(*args, **kwargs)

            if (not self.skip_set_output_type):
                self_val._set_output_type(arg_val)

            if (not self.skip_set_output_dtype):
                self_val._set_target_dtype(arg_val)

            if (not self.skip_set_n_features_in):
                self_val._set_n_features_in(arg_val)

            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        @wraps(func)
        def inner_skip_autowrap(*args, **kwargs):

            with self._recreate_cm(func, args):
                return func(*args, **kwargs)

        # Return the function depending on whether or not we do any automatic wrapping
        return inner if self.do_autowrap else inner_skip_autowrap

    def _recreate_cm(self, func, args):
        return BaseReturnAnyCM(func, args)


class ReturnArrayDecorator(ReturnDecorator, InputArgDecoratorMixin):
    def __init__(self,
                 input_arg: str = None,
                 skip_get_output_type=True,
                 skip_get_output_dtype=True) -> None:

        ReturnDecorator.__init__(self)
        InputArgDecoratorMixin.__init__(self,
                                        input_arg=input_arg,
                                        should_have_self=False)

        self.skip_get_output_type = skip_get_output_type
        self.skip_get_output_dtype = skip_get_output_dtype

        self.has_getters = not (self.skip_get_output_type
                                and self.skip_get_output_dtype)

        self.do_autowrap = self.has_getters

    def __call__(self, func):

        if (self.do_autowrap):
            self.do_autowrap = self.prep_arg_to_use(func)

        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                _, arg_val = self.get_arg_value(*args, **kwargs)

                # Now execute the getters
                if (not self.skip_get_output_type):
                    set_api_output_type(
                        cuml.common.base._input_to_type(arg_val))

                if (not self.skip_get_output_dtype):
                    set_api_output_dtype(
                        cuml.common.base._input_target_to_dtype(arg_val))

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        @wraps(func)
        def inner_skip_autowrap(*args, **kwargs):
            with self._recreate_cm(func, args) as cm:

                ret_val = func(*args, **kwargs)

            return cm.process_return(ret_val)

        return inner if self.do_autowrap else inner_skip_autowrap

    def _recreate_cm(self, func, args):

        return ReturnArrayCM(func, args)


class BaseReturnArrayDecorator(ReturnArrayDecorator):
    def __init__(self,
                 input_arg: str = None,
                 skip_get_output_type=False,
                 skip_get_output_dtype=True,
                 skip_set_output_type=True,
                 skip_set_output_dtype=True,
                 skip_set_n_features_in=True) -> None:

        super().__init__(input_arg=input_arg,
                         skip_get_output_type=skip_get_output_type,
                         skip_get_output_dtype=skip_get_output_dtype)

        self.skip_set_output_type = skip_set_output_type
        self.skip_set_output_dtype = skip_set_output_dtype
        self.skip_set_n_features_in = skip_set_n_features_in

        self.has_setters = not (self.skip_set_output_type
                                and self.skip_set_output_dtype
                                and self.skip_set_n_features_in)

        self.do_autowrap = self.has_setters or self.has_getters

    def __call__(self, func):

        try:
            if (self.do_autowrap):
                self.do_autowrap = self.prep_arg_to_use(func)

            @wraps(func)
            def inner(*args, **kwargs):
                with self._recreate_cm(func, args) as cm:

                    self_val, arg_val = self.get_arg_value(*args, **kwargs)

                    # Must do the setters first
                    if (not self.skip_set_output_type):
                        self_val._set_output_type(arg_val)

                    if (not self.skip_set_output_dtype):
                        self_val._set_target_dtype(arg_val)

                    if (not self.skip_set_output_dtype):
                        self_val._set_n_features_in(arg_val)

                    # Now execute the getters
                    if (not self.skip_get_output_type):
                        set_api_output_type(self_val._get_output_type(arg_val))

                    if (not self.skip_get_output_dtype):
                        set_api_output_dtype(self_val._get_target_dtype())

                    ret_val = func(*args, **kwargs)

                return cm.process_return(ret_val)

            @wraps(func)
            def inner_no_setters(*args, **kwargs):
                with self._recreate_cm(func, args) as cm:

                    self_val, arg_val = self.get_arg_value(*args, **kwargs)

                    # Now execute the getters
                    if (not self.skip_get_output_type):
                        set_api_output_type(self_val._get_output_type(arg_val))

                    if (not self.skip_get_output_dtype):
                        set_api_output_dtype(self_val._get_target_dtype())

                    ret_val = func(*args, **kwargs)

                return cm.process_return(ret_val)

            @wraps(func)
            def inner_skip_autowrap(*args, **kwargs):
                with self._recreate_cm(func, args) as cm:

                    ret_val = func(*args, **kwargs)

                return cm.process_return(ret_val)

            # Return the function depending on whether or not we do any automatic wrapping
            if (not self.do_autowrap):
                return inner_skip_autowrap

            return inner if self.has_setters else inner_no_setters

        except Exception as ex:

            # TODO: Do we print? Assert?
            return super().__call__(func)

    def _recreate_cm(self, func, args):

        # TODO: Should we return just ReturnArrayCM if `do_autowrap` == False?
        return BaseReturnArrayCM(func, args)

# TODO: Static check the typings for valid values
class BaseReturnGenericDecorator(BaseReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return BaseReturnGenericCM(func, args)


class BaseReturnArrayFitTransformDecorator(BaseReturnArrayDecorator):
    """
    Identical to `BaseReturnArrayDecorator`, however the defaults have been
    changed to better suit `fit_transform` methods
    """
    def __init__(self,
                 input_arg: str = None,
                 skip_get_output_type=False,
                 skip_get_output_dtype=True,
                 skip_set_output_type=False,
                 skip_set_output_dtype=True,
                 skip_set_n_features_in=False) -> None:

        super().__init__(input_arg=input_arg,
                         skip_get_output_type=skip_get_output_type,
                         skip_get_output_dtype=skip_get_output_dtype,
                         skip_set_output_type=skip_set_output_type,
                         skip_set_output_dtype=skip_set_output_dtype,
                         skip_set_n_features_in=skip_set_n_features_in)


api_return_any = ReturnAnyDecorator
api_base_return_any = BaseReturnAnyDecorator
api_return_array = ReturnArrayDecorator
api_base_return_array = BaseReturnArrayDecorator
api_base_return_generic = BaseReturnGenericDecorator
api_base_fit_transform = BaseReturnArrayFitTransformDecorator

api_return_array_skipall = ReturnArrayDecorator(skip_get_output_dtype=True, skip_get_output_type=True)

api_base_return_any_skipall = BaseReturnAnyDecorator(
    skip_set_output_type=True, skip_set_n_features_in=True)
api_base_return_array_skipall = BaseReturnArrayDecorator(
    skip_get_output_type=True)
api_base_return_generic_skipall = BaseReturnGenericDecorator(
    skip_get_output_type=True)

@contextlib.contextmanager
def exit_internal_api(*args, **kwds):

    assert (global_output_type_data.root_cm is not None)

    try:
        old_root_cm = global_output_type_data.root_cm

        # TODO: DO we set the global output type here?
        global_output_type_data.root_cm = None

        yield

    finally:
        global_output_type_data.root_cm = old_root_cm
        

