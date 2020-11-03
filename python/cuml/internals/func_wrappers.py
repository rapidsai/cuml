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
import functools
import inspect
import threading
import typing
from collections import deque
from functools import wraps

import cuml
import cuml.common
import cuml.common.array
import cuml.common.array_sparse
import cuml.common.base
import cuml.common.input_utils
import rmm
# from cuml.common.array_outputable import ArrayOutputable
from cuml.common.input_utils import determine_array_type
from cuml.common.input_utils import determine_array_type_full
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import is_array_like
from cuml.internals.base_helpers import _get_base_return_type

try:
    from cupy.cuda import using_allocator as cupy_using_allocator
except ImportError:
    try:
        from cupy.cuda.memory import using_allocator as cupy_using_allocator
    except ImportError:
        pass

# Use _F as a type variable for decorators. See:
# https://github.com/python/mypy/pull/8336/files#diff-eb668b35b7c0c4f88822160f3ca4c111f444c88a38a3b9df9bb8427131538f9cR260
_F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])


@contextlib.contextmanager
def _using_mirror_output_type():
    prev_output_type = cuml.global_output_type
    try:
        cuml.global_output_type = "mirror"
        yield prev_output_type
    finally:
        cuml.global_output_type = prev_output_type


global_output_type_data = threading.local()
global_output_type_data.root_cm = None


def in_internal_api():
    return global_output_type_data.root_cm is not None


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
    if (output_dtype is not None
            and cuml.common.input_utils.is_array_like(output_dtype)):
        output_dtype = cuml.common.input_utils.determine_array_dtype(
            output_dtype)

        assert (output_dtype is not None)

    global_output_type_data.root_cm.output_dtype = output_dtype


class InternalAPIContext(contextlib.ExitStack):
    def __init__(self):
        super().__init__()

        def cleanup():
            global_output_type_data.root_cm = None

        self.callback(cleanup)

        self.enter_context(cupy_using_allocator(rmm.rmm_cupy_allocator))
        self.prev_output_type = self.enter_context(_using_mirror_output_type())

        self._output_type = None
        self.output_dtype = None

        # Set the output type to the prev_output_type. If "input", set to None
        # to allow inner functions to specify the input
        self.output_type = (None if self.prev_output_type == "input" else
                            self.prev_output_type)

        self._count = 0

        self.call_stack = {}

        global_output_type_data.root_cm = self

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value: str):
        self._output_type = value

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

        # del self.call_stack[self._count]

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

        self.is_root = False

        self._enter_obj: ProcessEnter = self.ProcessEnter_Type(self)
        self._process_obj: ProcessReturn = None

    def __enter__(self):

        # Enter the root context to know if we are the root cm
        self.is_root = self.enter_context(self.root_cm) == 1

        # self.root_cm.push_func(self._func)

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
    pass


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

        # IMPORTANT: Only perform output type processing if
        # `root_cm.output_type` is None. Since we default to using the incoming
        # value if its set, there is no need to do any processing if the user
        # has specified the output type
        if (self._context.root_cm.prev_output_type is None
                or self._context.root_cm.prev_output_type == "input"):
            self._process_enter_cbs.append(self.base_output_type_callback)

    def base_output_type_callback(self):

        root_cm = self._context.root_cm

        def set_output_type():
            output_type = (root_cm.output_type if root_cm.output_type
                           is not None else root_cm.prev_output_type)

            # TODO: (MDD) Determine why this fails only when all tests are run
            # and not when just a single test is run
            assert output_type == root_cm.output_type, \
                "MDD: Unclear why this is necessary. Calculated output_type: {}, root_cm.output_type: {}, root_cm.prev_output_type: {}".format(output_type, root_cm.output_type, root_cm.prev_output_type) # noqa

            # Check if output_type is None, can happen if no output type has
            # been set by estimator
            if (output_type is None):
                output_type = self.base_obj.output_type

            if (output_type == "input"):
                output_type = self.base_obj._input_type

            if (output_type != root_cm.output_type):
                set_api_output_type(output_type)

            assert (output_type != "mirror")

        self._context.callback(set_output_type)


class ProcessEnterBaseSetOutputTypes(ProcessEnterBaseMixin):
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
            ret_val = input_to_cuml_array(ret_val, order="K").array

        return ret_val

    def convert_to_outputtype(self, ret_val):

        # # TODO: Simple workaround for sparse arrays. Should not be released
        # if (not isinstance(ret_val, ArrayOutputable)):
        #     assert False, \
        #         "Must be array by this point. Obj: {}".format(ret_val)
        #     return ret_val

        assert (self._context.root_cm.output_type is not None
                and self._context.root_cm.output_type != "mirror"
                and self._context.root_cm.output_type != "input"), \
            ("Invalid root_cm.output_type: "
             "'{}'.").format(self._context.root_cm.output_type)

        return ret_val.to_output(
            output_type=self._context.root_cm.output_type,
            output_dtype=self._context.root_cm.output_dtype)


class ProcessReturnSparseArray(ProcessReturn):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self._process_return_cbs.append(self.convert_to_cumlarray)

        if (self._context.is_root):
            self._process_return_cbs.append(self.convert_to_outputtype)

    def convert_to_cumlarray(self, ret_val):

        # Get the output type
        ret_val_type_str, is_sparse = determine_array_type_full(ret_val)

        # If we are a supported array and not already cuml, convert to cuml
        if (ret_val_type_str is not None and ret_val_type_str != "cuml"):
            if is_sparse:
                ret_val = cuml.common.array_sparse.SparseCumlArray(
                    ret_val, convert_index=False)
            else:
                ret_val = input_to_cuml_array(ret_val, order="K").array

        return ret_val

    def convert_to_outputtype(self, ret_val):

        # # TODO: Simple workaround for sparse arrays. Should not be released
        # if (not isinstance(ret_val, ArrayOutputable)):
        #     return ret_val

        assert (self._context.root_cm.output_type is not None
                and self._context.root_cm.output_type != "mirror"
                and self._context.root_cm.output_type != "input")

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

            # If we are a Union, the supported types must only be either
            # CumlArray, Tuple, Dict, or List
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

        for name, item in ret_val.items():

            ret_val[name] = self.process_generic(item)

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


class ReturnSparseArrayCM(InternalAPIContextBase[ProcessEnterReturnArray,
                                                 ProcessReturnSparseArray]):
    pass


class ReturnGenericCM(InternalAPIContextBase[ProcessEnterReturnArray,
                                             ProcessReturnGeneric]):
    pass


class BaseReturnAnyCM(InternalAPIContextBase[ProcessEnterReturnAny,
                                             ProcessReturnAny]):
    pass


class BaseReturnArrayCM(InternalAPIContextBase[ProcessEnterBaseReturnArray,
                                               ProcessReturnArray]):
    pass


class BaseReturnSparseArrayCM(
        InternalAPIContextBase[ProcessEnterBaseReturnArray,
                               ProcessReturnSparseArray]):
    pass


class BaseReturnGenericCM(InternalAPIContextBase[ProcessEnterBaseReturnArray,
                                                 ProcessReturnGeneric]):
    pass


def get_internal_context() -> InternalAPIContext:
    if (global_output_type_data.root_cm is None):
        return InternalAPIContext()

    return global_output_type_data.root_cm


class DecoratorMetaClass(type):
    def __new__(cls, classname, bases, classDict):

        if ("__call__" in classDict):

            func = classDict["__call__"]

            @wraps(func)
            def wrap_call(*args, **kwargs):
                ret_val = func(*args, **kwargs)

                ret_val.__dict__["__cuml_is_wrapped"] = True

                return ret_val

            classDict["__call__"] = wrap_call

        return type.__new__(cls, classname, bases, classDict)


class WithArgsDecoratorMixin(object):
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

            self.target_arg_to_use = target_arg_to_use
            self.target_arg_to_use_name = target_arg_to_use_name

        return True

    def get_arg_values(self, *args, **kwargs):

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

    def __call__(self, func: _F) -> _F:
        raise NotImplementedError()

    def _recreate_cm(self, func, args) -> InternalAPIContextBase:
        raise NotImplementedError()


class ReturnAnyDecorator(ReturnDecorator):
    def __call__(self, func: _F) -> _F:
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

    def __call__(self, func: _F) -> _F:

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

    def __call__(self, func: _F) -> _F:

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

    def __call__(self, func: _F) -> _F:

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

        # TODO: Should we return just ReturnArrayCM if `do_autowrap` == False?
        return BaseReturnArrayCM(func, args)


class BaseReturnSparseArrayDecorator(BaseReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return BaseReturnSparseArrayCM(func, args)


# TODO: Static check the typings for valid values
class ReturnGenericDecorator(ReturnArrayDecorator):
    def _recreate_cm(self, func, args):

        return ReturnGenericCM(func, args)


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


def api_ignore(func: _F) -> _F:

    func.__dict__["__cuml_is_wrapped"] = True

    return func


@contextlib.contextmanager
def exit_internal_api():

    assert (global_output_type_data.root_cm is not None)

    try:
        old_root_cm = global_output_type_data.root_cm

        global_output_type_data.root_cm = None

        # TODO: DO we set the global output type here?
        with cuml.using_output_type(old_root_cm.prev_output_type):

            yield

    finally:
        global_output_type_data.root_cm = old_root_cm


def mirror_args(
        wrapped: _F,
        assigned=('__doc__', '__annotations__'),
        updated=functools.WRAPPER_UPDATES) -> typing.Callable[[_F], _F]:
    return wraps(wrapped=wrapped, assigned=assigned, updated=updated)


@mirror_args(BaseReturnArrayDecorator)
def api_base_return_autoarray(*args, **kwargs):
    def inner(func: _F) -> _F:
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
