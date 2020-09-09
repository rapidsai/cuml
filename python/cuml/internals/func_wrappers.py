from dataclasses import dataclass
import threading
from functools import wraps
import collections
from types import TracebackType
from typing import List, Optional, Type
import contextlib
import cuml
import cuml.common
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
    assert(global_output_type_data.root_cm is not None)
    
    global_output_type_data.root_cm.target_type = output_type


def set_api_target_dtype(target_dtype):
    assert(global_output_type_data.root_cm is not None)
    
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

                    # Determine what the target type and dtype should be. Use the non-None value from the lowest value in the stack
                    global_output_type_data.target_type = old_target_type if old_target_type is not None else global_output_type_data.target_type
                    global_output_type_data.target_dtype = old_target_dtype if old_target_dtype is not None else global_output_type_data.target_dtype

                    if isinstance(
                            ret_val, cuml.common.CumlArray
                    ) and global_output_type_data.internal_func_count == 1:

                        target_type = global_output_type_data.target_type if global_output_type_data.target_type is not None else prev_type

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
def internal_func_ret_base():
    try:
        old_target_type = global_output_type_data.target_type
        old_target_dtype = global_output_type_data.target_dtype

        yield

    finally:
        global_output_type_data.target_type = old_target_type if old_target_type is not None else global_output_type_data.target_type
        global_output_type_data.target_dtype = old_target_dtype if old_target_dtype is not None else global_output_type_data.target_dtype


class OutputTypeContextManager(contextlib.ExitStack):
    def __init__(self, func, args):
        super().__init__()

        self._func = func
        self._args = args

    def __enter__(self):

        if (len(self._args) > 0 and isinstance(self._args[0], cuml.Base)
                and self._args[0]._mirror_input):
            self.enter_context(global_output_type_data.root_cm.internal_func_ret_base())
        elif (global_output_type_data.root_cm.prev_output_type == "input"):
            self.enter_context(global_output_type_data.root_cm.internal_func_ret_base())

        return super().__enter__()

    # def __exit__(self,
    #              __exc_type: Optional[Type[BaseException]],
    #              __exc_value: Optional[BaseException],
    #              __traceback: Optional[TracebackType]) -> Optional[bool]:

    #     return False

    def return_to_output(self, ret_val):

        return ret_val


class RootOutputTypeContextManager(OutputTypeContextManager):
    def __init__(self, func, args):
        super().__init__(func, args)

        def cleanup():
            global_output_type_data.root_cm = None

        self.callback(cleanup)

        self.enter_context(cupy_using_allocator(rmm.rmm_cupy_allocator))
        self.prev_output_type = self.enter_context(
            cuml.using_output_type("mirror"))

        self.target_type = None
        self.target_dtype = None

        global_output_type_data.root_cm = self

    @contextlib.contextmanager
    def internal_func_ret_base(self):
        try:
            old_target_type = self.target_type
            old_target_dtype = self.target_dtype

            yield

        finally:
            self.target_type = old_target_type if old_target_type is not None else self.target_type
            self.target_dtype = old_target_dtype if old_target_dtype is not None else self.target_dtype


    def return_to_output(self, ret_val):

        if isinstance(ret_val, cuml.common.CumlArray):

            target_type = self.target_type if self.target_type is not None else self.prev_output_type

            if (target_type == "input"):

                # If we are on the Base object, get output_type
                if (len(self._args) > 0 and isinstance(self._args[0], cuml.Base)):
                    target_type = self._args[0].output_type

            return ret_val.to_output(
                output_type=target_type,
                output_dtype=self.target_dtype)
        else:
            return ret_val


def get_internal_cm(func, args, kwargs):

    if (global_output_type_data.root_cm is None):
        return RootOutputTypeContextManager(func, args)
    else:
        return OutputTypeContextManager(func, args)


def cuml_internal(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with get_internal_cm(func, args, kwargs) as cm:

            ret_val = func(*args, **kwargs)

        return cm.return_to_output(ret_val)

    return wrapped


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


def cuml_ignore_base_wrapper(func):

    func.__dict__["__cuml_do_not_wrap"] = True

    return func
