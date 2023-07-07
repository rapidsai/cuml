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
import typing
from collections import deque

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False

import cuml.internals.input_utils
import cuml.internals.memory_utils

from cuml.internals.array_sparse import SparseCumlArray

if TYPE_CHECKING:
    from cuml.internals.base import Base
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType
from cuml.internals.safe_imports import (
    gpu_only_import_from,
    UnavailableNullContext,
)

cupy_using_allocator = gpu_only_import_from(
    "cupy.cuda", "using_allocator", alt=UnavailableNullContext
)
rmm_cupy_allocator = gpu_only_import_from(
    "rmm.allocators.cupy", "rmm_cupy_allocator"
)


@contextlib.contextmanager
def _using_mirror_output_type():
    """
    Sets global_settings.output_type to "mirror" for internal API
    handling. We need a separate function since `cuml.using_output_type()`
    doesn't accept "mirror"

    Yields
    -------
    string
        Returns the previous value in global_settings.output_type
    """
    prev_output_type = GlobalSettings().output_type
    try:
        GlobalSettings().output_type = "mirror"
        yield prev_output_type
    finally:
        GlobalSettings().output_type = prev_output_type


def in_internal_api():
    return GlobalSettings().root_cm is not None


def set_api_output_type(output_type: str):
    assert GlobalSettings().root_cm is not None

    # Quick exit
    if isinstance(output_type, str):
        GlobalSettings().root_cm.output_type = output_type
        return

    # Try to convert any array objects to their type
    array_type = cuml.internals.input_utils.determine_array_type(output_type)

    # Ensure that this is an array-like object
    assert output_type is None or array_type is not None

    GlobalSettings().root_cm.output_type = array_type


def set_api_memory_type(mem_type):
    assert GlobalSettings().root_cm is not None

    try:
        mem_type = MemoryType.from_str(mem_type)
    except ValueError:
        mem_type = cuml.internals.memory_utils.determine_array_memtype(
            mem_type
        )

    GlobalSettings().root_cm.memory_type = mem_type


def set_api_output_dtype(output_dtype):
    assert GlobalSettings().root_cm is not None

    # Try to convert any array objects to their type
    if output_dtype is not None and cuml.internals.input_utils.is_array_like(
        output_dtype
    ):
        output_dtype = cuml.internals.input_utils.determine_array_dtype(
            output_dtype
        )

        assert output_dtype is not None

    GlobalSettings().root_cm.output_dtype = output_dtype


class InternalAPIContext(contextlib.ExitStack):
    def __init__(self):
        super().__init__()

        def cleanup():
            GlobalSettings().root_cm = None

        self.callback(cleanup)

        self.enter_context(cupy_using_allocator(rmm_cupy_allocator))
        self.prev_output_type = self.enter_context(_using_mirror_output_type())

        self._output_type = None
        self._memory_type = None
        self.output_dtype = None

        # Set the output type to the prev_output_type. If "input", set to None
        # to allow inner functions to specify the input
        self.output_type = (
            None if self.prev_output_type == "input" else self.prev_output_type
        )

        self._count = 0

        self.call_stack = {}

        GlobalSettings().root_cm = self

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value: str):
        self._output_type = value

    @property
    def memory_type(self):
        return self._memory_type

    @memory_type.setter
    def memory_type(self, value):
        self._memory_type = MemoryType.from_str(value)

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

        self._count -= 1

        return

    @contextlib.contextmanager
    def push_output_types(self):
        try:
            old_output_type = self.output_type
            old_output_dtype = self.output_dtype

            self.output_type = None
            self.output_dtype = None

            yield

        finally:
            self.output_type = (
                old_output_type
                if old_output_type is not None
                else self.output_type
            )
            self.output_dtype = (
                old_output_dtype
                if old_output_dtype is not None
                else self.output_dtype
            )


def get_internal_context() -> InternalAPIContext:
    """Return the current "root" context manager used to control output type
    for external API calls and minimize unnecessary internal output
    conversions"""

    if GlobalSettings().root_cm is None:
        GlobalSettings().root_cm = InternalAPIContext()

    return GlobalSettings().root_cm


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

        self._process_return_cbs: typing.Deque[
            typing.Callable[[typing.Any], typing.Any]
        ] = deque()

    def process_return(self, ret_val):

        for cb in self._process_return_cbs:
            ret_val = cb(ret_val)

        return ret_val


EnterT = typing.TypeVar("EnterT", bound=ProcessEnter)
ProcessT = typing.TypeVar("ProcessT", bound=ProcessReturn)


class InternalAPIContextBase(
    contextlib.ExitStack, typing.Generic[EnterT, ProcessT]
):

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
            param.__name__ if hasattr(param, "__name__") else str(param)
            for param in params
        ]

        type_name = f'{cls.__name__}[{", ".join(param_names)}]'

        ns = {
            "ProcessEnter_Type": params[0],
            "ProcessReturn_Type": params[1],
        }

        return type(type_name, (cls,), ns)


class ProcessEnterBaseMixin(ProcessEnter):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self.base_obj: Base = self._context._args[0]


class ProcessEnterReturnAny(ProcessEnter):
    pass


class ProcessEnterReturnArray(ProcessEnter):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self._process_enter_cbs.append(self.push_output_types)

    def push_output_types(self):

        self._context.enter_context(self._context.root_cm.push_output_types())


class ProcessEnterBaseReturnArray(
    ProcessEnterReturnArray, ProcessEnterBaseMixin
):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        # IMPORTANT: Only perform output type processing if
        # `root_cm.output_type` is None. Since we default to using the incoming
        # value if its set, there is no need to do any processing if the user
        # has specified the output type
        if (
            self._context.root_cm.prev_output_type is None
            or self._context.root_cm.prev_output_type == "input"
        ):
            self._process_enter_cbs.append(self.base_output_type_callback)

    def base_output_type_callback(self):

        root_cm = self._context.root_cm

        def set_output_type():
            output_type = root_cm.output_type
            mem_type = root_cm.memory_type

            # Check if output_type is None, can happen if no output type has
            # been set by estimator
            if output_type is None:
                output_type = self.base_obj.output_type
            if mem_type is None:
                mem_type = self.base_obj.output_mem_type

            if output_type == "input":
                output_type = self.base_obj._input_type
                mem_type = self.base_obj._input_mem_type

            if mem_type is None:
                mem_type = GlobalSettings().memory_type

            if output_type != root_cm.output_type:
                set_api_output_type(output_type)
            if mem_type != root_cm.memory_type:
                set_api_memory_type(mem_type)

            assert output_type != "mirror"

        self._context.callback(set_output_type)


class ProcessReturnAny(ProcessReturn):
    pass


class ProcessReturnArray(ProcessReturn):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        self._process_return_cbs.append(self.convert_to_cumlarray)

        if self._context.is_root or GlobalSettings().output_type != "mirror":
            self._process_return_cbs.append(self.convert_to_outputtype)

    def convert_to_cumlarray(self, ret_val):

        # Get the output type
        (
            ret_val_type_str,
            is_sparse,
        ) = cuml.internals.input_utils.determine_array_type_full(ret_val)

        # If we are a supported array and not already cuml, convert to cuml
        if ret_val_type_str is not None and ret_val_type_str != "cuml":
            if is_sparse:
                ret_val = SparseCumlArray(
                    ret_val,
                    convert_to_mem_type=GlobalSettings().memory_type,
                    convert_index=False,
                )
            else:
                ret_val = cuml.internals.input_utils.input_to_cuml_array(
                    ret_val,
                    convert_to_mem_type=GlobalSettings().memory_type,
                    order="K",
                ).array

        return ret_val

    def convert_to_outputtype(self, ret_val):

        output_type = GlobalSettings().output_type
        memory_type = GlobalSettings().memory_type

        if (
            output_type is None
            or output_type == "mirror"
            or output_type == "input"
        ):
            output_type = self._context.root_cm.output_type

        if GlobalSettings().memory_type in (None, MemoryType.mirror):
            memory_type = self._context.root_cm.memory_type

        assert (
            output_type is not None
            and output_type != "mirror"
            and output_type != "input"
        ), ("Invalid root_cm.output_type: " "'{}'.").format(output_type)

        return ret_val.to_output(
            output_type=output_type,
            output_dtype=self._context.root_cm.output_dtype,
            output_mem_type=memory_type,
        )


class ProcessReturnSparseArray(ProcessReturnArray):
    def convert_to_cumlarray(self, ret_val):

        # Get the output type
        (
            ret_val_type_str,
            is_sparse,
        ) = cuml.internals.input_utils.determine_array_type_full(ret_val)

        # If we are a supported array and not already cuml, convert to cuml
        if ret_val_type_str is not None and ret_val_type_str != "cuml":
            if is_sparse:
                ret_val = SparseCumlArray(
                    ret_val,
                    convert_to_mem_type=GlobalSettings().memory_type,
                    convert_index=False,
                )
            else:
                ret_val = cuml.internals.input_utils.input_to_cuml_array(
                    ret_val,
                    convert_to_mem_type=GlobalSettings().memory_type,
                    order="K",
                ).array

        return ret_val


class ProcessReturnGeneric(ProcessReturnArray):
    def __init__(self, context: "InternalAPIContextBase"):
        super().__init__(context)

        # Clear the existing callbacks to allow processing one at a time
        self._single_array_cbs = self._process_return_cbs

        # Make a new queue
        self._process_return_cbs = deque()

        self._process_return_cbs.append(self.process_generic)

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

        if cuml.internals.input_utils.is_array_like(ret_val):
            return self.process_single(ret_val)

        if isinstance(ret_val, tuple):
            return self.process_tuple(ret_val)

        if isinstance(ret_val, dict):
            return self.process_dict(ret_val)

        if isinstance(ret_val, list):
            return self.process_list(ret_val)

        return ret_val


class ReturnAnyCM(
    InternalAPIContextBase[ProcessEnterReturnAny, ProcessReturnAny]
):
    pass


class ReturnArrayCM(
    InternalAPIContextBase[ProcessEnterReturnArray, ProcessReturnArray]
):
    pass


class ReturnSparseArrayCM(
    InternalAPIContextBase[ProcessEnterReturnArray, ProcessReturnSparseArray]
):
    pass


class ReturnGenericCM(
    InternalAPIContextBase[ProcessEnterReturnArray, ProcessReturnGeneric]
):
    pass


class BaseReturnAnyCM(
    InternalAPIContextBase[ProcessEnterReturnAny, ProcessReturnAny]
):
    pass


class BaseReturnArrayCM(
    InternalAPIContextBase[ProcessEnterBaseReturnArray, ProcessReturnArray]
):
    pass


class BaseReturnSparseArrayCM(
    InternalAPIContextBase[
        ProcessEnterBaseReturnArray, ProcessReturnSparseArray
    ]
):
    pass


class BaseReturnGenericCM(
    InternalAPIContextBase[ProcessEnterBaseReturnArray, ProcessReturnGeneric]
):
    pass
